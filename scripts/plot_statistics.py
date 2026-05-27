import os
import sys
import glob
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─────────────────────────────────────────────────────────────────────────────
# Regex catalogue – every pattern corresponds to a log line added by our edits
# ─────────────────────────────────────────────────────────────────────────────
RE = {
    # "Simulation Finished at 12345 cycle 12 us"
    "total_cycles": re.compile(r"Simulation Finished at (\d+) cycle"),

    "core_util": re.compile(
        r"Core\s+\[(\d+)\]\s*:\s*"
        r"PE Utilization\(%\)\s+([\d.]+),\s*"
        r"Systolic Bubble\(%\)\s+([\d.]+),\s*"
        r"Vector Unit Utilization\(%\)\s+([\d.]+),\s*"
        r"Total cycle:\s+(\d+)"
    ),

    # "Core [0] : Memory unit idle cycle 1000 Systolic bubble cycle 500 Core idle cycle 200"
    "core_idle": re.compile(
        r"Core\s+\[(\d+)\].*?Memory unit idle cycle\s+(\d+)"
        r"\s+Systolic bubble cycle\s+(\d+)"
        r"\s+Core idle cycle\s+(\d+)"
    ),

    # "Runtime Instructions [123456] Avg Instructions Per Finished Tile [42.00]"
    "inst_per_tile": re.compile(
        r"Avg Instructions Per Finished Tile\s+\[([\d.]+)\]"
    ),


    "row_stats": re.compile(
        r"Row hits:\s+(\d+),\s+Row misses:\s+(\d+),\s+Row conflicts:\s+(\d+)"
    ),

    "mem_cycles": re.compile(
        r"memory_system_cycles:\s+(\d+)"
    ),

    "total_reads": re.compile(
        r"total_num_read_requests:\s+(\d+)"
    ),

    "total_writes": re.compile(
        r"total_num_write_requests:\s+(\d+)"
    ),

    "dram_avg_bw": re.compile(
        r"HBM2-CH_(\d+): avg BW utilization (\d+)%"
    ),

    # "[MEMORY_FOOTPRINT] Total Core->L2/DRAM Request Bytes: 123456789"
    "mem_bytes": re.compile(
        r"Total Core->L2/DRAM Request Bytes:\s+(\d+)"
    ),

    # "[Core 0] Input SRAM Hit Rate: 75.00% (1000 hits, 333 misses)"
    "input_sram": re.compile(
        r"\[Core\s+(\d+)\]\s+Input SRAM Hit Rate:\s+([\d.]+)%\s+\((\d+)\s+hits,\s+(\d+)\s+misses\)"
    ),

    # "[Core 0] Accumulator SRAM Hit Rate: 60.00% (800 hits, 533 misses)"
    "acc_sram": re.compile(
        r"\[Core\s+(\d+)\]\s+Accumulator SRAM Hit Rate:\s+([\d.]+)%\s+\((\d+)\s+hits,\s+(\d+)\s+misses\)"
    ),

    # "[Core 0] Average Core->L2 Request Size: 64.00 bytes"
    "avg_req_size": re.compile(
        r"\[Core\s+(\d+)\]\s+Average Core->L2 Request Size:\s+([\d.]+)\s+bytes"
    ),

    # "[Core 0] Object Histogram: Top-1 size 64 B (52.30%), Top-2 size 128 B (25.10%)"
    "obj_hist": re.compile(
        r"\[Core\s+\d+\]\s+Object Histogram: Top-1 size\s+(\d+)\s+B\s+\(([\d.]+)%\),"
        r"\s+Top-2 size\s+(\d+)\s+B\s+\(([\d.]+)%\)"
    ),

    # Layer finish lines (printed by scheduler):
    # "Layer p0.layer0.attn.QKVgen finish at 12345"  /  "Total compute time 200"
    "layer_finish": re.compile(r"Layer\s+([\w.\-]+)\s+finish at\s+(\d+)"),
    "compute_time": re.compile(r"Total compute time\s+(\d+)"),

    # "[ICNT] ICNT<-MEM {:.2f} GB/s" – DRAM read bandwidth
    "dram_read_bw": re.compile(r"ICNT\s*<\-\s*MEM\s+([\d.]+)\s+GB/s", re.IGNORECASE),
}

LAYER_BUCKETS = ["QKV Projection", "Attention", "Attn Projection", "FFN FC1", "FFN FC2"]

def classify_layer(name: str) -> str:
    n = name.lower()
    if "qkvgen" in n or "qkgen" in n or "vgen" in n:
        return "QKV Projection"
    if "attention" in n:
        return "Attention"
    if "attn.proj" in n or "attn.in" in n:
        return "Attn Projection"
    if "ffn.fc1" in n:
        return "FFN FC1"
    if "ffn.fc2" in n:
        return "FFN FC2"
    return "Other"


def parse_log(path: str) -> dict:
    """Parse a single ONNXim log file and return a flat dict of stats."""
    s = {
        "total_cycles": 0,
        "sys_util": 0.0,
        "pe_util": 0.0,
        "vec_util": 0.0,
        "mem_idle_raw": 0.0,
        "core_idle_raw": 0.0,
        "inst_per_tile": 0.0,
        "dram_util_pct": 0.0,
        "dram_bw_gbps": 0.0,
        "mem_bytes": 0,
        # summed over all cores
        "input_hits": 0, "input_misses": 0,
        "acc_hits": 0,   "acc_misses": 0,
        "avg_req_size": 0.0,
        "top1_share": 0.0, "top2_share": 0.0,
        # ramulator row stats (totalled over channels)
        "row_hits": 0, "row_misses": 0, "row_conflicts": 0,
        "dram_read_bw_peak": 0.0,
        # layer cycles dict
        **{b: 0 for b in LAYER_BUCKETS}, "Other": 0,
    }

    cur_layer = None
    core_util_seen = {}     # core_id -> dict
    core_idle_seen = {}
    dram_utils_seen = []
    inst_per_tile_vals = []
    req_sizes = []

    row_hits = 0
    row_misses = 0
    row_conflicts = 0

    total_reads = 0
    total_writes = 0

    mem_cycles = 0

    with open(path, "r", errors="replace") as f:
        for line in f:
            if m := RE["row_stats"].search(line):
                row_hits += int(m.group(1))
                row_misses += int(m.group(2))
                row_conflicts += int(m.group(3))

            if m := RE["mem_cycles"].search(line):
                mem_cycles = max(
                    mem_cycles,
                    int(m.group(1))
                )
            
            if m := RE["total_reads"].search(line):
                total_reads += int(m.group(1))
            
            if m := RE["total_writes"].search(line):
                total_writes += int(m.group(1))

            # ── Total cycles ──────────────────────────────────────────────
            if m := RE["total_cycles"].search(line):
                s["total_cycles"] = max(s["total_cycles"], int(m.group(1)))

            # ── Core utilisation ──────────────────────────────────────────
            if m := RE["core_util"].search(line):
                cid = int(m.group(1))
                core_util_seen[cid] = {
                    "pe":     float(m.group(2)),
                    "bubble": float(m.group(3)),
                    "vec":    float(m.group(4)),
                }
                

            # ── Core idle cycles ──────────────────────────────────────────
            if m := RE["core_idle"].search(line):
                cid = int(m.group(1))
                core_idle_seen[cid] = {
                    "mem":  int(m.group(2)),
                    "core": int(m.group(4)),
                }

            # ── Avg instr per tile ────────────────────────────────────────
            if m := RE["inst_per_tile"].search(line):
                inst_per_tile_vals.append(float(m.group(1)))


            # Derived hit rates# ── HBM Channel Utilization ──────────────────────────────────
            if m := RE["dram_avg_bw"].search(line):
                util = float(m.group(2))
                dram_utils_seen.append(util)

            # ── DRAM read bandwidth (ICNT<-MEM) ───────────────────────────
            if m := RE["dram_read_bw"].search(line):
                s["dram_read_bw_peak"] = max(s["dram_read_bw_peak"], float(m.group(1)))

            # ── Memory footprint ──────────────────────────────────────────
            if m := RE["mem_bytes"].search(line):
                s["mem_bytes"] = max(s["mem_bytes"], int(m.group(1)))

            # ── SRAM hit / miss ───────────────────────────────────────────
            if m := RE["input_sram"].search(line):
                s["input_hits"]   += int(m.group(3))
                s["input_misses"] += int(m.group(4))
            if m := RE["acc_sram"].search(line):
                s["acc_hits"]   += int(m.group(3))
                s["acc_misses"] += int(m.group(4))

            # ── Avg request size & object histogram ───────────────────────
            if m := RE["avg_req_size"].search(line):
                req_sizes.append(float(m.group(2)))
            if m := RE["obj_hist"].search(line):
                s["top1_share"] = max(s["top1_share"], float(m.group(2)))
                s["top2_share"] = max(s["top2_share"], float(m.group(4)))

            # ── Layer cycles ──────────────────────────────────────────────
            if m := RE["layer_finish"].search(line):
                cur_layer = m.group(1)
            if cur_layer and (m := RE["compute_time"].search(line)):
                bucket = classify_layer(cur_layer)
                s[bucket] += int(m.group(1))
                cur_layer = None

    if req_sizes:
        s["avg_req_size"] = np.mean(req_sizes)

    if inst_per_tile_vals:
        s["inst_per_tile"] = np.mean(inst_per_tile_vals)

    # Aggregate per-core util (use core 0 or average)
    if core_util_seen:
        for cid, vals in core_util_seen.items():
            # SA util = 100 - bubble
            sa_util = 100.0 - vals["bubble"]

            s[f"core{cid}_sys_util"] = sa_util
            s[f"core{cid}_pe_util"]  = vals["pe"]
            s[f"core{cid}_vec_util"] = vals["vec"]
            s[f"core{cid}_bubble"]   = vals["bubble"]

        s["sys_util"] = np.mean([
            100.0 - v["bubble"]
            for v in core_util_seen.values()
        ])

        s["pe_util"] = np.mean([
            v["pe"]
            for v in core_util_seen.values()
        ])

        s["vec_util"] = np.mean([
            v["vec"]
            for v in core_util_seen.values()
        ])

        s["bubble_util"] = np.mean([
            v["bubble"]
            for v in core_util_seen.values()
        ])

    if core_idle_seen and s["total_cycles"] > 0:
        avg_mem  = np.mean([v["mem"]  for v in core_idle_seen.values()])
        avg_core = np.mean([v["core"] for v in core_idle_seen.values()])

        for cid, vals in core_idle_seen.items():
            s[f"core{cid}_mem_idle_pct"] = (
            vals["mem"] / s["total_cycles"] * 100.0
            )
            s[f"core{cid}_core_idle_pct"] = (
                vals["core"] / s["total_cycles"] * 100.0
            )

        s["mem_idle_pct"]  = avg_mem  / s["total_cycles"] * 100.0
        s["core_idle_pct"] = avg_core / s["total_cycles"] * 100.0
    else:
        s["mem_idle_pct"]  = 0.0
        s["core_idle_pct"] = 0.0

    # Average DRAM utilization across all HBM channel snapshots
    if len(dram_utils_seen) > 0:
        s["dram_util_pct"] = np.mean(dram_utils_seen)
    else:
        s["dram_util_pct"] = 0.0

    # Average DRAM activity
    REQUEST_SIZE = 32       # bytes
    DRAM_FREQ = 1200e6      # Hz

    if mem_cycles > 0:

        total_bytes = (
            (total_reads + total_writes) * REQUEST_SIZE)

        total_bw = (
            total_bytes / mem_cycles) * DRAM_FREQ

        read_bw = (
            (total_reads * REQUEST_SIZE) / mem_cycles) * DRAM_FREQ

        # Convert to GB/s
        s["dram_bw_gbps"] = total_bw / 1e9
        s["dram_read_bw_peak"] = read_bw / 1e9

    else:
        s["dram_bw_gbps"] = 0.0
        s["dram_read_bw_peak"] = 0.0

    s["row_hits"] = row_hits
    s["row_misses"] = row_misses
    s["row_conflicts"] = row_conflicts
    # Derived hit rates
    s["input_hit_rate"] = s["input_hits"] / max(s["input_hits"] + s["input_misses"], 1)
    s["acc_hit_rate"]   = s["acc_hits"]   / max(s["acc_hits"]   + s["acc_misses"],   1)

    total_row = max(s["row_hits"] + s["row_misses"] + s["row_conflicts"], 1)
    s["row_hit_rate"]      = s["row_hits"]      / total_row * 100.0
    s["row_miss_rate"]     = s["row_misses"]    / total_row * 100.0
    s["row_conflict_rate"] = s["row_conflicts"] / total_row * 100.0

    return s


def extract_seq_len(filename: str) -> int:
    """Pull the first integer from a log filename as the sequence length."""
    nums = re.findall(r"\d+", os.path.basename(filename))
    return int(nums[0]) if nums else 0


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
STYLE = dict(linewidth=2, markersize=7)

def _save(fig, name, out_dir):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓  {path}")

def _line(ax, x, y, label=None, color=None, logy=False, **kw):
    ax.plot(x, y, marker="o", label=label, color=color, **STYLE, **kw)
    if logy:
        ax.set_yscale("log")

def _setup(ax, seqs, xlabel="Sequence Length"):
    ax.set_xticks(range(len(seqs)))
    ax.set_xticklabels(seqs, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)


def plot_all(df: pd.DataFrame, seqs: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    x = list(range(len(seqs)))
    print(f"\nWriting plots to: {os.path.abspath(out_dir)}")

    # ── 1. Total Cycles ───────────────────────────────────────────────────────
    #fig, ax = plt.subplots(figsize=(10, 5))
    #_line(ax, x, df["total_cycles"], color="steelblue")
    #_setup(ax, seqs)
    #ax.set_ylabel("Total Cycles")
    #ax.set_title("Total Cycles vs Sequence Length")
    #_save(fig, "01_total_cycles.png", out_dir)

    # ── 2. Systolic Array Utilization ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    _line(ax, x, df["sys_util"], label="Systolic Array Util", color="royalblue")
    _line(ax, x, df["pe_util"],  label="PE Util",             color="cornflowerblue", linestyle="--")
    _setup(ax, seqs)
    ax.set_ylabel("Utilization (%)")
    ax.set_title("Systolic Array Utilization vs Sequence Length")
    ax.legend()
    _save(fig, "02_systolic_util.png", out_dir)

    # ── 3. Memory Idle vs Sequence Length ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    _line(ax, x, df["mem_idle_pct"], color="orangered")
    _setup(ax, seqs)
    ax.set_ylabel("Memory Idle (%)")
    ax.set_title("Memory Idle % vs Sequence Length")
    _save(fig, "03_memory_idle.png", out_dir)

    # ── 4. Core Idle vs Sequence Length ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    _line(ax, x, df["core_idle_pct"], color="purple")
    _setup(ax, seqs)
    ax.set_ylabel("Core Idle (%)")
    ax.set_title("Core Idle % vs Sequence Length")
    _save(fig, "04_core_idle.png", out_dir)

    # ── 5. Instructions Per Tile ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    _line(ax, x, df["inst_per_tile"], color="darkorange")
    _setup(ax, seqs)
    ax.set_ylabel("Avg Instructions Per Tile")
    ax.set_title("Instructions Per Tile vs Sequence Length")
    _save(fig, "05_inst_per_tile.png", out_dir)

    # ── 6. DRAM Bandwidth Utilization ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    _line(ax, x, df["dram_util_pct"], color="crimson")
    _setup(ax, seqs)
    ax.set_ylabel("DRAM BW Utilisation (%)")
    ax.set_title("DRAM Bandwidth Utilisation vs Sequence Length")
    _save(fig, "06_dram_bw_util.png", out_dir)

    # ── 7. DRAM Read Bandwidth (GB/s) ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    _line(ax, x, df["dram_read_bw_peak"], color="firebrick")
    _setup(ax, seqs)
    ax.set_ylabel("DRAM Read Bandwidth (GB/s)")
    ax.set_title("DRAM Read Bandwidth vs Sequence Length")
    _save(fig, "07_dram_read_bw.png", out_dir)

    # ── 8. Input SRAM Hit Rate ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    _line(ax, x, df["input_hit_rate"], color="darkorange")
    _setup(ax, seqs)
    ax.set_ylabel("Input-SRAM Hit Rate")
    ax.set_title("Input-SRAM Hit Rate vs Sequence Length")
    _save(fig, "08_input_sram_hit_rate.png", out_dir)

    # ── 9. Acc SRAM Hit Rate ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    _line(ax, x, df["acc_hit_rate"], color="teal")
    _setup(ax, seqs)
    ax.set_ylabel("Accumulator-SRAM Hit Rate")
    ax.set_title("Accumulator-SRAM Hit Rate vs Sequence Length")
    _save(fig, "09_acc_sram_hit_rate.png", out_dir)

    # ── 10. SRAM Hit & Miss Counts (log) ─────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    for ax, hits, misses, title in [
        (ax1, df["input_hits"], df["input_misses"], "Input-SRAM"),
        (ax2, df["acc_hits"],   df["acc_misses"],   "Accumulator-SRAM"),
    ]:
        _line(ax, x, hits,   label="Hits",   color="green",  logy=True)
        _line(ax, x, misses, label="Misses", color="red",    logy=True)
        _setup(ax, seqs)
        ax.set_ylabel("Count (log scale)")
        ax.set_title(f"{title} Hit & Miss Counts")
        ax.legend()
    fig.suptitle("SRAM Hits & Misses vs Sequence Length", fontsize=13, fontweight="bold")
    _save(fig, "10_sram_hit_miss_counts.png", out_dir)

    # ── 11. DRAM Row Access Rates ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    _line(ax, x, df["row_hit_rate"],      label="Row Hit Rate %",      color="green")
    _line(ax, x, df["row_miss_rate"],     label="Row Miss Rate %",     color="red")
    _line(ax, x, df["row_conflict_rate"], label="Row Conflict Rate %", color="purple")
    _setup(ax, seqs)
    ax.set_ylabel("Rate (%)")
    ax.set_title("DRAM Row Access Rates vs Sequence Length")
    ax.legend()
    _save(fig, "11_dram_row_access.png", out_dir)

    # ── 12. Total Memory Bytes to L2 ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    _line(ax, x, df["mem_bytes"], color="darkorange", logy=True)
    _setup(ax, seqs)
    ax.set_ylabel("Total Core→L2 Request Bytes")
    ax.set_title("Total Memory-Request Bytes to L2 vs Sequence Length")
    _save(fig, "12_total_mem_bytes.png", out_dir)

    # ── 13. Avg Core→L2 Request Size ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    _line(ax, x, df["avg_req_size"], color="slateblue")
    _setup(ax, seqs)
    ax.set_ylabel("Average Object Size (bytes)")
    ax.set_title("Average Core→L2 Request Size vs Sequence Length")
    _save(fig, "13_avg_req_size.png", out_dir)

    # ── 14. Top-2 Size-Class Byte Share ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.35
    ax.bar(x, df["top1_share"], width=w, align="center", label="Top-1 size class", color="tab:orange")
    ax.bar([v + w for v in x], df["top2_share"], width=w, align="center", label="Top-2 size class", color="peachpuff", edgecolor="darkorange")
    _setup(ax, seqs)
    ax.set_xticks([v + w/2 for v in x])
    ax.set_xticklabels(seqs, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Share of Total Request Bytes (%)")
    ax.set_title("Object Histogram: Top-2 Size-Class Byte Share vs Sequence Length")
    ax.legend()
    _save(fig, "14_obj_size_class_share.png", out_dir)

    # ── 15. % of Total Cycles per Layer Type (stacked bar) ───────────────────
    layer_totals = df[LAYER_BUCKETS + ["Other"]].sum(axis=1).replace(0, 1)  # avoid /0
    pcts = df[LAYER_BUCKETS + ["Other"]].div(layer_totals, axis=0) * 100.0

    colors = {
        "QKV Projection": "tab:green",
        "Attention":      "tab:red",
        "Attn Projection":"tab:purple",
        "FFN FC1":        "tab:blue",
        "FFN FC2":        "tab:orange",
        "Other":          "grey",
    }
    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(len(x))
    for bucket in LAYER_BUCKETS + ["Other"]:
        col = pcts[bucket].to_numpy()
        if col.sum() == 0:
            continue
        ax.bar(x, col, bottom=bottom, label=bucket, color=colors[bucket])
        bottom += col
    _setup(ax, seqs)
    ax.set_ylabel("Percentage of Total Cycles (%)")
    ax.set_title("Percentage of Total Cycles by Layer Type vs Sequence Length")
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, "15_layer_cycle_pct.png", out_dir)


    styles = ["-", "--", "-.", ":"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for cid in range(4):
        col = f"core{cid}_pe_util"

        if col in df.columns:
            _line(ax, x, df[col], label=f"Core {cid}", linestyle=styles[cid])

    _setup(ax, seqs)

    ax.set_ylabel("PE Utilization (%)")
    ax.set_title("Per-Core PE Utilization vs Sequence Length")
    ax.legend()
    _save(fig, "per_core_pe_util.png", out_dir)

# ── 17. Per-Core Systolic Array Utilization ────────────────

    fig, ax = plt.subplots(figsize=(12, 6))
    styles = ["-", "--", "-.", ":"]

    for cid in range(4):
        col = f"core{cid}_sys_util"

        if col in df.columns:
            _line(
                ax,
                x,
                df[col],
                label=f"Core {cid}",
                linestyle=styles[cid])

    _setup(ax, seqs)
    ax.set_ylabel("Systolic Array Utilization (%)")
    ax.set_title("Per-Core Systolic Array Utilization vs Sequence Length")
    ax.legend()
    _save(fig, "17_per_core_systolic_util.png", out_dir)

    # ── 18. Per-Core Core Idle vs Sequence Length ─────────────

    fig, ax = plt.subplots(figsize=(12, 6))

    styles = ["-", "--", "-.", ":"]

    for cid in range(4):

        col = f"core{cid}_core_idle_pct"

        if col in df.columns:

            _line(
                ax,
                x,
                df[col],
                label=f"Core {cid}",
                linestyle=styles[cid]
            )

    _setup(ax, seqs)

    ax.set_ylabel("Core Idle (%)")
    ax.set_title("Per-Core Core Idle vs Sequence Length")

    ax.legend()

    _save(fig, "18_per_core_core_idle.png", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ONNXim statistics plotter")
    parser.add_argument("log_dir", help="Directory containing ONNXim output log files")
    parser.add_argument("--output_dir", default="plots", help="Where to write PNGs (default: plots/)")
    args = parser.parse_args()

    log_files = sorted(
        glob.glob(os.path.join(args.log_dir, "*.txt")) +
        glob.glob(os.path.join(args.log_dir, "*.log"))
    )
    if not log_files:
        print(f"No .txt/.log files found in '{args.log_dir}'. Exiting.")
        sys.exit(1)

    print(f"Found {len(log_files)} log file(s).")
    records = []
    for f in log_files:
        seq = extract_seq_len(f)
        print(f"  Parsing {os.path.basename(f)}  (seq_len={seq}) …")
        st = parse_log(f)
        st["seq_len"] = seq
        records.append(st)

    df = pd.DataFrame(records).sort_values("seq_len").reset_index(drop=True)
    seqs = [str(s) for s in df["seq_len"]]

    print(f"\nParsed {len(df)} run(s). Generating plots …")
    plot_all(df, seqs, args.output_dir)


if __name__ == "__main__":
    main()