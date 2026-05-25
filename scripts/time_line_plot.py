"""
Usage:
    python time_line_plot.py --log path/to/log_seq_1.log --model tiny_transformer --out-dir ./plots
"""

import re, os, argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.size":            18,
    "axes.labelsize":       20,
    "axes.titlesize":       22,
    "xtick.labelsize":      18,
    "ytick.labelsize":      18,
    "legend.fontsize":      16,
    "legend.title_fontsize":18,
    "figure.titlesize":     24,
})

# ── Colour palette ─────────────────────────────────────────────────────
LAYER_COLORS = [
    "#4C72B0", "#55A868", "#C44E52", "#8172B3",
    "#CCB974", "#64B5CD", "#8C8C8C", "#E17C05",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
]

# Metrics exposed in plots  (key → y-axis label)
METRIC_LABELS = {
    "mem_idle_pct":  "Memory Unit Idle (%)",
    "pe_util_pct":   "PE Util (%)",
    "vec_util_pct":  "Vector Unit Util (%)",
    "core_idle_pct": "Core Idle (%)",
    "bubble_pct":    "Systolic Bubble (%)",
}

# ── Regexes ────────────────────────────────────────────────────────────
RE_SCHED  = re.compile(r"Schedule model")
RE_LAYER  = re.compile(r"Start layer (.+)")
RE_FINISH = re.compile(r"Layer (.+) finish at (\d+)")

# Per-core idle / bubble / core-idle  (same format as original)
RE_IDLE   = re.compile(
    r"Core \[(\d+)\] : Memory unit idle cycle (\d+)"
    r" Systolic bubble cycle (\d+) Core idle cycle (\d+)"
)
# Per-core utilisation + **absolute cumulative** Total cycle
RE_UTIL   = re.compile(
    r"Core \[(\d+)\]\s*:\s*"
    r"PE Utilization\(%\)\s*([\d.]+),\s*"
    r"Systolic Bubble\(%\)\s*([\d.]+),\s*"
    r"Vector Unit Utilization\(%\)\s*([\d.]+),\s*"
    r"Total cycle:\s*(\d+)"
)


# ── Parser ─────────────────────────────────────────────────────────────
def parse_log(path: str):
    """
    Returns a flat list of records, one per (core, 1000-cycle window).

    The ONNXim log emits periodic stat lines for every 1000-cycle interval.
    Total cycle on the Util line is the *cumulative* cycle count – we use it
    directly as the x-axis value (no separate "cycle: [N]" event needed).
    The idle / bubble / core-idle counts are *per-interval* (they reset each
    window), so we divide by 1000 to get percentages.
    """
    records  = []
    staged   = {}          # core → partial record
    current_layer = "unknown"
    pass_idx = 0

    with open(path) as f:
        for raw in f:
            line = raw.strip()

            # ── pass boundary ──────────────────────────────────────────
            if RE_SCHED.search(line):
                pass_idx += 1
                staged = {}
                continue

            # ── layer start ────────────────────────────────────────────
            m = RE_LAYER.search(line)
            if m:
                current_layer = m.group(1).strip()
                staged = {}
                continue

            # ── idle / bubble / core-idle ──────────────────────────────
            m = RE_IDLE.search(line)
            if m:
                core = int(m.group(1))
                staged.setdefault(core, {}).update({
                    "mem":    int(m.group(2)),
                    "bubble": int(m.group(3)),
                    "ci":     int(m.group(4)),
                    "layer":  current_layer,
                    "pass":   pass_idx,
                })
                continue

            # ── util + total cycle  (self-contained trigger) ───────────
            m = RE_UTIL.search(line)
            if m:
                core      = int(m.group(1))
                pe_util   = float(m.group(2))
                bubble    = float(m.group(3))
                vec_util  = float(m.group(4))
                abs_cycle = int(m.group(5))

                s = staged.get(core, {})
                # window size = 10 cycles (fixed by simulator)
                window = 10

                records.append({
                    "core":          core,
                    "cycle":         abs_cycle,
                    "layer":         s.get("layer", current_layer),
                    "pass":          s.get("pass",  pass_idx),
                    "layer_key":     f"p{s.get('pass', pass_idx)}.{s.get('layer', current_layer)}",
                    "mem_idle_pct":  min(s.get("mem",    0) / window * 100, 100),
                    "core_idle_pct": min(s.get("ci",     0) / window * 100, 100),
                    "bubble_pct": min(bubble, 100),
                    "pe_util_pct":   min(pe_util,  100),
                    "vec_util_pct":  min(vec_util, 100),
                })
                staged.pop(core, None)

    return records


# ── Layer span builder ─────────────────────────────────────────────────
def get_layer_spans(records):
    """Returns [(start_cycle, end_cycle, layer_key), …] sorted by start."""
    if not records:
        return []
    data   = sorted(records, key=lambda r: r["cycle"])
    spans  = []
    prev   = data[0]["layer_key"]
    start  = data[0]["cycle"]
    for r in data[1:]:
        if r["layer_key"] != prev:
            spans.append((start, r["cycle"], prev))
            start = r["cycle"]
            prev  = r["layer_key"]
    spans.append((start, data[-1]["cycle"], prev))
    return spans


def _color_map(spans):
    keys = list(dict.fromkeys(s[2] for s in spans))
    return {k: LAYER_COLORS[i % len(LAYER_COLORS)] for i, k in enumerate(keys)}


def _strip_pass(lkey):
    parts = lkey.split(".", 1)
    return parts[1] if len(parts) == 2 else lkey


# ── Shared plot helpers ────────────────────────────────────────────────
def _shade_layers(ax, spans, cmap, annotate=False):
    for start, end, lkey in spans:
        color = cmap.get(lkey, "#aaaaaa")
        ax.axvspan(start, end, color=color, alpha=0.13, zorder=0)
        ax.axvline(start, color=color, linestyle=":", linewidth=0.8, zorder=1)
        if annotate:
            mid = (start + end) / 2
            ax.text(mid, 103, _strip_pass(lkey), fontsize=6,
                    ha="center", va="bottom", color=color,
                    rotation=45, clip_on=True)


def _axis_style(ax):
    ax.set_ylim(-3, 108)
    ax.set_yticks(np.linspace(0, 100, 6))
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _layer_legend(fig_or_ax, cmap, use_base=True, **kwargs):
    patches = []
    for k, c in cmap.items():
        label = _strip_pass(k) if use_base else k
        patches.append(mpatches.Patch(facecolor=c, alpha=0.4, label=label))
    defaults = dict(title="Layer", bbox_to_anchor=(1.01, 1),
                    loc="upper left", fontsize=14, frameon=False)
    defaults.update(kwargs)
    return fig_or_ax.legend(handles=patches, **defaults)


# ══════════════════════════════════════════════════════════════════════
#  Plot functions
# ══════════════════════════════════════════════════════════════════════

def plot_all_metrics_single_pass(records, model, pass_id, out_path):
    """
    One figure per pass, one subplot per metric.
    X-axis rebased to 0.  Matches Figures 4.16 / 4.17 style.
    """
    metrics = list(METRIC_LABELS.keys())
    precs   = [r for r in records if r["pass"] == pass_id]
    if not precs:
        print(f"  [SKIP] pass {pass_id} – no data")
        return

    base  = min(r["cycle"] for r in precs)
    precs = [{**r, "cycle": r["cycle"] - base} for r in precs]

    spans = get_layer_spans(precs)
    # build cmap keyed on *base* layer name (no pass prefix)
    base_layers = list(dict.fromkeys(_strip_pass(s[2]) for s in spans))
    cmap = {l: LAYER_COLORS[i % len(LAYER_COLORS)] for i, l in enumerate(base_layers)}
    # remap spans to base-layer keys
    spans_base = [(_s, _e, _strip_pass(_k)) for _s, _e, _k in spans]

    cores = sorted(set(r["core"] for r in precs))
    n     = len(metrics)

    PHASE = {0: "Prompt Phase", 1: "Token Generation Phase"}
    phase_title = PHASE.get(pass_id, f"Pass {pass_id}")

    fig, axes = plt.subplots(n, 1, figsize=(26, 5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{model}  —  {phase_title}  (All Metrics)", y=1.002)

    for ax, metric in zip(axes, metrics):
        _shade_layers(ax, spans_base, cmap)
        for core in cores:
            data = sorted([r for r in precs if r["core"] == core],
                          key=lambda x: x["cycle"])
            ax.plot([r["cycle"] for r in data],
                    [r[metric]  for r in data],
                    label=f"Core {core}", linewidth=1.4)
        ax.set_ylabel(METRIC_LABELS[metric])
        _axis_style(ax)
        ax.legend(loc="upper left", fontsize=14)

    axes[-1].set_xlabel("Cycle (rebased to 0)")

    patches = [mpatches.Patch(facecolor=c, alpha=0.4, label=l)
               for l, c in cmap.items()]
    fig.legend(handles=patches, title="Layer",
               bbox_to_anchor=(0.97, 0.98), loc="upper left",
               fontsize=14, frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved:", out_path)


def plot_single_metric_pass(records, model, metric, pass_id, out_path):
    """Single metric, single pass – x rebased, layer bands + annotations."""
    precs = [r for r in records if r["pass"] == pass_id]
    if not precs:
        return
    base  = min(r["cycle"] for r in precs)
    precs = [{**r, "cycle": r["cycle"] - base} for r in precs]

    spans      = get_layer_spans(precs)
    base_layers= list(dict.fromkeys(_strip_pass(s[2]) for s in spans))
    cmap       = {l: LAYER_COLORS[i % len(LAYER_COLORS)]
                  for i, l in enumerate(base_layers)}
    spans_base = [(_s, _e, _strip_pass(_k)) for _s, _e, _k in spans]

    PHASE = {0: "Prompt Phase", 1: "Token Generation Phase"}
    phase_title = PHASE.get(pass_id, f"Pass {pass_id}")

    fig, ax = plt.subplots(figsize=(22, 5))
    _shade_layers(ax, spans_base, cmap, annotate=True)

    cores = sorted(set(r["core"] for r in precs))
    for core in cores:
        data = sorted([r for r in precs if r["core"] == core],
                      key=lambda x: x["cycle"])
        ax.plot([r["cycle"] for r in data],
                [r[metric]  for r in data],
                label=f"Core {core}", linewidth=1.4)

    ax.set_title(f"{model}  |  {phase_title}  –  {METRIC_LABELS[metric]}")
    ax.set_xlabel("Cycle (rebased to 0)")
    ax.set_ylabel(METRIC_LABELS[metric])
    _axis_style(ax)

    core_leg = ax.legend(loc="upper left", fontsize=14)
    ax.add_artist(core_leg)
    _layer_legend(ax, cmap)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved:", out_path)


def plot_pass_comparison(records, model, pa, pb, out_path):
    """Side-by-side: prompt phase (left) vs token-gen phase (right)."""
    metrics = list(METRIC_LABELS.keys())
    recs_a  = [r for r in records if r["pass"] == pa]
    recs_b  = [r for r in records if r["pass"] == pb]
    if not recs_a or not recs_b:
        print(f"  [SKIP] comparison pass {pa} vs {pb} – missing data")
        return

    def rebase(recs):
        base = min(r["cycle"] for r in recs)
        return [{**r, "cycle": r["cycle"] - base} for r in recs]

    recs_a, recs_b = rebase(recs_a), rebase(recs_b)

    def make_spans(recs):
        spans = get_layer_spans(recs)
        base_layers = list(dict.fromkeys(_strip_pass(s[2]) for s in spans))
        cmap = {l: LAYER_COLORS[i % len(LAYER_COLORS)]
                for i, l in enumerate(base_layers)}
        spans_base = [(_s, _e, _strip_pass(_k)) for _s, _e, _k in spans]
        return spans_base, cmap

    spans_a, cmap_a = make_spans(recs_a)
    spans_b, cmap_b = make_spans(recs_b)

    # unified colour map so same layer = same colour in both panels
    all_layers = list(dict.fromkeys(list(cmap_a) + list(cmap_b)))
    uni_cmap   = {l: LAYER_COLORS[i % len(LAYER_COLORS)]
                  for i, l in enumerate(all_layers)}

    PHASE = {0: "Prompt Phase", 1: "Token Generation Phase"}
    label_a = PHASE.get(pa, f"Pass {pa}")
    label_b = PHASE.get(pb, f"Pass {pb}")

    n      = len(metrics)
    cores  = sorted(set(r["core"] for r in records))
    fig, axes = plt.subplots(n, 2, figsize=(34, 4.5 * n), sharex="col")
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{model}  |  {label_a}  vs  {label_b}", y=1.002)

    for row, metric in enumerate(metrics):
        for col, (recs, spans, pid) in enumerate([
            (recs_a, spans_a, pa),
            (recs_b, spans_b, pb),
        ]):
            ax = axes[row][col]
            _shade_layers(ax, spans, uni_cmap)
            for core in cores:
                data = sorted([r for r in recs if r["core"] == core],
                              key=lambda x: x["cycle"])
                if data:
                    ax.plot([r["cycle"] for r in data],
                            [r[metric]  for r in data],
                            label=f"Core {core}", linewidth=1.2)
            _axis_style(ax)
            if row == 0:
                ax.set_title(label_a if col == 0 else label_b, pad=6)
            if col == 0:
                ax.set_ylabel(METRIC_LABELS[metric])
            if row == n - 1:
                ax.set_xlabel("Cycle (rebased to 0)")
            if col == 0:
                ax.legend(loc="upper left", fontsize=12)

    patches = [mpatches.Patch(facecolor=c, alpha=0.4, label=l)
               for l, c in uni_cmap.items()]
    fig.legend(handles=patches, title="Layer",
               bbox_to_anchor=(1.01, 0.98), loc="upper left",
               fontsize=12, frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved:", out_path)


def plot_all_metrics_all_passes(records, model, out_path):
    """All passes overlaid on one figure (full timeline)."""
    metrics = list(METRIC_LABELS.keys())
    spans   = get_layer_spans(records)
    cmap    = _color_map(spans)
    cores   = sorted(set(r["core"] for r in records))

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(28, 5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{model}  —  all metrics, all passes", y=1.001)

    for ax, metric in zip(axes, metrics):
        _shade_layers(ax, spans, cmap)
        for core in cores:
            data = sorted([r for r in records if r["core"] == core],
                          key=lambda x: x["cycle"])
            ax.plot([r["cycle"] for r in data],
                    [r[metric]  for r in data],
                    label=f"Core {core}", linewidth=1.2)
        ax.set_ylabel(METRIC_LABELS[metric])
        _axis_style(ax)
        ax.legend(loc="upper left", fontsize=12)

    axes[-1].set_xlabel("Cycle")

    patches = [mpatches.Patch(facecolor=c, alpha=0.4, label=k)
               for k, c in cmap.items()]
    fig.legend(handles=patches, title="Pass.Layer",
               bbox_to_anchor=(0.97, 0.98), loc="upper left",
               fontsize=12, frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved:", out_path)


def plot_gantt(records, model, out_path):
    """Horizontal bar (Gantt) chart – one row per pass.layer."""
    spans       = get_layer_spans(records)
    unique_keys = list(dict.fromkeys(s[2] for s in spans))
    cmap        = {k: LAYER_COLORS[i % len(LAYER_COLORS)]
                   for i, k in enumerate(unique_keys)}

    fig, ax = plt.subplots(figsize=(24, max(4, len(unique_keys) * 0.6)))
    for i, lkey in enumerate(unique_keys):
        for s, e, k in spans:
            if k == lkey:
                ax.barh(i, e - s, left=s, height=0.7,
                        color=cmap[lkey], alpha=0.85)

    ax.set_yticks(range(len(unique_keys)))
    ax.set_yticklabels(unique_keys, fontsize=9)
    ax.set_xlabel("Cycle")
    ax.set_title(f"{model}  —  layer timeline (all passes)")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved:", out_path)


# ── Main ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Plot ONNXim tiny_transformer simulation logs")
    ap.add_argument("--log",     required=True,          help="Path to .log file")
    ap.add_argument("--model",   default="tiny_transformer")
    ap.add_argument("--out-dir", default="plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = args.model

    print(f"Parsing: {args.log}")
    records = parse_log(args.log)
    if not records:
        print("ERROR: no records parsed – check log format.")
        return

    passes = sorted(set(r["pass"] for r in records))
    layers = sorted(set(r["layer_key"] for r in records))
    print(f"  Passes found : {passes}")
    print(f"  Unique layer keys: {len(layers)}")
    print(f"  Total records    : {len(records)}")

    def out(fname):
        return os.path.join(args.out_dir, fname)

    # ── full-timeline plots ───────────────────────────────────────────
    plot_all_metrics_all_passes(records, model,
                                out(f"{model}_ALL_metrics_all_passes.png"))
    plot_gantt(records, model, out(f"{model}_gantt.png"))

    # ── per-pass: all-metrics combined + individual metrics ───────────
    PHASE_NAMES = {0: "prompt", 1: "token_gen"}
    for p in passes:
        phase = PHASE_NAMES.get(p, f"pass{p}")
        plot_all_metrics_single_pass(
            records, model, p,
            out(f"{model}_pass{p}_{phase}_ALL_metrics.png"))
        for metric in METRIC_LABELS:
            plot_single_metric_pass(
                records, model, metric, p,
                out(f"{model}_pass{p}_{phase}_{metric}.png"))

    # ── side-by-side pass comparisons ────────────────────────────────
    for i in range(len(passes) - 1):
        pa, pb = passes[i], passes[i + 1]
        plot_pass_comparison(
            records, model, pa, pb,
            out(f"{model}_pass{pa}_vs_pass{pb}_comparison.png"))

    print("\nDone. All plots saved to:", args.out_dir)


if __name__ == "__main__":
    main()