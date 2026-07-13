#!/usr/bin/env python3
"""
plot_idle_intervals.py
======================
Parse ONNXim simulation logs containing per-core SA and Vector idle interval
sequences, then produce clean publication-quality visualisations.

Plots generated
---------------
1. Per-core SA idle interval histogram  (log-scale x, linear count y)
2. Per-core Vector idle interval histogram
3. Summary bar chart  – mean / median / max idle per component per core
4. Idle-period distribution stacked bar  (inspired by Slumber Fig-5)
   Bins: 0-3 cc, 4-6 cc, 7-8 cc, 9-10 cc, 11-12 cc, 13-15 cc, 16+ cc
   (customisable via IDLE_BINS below)

Usage
-----
    python plot_idle_intervals.py <logfile> [--out-dir <dir>]

If no logfile is given the script looks for "sim_output.log" in the cwd.
"""

import argparse
import re
import sys
import os
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import to_rgba
from scipy.stats import gaussian_kde
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd


# ── Tuneable constants ───────────────────────────────────────────────────────

# Idle-duration bins used for the stacked-percentage chart.
# Each entry: (label, upper_bound_inclusive)  – last entry catches everything above.
IDLE_BINS = [
    ("0–999",     999),
    ("1k–4.9k",   4_999),
    ("5k–19.9k",  19_999),
    ("20k–49.9k", 49_999),
    ("50k–99.9k", 99_999),
    ("100k+",     math.inf),
]

# Histogram settings
HIST_BINS = 60          # number of histogram bins (log-spaced)
HIST_MIN_VAL = 1        # smallest value to include (filters 0-length intervals)

# Figure DPI
DPI = 150

# Colour palette (one per core)
CORE_COLORS = plt.cm.tab10.colors

# ── Regex patterns ────────────────────────────────────────────────────────────
RE_CORE   = re.compile(r"Core\s*\[(\d+)\]")
RE_SA     = re.compile(r"SA idle\s*=\s*(\d+)")
RE_VEC    = re.compile(r"Vector idle\s*=\s*(\d+)")
RE_TOTAL_CYCLES = re.compile(r"Simulation Finished at\s+(\d+)\s+cycle")


def parse_log(path: str) -> dict:
    """
    Returns
    -------
    data : dict  { core_id(int): { 'sa': [int,...], 'vec': [int,...] } }
    """
    data = defaultdict(lambda: {"sa": [], "vec": []})
    current_core = 0          # default core if header hasn't been seen yet
    total_cycles = None

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = RE_TOTAL_CYCLES.search(line)
            if m:
                total_cycles = int(m.group(1))
            m = RE_CORE.search(line)
            if m:
                current_core = int(m.group(1))
                continue
            m = RE_SA.search(line)
            if m:
                v = int(m.group(1))
                if v >= HIST_MIN_VAL:
                    data[current_core]["sa"].append(v)
                continue
            m = RE_VEC.search(line)
            if m:
                v = int(m.group(1))
                if v >= HIST_MIN_VAL:
                    data[current_core]["vec"].append(v)

    return dict(data), total_cycles


# ── Helpers ───────────────────────────────────────────────────────────────────

def log_bins(values, n_bins=HIST_BINS):
    """Return log-spaced bin edges covering all values."""
    vmin = max(1, min(values))
    vmax = max(values)
    if vmin == vmax:
        vmin = vmax / 10
    return np.logspace(np.log10(vmin), np.log10(vmax), n_bins + 1)


def assign_bin(val):
    for label, ub in IDLE_BINS:
        if val <= ub:
            return label
    return IDLE_BINS[-1][0]


def bin_distribution(values):
    """Return ordered list of (label, count)."""
    from collections import Counter
    counts = Counter(assign_bin(v) for v in values)
    return [(lbl, counts.get(lbl, 0)) for lbl, _ in IDLE_BINS]


def summary_stats(values):
    if not values:
        return dict(mean=0, median=0, p95=0, maximum=0, total=0, count=0)
    a = np.array(values)
    return dict(
        mean=np.mean(a),
        median=np.median(a),
        p95=np.percentile(a, 95),
        maximum=np.max(a),
        total=np.sum(a),
        count=len(a),
    )


# ── Plot functions ────────────────────────────────────────────────────────────

def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)


def plot_histograms(data: dict, component: str, out_dir: Path):
    """One subplot per core; log-x histogram of idle intervals."""
    cores = sorted(data.keys())
    n = len(cores)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.5 * nrows),
                             squeeze=False)
    fig.suptitle(f"{component.upper()} Idle-Interval Distribution per Core",
                 fontsize=13, fontweight="bold", y=1.01)

    key = "sa" if component == "sa" else "vec"
    for idx, core in enumerate(cores):
        ax = axes[idx // ncols][idx % ncols]
        vals = data[core][key]
        color = CORE_COLORS[core % len(CORE_COLORS)]

        if vals:
            edges = log_bins(vals)
            counts, _ = np.histogram(vals, bins=edges)
            # Draw as step histogram
            ax.bar(edges[:-1], counts,
                   width=np.diff(edges),
                   align="edge",
                   color=color,
                   alpha=0.80,
                   edgecolor="white",
                   linewidth=0.4)
            ax.set_xscale("log")
            # Annotate with stats
            s = summary_stats(vals)
            ax.axvline(s["mean"],   color="red",    lw=1.2, ls="--", label=f"mean={s['mean']:.0f}")
            ax.axvline(s["median"], color="orange", lw=1.2, ls=":",  label=f"med={s['median']:.0f}")
            ax.legend(fontsize=6.5, loc="upper right", framealpha=0.6)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")

        style_ax(ax, f"Core {core}  (n={len(vals):,})",
                 "Idle duration (cycles, log scale)", "Count")

    # Hide unused subplots
    for i in range(len(cores), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.tight_layout()
    out = out_dir / f"hist_{component}_idle.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_summary_bars(data: dict, out_dir: Path):
    """Mean / median / P95 / max for SA and Vector per core."""
    cores = sorted(data.keys())
    components = [("sa", "SA"), ("vec", "Vector")]

    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(cores) * 1.8 + 2), 5),
                             sharey=False)
    fig.suptitle("Idle Interval Statistics per Core", fontsize=13, fontweight="bold")

    metrics = ["mean", "median", "p95", "maximum"]
    metric_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bar_w = 0.18
    x = np.arange(len(cores))

    for ax, (key, label) in zip(axes, components):
        for mi, (met, mc) in enumerate(zip(metrics, metric_colors)):
            vals = [summary_stats(data[c][key])[met] for c in cores]
            offset = (mi - 1.5) * bar_w
            bars = ax.bar(x + offset, vals, width=bar_w,
                          label=met, color=mc, alpha=0.85, edgecolor="white")
            # Label bars only if not too many cores
            if len(cores) <= 6:
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                                f"{h:.0f}", ha="center", va="bottom",
                                fontsize=5.5, rotation=45)

        ax.set_yscale("symlog", linthresh=100)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else str(int(v))))
        style_ax(ax, f"{label} Idle Statistics", "Core", "Cycles (symlog)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Core {c}" for c in cores], rotation=20, ha="right")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.7)

    fig.tight_layout()
    out = out_dir / "summary_stats.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_stacked_distribution(data: dict, out_dir: Path):
    """
    Stacked percentage bar chart of idle-duration bins per core,
    separately for SA and Vector (inspired by Slumber Fig-5).
    """
    cores = sorted(data.keys())
    bin_labels = [lbl for lbl, _ in IDLE_BINS]
    cmap = plt.cm.RdYlGn_r
    bin_colors = [cmap(i / (len(IDLE_BINS) - 1)) for i in range(len(IDLE_BINS))]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(cores) * 1.6 + 3), 5),
                             sharey=True)
    fig.suptitle("Idle-Period Distribution by Duration Bin",
                 fontsize=13, fontweight="bold")

    for ax, (key, comp_label) in zip(axes, [("sa", "SA"), ("vec", "Vector")]):
        bottoms = np.zeros(len(cores))
        for bi, (blabel, bcolor) in enumerate(zip(bin_labels, bin_colors)):
            heights = []
            for c in cores:
                dist = dict(bin_distribution(data[c][key]))
                total = sum(dist.values())
                heights.append(100.0 * dist.get(blabel, 0) / total if total else 0)
            ax.bar(range(len(cores)), heights, bottom=bottoms,
                   label=blabel, color=bcolor, edgecolor="white", linewidth=0.3)
            bottoms += np.array(heights)

        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        style_ax(ax, f"{comp_label} Idle – Duration Bin Distribution",
                 "Core", "% of idle intervals")
        ax.set_xticks(range(len(cores)))
        ax.set_xticklabels([f"Core {c}" for c in cores], rotation=20, ha="right")
        ax.legend(fontsize=7.5, loc="upper right",
                  title="Duration (cycles)", title_fontsize=7,
                  framealpha=0.75, ncol=1)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = out_dir / "stacked_distribution.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_total_idle_share(data: dict, out_dir: Path):
    """
    Pie / donut chart: how much of total idle cycles belong to each core,
    split by SA vs Vector.
    """
    cores = sorted(data.keys())
    sa_totals  = [sum(data[c]["sa"])  for c in cores]
    vec_totals = [sum(data[c]["vec"]) for c in cores]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.suptitle("Total Idle Cycles Share per Core", fontsize=13, fontweight="bold")

    for ax, totals, title in zip(axes,
                                  [sa_totals, vec_totals],
                                  ["SA Idle", "Vector Idle"]):
        if sum(totals) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title)
            continue
        wedge_colors = [CORE_COLORS[c % len(CORE_COLORS)] for c in cores]
        wedges, texts, autotexts = ax.pie(
            totals,
            labels=[f"Core {c}" for c in cores],
            autopct="%1.1f%%",
            colors=wedge_colors,
            wedgeprops=dict(width=0.55, edgecolor="white"),
            startangle=90,
            pctdistance=0.75,
        )
        for t in autotexts:
            t.set_fontsize(8)
        ax.set_title(title, fontsize=11, fontweight="bold")

    fig.tight_layout()
    out = out_dir / "total_idle_share.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_idle_cycle_contribution(data, out_dir):
    """
    Shows what fraction of TOTAL idle cycles comes from each
    idle-duration bucket.

    Much more relevant for slumber than interval count.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    cores = sorted(data.keys())

    bins = [
        ("0-1K", 0, 1000),
        ("1K-10K", 1000, 10000),
        ("10K-100K", 10000, 100000),
        ("100K+", 100000, float("inf"))
    ]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5),
        sharey=True
    )

    for ax, key, title in zip(
        axes,
        ["sa", "vec"],
        ["Systolic Array", "Vector Unit"]
    ):

        bottoms = np.zeros(len(cores))

        for label, low, high in bins:

            contributions = []

            for core in cores:

                vals = np.array(data[core][key])

                if len(vals) == 0:
                    contributions.append(0)
                    continue

                total_cycles = vals.sum()

                mask = (vals >= low) & (vals < high)

                bucket_cycles = vals[mask].sum()

                contributions.append(
                    100 * bucket_cycles / total_cycles
                )

            ax.bar(
                range(len(cores)),
                contributions,
                bottom=bottoms,
                label=label
            )

            bottoms += np.array(contributions)

        ax.set_title(
            f"{title}\nIdle Cycle Contribution",
            fontsize=12,
            weight="bold"
        )

        ax.set_ylabel("% of Total Idle Cycles")
        ax.set_xlabel("Core")

        ax.set_xticks(range(len(cores)))
        ax.set_xticklabels(
            [f"Core {c+1}" for c in cores]
        )

        ax.legend()

    plt.tight_layout()

    plt.savefig(
        out_dir / "idle_cycle_contribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def plot_idle_timeline(data, component, out_dir, max_points=5000):
    cores = sorted(data.keys())

    fig, axes = plt.subplots(
        len(cores), 1,
        figsize=(12, 2.5 * len(cores)),
        sharex=True
    )

    if len(cores) == 1:
        axes = [axes]

    key = component

    for ax, core in zip(axes, cores):

        vals = data[core][key]

        if not vals:
            continue

        vals = vals[:max_points]

        ax.plot(
            np.arange(len(vals)),
            vals,
            linewidth=0.8
        )

        ax.set_yscale("log")

        ax.set_title(
            f"Core {core}",
            fontsize=10
        )

        ax.set_ylabel("Cycles")

        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Idle Interval Index")

    fig.suptitle(
        f"{component.upper()} Idle Timeline",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    out = out_dir / f"{component}_timeline.png"

    plt.savefig(
        out,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f" Saved: {out}")

def plot_cdf(data, component, out_dir):

    cores = sorted(data.keys())

    fig, axes = plt.subplots(
        len(cores), 1,
        figsize=(10, 2.5 * len(cores)),
        sharex=True
    )

    if len(cores) == 1:
        axes = [axes]

    for ax, core in zip(axes, cores):

        vals = np.array(data[core][component])

        if len(vals) == 0:
            continue

        vals = np.sort(vals)

        cdf = np.arange(1, len(vals)+1) / len(vals)

        ax.plot(vals, cdf)

        ax.set_xscale("log")

        ax.set_ylabel("CDF")

        ax.set_title(f"Core {core}")

        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Idle Duration (cycles)")

    fig.suptitle(
        f"{component.upper()} Idle CDF",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        out_dir / f"{component}_cdf.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def plot_survival(data, component, out_dir):

    cores = sorted(data.keys())

    fig, axes = plt.subplots(
        len(cores), 1,
        figsize=(10, 2.5 * len(cores)),
        sharex=True
    )

    if len(cores) == 1:
        axes = [axes]

    for ax, core in zip(axes, cores):

        vals = np.array(data[core][component])

        if len(vals) == 0:
            continue

        vals = np.sort(vals)

        survival = 1.0 - (
            np.arange(1, len(vals)+1) / len(vals)
        )

        ax.plot(vals, survival)

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_title(f"Core {core}")

        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Idle Duration (cycles)")
    axes[-1].set_ylabel("P(idle > x)")

    fig.suptitle(
        f"{component.upper()} Survival Curve",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        out_dir / f"{component}_survival.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def plot_lag(data, component, out_dir):

    cores = sorted(data.keys())

    fig, axes = plt.subplots(
        2, 2,
        figsize=(10,8)
    )

    axes = axes.flatten()

    for ax, core in zip(axes, cores):

        vals = np.array(data[core][component])

        if len(vals) < 2:
            continue

        x = vals[:-1]
        y = vals[1:]

        ax.scatter(
            x,
            y,
            s=4,
            alpha=0.4
        )

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_title(f"Core {core}")

        ax.set_xlabel("Idle[n]")
        ax.set_ylabel("Idle[n+1]")

        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{component.upper()} Lag Plot",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        out_dir / f"{component}_lag.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def plot_autocorrelation(data, component, out_dir):

    cores = sorted(data.keys())

    fig, axes = plt.subplots(
        len(cores), 1,
        figsize=(10, 3 * len(cores))
    )

    if len(cores) == 1:
        axes = [axes]

    for ax, core in zip(axes, cores):

        vals = np.array(data[core][component])

        if len(vals) < 20:
            continue

        plot_acf(
            vals,
            lags=100,
            ax=ax
        )

        ax.set_title(f"Core {core}")

    fig.suptitle(
        f"{component.upper()} Autocorrelation",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        out_dir / f"{component}_acf.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def plot_rolling_mean(data, component, out_dir, window=100):

    cores = sorted(data.keys())

    fig, axes = plt.subplots(
        len(cores), 1,
        figsize=(12, 3 * len(cores)),
        sharex=True
    )

    if len(cores) == 1:
        axes = [axes]

    for ax, core in zip(axes, cores):

        vals = np.array(data[core][component])

        if len(vals) < window:
            continue

        rolling = (
            pd.Series(vals)
            .rolling(window)
            .mean()
        )

        ax.plot(rolling)

        ax.set_yscale("log")

        ax.set_title(f"Core {core}")

        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Idle Interval Index")

    fig.suptitle(
        f"{component.upper()} Rolling Mean (window={window})",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        out_dir / f"{component}_rolling_mean.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def print_table(data: dict):
    """Console summary table."""
    print("\n" + "═" * 72)
    print(f"{'Core':>6}  {'Comp':>7}  {'Count':>7}  {'Mean':>10}  "
          f"{'Median':>10}  {'P95':>10}  {'Max':>10}  {'Total':>12}")
    print("─" * 72)
    for core in sorted(data.keys()):
        for key, label in [("sa", "SA"), ("vec", "Vector")]:
            s = summary_stats(data[core][key])
            print(f"{core:>6}  {label:>7}  {s['count']:>7,}  "
                  f"{s['mean']:>10.1f}  {s['median']:>10.1f}  "
                  f"{s['p95']:>10.1f}  {s['maximum']:>10.0f}  "
                  f"{s['total']:>12,}")
    print("═" * 72 + "\n")


def print_idle_percentage(data, total_cycles):
    print("\n")
    print("=" * 60)
    print("Average Idle Percentage")
    print("=" * 60)

    num_cores = len(data)

    total_sa_idle = sum(
        sum(data[c]["sa"])
        for c in data
    )

    total_vec_idle = sum(
        sum(data[c]["vec"])
        for c in data
    )

    sa_percent = (
        total_sa_idle /
        (num_cores * total_cycles)
    ) * 100

    vec_percent = (
        total_vec_idle /
        (num_cores * total_cycles)
    ) * 100

    labels = ["Systolic\nArray", "Vector\nUnit"]
    values = [sa_percent, vec_percent]

    plt.figure(figsize=(8,6))

    colors = ["#4C72B0", "#DD8452"]   # blue and orange

    bars = plt.bar(
        labels,
        values,
        color=colors,
        width=0.55,
        edgecolor="black",
        linewidth=1.2
    )

    plt.ylabel("Idle Cycles (%)", fontsize=16)
    plt.title("Idle Cycles as Percentage of Total Simulation Cycles",
            fontsize=20,
            weight="bold")

    plt.ylim(0, 110)          # <-- extra space for labels

    plt.grid(axis='y',
            linestyle='--',
            alpha=0.35)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=14)

    for bar in bars:
        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width()/2,
            min(height + 2, 107),      # keep text inside figure
            f"{height:.2f}%",
            ha='center',
            va='bottom',
            fontsize=15,
            fontweight='bold'
        )

    plt.tight_layout()

    plt.savefig("idle_percentage.png",
                dpi=300,
                bbox_inches="tight")

    plt.close()



# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot SA/Vector idle intervals from ONNXim simulation log.")
    parser.add_argument("logfile", nargs="?", default="sim_output.log",
                        help="Path to the simulator log file")
    parser.add_argument("--out-dir", default="idle_plots",
                        help="Directory to write output PNGs (default: ./idle_plots)")
    args = parser.parse_args()

    if not os.path.isfile(args.logfile):
        print(f"[ERROR] Log file not found: {args.logfile}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing: {args.logfile}")
    data, total_cycles = parse_log(args.logfile)

    if not data:
        print("[ERROR] No idle-interval data found in log.", file=sys.stderr)
        sys.exit(1)

    total_cores = len(data)
    total_sa  = sum(len(data[c]["sa"])  for c in data)
    total_vec = sum(len(data[c]["vec"]) for c in data)
    print(f"Found {total_cores} core(s) | "
          f"{total_sa} SA intervals | {total_vec} Vector intervals")

    print_table(data)

    print("Generating plots …")
    plot_idle_timeline(data, "sa", out_dir)
    plot_idle_timeline(data, "vec", out_dir)
    plot_cdf(data, "sa", out_dir)
    plot_cdf(data, "vec", out_dir)
    plot_survival(data, "sa", out_dir)
    plot_survival(data, "vec", out_dir)
    plot_lag(data, "sa", out_dir)
    plot_lag(data, "vec", out_dir)
    # plot_autocorrelation(data, "sa", out_dir)
    # plot_autocorrelation(data, "vec", out_dir)
    # plot_rolling_mean(data, "sa", out_dir)
    # plot_rolling_mean(data, "vec", out_dir)
    print_idle_percentage(data,total_cycles)

    print(f"\nAll plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()