"""
make_idle_figures.py
---------------------------------------------------------------------
Generates a publication-quality, IEEE two-column-ready figure showing
Systolic-Array idle-interval statistics (Mean, Median, P95, Maximum)
for four NN models (ResNet18, OPT-125M, Llama3-8B, OPT-66B) under
three workload levels (Low, Medium, Heavy).

Pipeline:
  1. Load raw per-core CSV (Core 0..3).
  2. Average the four cores per (Model, Load, Statistic) row.
     -> No other transformation is applied to the data itself.
  3. Plot a 2x2 grid of grouped bar charts (one panel per statistic),
     log-scale y-axis (only the axis scale, not the data, is log).
  4. Export both vector (PDF) and raster (PNG, 300 dpi) versions.

Usage:
    python make_idle_figures.py --input idle_intervals_raw.csv --outdir figs
---------------------------------------------------------------------
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 1. IEEE-style matplotlib configuration
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,   # embed fonts as true text, not paths
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# Fixed ordering (logical progression: CNN -> small LM -> mid LM -> large LM)
MODEL_ORDER = ["ResNet18", "OPT-125M", "Llama3-8B", "OPT-66B"]
LOAD_ORDER = ["Low", "Medium", "Heavy"]
STAT_ORDER = ["Mean", "Median", "P95", "Maximum"]

# Colorblind-safe palette (Wong, 2011) + hatch patterns for grayscale print
LOAD_COLORS = {
    "Low":    "#0072B2",   # blue
    "Medium": "#E69F00",   # orange
    "Heavy":  "#009E73",   # green
}
LOAD_HATCH = {
    "Low":    "",
    "Medium": "///",
    "Heavy":  "xxx",
}


def load_and_average(csv_path: str) -> pd.DataFrame:
    """Load per-core CSV and average Core 0-3 (exact values, no transform)."""
    df = pd.read_csv(csv_path)
    core_cols = ["Core 0", "Core 1", "Core 2", "Core 3"]
    df["Value"] = df[core_cols].mean(axis=1)
    out = df[["Model Name", "Load Type", "Statistic", "Value"]].rename(
        columns={"Model Name": "Model", "Load Type": "Load"}
    )
    return out


def plot_figure(df: pd.DataFrame, outdir: str, basename: str = "idle_interval_stats"):
    os.makedirs(outdir, exist_ok=True)

    n_models = len(MODEL_ORDER)
    n_loads = len(LOAD_ORDER)
    bar_width = 0.25
    x = np.arange(n_models)

    # Double-column IEEE width ~= 7.16 in; leave a touch of margin.
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    axes = axes.flatten()


    legend_handles = None
    for ax, stat in zip(axes, STAT_ORDER):
        sub = df[df["Statistic"] == stat]
        for i, load in enumerate(LOAD_ORDER):
            vals = [
                sub[(sub["Model"] == m) & (sub["Load"] == load)]["Value"].values[0]
                for m in MODEL_ORDER
            ]
            offset = (i - (n_loads - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset, vals, width=bar_width,
                color=LOAD_COLORS[load], hatch=LOAD_HATCH[load],
                edgecolor="black", linewidth=0.5,
                label=load, zorder=3,
            )
            if stat == STAT_ORDER[0]:
                pass
        legend_handles, legend_labels = ax.get_legend_handles_labels()

        ax.set_yscale("log")
        ax.set_title(stat, fontweight="bold", pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=15, ha="right")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=6))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
        ax.grid(axis="y", which="major", linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.text(0.015, 0.5, "Idle Interval (cycles, log scale)",
              va="center", rotation="vertical", fontsize=8.5)

    # One shared legend above all panels
    fig.legend(
        legend_handles, legend_labels,
        loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.965), handletextpad=0.5, columnspacing=1.2,
    )

    fig.tight_layout(rect=[0.035, 0.0, 1.0, 0.93])
    
    pdf_path = os.path.join(outdir, f"{basename}.pdf")
    png_path = os.path.join(outdir, f"{basename}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="idle_intervals_raw.csv",
                         help="Path to raw per-core CSV")
    parser.add_argument("--outdir", default="figs",
                         help="Output directory for figures / averaged CSV")
    args = parser.parse_args()

    df = load_and_average(args.input)

    # Save the averaged (core-collapsed) table for reference / reuse in the paper
    os.makedirs(args.outdir, exist_ok=True)
    avg_csv_path = os.path.join(args.outdir, "idle_intervals_core_averaged.csv")
    df.to_csv(avg_csv_path, index=False)

    pdf_path, png_path = plot_figure(df, args.outdir)

    print(f"Averaged data saved to : {avg_csv_path}")
    print(f"Vector figure (PDF)    : {pdf_path}")
    print(f"Raster preview (PNG)   : {png_path}")


if __name__ == "__main__":
    main()