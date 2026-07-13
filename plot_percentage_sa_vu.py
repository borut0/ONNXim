import argparse
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MODEL_ORDER = ["ResNet18", "OPT-125M", "Llama3-8B", "OPT-66B"]
LOAD_ORDER = ["Low", "Medium", "Heavy"]

LOAD_COLORS = {
    "Low": "#0072B2",
    "Medium": "#E69F00",
    "Heavy": "#009E73",
}

LOAD_HATCH = {
    "Low": "",
    "Medium": "///",
    "Heavy": "xxx",
}


def plot(csv_file, outdir):

    df = pd.read_csv(csv_file)

    # Remove accidental spaces from CSV column names
    df.columns = df.columns.str.strip()

    # Convert metric columns to numeric.
    # Missing values become NaN.
    metric_columns = [
        "SA Idle %",
        "Vector Idle %",
        "MC Idle %",
        "NoC Idle %",
    ]

    for column in metric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # 2 x 2 layout for four hardware components
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.0, 5.2),
        sharey=True,
    )

    axes = axes.flatten()

    width = 0.25
    x = np.arange(len(MODEL_ORDER))

    metrics = [
        ("SA Idle %", "Systolic Array"),
        ("Vector Idle %", "Vector Unit"),
        ("MC Idle %", "Memory Controller"),
        ("NoC Idle %", "Interconnect"),
    ]

    legend_handles = None
    legend_labels = None

    for ax, (column, title) in zip(axes, metrics):

        for i, load in enumerate(LOAD_ORDER):

            vals = []

            for model in MODEL_ORDER:

                selected = df[
                    (df["Model"] == model) &
                    (df["Load"] == load)
                ][column]

                if selected.empty:
                    value = np.nan
                else:
                    value = selected.iloc[0]

                vals.append(value)

            ax.bar(
                x + (i - 1) * width,
                vals,
                width=width,
                color=LOAD_COLORS[load],
                hatch=LOAD_HATCH[load],
                edgecolor="black",
                linewidth=0.5,
                label=load,
                zorder=3,
            )

        if legend_handles is None:
            legend_handles, legend_labels = (
                ax.get_legend_handles_labels()
            )

        ax.set_title(title, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(
            MODEL_ORDER,
            rotation=15,
            ha="right",
        )

        ax.set_ylabel("Idle (%)")

        ax.set_ylim(0, 101)

        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.4,
            alpha=0.5,
            zorder=0,
        )

        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(
        legend_handles,
        legend_labels,
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(outdir, exist_ok=True)

    fig.savefig(
        os.path.join(outdir, "idle_percentage.pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        os.path.join(outdir, "idle_percentage.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="idle_percentage.csv",
    )

    parser.add_argument(
        "--outdir",
        default="figs",
    )

    args = parser.parse_args()

    plot(args.input, args.outdir)