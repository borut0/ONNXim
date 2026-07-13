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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# ----------------------------------------------------------------------
# 2. Fixed ordering
# ----------------------------------------------------------------------

MODEL_ORDER = [
    "ResNet18",
    "OPT-125M",
    "Llama3-8B",
    "OPT-66B",
]

LOAD_ORDER = [
    "Low",
    "Medium",
    "Heavy",
]

STAT_ORDER = [
    "Mean",
    "Median",
    "P95",
    "Maximum",
]


# ----------------------------------------------------------------------
# 3. Colorblind-safe palette + grayscale hatches
# ----------------------------------------------------------------------

LOAD_COLORS = {
    "Low":    "#0072B2",
    "Medium": "#E69F00",
    "Heavy":  "#009E73",
}

LOAD_HATCH = {
    "Low":    "",
    "Medium": "///",
    "Heavy":  "xxx",
}


# ----------------------------------------------------------------------
# 4. Load CSV
# ----------------------------------------------------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load already-aggregated MC/NoC idle interval statistics.

    Expected format:

    Model Name,Load Type,Mean,Median,P95,Maximum
    """

    df = pd.read_csv(csv_path)


    required_columns = [
        "Model Name",
        "Load Type",
        "Mean",
        "Median",
        "P95",
        "Maximum",
    ]


    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]


    if missing_columns:
        raise ValueError(
            "Missing required CSV columns: "
            + ", ".join(missing_columns)
        )


    # --------------------------------------------------------------
    # Rename columns for simpler plotting
    # --------------------------------------------------------------

    df = df.rename(
        columns={
            "Model Name": "Model",
            "Load Type": "Load",
        }
    )


    # --------------------------------------------------------------
    # Convert statistics to numeric values
    # --------------------------------------------------------------

    for stat in STAT_ORDER:

        df[stat] = pd.to_numeric(
            df[stat],
            errors="raise",
        )


    return df


# ----------------------------------------------------------------------
# 5. Validate data
# ----------------------------------------------------------------------

def validate_data(df: pd.DataFrame):

    for model in MODEL_ORDER:

        for load in LOAD_ORDER:

            rows = df[
                (df["Model"] == model)
                &
                (df["Load"] == load)
            ]

            # Missing OPT-66B Medium/Heavy MC data is allowed
            if len(rows) == 0:

                if (
                    model == "OPT-66B"
                    and load in ["Medium", "Heavy"]
                ):
                    continue

                raise ValueError(
                    f"Missing data for: "
                    f"{model} / {load}"
                )

            if len(rows) > 1:

                raise ValueError(
                    f"Duplicate data for: "
                    f"{model} / {load}"
                )


# ----------------------------------------------------------------------
# 6. Plot figure
# ----------------------------------------------------------------------

def plot_figure(
    df: pd.DataFrame,
    outdir: str,
    basename: str = "mc_idle_interval_stats",
):


    os.makedirs(
        outdir,
        exist_ok=True,
    )


    n_models = len(MODEL_ORDER)

    n_loads = len(LOAD_ORDER)


    bar_width = 0.25


    x = np.arange(n_models)


    # ------------------------------------------------------------------
    # Create 2 × 2 figure
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.0, 4.6),
    )


    axes = axes.flatten()


    legend_handles = None
    legend_labels = None


    # ------------------------------------------------------------------
    # Plot Mean, Median, P95, Maximum
    # ------------------------------------------------------------------

    for ax, stat in zip(
        axes,
        STAT_ORDER,
    ):


        for i, load in enumerate(
            LOAD_ORDER
        ):


            values = []


            for model in MODEL_ORDER:
                row = df[
                    (df["Model"] == model)
                    &
                    (df["Load"] == load)
                ]

                if row.empty:
                    value = np.nan
                else:
                    value = row[stat].iloc[0]

                values.append(value)


            # ----------------------------------------------------------
            # Calculate grouped-bar offset
            # ----------------------------------------------------------

            offset = (
                i
                -
                (n_loads - 1) / 2
            ) * bar_width


            # ----------------------------------------------------------
            # Draw bars
            # ----------------------------------------------------------

            ax.bar(
                x + offset,
                values,
                width=bar_width,
                color=LOAD_COLORS[load],
                hatch=LOAD_HATCH[load],
                edgecolor="black",
                linewidth=0.5,
                label=load,
                zorder=3,
            )


        # --------------------------------------------------------------
        # Save legend handles
        # --------------------------------------------------------------

        legend_handles, legend_labels = (
            ax.get_legend_handles_labels()
        )


        # --------------------------------------------------------------
        # Logarithmic Y-axis
        # --------------------------------------------------------------

        ax.set_yscale("log")


        # --------------------------------------------------------------
        # Panel title
        # --------------------------------------------------------------

        ax.set_title(
            stat,
            fontweight="bold",
            pad=3,
        )


        # --------------------------------------------------------------
        # X-axis
        # --------------------------------------------------------------

        ax.set_xticks(x)


        ax.set_xticklabels(
            MODEL_ORDER,
            rotation=15,
            ha="right",
        )


        # --------------------------------------------------------------
        # Logarithmic tick configuration
        # --------------------------------------------------------------

        ax.yaxis.set_major_locator(

            mticker.LogLocator(
                base=10.0,
                numticks=6,
            )

        )


        ax.yaxis.set_minor_locator(

            mticker.LogLocator(
                base=10.0,
                subs=np.arange(2, 10) * 0.1,
                numticks=12,
            )

        )


        # --------------------------------------------------------------
        # Grid
        # --------------------------------------------------------------

        ax.grid(
            axis="y",
            which="major",
            linestyle="--",
            linewidth=0.4,
            alpha=0.5,
            zorder=0,
        )


        ax.set_axisbelow(True)


        # --------------------------------------------------------------
        # Remove top/right borders
        # --------------------------------------------------------------

        for spine in (
            "top",
            "right",
        ):

            ax.spines[spine].set_visible(False)


    # ------------------------------------------------------------------
    # Shared Y-axis label
    # ------------------------------------------------------------------

    fig.text(
        0.015,
        0.5,
        "MC Idle Interval (cycles, log scale)",
        va="center",
        rotation="vertical",
        fontsize=8.5,
    )


    # ------------------------------------------------------------------
    # Shared legend
    # ------------------------------------------------------------------

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
        handletextpad=0.5,
        columnspacing=1.2,
    )


    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    fig.tight_layout(
        rect=[
            0.035,
            0.0,
            1.0,
            0.93,
        ]
    )


    # ------------------------------------------------------------------
    # Output files
    # ------------------------------------------------------------------

    pdf_path = os.path.join(
        outdir,
        f"{basename}.pdf",
    )


    png_path = os.path.join(
        outdir,
        f"{basename}.png",
    )


    # ------------------------------------------------------------------
    # Save PDF
    # ------------------------------------------------------------------

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )


    # ------------------------------------------------------------------
    # Save PNG
    # ------------------------------------------------------------------

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(fig)


    return (
        pdf_path,
        png_path,
    )


# ----------------------------------------------------------------------
# 7. Main
# ----------------------------------------------------------------------

def main():


    parser = argparse.ArgumentParser(
        description=(
            "Plot MC idle interval statistics."
        )
    )


    parser.add_argument(
        "--input",
        default="mc_idle_stats.csv",
        help="Path to aggregated MC idle statistics CSV",
    )


    parser.add_argument(
        "--outdir",
        default="figs",
        help="Output directory",
    )


    parser.add_argument(
        "--basename",
        default="mc_idle_interval_stats",
        help="Output filename without extension",
    )


    args = parser.parse_args()


    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    df = load_data(
        args.input
    )


    # ------------------------------------------------------------------
    # Validate data
    # ------------------------------------------------------------------

    validate_data(
        df
    )


    # ------------------------------------------------------------------
    # Generate figure
    # ------------------------------------------------------------------

    pdf_path, png_path = plot_figure(
        df,
        args.outdir,
        args.basename,
    )


    # ------------------------------------------------------------------
    # Print output paths
    # ------------------------------------------------------------------

    print(
        f"Input data          : "
        f"{args.input}"
    )


    print(
        f"Vector figure (PDF) : "
        f"{pdf_path}"
    )


    print(
        f"Raster preview (PNG): "
        f"{png_path}"
    )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()