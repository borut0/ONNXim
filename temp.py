import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================
# CONFIG
# ============================================================

METRICS_CSV = "metrics.csv"
LAYERS_CSV = "layers.csv"

OUTPUT_FIG = "timeline_plot.png"

# ============================================================
# LOAD CSVs
# ============================================================

metrics = pd.read_csv(METRICS_CSV)
layers = pd.read_csv(LAYERS_CSV)

# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

metrics.columns = [c.strip() for c in metrics.columns]
layers.columns = [c.strip() for c in layers.columns]

# ============================================================
# DEBUG PRINT
# ============================================================

print("Metrics columns:")
print(metrics.columns)

print("\nLayers columns:")
print(layers.columns)

# ============================================================
# EXTRACT REQUIRED COLUMNS
# ============================================================

# IMPORTANT:
# Adjust these names if your CSV headers differ slightly

cycle_col = "Cycle"

pe_col = None
vec_col = None
mem_col = None
core_idle_col = None

for c in metrics.columns:
    lc = c.lower()

    if "pe" in lc and "%" in lc:
        pe_col = c

    if "vector" in lc:
        vec_col = c

    if "memoryidle" in lc or "memory idle" in lc:
        mem_col = c

    if "coreidle" in lc or "core idle" in lc:
        core_idle_col = c

print("\nDetected columns:")
print("PE:", pe_col)
print("Vector:", vec_col)
print("MemoryIdle:", mem_col)
print("CoreIdle:", core_idle_col)

# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    4,
    1,
    figsize=(18, 12),
    sharex=True
)

# ============================================================
# HELPER: COLOR MAP
# ============================================================

def get_layer_color(layer_name):
    lname = layer_name.lower()

    if "attention" in lname:
        return "lightblue"

    if "qkv" in lname:
        return "lavender"

    if "proj" in lname:
        return "lightgreen"

    if "fc1" in lname:
        return "moccasin"

    if "fc2" in lname:
        return "peachpuff"

    if "act" in lname:
        return "lightcoral"

    if "ln" in lname:
        return "lightgray"

    return "whitesmoke"

# ============================================================
# ADD LAYER SPANS TO ALL SUBPLOTS
# ============================================================

for _, row in layers.iterrows():

    start = row["StartCycle"]
    end = row["EndCycle"]
    layer = row["Layer"]

    color = get_layer_color(layer)

    for ax in axes:
        ax.axvspan(
            start,
            end,
            alpha=0.25,
            color=color
        )

# ============================================================
# PLOT 1 : PE UTIL
# ============================================================

if pe_col is not None:
    axes[0].plot(
        metrics[cycle_col],
        metrics[pe_col],
        linewidth=2
    )

axes[0].set_ylabel("PE Util (%)")
axes[0].set_title("PE Utilization vs Cycle")

# ============================================================
# PLOT 2 : VECTOR UTIL
# ============================================================

if vec_col is not None:
    axes[1].plot(
        metrics[cycle_col],
        metrics[vec_col],
        linewidth=2
    )

axes[1].set_ylabel("Vector Util (%)")
axes[1].set_title("Vector Utilization vs Cycle")

# ============================================================
# PLOT 3 : MEMORY IDLE
# ============================================================

if mem_col is not None:
    axes[2].plot(
        metrics[cycle_col],
        metrics[mem_col],
        linewidth=2
    )

axes[2].set_ylabel("Memory Idle")
axes[2].set_title("Memory Idle vs Cycle")

# ============================================================
# PLOT 4 : CORE IDLE
# ============================================================

if core_idle_col is not None:
    axes[3].plot(
        metrics[cycle_col],
        metrics[core_idle_col],
        linewidth=2
    )

axes[3].set_ylabel("Core Idle")
axes[3].set_title("Core Idle vs Cycle")

# ============================================================
# X LABEL
# ============================================================

axes[3].set_xlabel("Cycle")

# ============================================================
# GRID
# ============================================================

for ax in axes:
    ax.grid(True, alpha=0.3)

# ============================================================
# LEGEND
# ============================================================

legend_items = [
    Patch(color="lightblue", label="Attention"),
    Patch(color="lavender", label="QKV"),
    Patch(color="lightgreen", label="Projection"),
    Patch(color="moccasin", label="FC1"),
    Patch(color="peachpuff", label="FC2"),
    Patch(color="lightcoral", label="Activation"),
    Patch(color="lightgray", label="LayerNorm"),
]

axes[0].legend(
    handles=legend_items,
    loc="upper right"
)

# ============================================================
# SAVE
# ============================================================

plt.tight_layout()

plt.savefig(
    OUTPUT_FIG,
    dpi=300
)

print(f"\nSaved plot to: {OUTPUT_FIG}")

plt.show()