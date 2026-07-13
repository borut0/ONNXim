import re
import csv
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("log")
parser.add_argument("--model", required=True)
parser.add_argument("--load", required=True)
parser.add_argument("--output", default="summary.csv")
args = parser.parse_args()

with open(args.log, "r", errors="ignore") as f:
    text = f.read()

# --------------------------------------------------
# Parse ONLY the final summary
# --------------------------------------------------

finish_pos = text.find("Simulation Finished at")

if finish_pos == -1:
    raise RuntimeError("Could not find 'Simulation Finished at' in log.")

final_text = text[finish_pos:]

summary = {}

summary["Model"] = args.model
summary["Load"] = args.load

# --------------------------------------------------
# Overall simulation
# --------------------------------------------------

m = re.search(r"Simulation Finished at\s+(\d+)\s+cycle", final_text)
summary["Total_Cycles"] = int(m.group(1)) if m else ""

m = re.search(r"Simulation time:\s*([\d\.]+)\s*seconds", final_text)
summary["Simulation_Time_sec"] = float(m.group(1)) if m else ""

m = re.search(
    r"Total tile:\s*(\d+),\s*simulated tile per seconds\(TPS\):\s*([\d\.]+)",
    final_text,
)

if m:
    summary["Total_Tiles"] = int(m.group(1))
    summary["TPS"] = float(m.group(2))
else:
    summary["Total_Tiles"] = ""
    summary["TPS"] = ""

# --------------------------------------------------
# Core statistics
# --------------------------------------------------

for core in range(4):

    pattern = rf"""
Core\s+\[{core}\]\s*:\s*
MatMul\s+active\s+cycle\s+(\d+)\s+
Vector\s+active\s+cycle\s+(\d+).*?

Core\s+\[{core}\]\s*:\s*
Memory\s+unit\s+idle\s+cycle\s+(\d+)\s+
Systolic\s+bubble\s+cycle\s+(\d+)\s+
Core\s+idle\s+cycle\s+(\d+).*?

Core\s+\[{core}\]\s*:\s*
Systolic\s+Array\s+Utilization\(%\)\s*
([\d\.]+)\s*
\(([\d\.]+)%\s*PE\s*util\),\s*
Vector\s+Unit\s+Utilization\(%\)\s*
([\d\.]+).*?

Core\s+\[{core}\]\s*:\s*
Systolic\s+Inst\s+Issue\s+Count\s*:\s*(\d+).*?

Core\s+\[{core}\]\s*:\s*
Systolic\s+PRELOAD\s+Issue\s+Count\s*:\s*(\d+)
"""

    m = re.search(pattern, final_text, re.S | re.X)

    if not m:
        raise RuntimeError(f"Failed to parse statistics for Core {core}")

    summary[f"Core{core}_MatMul_Active_Cycles"] = int(m.group(1))
    summary[f"Core{core}_Vector_Active_Cycles"] = int(m.group(2))
    summary[f"Core{core}_Memory_Idle_Cycles"] = int(m.group(3))
    summary[f"Core{core}_Systolic_Bubble_Cycles"] = int(m.group(4))
    summary[f"Core{core}_Core_Idle_Cycles"] = int(m.group(5))
    summary[f"Core{core}_SA_Util"] = float(m.group(6))
    summary[f"Core{core}_PE_Util"] = float(m.group(7))
    summary[f"Core{core}_Vector_Util"] = float(m.group(8))
    summary[f"Core{core}_SA_Issue_count"] = int(m.group(9))
    summary[f"Core{core}_Preload_Issue_count"] = int(m.group(10))

# --------------------------------------------------
# Channel 0 DRAM statistics
# --------------------------------------------------

m = re.search(
    r"Row hits:\s*(\d+),\s*"
    r"Row misses:\s*(\d+),\s*"
    r"Row conflicts:\s*(\d+).*?"
    r"total_num_write_requests:\s*(\d+).*?"
    r"total_num_read_requests:\s*(\d+).*?"
    r"HBM2-CH_0:\s*avg BW utilization\s*(\d+)%",
    final_text,
    re.S,
)

if m:
    summary["CH0_RowHits"] = int(m.group(1))
    summary["CH0_RowMisses"] = int(m.group(2))
    summary["CH0_RowConflicts"] = int(m.group(3))
    summary["CH0_Writes"] = int(m.group(4))
    summary["CH0_Reads"] = int(m.group(5))
    summary["CH0_BW_Util_%"] = int(m.group(6))
else:
    print("Warning: Could not parse CH0 DRAM statistics.")

# --------------------------------------------------
# Write CSV
# --------------------------------------------------

file_exists = os.path.exists(args.output)

with open(args.output, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary.keys())

    if not file_exists:
        writer.writeheader()

    writer.writerow(summary)

print(f"Summary written to {args.output}")