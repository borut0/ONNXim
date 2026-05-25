#!/usr/bin/env python3
"""
Automates ONNXim simulation sweep over prompt_length = 2^0 to 2^5 (1, 2, 4, 8, 16, 32).

For each prompt_length:
  1. Writes traces/input.csv with the new prompt_length
  2. Runs the simulator and saves the log to logs/log_seq_<prompt_length>.log
After all runs:
  3. Calls python3 scripts/plot_statistics.py logs/
"""

import subprocess
import csv
import os
import sys

# ── Configuration ─────────────────────────────────────────────────────────────

PROMPT_LENGTHS = [2**i for i in range(15)]   # [1, 2, 4, 8, 16, 32]

TRACE_FILE     = "traces/input.csv"
LOGS_DIR       = "logs"

SIMULATOR_CMD  = [
    "./build/bin/Simulator",
    "--config",  "configs/systolic_ws_128x128_c4_booksim2_transformer_tpuv4.json",
    "--models_list", "example/tiny_transformer.json",
    "--mode",    "language",
    "--trace_file", "input.csv",          # simulator resolves relative to its cwd
    "--log_level", "debug",
]

PLOT_CMD = ["python3", "scripts/plot_statistics.py", LOGS_DIR]

# Fixed fields (everything except prompt_length stays the same)
TIME           = 0
TARGET_LENGTH  = 2
CACHED_LENGTH  = 1024

# ── Helpers ───────────────────────────────────────────────────────────────────

def write_trace(prompt_length: int) -> None:
    """Overwrite traces/input.csv with the given prompt_length."""
    os.makedirs(os.path.dirname(TRACE_FILE) or ".", exist_ok=True)
    with open(TRACE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "prompt_length", "target_length", "cached_length"])
        writer.writerow([TIME, prompt_length, TARGET_LENGTH, CACHED_LENGTH])
    print(f"  [trace] wrote prompt_length={prompt_length} → {TRACE_FILE}")


def run_simulation(prompt_length):
    """Run the simulator and redirect stdout+stderr to the log file."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f"log_seq_{prompt_length}.log")

    print(f"  [sim]   running → {log_path}")
    with open(log_path, "w") as log_file:
        result = subprocess.run(
            SIMULATOR_CMD,
            stdout=log_file,
            stderr=subprocess.STDOUT,   # merge stderr into the same log
        )

    if result.returncode != 0:
        print(f"  [WARN]  simulator exited with code {result.returncode} "
              f"for prompt_length={prompt_length}")
        return False

    print(f"  [sim]   done (exit 0)")
    return True


def run_plot():
    """Call the plotting script on the logs directory."""
    print(f"\n[plot]  running: {' '.join(PLOT_CMD)}")
    result = subprocess.run(PLOT_CMD)
    if result.returncode != 0:
        print(f"[WARN]  plot script exited with code {result.returncode}")
    else:
        print("[plot]  done")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"Prompt-length sweep: {PROMPT_LENGTHS}")
    print("=" * 60)

    failed = []

    for pl in PROMPT_LENGTHS:
        print(f"\n── prompt_length = {pl} ──")
        write_trace(pl)
        ok = run_simulation(pl)
        if not ok:
            failed.append(pl)

    if failed:
        print(f"\n[WARN]  {len(failed)} run(s) failed: prompt_lengths = {failed}")
    else:
        print(f"\n[OK]    All {len(PROMPT_LENGTHS)} simulations completed successfully.")

    run_plot()

    print("\nDone.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()