#!/usr/bin/env python3

import argparse
import re
import sys

import numpy as np


# ============================================================
# Regex patterns
# ============================================================

RE_NOC_IDLE = re.compile(
    r"NoC idle\s*=\s*(\d+)"
)

RE_CONTROLLER_IDLE = re.compile(
    r"Controller idle\s*=\s*(\d+)"
)

RE_ROW_HITS = re.compile(
    r"Row hits:"
)

RE_MEMORY_SYSTEM = re.compile(
    r"^\s*MemorySystem:"
)

RE_TOTAL_CYCLES = re.compile(
    r"Simulation Finished at\s+(\d+)\s+cycle"
)

# NEW:
# Extract:
# memory_system_cycles: 555249
RE_MEMORY_SYSTEM_CYCLES = re.compile(
    r"memory_system_cycles:\s*(\d+)"
)


# ============================================================
# Statistics
# ============================================================

def calculate_stats(values):
    """
    Calculate Mean, Median, P95, Maximum, Total and Count.
    """

    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "maximum": 0.0,
            "total": 0,
            "count": 0,
        }

    values = np.asarray(values, dtype=np.float64)

    return {
        "mean": np.mean(values),
        "median": np.median(values),
        "p95": np.percentile(values, 95),
        "maximum": np.max(values),
        "total": np.sum(values),
        "count": len(values),
    }


# ============================================================
# Parse log
# ============================================================

def parse_log(logfile):

    noc_values = []

    controller_blocks = []

    current_controller_values = []

    collecting_controller = False

    total_cycles = None

    # NEW:
    # Store all memory_system_cycles values found in the log.
    memory_system_cycles_values = []

    with open(logfile, "r", errors="replace") as f:

        for line in f:

            # ------------------------------------------------
            # Total NPU simulation cycles
            # ------------------------------------------------

            match = RE_TOTAL_CYCLES.search(line)

            if match:
                total_cycles = int(match.group(1))


            # ------------------------------------------------
            # Memory system cycles
            # ------------------------------------------------

            match = RE_MEMORY_SYSTEM_CYCLES.search(line)

            if match:
                value = int(match.group(1))
                memory_system_cycles_values.append(value)


            # ------------------------------------------------
            # NoC idle
            # ------------------------------------------------

            match = RE_NOC_IDLE.search(line)

            if match:
                value = int(match.group(1))
                noc_values.append(value)


            # ------------------------------------------------
            # Start of controller block
            # ------------------------------------------------

            if RE_ROW_HITS.search(line):

                # Safety check:
                # Save previous unfinished block if necessary.

                if current_controller_values:

                    controller_blocks.append(
                        current_controller_values
                    )

                current_controller_values = []

                collecting_controller = True

                continue


            # ------------------------------------------------
            # Collect Controller idle values
            # ------------------------------------------------

            if collecting_controller:

                match = RE_CONTROLLER_IDLE.search(line)

                if match:

                    value = int(match.group(1))

                    current_controller_values.append(value)

                    continue


            # ------------------------------------------------
            # End of controller block
            # ------------------------------------------------

            if (
                collecting_controller
                and RE_MEMORY_SYSTEM.search(line)
            ):

                if current_controller_values:

                    controller_blocks.append(
                        current_controller_values
                    )

                current_controller_values = []

                collecting_controller = False


    # --------------------------------------------------------
    # Save final unfinished controller block if necessary
    # --------------------------------------------------------

    if current_controller_values:

        controller_blocks.append(
            current_controller_values
        )


    # --------------------------------------------------------
    # Determine memory_system_cycles
    # --------------------------------------------------------

    memory_system_cycles = None

    if memory_system_cycles_values:

        # Find all unique values.
        unique_values = set(memory_system_cycles_values)

        if len(unique_values) == 1:

            # All MC reports contain the same value.
            memory_system_cycles = (
                memory_system_cycles_values[0]
            )

        else:

            print()
            print("WARNING:")
            print(
                "Different memory_system_cycles values "
                "were found:"
            )

            for value in sorted(unique_values):
                print(f"  {value:,}")

            print()
            print(
                "Using the maximum "
                "memory_system_cycles value."
            )

            memory_system_cycles = max(
                memory_system_cycles_values
            )


    return (
        noc_values,
        controller_blocks,
        total_cycles,
        memory_system_cycles,
        memory_system_cycles_values,
    )


# ============================================================
# Print NoC statistics
# ============================================================

def print_noc_stats(noc_values):

    print()
    print("=" * 70)
    print("NoC IDLE INTERVAL STATISTICS")
    print("=" * 70)

    if not noc_values:

        print("No NoC idle values found.")

        return None


    stats = calculate_stats(noc_values)


    print(f"Number of intervals : {stats['count']:,}")
    print(f"Total idle cycles   : {stats['total']:,.0f}")

    print()

    print(f"Mean                : {stats['mean']:,.2f}")
    print(f"Median              : {stats['median']:,.2f}")
    print(f"P95                 : {stats['p95']:,.2f}")
    print(f"Maximum             : {stats['maximum']:,.2f}")


    return stats


# ============================================================
# Print Controller statistics
# ============================================================

def print_controller_stats(controller_blocks):

    print()
    print("=" * 70)
    print("PER-MC IDLE INTERVAL STATISTICS")
    print("=" * 70)


    if not controller_blocks:

        print("No MC idle blocks found.")

        return None


    controller_stats = []


    # --------------------------------------------------------
    # Calculate statistics separately for every MC
    # --------------------------------------------------------

    for controller_id, values in enumerate(
        controller_blocks
    ):

        stats = calculate_stats(values)

        controller_stats.append(stats)


        print(
            f"MC {controller_id:2d} | "
            f"Count = {stats['count']:6,d} | "
            f"Mean = {stats['mean']:12,.2f} | "
            f"Median = {stats['median']:10,.2f} | "
            f"P95 = {stats['p95']:12,.2f} | "
            f"Max = {stats['maximum']:12,.2f}"
        )


    # --------------------------------------------------------
    # Average statistics across all MCs
    # --------------------------------------------------------

    final_stats = {

        "mean": np.mean([
            s["mean"]
            for s in controller_stats
        ]),

        "median": np.mean([
            s["median"]
            for s in controller_stats
        ]),

        "p95": np.mean([
            s["p95"]
            for s in controller_stats
        ]),

        "maximum": np.mean([
            s["maximum"]
            for s in controller_stats
        ]),
    }


    print()

    print("=" * 70)
    print("MC-AVERAGED IDLE INTERVAL STATISTICS")
    print("=" * 70)

    print(
        f"Mean    : {final_stats['mean']:,.2f}"
    )

    print(
        f"Median  : {final_stats['median']:,.2f}"
    )

    print(
        f"P95     : {final_stats['p95']:,.2f}"
    )

    print(
        f"Maximum : {final_stats['maximum']:,.2f}"
    )


    return final_stats


# ============================================================
# Print idle percentages
# ============================================================

def print_idle_percentages(
    noc_values,
    controller_blocks,
    total_cycles,
    memory_system_cycles,
):

    print()
    print("=" * 70)
    print("IDLE PERCENTAGES")
    print("=" * 70)


    # --------------------------------------------------------
    # NoC idle percentage
    #
    # NoC idle intervals use Interconnect clock.
    # Existing logs use Simulation Finished cycles.
    # --------------------------------------------------------

    if total_cycles is not None:

        total_noc_idle = sum(noc_values)

        noc_percentage = (
            total_noc_idle
            /
            total_cycles
        ) * 100


        print(
            f"Simulation cycles          : "
            f"{total_cycles:,}"
        )

        print(
            f"Total NoC idle cycles      : "
            f"{total_noc_idle:,}"
        )

        print(
            f"NoC idle percentage        : "
            f"{noc_percentage:.4f}%"
        )

    else:

        print(
            "NoC percentage: Simulation cycles "
            "were not found."
        )


    print()


    # --------------------------------------------------------
    # MC idle percentage
    #
    # CORRECT FORMULA:
    #
    # Total MC idle
    # ------------------------------- × 100
    # Number of MCs × Memory cycles
    #
    # --------------------------------------------------------

    number_of_controllers = len(
        controller_blocks
    )


    total_controller_idle = sum(

        sum(values)

        for values in controller_blocks

    )


    if (
        number_of_controllers > 0
        and memory_system_cycles is not None
    ):

        total_available_mc_cycles = (

            number_of_controllers
            *
            memory_system_cycles

        )


        controller_percentage = (

            total_controller_idle
            /
            total_available_mc_cycles

        ) * 100


        print(
            f"Memory system cycles       : "
            f"{memory_system_cycles:,}"
        )

        print(
            f"Number of MCs              : "
            f"{number_of_controllers}"
        )

        print(
            f"Available MC cycles        : "
            f"{total_available_mc_cycles:,}"
        )

        print(
            f"Total MC idle cycles       : "
            f"{total_controller_idle:,}"
        )

        print(
            f"MC idle percentage         : "
            f"{controller_percentage:.4f}%"
        )


        # ----------------------------------------------------
        # Sanity check
        # ----------------------------------------------------

        if controller_percentage > 100.0:

            print()

            print(
                "WARNING: MC idle percentage "
                "is greater than 100%."
            )

            print(
                "Check MC idle interval tracking "
                "or clock-domain assumptions."
            )


    else:

        print(
            "MC percentage could not be calculated."
        )

        if memory_system_cycles is None:

            print(
                "Reason: memory_system_cycles "
                "was not found."
            )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Calculate NoC and MC idle "
            "interval statistics."
        )

    )


    parser.add_argument(

        "logfile",

        help="Path to simulation log file"

    )


    parser.add_argument(

        "--controllers",

        type=int,

        default=16,

        help=(
            "Expected number of MCs "
            "(default: 16)"
        )

    )


    args = parser.parse_args()


    # --------------------------------------------------------
    # Parse log
    # --------------------------------------------------------

    try:

        (
            noc_values,
            controller_blocks,
            total_cycles,
            memory_system_cycles,
            memory_system_cycles_values,

        ) = parse_log(args.logfile)


    except FileNotFoundError:

        print(
            f"ERROR: File not found: "
            f"{args.logfile}"
        )

        sys.exit(1)


    # --------------------------------------------------------
    # Log parsing summary
    # --------------------------------------------------------

    print()

    print("=" * 70)
    print("LOG PARSING SUMMARY")
    print("=" * 70)


    print(
        f"Log file                   : "
        f"{args.logfile}"
    )


    print(
        f"NoC idle intervals          : "
        f"{len(noc_values):,}"
    )


    print(
        f"MC blocks found             : "
        f"{len(controller_blocks)}"
    )


    print(
        f"Expected MCs                : "
        f"{args.controllers}"
    )


    if total_cycles is not None:

        print(
            f"Simulation cycles           : "
            f"{total_cycles:,}"
        )

    else:

        print(
            "Simulation cycles           : "
            "NOT FOUND"
        )


    if memory_system_cycles is not None:

        print(
            f"Memory system cycles        : "
            f"{memory_system_cycles:,}"
        )

        print(
            f"memory_system_cycles entries: "
            f"{len(memory_system_cycles_values)}"
        )

    else:

        print(
            "Memory system cycles        : "
            "NOT FOUND"
        )


    # --------------------------------------------------------
    # Validate MC count
    # --------------------------------------------------------

    if len(controller_blocks) != args.controllers:

        print()

        print("WARNING:")

        print(
            f"Expected {args.controllers} MC blocks, "
            f"but found {len(controller_blocks)}."
        )


    # --------------------------------------------------------
    # Calculate and print statistics
    # --------------------------------------------------------

    print_noc_stats(
        noc_values
    )


    print_controller_stats(
        controller_blocks
    )


    print_idle_percentages(

        noc_values,

        controller_blocks,

        total_cycles,

        memory_system_cycles,

    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()