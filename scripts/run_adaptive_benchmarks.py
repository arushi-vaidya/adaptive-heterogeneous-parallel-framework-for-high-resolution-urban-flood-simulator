#!/usr/bin/env python3
"""Run repeatable adaptive-solver baselines into a separate raw-results CSV."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time


SCENARIOS = ("dam-break", "rain-drain", "localized-refinement")
CSV_FIELDS = (
    "scenario", "backend", "rows", "cols", "patch_extent", "max_level", "base_cells",
    "adaptive_leaf_cells", "cell_count_ratio", "level0_cells", "level1_cells",
    "level2_cells", "active_patches", "maximum_active_level",
    "refined_patches", "coarsened_patches", "steps", "threads", "rank_count",
    "dynamic_load_balancing", "migration_count", "migrated_patches",
    "migrated_cells", "migration_time_seconds", "pre_rebalance_imbalance",
    "post_rebalance_imbalance", "last_rebalance_timestep",
    "runtime_seconds", "wall_seconds", "compute_seconds", "mesh_seconds",
    "cfl_seconds", "interface_seconds", "update_seconds", "activity_seconds",
    "regridding_seconds",
    "communication_seconds", "diagnostics_seconds", "minimum_rank_work",
    "maximum_rank_work", "mean_rank_work", "load_imbalance",
    "rank_workload_distribution", "workload_snapshots", "initial_volume", "final_volume",
    "rainfall_volume", "infiltration_volume", "outflow_volume",
    "mass_balance_residual", "maximum_velocity",
    "maximum_interface_mass_flux_residual", "maximum_regrid_volume_delta",
    "repeat", "timestamp_utc",
)


def run_one(binary, backend, scenario, rows, cols, patch_extent, max_level,
            threads, processes, cell_size, end_time, mpi_launcher,
            mpi_numproc_flag, timeout, dynamic_load_balancing=False,
            imbalance_threshold=1.20, rebalance_cooldown=10,
            output_directory=None):
    command = [
        str(binary), "--backend", backend, "--scenario", scenario,
        "--rows", str(rows), "--cols", str(cols),
        "--patch-extent", str(patch_extent), "--max-level", str(max_level),
        "--threads", str(threads), "--cell-size", str(cell_size),
        "--end-time", str(end_time),
    ]
    if output_directory is not None:
        command.extend(("--output-dir", str(output_directory)))
    if dynamic_load_balancing:
        command.extend(("--dynamic-load-balancing", "true",
                        "--imbalance-threshold", str(imbalance_threshold),
                        "--rebalance-cooldown", str(rebalance_cooldown)))
    if backend == "mpi":
        command = [mpi_launcher, mpi_numproc_flag, str(processes), *command]
    started = time.perf_counter()
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=timeout)
    wall_seconds = time.perf_counter() - started
    result = next(csv.DictReader(completed.stdout.splitlines()))
    result["patch_extent"] = patch_extent
    result["max_level"] = max_level
    result["wall_seconds"] = wall_seconds
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path,
                        default=Path("build-mpi/flood_adaptive_benchmark"))
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--backends", default="serial,openmp,mpi")
    parser.add_argument("--threads", default="1,2,4")
    parser.add_argument("--processes", default="1,2,4")
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--cols", type=int, default=512)
    parser.add_argument("--patch-extent", type=int, default=8)
    parser.add_argument("--max-level", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cell-size", type=float, default=1.0)
    parser.add_argument("--end-time", type=float, default=0.12)
    parser.add_argument("--mpi-exec", default=None)
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--imbalance-threshold", type=float, default=1.20)
    parser.add_argument("--rebalance-cooldown", type=int, default=10)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/adaptive/final-adaptive-raw.csv"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sizes", default="")
    args = parser.parse_args()

    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"adaptive benchmark executable does not exist: {binary}")
    if args.output.exists() and not args.overwrite:
        parser.error(f"refusing to overwrite existing raw results: {args.output}")
    if min(args.rows, args.cols, args.patch_extent, args.repeats) <= 0:
        parser.error("rows, cols, patch extent, and repeats must be positive")
    if args.rows < 2 or args.cols < 2 or args.max_level not in (0, 1, 2):
        parser.error("grid dimensions must be >=2 and max level must be 0, 1, or 2")
    if args.cell_size <= 0 or args.end_time <= 0 or args.timeout <= 0:
        parser.error("cell size, end time, and timeout must be positive")
    if args.imbalance_threshold < 1.0 or args.rebalance_cooldown < 0:
        parser.error("imbalance threshold must be >= 1 and cooldown nonnegative")

    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    if not scenarios or any(item not in SCENARIOS for item in scenarios):
        parser.error(f"scenarios must be selected from {', '.join(SCENARIOS)}")
    if not backends or any(item not in ("serial", "openmp", "mpi") for item in backends):
        parser.error("backends must be selected from serial, openmp, mpi")
    try:
        threads = [int(item) for item in args.threads.split(",")]
        processes = [int(item) for item in args.processes.split(",")]
    except ValueError:
        parser.error("thread and process counts must be comma-separated integers")
    if any(value <= 0 for value in threads + processes):
        parser.error("thread and process counts must be positive")
    try:
        sizes = ([tuple(int(part) for part in item.lower().split("x", 1))
                  for item in args.sizes.split(",") if item]
                 if args.sizes else [(args.rows, args.cols)])
    except ValueError:
        parser.error("sizes must use comma-separated ROWSxCOLS values")
    if any(len(size) != 2 or min(size) < 2 for size in sizes):
        parser.error("every grid size must have two dimensions >= 2")
    mpi_launcher = args.mpi_exec or shutil.which("mpiexec") or shutil.which("mpirun")
    if "mpi" in backends and not mpi_launcher:
        parser.error("MPI was requested but no mpiexec/mpirun launcher was found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_name(args.output.stem + "-metadata.json")
    if metadata_path.exists() and not args.overwrite:
        parser.error(f"refusing to overwrite existing metadata: {metadata_path}")
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "binary": str(binary),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "scenarios": scenarios,
        "backends": backends,
        "thread_counts": threads,
        "process_counts": processes,
        "repeats": args.repeats,
        "grid_sizes": [f"{rows}x{cols}" for rows, cols in sizes],
        "patch_extent": args.patch_extent,
        "max_level": args.max_level,
        "cell_size": args.cell_size,
        "end_time": args.end_time,
        "imbalance_threshold": args.imbalance_threshold,
        "rebalance_cooldown": args.rebalance_cooldown,
        "mpi_launcher": mpi_launcher,
    }

    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for rows, cols in sizes:
            for scenario in scenarios:
                configurations = [("serial", 1, 1, False)]
                if "openmp" in backends:
                    configurations.extend(("openmp", count, 1, False) for count in threads)
                if "mpi" in backends:
                    configurations.extend(("mpi", 1, count, False) for count in processes)
                    configurations.extend(("mpi", 1, count, True) for count in processes)
                for backend, thread_count, process_count, dynamic in configurations:
                    if backend not in backends:
                        continue
                    durations = []
                    for repeat in range(1, args.repeats + 1):
                        row = run_one(
                            binary, backend, scenario, rows, cols,
                            args.patch_extent, args.max_level, thread_count,
                            process_count, args.cell_size, args.end_time,
                            mpi_launcher, args.mpi_numproc_flag, args.timeout,
                            dynamic, args.imbalance_threshold,
                            args.rebalance_cooldown)
                        row["repeat"] = repeat
                        row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
                        durations.append(float(row["runtime_seconds"]))
                        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
                        stream.flush()
                    median = statistics.median(durations)
                    print(f"{rows}x{cols} {scenario}: {backend} dynamic={dynamic} count="
                          f"{process_count if backend == 'mpi' else thread_count} "
                          f"median solver seconds={median:.9g}")

    metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
