#!/usr/bin/env python3
"""Combine the measured uniform and adaptive final-study CSVs."""

import argparse
import csv
from pathlib import Path
import statistics


FIELDS = (
    "family", "scenario", "rows", "cols", "backend", "threads", "ranks",
    "dynamic_load_balancing", "max_level", "patch_extent", "repeats",
    "median_runtime_seconds", "serial_speedup", "parallel_efficiency",
    "adaptive_overhead_vs_uniform_serial", "static_over_dynamic_speedup",
    "active_leaf_cells", "active_cell_ratio", "workload_imbalance",
    "pre_rebalance_imbalance", "post_rebalance_imbalance",
    "migration_count", "migrated_patches", "migrated_cells",
    "migration_time_seconds", "migration_overhead_fraction",
    "compute_seconds", "communication_seconds", "mesh_seconds",
    "regridding_seconds", "steps", "mass_balance_residual",
    "interface_conservation_residual", "refined_patches", "coarsened_patches",
)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def median_value(rows, name, default=""):
    values = []
    for row in rows:
        raw = row.get(name, "")
        if raw not in ("", None):
            try:
                values.append(float(raw))
            except ValueError:
                pass
    return statistics.median(values) if values else default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniform", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    uniform_rows = read_rows(args.uniform)
    adaptive_rows = read_rows(args.adaptive)

    uniform_baselines = {}
    for row in uniform_rows:
        key = (row["scenario"], row["grid_rows"], row["grid_cols"])
        if row["backend"] == "serial":
            uniform_baselines[key] = float(row["backend_runtime"])

    adaptive_groups = {}
    for row in adaptive_rows:
        key = (
            row["scenario"], row["rows"], row["cols"], row["backend"],
            row["threads"], row["rank_count"], row["dynamic_load_balancing"],
        )
        adaptive_groups.setdefault(key, []).append(row)
    adaptive_serial = {}
    adaptive_mpi = {}
    for key, rows in adaptive_groups.items():
        scenario, grid_rows, grid_cols, backend, threads, ranks, dynamic = key
        runtime = median_value(rows, "runtime_seconds")
        if backend == "serial":
            adaptive_serial[(scenario, grid_rows, grid_cols)] = runtime
        if backend == "mpi":
            adaptive_mpi[(scenario, grid_rows, grid_cols, ranks, dynamic)] = runtime

    uniform_groups = {}
    for row in uniform_rows:
        key = (row["scenario"], row["grid_rows"], row["grid_cols"],
               row["backend"], row.get("threads", ""), row.get("processes", ""))
        uniform_groups.setdefault(key, []).append(row)

    output_rows = []
    for key, rows in uniform_groups.items():
        scenario, grid_rows, grid_cols, backend, threads, ranks = key
        rows_num, cols_num = int(grid_rows), int(grid_cols)
        runtime = median_value(rows, "backend_runtime")
        serial_runtime = median_value(rows, "serial_runtime")
        parallelism = (int(ranks) if backend == "mpi" else
                       int(threads) if backend == "openmp" else 1)
        speedup = serial_runtime / runtime if runtime else ""
        output_rows.append({
            "family": "uniform", "scenario": scenario,
            "rows": rows_num, "cols": cols_num,
            "backend": backend, "threads": threads, "ranks": ranks,
            "dynamic_load_balancing": "false",
            "repeats": int(median_value(rows, "repetitions", len(rows))),
            "median_runtime_seconds": runtime,
            "serial_speedup": speedup,
            "parallel_efficiency": speedup / parallelism if speedup != "" else "",
            "active_leaf_cells": rows_num * cols_num, "active_cell_ratio": 1.0,
            "compute_seconds": median_value(rows, "computation_seconds"),
            "communication_seconds": median_value(rows, "communication_seconds"),
            "steps": median_value(rows, "steps"),
            "mass_balance_residual": median_value(rows, "conservation_residual_m3"),
        })

    for key, rows in adaptive_groups.items():
        scenario, grid_rows, grid_cols, backend, threads, ranks, dynamic = key
        rows_num, cols_num = int(grid_rows), int(grid_cols)
        runtime = median_value(rows, "runtime_seconds")
        base_key = (scenario, grid_rows, grid_cols)
        serial_runtime = adaptive_serial.get(base_key, "")
        count = int(ranks) if backend == "mpi" else int(threads) if backend == "openmp" else 1
        speedup = (serial_runtime / runtime
                   if serial_runtime not in ("", 0) and runtime else "")
        efficiency = speedup / count if speedup != "" else ""
        cell_count = median_value(rows, "adaptive_leaf_cells")
        active_ratio = cell_count / (rows_num * cols_num) if cell_count != "" else ""
        uniform_key = (scenario, grid_rows, grid_cols)
        uniform_serial_runtime = uniform_baselines.get(uniform_key, "")
        adaptive_overhead = (runtime / uniform_serial_runtime
                             if uniform_serial_runtime and runtime else "")
        static_dynamic = ""
        if backend == "mpi" and dynamic == "true":
            static_runtime = adaptive_mpi.get(
                (scenario, grid_rows, grid_cols, ranks, "false"), "")
            if static_runtime and runtime:
                static_dynamic = static_runtime / runtime
        migration_time = median_value(rows, "migration_time_seconds")
        record = {
            "family": "adaptive", "scenario": scenario,
            "rows": rows_num, "cols": cols_num, "backend": backend,
            "threads": threads, "ranks": ranks,
            "dynamic_load_balancing": dynamic,
            "max_level": median_value(rows, "max_level"),
            "patch_extent": median_value(rows, "patch_extent"),
            "repeats": len(rows), "median_runtime_seconds": runtime,
            "serial_speedup": speedup, "parallel_efficiency": efficiency,
            "adaptive_overhead_vs_uniform_serial": adaptive_overhead,
            "static_over_dynamic_speedup": static_dynamic,
            "active_leaf_cells": cell_count, "active_cell_ratio": active_ratio,
            "workload_imbalance": median_value(rows, "load_imbalance"),
            "pre_rebalance_imbalance": median_value(rows, "pre_rebalance_imbalance"),
            "post_rebalance_imbalance": median_value(rows, "post_rebalance_imbalance"),
            "migration_count": median_value(rows, "migration_count"),
            "migrated_patches": median_value(rows, "migrated_patches"),
            "migrated_cells": median_value(rows, "migrated_cells"),
            "migration_time_seconds": migration_time,
            "migration_overhead_fraction": (
                migration_time / runtime if migration_time != "" and runtime else ""),
            "compute_seconds": median_value(rows, "compute_seconds"),
            "communication_seconds": median_value(rows, "communication_seconds"),
            "mesh_seconds": median_value(rows, "mesh_seconds"),
            "regridding_seconds": median_value(rows, "regridding_seconds"),
            "steps": median_value(rows, "steps"),
            "mass_balance_residual": median_value(rows, "mass_balance_residual"),
            "interface_conservation_residual": median_value(
                rows, "maximum_interface_mass_flux_residual"),
            "refined_patches": median_value(rows, "refined_patches"),
            "coarsened_patches": median_value(rows, "coarsened_patches"),
        }
        output_rows.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} median benchmark rows to {args.output}")


if __name__ == "__main__":
    main()
