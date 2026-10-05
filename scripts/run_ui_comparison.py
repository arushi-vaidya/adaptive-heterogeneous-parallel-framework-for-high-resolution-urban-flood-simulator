#!/usr/bin/env python3
"""Run a serial reference and selected backend; write measured UI comparison metrics."""
import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys


def read_one(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return next(csv.DictReader(stream))


def run_adaptive(binary, scenario, rows, cols, duration, backend, threads,
                 processes, output, adaptive_binary, patch_extent, max_level,
                 regrid_interval, refine_threshold, coarsen_threshold,
                 dynamic_load_balancing, imbalance_threshold,
                 rebalance_cooldown, mpi_exec="", mpi_numproc_flag="-n",
                 mpi_preflags="", mpi_postflags=""):
    command = [
        str(adaptive_binary), "--backend", backend, "--scenario", scenario,
        "--rows", str(rows), "--cols", str(cols),
        "--patch-extent", str(patch_extent), "--max-level", str(max_level),
        "--end-time", str(duration), "--regrid-interval", str(regrid_interval),
        "--refine-threshold", str(refine_threshold),
        "--coarsen-threshold", str(coarsen_threshold),
        "--output-dir", str(output),
    ]
    if backend == "openmp":
        command.extend(("--threads", str(threads)))
    if backend == "mpi":
        if not mpi_exec:
            raise RuntimeError("MPI launcher was not detected for UI execution")
        command.extend(("--dynamic-load-balancing",
                        "true" if dynamic_load_balancing else "false",
                        "--imbalance-threshold", str(imbalance_threshold),
                        "--rebalance-cooldown", str(rebalance_cooldown)))
        command = [mpi_exec, mpi_numproc_flag, str(processes),
                   *shlex.split(mpi_preflags), *command,
                   *shlex.split(mpi_postflags)]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True)
    row = next(csv.DictReader(completed.stdout.splitlines()))
    row["backend"] = backend
    row["threads"] = threads if backend == "openmp" else 1
    row["processes"] = processes if backend == "mpi" else 1
    return row


def run(binary, scenario, rows, cols, duration, backend, threads, processes,
    output, mpi_exec="", mpi_numproc_flag="-n", mpi_preflags="", mpi_postflags=""):
    command = [str(binary), "--scenario", scenario, "--rows", str(rows), "--cols", str(cols),
               "--duration", str(duration), "--backend", backend, "--output", str(output)]
    if backend == "openmp":
        command.extend(("--threads", str(threads)))
    elif backend == "mpi":
        if not mpi_exec:
            raise RuntimeError("MPI launcher was not detected for UI execution")
        command = [mpi_exec, mpi_numproc_flag, str(processes), *shlex.split(mpi_preflags),
               *command, *shlex.split(mpi_postflags)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return read_one(output / "stats.csv"), read_one(output / "benchmark.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--backend", choices=("serial", "openmp", "mpi"), required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--mpi-exec", default="")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflags", default="")
    parser.add_argument("--mpi-postflags", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--adaptive-binary", type=Path)
    parser.add_argument("--patch-extent", type=int, default=8)
    parser.add_argument("--max-level", type=int, default=2)
    parser.add_argument("--regrid-interval", type=int, default=2)
    parser.add_argument("--refine-threshold", type=float, default=0.10)
    parser.add_argument("--coarsen-threshold", type=float, default=0.05)
    parser.add_argument("--dynamic-load-balancing", action="store_true")
    parser.add_argument("--imbalance-threshold", type=float, default=1.20)
    parser.add_argument("--rebalance-cooldown", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--momentum-tolerance", type=float, default=1e-8)
    parser.add_argument("--conservation-tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.adaptive:
        adaptive_binary = args.adaptive_binary
        if not adaptive_binary or not adaptive_binary.is_file():
            raise SystemExit("Adaptive benchmark executable is unavailable")
        if args.max_level not in (0, 1, 2) or args.patch_extent <= 0:
            raise SystemExit("Adaptive level and patch extent are invalid")
        if args.regrid_interval <= 0 or args.coarsen_threshold < 0 or \
                args.refine_threshold <= args.coarsen_threshold:
            raise SystemExit("Adaptive regridding parameters are invalid")
        if args.backend != "mpi" and args.dynamic_load_balancing:
            raise SystemExit("Dynamic load balancing requires the MPI backend")
        reference = run_adaptive(
            adaptive_binary, args.scenario, args.rows, args.cols, args.duration,
            "serial", 1, 1, args.output / "reference", adaptive_binary,
            args.patch_extent, args.max_level, args.regrid_interval,
            args.refine_threshold, args.coarsen_threshold, False,
            args.imbalance_threshold, args.rebalance_cooldown)
        candidate = run_adaptive(
            adaptive_binary, args.scenario, args.rows, args.cols, args.duration,
            args.backend, args.threads, args.processes, args.output / "current",
            adaptive_binary, args.patch_extent, args.max_level,
            args.regrid_interval, args.refine_threshold, args.coarsen_threshold,
            args.dynamic_load_balancing, args.imbalance_threshold,
            args.rebalance_cooldown, args.mpi_exec, args.mpi_numproc_flag,
            args.mpi_preflags, args.mpi_postflags)
        reference_runtime = float(reference["runtime_seconds"])
        runtime = float(candidate["runtime_seconds"])
        speedup = reference_runtime / runtime if runtime > 0 else 0.0
        rank_count = int(candidate["processes"])
        thread_count = int(candidate["threads"])
        parallelism = rank_count if args.backend == "mpi" else thread_count
        stats = {
            "time_s": float(candidate["final_time"]),
            "stored_m3": float(candidate["final_volume"]),
            "max_depth_m": float(candidate["max_depth"]),
            "flooded_area_m2": "",
            "wet_cells": candidate["wet_cells"],
            "rainfall_m3": float(candidate["rainfall_volume"]),
            "mass_residual_m3": float(candidate["mass_balance_residual"]),
            "active_leaf_cells": int(candidate["adaptive_leaf_cells"]),
            "maximum_velocity": float(candidate["maximum_velocity"]),
            "coarse_fine_interface_segments": int(candidate["coarse_fine_segments"]),
            "maximum_interface_mass_flux_residual": float(
                candidate["maximum_interface_mass_flux_residual"]),
            "communication_seconds": float(candidate["communication_seconds"]),
            "load_imbalance": float(candidate["load_imbalance"]),
            "migration_count": int(candidate["migration_count"]),
            "migrated_patches": int(candidate["migrated_patches"]),
            "migrated_cells": int(candidate["migrated_cells"]),
            "migration_time_seconds": float(candidate["migration_time_seconds"]),
            "pre_rebalance_imbalance": float(candidate["pre_rebalance_imbalance"]),
            "post_rebalance_imbalance": float(candidate["post_rebalance_imbalance"]),
            "last_rebalance_timestep": int(candidate["last_rebalance_timestep"]),
            "mesh_seconds": float(candidate["mesh_seconds"]),
            "regridding_seconds": float(candidate["regridding_seconds"]),
            "steps": int(candidate["steps"]),
            "refined_patches": int(candidate["refined_patches"]),
            "coarsened_patches": int(candidate["coarsened_patches"]),
            "active_cell_ratio": float(candidate["cell_count_ratio"]),
        }
        conservation_pass = (
            abs(stats["mass_residual_m3"]) <= args.conservation_tolerance and
            stats["maximum_interface_mass_flux_residual"] <= args.conservation_tolerance)
        summary = {
            "backend": args.backend, "threads": thread_count,
            "processes": rank_count, "adaptive": True,
            "max_level": args.max_level,
            "dynamic_load_balancing": args.dynamic_load_balancing,
            "serial_runtime": reference_runtime, "backend_runtime": runtime,
            "speedup": speedup,
            "efficiency": speedup / parallelism if parallelism > 0 else 0.0,
            "max_error": 0.0, "conservation_pass": conservation_pass,
            "validation_pass": conservation_pass, "stats": stats,
            "refinement_map_available": (args.output / "current" / "refinement.csv").exists(),
        }
        with open(args.output / "summary.json", "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
        return

    reference_stats, reference_time = run(args.binary, args.scenario, args.rows, args.cols,
        args.duration, "serial", 1, 1, args.output / "reference")
    candidate_stats, candidate_time = run(args.binary, args.scenario, args.rows, args.cols,
        args.duration, args.backend, args.threads, args.processes,
        args.output / "current", args.mpi_exec, args.mpi_numproc_flag,
        args.mpi_preflags, args.mpi_postflags)
    comparison = subprocess.run([
        sys.executable, str(Path(__file__).with_name("compare_outputs.py")),
        str(args.output / "reference" / "depth.csv"),
        str(args.output / "current" / "depth.csv"),
        "--cell-size", "5", "--tolerance", str(args.tolerance),
        "--momentum-tolerance", str(args.momentum_tolerance),
        "--reference-hu", str(args.output / "reference" / "hu.csv"),
        "--candidate-hu", str(args.output / "current" / "hu.csv"),
        "--reference-hv", str(args.output / "reference" / "hv.csv"),
        "--candidate-hv", str(args.output / "current" / "hv.csv"),
    ], capture_output=True, text=True)
    metrics = {}
    for line in comparison.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            metrics[key] = value
    if comparison.returncode:
        raise SystemExit(comparison.stderr or "Selected backend failed serial comparison")
    serial_seconds = float(reference_time["solver_seconds"])
    backend_seconds = float(candidate_time["solver_seconds"])
    speedup = serial_seconds / backend_seconds if backend_seconds > 0 else float("nan")
    conservation_residual = abs(float(candidate_stats["mass_residual_m3"]))
    reference_residual = abs(float(reference_stats["mass_residual_m3"]))
    conservation_pass = (conservation_residual <= args.conservation_tolerance and
                         reference_residual <= args.conservation_tolerance)
    summary = {
        "backend": args.backend,
        "threads": int(candidate_time["threads"]),
        "processes": int(candidate_time.get("processes", 1)),
        "serial_runtime": serial_seconds,
        "backend_runtime": backend_seconds,
        "speedup": speedup,
        "efficiency": speedup / (int(candidate_time["processes"])
            if args.backend == "mpi" else int(candidate_time["threads"])),
        "max_error": metrics["max_absolute_error_m"],
        "mean_absolute_error": metrics["mean_absolute_error_m"],
        "rmse": metrics["rmse_m"],
        "relative_error": metrics["mean_relative_error"],
        "max_hu_error_m2_s": metrics["max_hu_error_m2_s"],
        "max_hv_error_m2_s": metrics["max_hv_error_m2_s"],
        "rmse_hu_m2_s": metrics["rmse_hu_m2_s"],
        "rmse_hv_m2_s": metrics["rmse_hv_m2_s"],
        "volume_difference_m3": metrics["water_volume_difference_m3"],
        "flooded_area_difference_m2": metrics["flooded_area_difference_m2"],
        "conservation_residual_m3": conservation_residual,
        "reference_conservation_residual_m3": reference_residual,
        "conservation_tolerance_m3": args.conservation_tolerance,
        "conservation_pass": conservation_pass,
        "validation_pass": metrics["within_tolerance"] == "true" and conservation_pass,
        "stats": candidate_stats,
    }
    with open(args.output / "summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)


if __name__ == "__main__":
    main()