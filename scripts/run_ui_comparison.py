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
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--momentum-tolerance", type=float, default=1e-8)
    parser.add_argument("--conservation-tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
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