#!/usr/bin/env python3
"""Run measured serial/OpenMP benchmarks and persist only observed results."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SCENARIOS = ("dam-break", "rain-drain", "localized-refinement")
FIELDS = (
    "scenario", "grid_rows", "grid_cols", "rows", "cols", "backend", "threads",
    "processes", "steps", "serial_runtime", "backend_runtime",
    "serial_wall_runtime", "backend_wall_runtime", "speedup", "efficiency",
    "max_depth_m", "max_depth_difference_m", "max_error", "mean_absolute_error_m",
    "rmse", "relative_error", "max_hu_error_m2_s", "max_hv_error_m2_s",
    "rmse_hu_m2_s", "rmse_hv_m2_s", "max_velocity_error_m_s", "flooded_area_m2",
    "flooded_area_difference_m2", "water_volume_m3", "water_volume_difference_m3",
    "conservation_residual_m3", "serial_conservation_residual_m3", "wet_cells_match",
    "memory_mb",
    "serial_memory_mb",
    "tolerance_m", "momentum_tolerance_m2_s", "conservation_tolerance_m3",
    "conservation_pass", "within_tolerance", "status", "repetitions", "timestamp_utc",
    "compiler", "compiler_version", "operating_system", "cpu",
    "setup_seconds", "computation_seconds", "communication_seconds", "synchronization_seconds",
    "other_seconds", "timing_sum_residual_seconds",
)


def run_once(binary, backend, scenario, rows, cols, threads, processes, cell_size,
             directory, timeout, duration=None, mpi_launcher=None, mpi_numproc_flag="-n",
             mpi_preflags=(), mpi_postflags=()):
    command = [str(binary), "--backend", backend, "--scenario", scenario,
               "--rows", str(rows), "--cols", str(cols), "--cell-size", str(cell_size),
               "--output", str(directory)]
    if duration is not None:
        command.extend(("--duration", str(duration)))
    if backend == "openmp":
        command.extend(("--threads", str(threads)))
    elif backend == "mpi":
        if not mpi_launcher:
            raise RuntimeError("MPI launcher was not detected")
        command = [mpi_launcher, mpi_numproc_flag, str(processes), *mpi_preflags,
               *command, *mpi_postflags]
    time_command = "/usr/bin/time"
    measure_memory = Path(time_command).is_file()
    timed_command = ([time_command, "-l"] if sys.platform == "darwin" else
                     [time_command, "-v"]) + command if measure_memory else command
    started = time.perf_counter()
    completed = subprocess.run(timed_command, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, text=True, timeout=timeout)
    wall_seconds = time.perf_counter() - started
    memory_mb = ""
    if sys.platform == "darwin":
        match = re.search(r"([\d,]+)\s+maximum resident set size", completed.stderr)
        if match:
            memory_mb = int(match.group(1).replace(",", "")) / (1024 * 1024)
    else:
        match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", completed.stderr)
        if match:
            memory_mb = int(match.group(1)) / 1024
    if backend == "mpi":
        memory_mb = ""
    with open(directory / "benchmark.csv", newline="", encoding="utf-8") as stream:
        timing = next(csv.DictReader(stream))
    depth = np.loadtxt(directory / "depth.csv", delimiter=",")
    with open(directory / "stats.csv", newline="", encoding="utf-8") as stream:
        stats = next(csv.DictReader(stream))
    hu = np.atleast_2d(np.loadtxt(directory / "hu.csv", delimiter=","))
    hv = np.atleast_2d(np.loadtxt(directory / "hv.csv", delimiter=","))
    stats["_peak_memory_mb"] = memory_mb
    return float(timing["solver_seconds"]), wall_seconds, np.atleast_2d(depth), hu, hv, stats


def compiler_metadata(binary):
    cache = binary.parent / "CMakeCache.txt"
    compiler = "unknown"
    if cache.is_file():
        for line in cache.read_text(errors="replace").splitlines():
            if line.startswith("CMAKE_CXX_COMPILER:FILEPATH="):
                compiler = line.split("=", 1)[1]
                break
    try:
        version = subprocess.run([compiler, "--version"], check=True,
            capture_output=True, text=True, timeout=5).stdout.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        version = "unknown"
    return compiler, version


def cmake_cache_value(binary, key):
    cache = binary.parent / "CMakeCache.txt"
    if cache.is_file():
        prefix = key + ":"
        for line in cache.read_text(errors="replace").splitlines():
            if line.startswith(prefix) and "=" in line:
                return line.split("=", 1)[1]
    return ""


def host_metadata():
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                check=True, capture_output=True, text=True, timeout=5).stdout.strip() or cpu
        except (OSError, subprocess.SubprocessError):
            pass
    return {"operating_system": platform.platform(), "cpu": cpu,
            "logical_cpu_count": os.cpu_count()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build-openmp/flood_sim"))
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--sizes", default="512x512,1024x1024")
    parser.add_argument("--threads", default="1,2,4")
    parser.add_argument("--processes", default="1,2,4")
    parser.add_argument("--mpi-exec", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cell-size", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=None,
                        help="override scenario duration in seconds")
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--momentum-tolerance", type=float, default=1e-8)
    parser.add_argument("--conservation-tolerance", type=float, default=1e-8)
    parser.add_argument("--dry-depth", type=float, default=1e-6)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/openmp/results.csv"))
    parser.add_argument("--backends", default="serial,openmp,mpi")
    parser.add_argument("--availability-output", type=Path)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--overwrite", action="store_true",
                        help="allow replacing an existing result CSV")
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"solver executable does not exist: {binary}")
    if args.output.exists() and not args.overwrite:
        parser.error(f"refusing to overwrite existing results: {args.output}; choose a new --output or pass --overwrite")
    if (args.repeats < 1 or args.tolerance < 0 or args.momentum_tolerance < 0 or
        args.conservation_tolerance < 0 or args.cell_size <= 0 or
        (args.duration is not None and args.duration <= 0)):
        parser.error("repeats/cell size must be positive and tolerances nonnegative")
    scenarios = args.scenarios.split(",")
    if not scenarios or any(name not in SCENARIOS for name in scenarios):
        parser.error(f"scenarios must be selected from {', '.join(SCENARIOS)}")
    try:
        sizes = [tuple(int(part) for part in value.lower().split("x", 1))
                 for value in args.sizes.split(",")]
        threads = [int(value) for value in args.threads.split(",")]
        processes = [int(value) for value in args.processes.split(",")]
    except ValueError:
        parser.error("sizes use ROWSxCOLS and threads are comma-separated integers")
    if any(len(size) != 2 or min(size) < 2 for size in sizes) or any(value < 1 for value in threads):
        parser.error("grid dimensions and thread counts must be positive (grid dimensions >= 2)")
    if any(value < 1 for value in processes):
        parser.error("MPI process counts must be positive")

    requested_backends = [name.strip() for name in args.backends.split(",") if name.strip()]
    known_backends = {"serial", "openmp", "mpi"}
    if not requested_backends or any(name not in known_backends for name in requested_backends):
        parser.error(f"backends must be selected from {', '.join(sorted(known_backends))}")
    try:
        capabilities = json.loads(subprocess.run([str(binary), "--capabilities"], check=True,
            capture_output=True, text=True, timeout=5).stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        parser.error(f"cannot query backend capabilities: {error}")
    status_rows = []
    for name in requested_backends:
        capability = capabilities.get(name, {})
        status = capability.get("status", "UNAVAILABLE")
        detail = capability.get("detail", "capability was not reported by the executable")
        status_rows.append({"backend": name, "status": status,
                            "implemented": capability.get("implemented", False), "detail": detail})
    availability_path = args.availability_output or args.output.with_name("availability.csv")
    availability_path.parent.mkdir(parents=True, exist_ok=True)
    with open(availability_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("backend", "status", "implemented", "detail"))
        writer.writeheader()
        writer.writerows(status_rows)

    compiler, compiler_version = compiler_metadata(binary)
    host = host_metadata()
    mpi_launcher = str(args.mpi_exec.resolve()) if args.mpi_exec else (
        cmake_cache_value(binary, "MPIEXEC_EXECUTABLE") or
        shutil.which("mpiexec") or shutil.which("mpirun") or "")
    mpi_numproc_flag = cmake_cache_value(binary, "MPIEXEC_NUMPROC_FLAG") or "-n"
    mpi_preflags = shlex.split(cmake_cache_value(binary, "MPIEXEC_PREFLAGS"))
    mpi_postflags = shlex.split(cmake_cache_value(binary, "MPIEXEC_POSTFLAGS"))
    started_at = datetime.now(timezone.utc)
    run_metadata = {
        "started_at_utc": started_at.isoformat(), "finished_at_utc": None,
        "binary": str(binary), "compiler": compiler, "compiler_version": compiler_version,
        "operating_system": host["operating_system"], "cpu": host["cpu"],
        "logical_cpu_count": host["logical_cpu_count"], "scenarios": scenarios,
        "grid_sizes": [f"{rows}x{cols}" for rows, cols in sizes],
        "thread_counts": threads, "repetitions": args.repeats,
        "process_counts": processes, "mpi_launcher": mpi_launcher or None,
        "requested_backends": requested_backends, "availability": status_rows,
        "depth_tolerance_m": args.tolerance,
        "momentum_tolerance_m2_s": args.momentum_tolerance,
        "conservation_tolerance_m3": args.conservation_tolerance,
        "duration_seconds": args.duration,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=FIELDS)
        writer.writeheader()
        for scenario in scenarios:
            for rows, cols in sizes:
                backend_runs = []
                if ("openmp" in requested_backends and
                    capabilities.get("openmp", {}).get("status") == "AVAILABLE"):
                    backend_runs.append(("openmp", threads))
                if ("mpi" in requested_backends and
                    capabilities.get("mpi", {}).get("status") == "AVAILABLE" and mpi_launcher):
                    backend_runs.append(("mpi", processes))
                with tempfile.TemporaryDirectory(prefix="flood-benchmark-") as temp:
                    root = Path(temp)
                    serial_runs = [run_once(binary, "serial", scenario, rows, cols, 1,
                        1, args.cell_size, root / f"serial-{repeat}", args.timeout,
                        args.duration)
                        for repeat in range(args.repeats)]
                    serial_runtime = statistics.median(run[0] for run in serial_runs)
                    reference_depth = serial_runs[-1][2]
                    reference_hu, reference_hv = serial_runs[-1][3:5]
                    reference_stats = serial_runs[-1][5]
                    serial_timing = float(serial_runs[-1][5].get("computation_seconds", 0.0) or 0.0)
                    serial_runtime = statistics.median(run[0] for run in serial_runs)
                    serial_row = {
                        "scenario": scenario, "grid_rows": rows, "grid_cols": cols,
                        "rows": rows, "cols": cols,
                        "steps": reference_stats.get("steps", ""),
                        "backend": "serial",
                        "threads": 1, "processes": 1,
                        "serial_runtime": serial_runtime,
                        "backend_runtime": serial_runtime,
                        "serial_wall_runtime": statistics.median(run[1] for run in serial_runs),
                        "backend_wall_runtime": statistics.median(run[1] for run in serial_runs),
                        "speedup": 1.0, "efficiency": 1.0,
                        "max_depth_m": reference_stats["max_depth_m"],
                        "max_depth_difference_m": 0.0, "max_error": 0.0,
                        "mean_absolute_error_m": 0.0, "rmse": 0.0,
                        "relative_error": 0.0, "max_hu_error_m2_s": 0.0,
                        "max_hv_error_m2_s": 0.0, "rmse_hu_m2_s": 0.0,
                        "rmse_hv_m2_s": 0.0, "max_velocity_error_m_s": 0.0,
                        "flooded_area_m2": reference_stats["flooded_area_m2"],
                        "flooded_area_difference_m2": 0.0,
                        "water_volume_m3": reference_stats["stored_m3"],
                        "water_volume_difference_m3": 0.0,
                        "conservation_residual_m3": abs(float(reference_stats["mass_residual_m3"])),
                        "serial_conservation_residual_m3": abs(float(reference_stats["mass_residual_m3"])),
                        "wet_cells_match": True,
                        "setup_seconds": reference_stats.get("setup_seconds", ""),
                        "computation_seconds": reference_stats.get("computation_seconds", serial_timing),
                        "communication_seconds": 0.0, "synchronization_seconds": 0.0,
                        "other_seconds": reference_stats.get("other_seconds", ""),
                        "timing_sum_residual_seconds": "",
                        "memory_mb": reference_stats["_peak_memory_mb"],
                        "serial_memory_mb": reference_stats["_peak_memory_mb"],
                        "tolerance_m": args.tolerance,
                        "momentum_tolerance_m2_s": args.momentum_tolerance,
                        "conservation_tolerance_m3": args.conservation_tolerance,
                        "conservation_pass": abs(float(reference_stats["mass_residual_m3"])) <= args.conservation_tolerance,
                        "within_tolerance": True, "status": "PASS",
                        "repetitions": args.repeats, "timestamp_utc": started_at.isoformat(),
                        "compiler": compiler, "compiler_version": compiler_version,
                        "operating_system": host["operating_system"], "cpu": host["cpu"],
                    }
                    writer.writerow(serial_row)
                    result_file.flush()
                    for backend, counts_to_run in backend_runs:
                      for parallel_count in counts_to_run:
                        thread_count = parallel_count if backend == "openmp" else 1
                        process_count = parallel_count if backend == "mpi" else 1
                        candidate_runs = [run_once(binary, backend, scenario, rows, cols,
                            thread_count, process_count, args.cell_size,
                            root / f"{backend}-{parallel_count}-{repeat}", args.timeout,
                            args.duration,
                            mpi_launcher, mpi_numproc_flag, mpi_preflags, mpi_postflags)
                            for repeat in range(args.repeats)]
                        runtime = statistics.median(run[0] for run in candidate_runs)
                        candidate_depth = candidate_runs[-1][2]
                        candidate_hu, candidate_hv = candidate_runs[-1][3:5]
                        candidate_stats = candidate_runs[-1][5]
                        difference = candidate_depth - reference_depth
                        absolute = np.abs(difference)
                        max_error = float(absolute.max())
                        mean_absolute_error = float(absolute.mean())
                        rmse = float(np.sqrt(np.mean(difference * difference)))
                        mean_reference = float(np.mean(np.abs(reference_depth)))
                        relative_error = mean_absolute_error / mean_reference if mean_reference > 0 else float("nan")
                        hu_difference = candidate_hu - reference_hu
                        hv_difference = candidate_hv - reference_hv
                        max_hu_error = float(np.abs(hu_difference).max())
                        max_hv_error = float(np.abs(hv_difference).max())
                        rmse_hu = float(np.sqrt(np.mean(hu_difference * hu_difference)))
                        rmse_hv = float(np.sqrt(np.mean(hv_difference * hv_difference)))
                        max_velocity_error = float(abs(
                            float(candidate_stats["max_velocity_m_s"]) -
                            float(reference_stats["max_velocity_m_s"])))
                        volume_difference = float(candidate_stats["stored_m3"]) - float(reference_stats["stored_m3"])
                        area_difference = float((
                            np.count_nonzero(candidate_depth > args.dry_depth) -
                            np.count_nonzero(reference_depth > args.dry_depth)) * args.cell_size**2)
                        conservation_residual = abs(float(candidate_stats["mass_residual_m3"]))
                        serial_conservation_residual = abs(float(reference_stats["mass_residual_m3"]))
                        conservation_pass = (conservation_residual <= args.conservation_tolerance and
                            serial_conservation_residual <= args.conservation_tolerance)
                        within_tolerance = (max_error <= args.tolerance and
                            max_hu_error <= args.momentum_tolerance and
                            max_hv_error <= args.momentum_tolerance and conservation_pass)
                        speedup = serial_runtime / runtime if runtime > 0 else float("nan")
                        efficiency = speedup / parallel_count
                        row = {
                            "scenario": scenario, "grid_rows": rows, "grid_cols": cols,
                            "rows": rows, "cols": cols,
                            "steps": candidate_stats.get("steps", ""),
                            "backend": backend,
                            "threads": thread_count if backend == "openmp" else "",
                            "processes": process_count if backend == "mpi" else "",
                            "serial_runtime": serial_runtime, "backend_runtime": runtime,
                            "serial_wall_runtime": statistics.median(run[1] for run in serial_runs),
                            "backend_wall_runtime": statistics.median(run[1] for run in candidate_runs),
                            "speedup": speedup, "efficiency": efficiency,
                            "max_depth_m": candidate_stats["max_depth_m"],
                            "max_depth_difference_m": abs(float(candidate_stats["max_depth_m"]) -
                                float(reference_stats["max_depth_m"])),
                            "max_error": max_error, "mean_absolute_error_m": mean_absolute_error,
                            "rmse": rmse, "relative_error": relative_error,
                            "max_hu_error_m2_s": max_hu_error, "max_hv_error_m2_s": max_hv_error,
                            "rmse_hu_m2_s": rmse_hu, "rmse_hv_m2_s": rmse_hv,
                            "max_velocity_error_m_s": max_velocity_error,
                            "flooded_area_m2": candidate_stats["flooded_area_m2"],
                            "water_volume_difference_m3": volume_difference,
                            "flooded_area_difference_m2": area_difference,
                            "water_volume_m3": candidate_stats["stored_m3"],
                            "conservation_residual_m3": conservation_residual,
                            "serial_conservation_residual_m3": serial_conservation_residual,
                            "wet_cells_match": area_difference == 0.0,
                            "setup_seconds": candidate_stats.get("setup_seconds", ""),
                            "computation_seconds": candidate_stats.get("computation_seconds", ""),
                            "communication_seconds": candidate_stats.get("communication_seconds", ""),
                            "synchronization_seconds": candidate_stats.get("synchronization_seconds", ""),
                            "other_seconds": candidate_stats.get("other_seconds", ""),
                            "timing_sum_residual_seconds": (
                                runtime - sum(float(candidate_stats.get(field, 0.0) or 0.0)
                                    for field in ("setup_seconds", "computation_seconds",
                                                  "communication_seconds", "synchronization_seconds",
                                                  "other_seconds")) if backend == "mpi" else ""),
                            "memory_mb": candidate_stats["_peak_memory_mb"],
                            "serial_memory_mb": reference_stats["_peak_memory_mb"],
                            "tolerance_m": args.tolerance,
                            "momentum_tolerance_m2_s": args.momentum_tolerance,
                            "conservation_tolerance_m3": args.conservation_tolerance,
                            "conservation_pass": conservation_pass,
                            "within_tolerance": within_tolerance,
                            "status": "PASS" if within_tolerance else "FAIL",
                            "repetitions": args.repeats,
                            "timestamp_utc": started_at.isoformat(),
                            "compiler": compiler, "compiler_version": compiler_version,
                            "operating_system": host["operating_system"], "cpu": host["cpu"],
                        }
                        writer.writerow(row)
                        result_file.flush()
                        parameter_name = "threads" if backend == "openmp" else "processes"
                        print(f"{scenario} {rows}x{cols} {parameter_name}={parallel_count} "
                            f"serial={serial_runtime:.6g}s {backend}={runtime:.6g}s "
                              f"speedup={speedup:.4g} max_error={max_error:.3g} "
                              f"pass={within_tolerance}")
                        if not within_tolerance:
                            raise SystemExit("Numerical tolerance exceeded; benchmark halted")
    print(f"Recorded measured results in {args.output}")
    run_metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata_path = args.metadata_output or args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded run metadata in {metadata_path}")
    print(f"Recorded backend availability in {availability_path}")


if __name__ == "__main__":
    main()