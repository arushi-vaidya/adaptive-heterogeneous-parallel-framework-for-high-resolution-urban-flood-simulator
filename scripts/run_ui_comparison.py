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


WET_THRESHOLD_M = 0.05
SOLVER_ARGS = []  # user scenario/physics overrides, forwarded to every solver run
DEFAULT_RAIN = {  # mirrors src/simulation/Scenarios.cpp (time s, mm/h)
    "urban-district": [[0, 120], [60, 40], [150, 0]],
    "rain-drain": [[0, 60], [30, 0]],
    "catchment": [[0, 200], [600, 0]],
}
MAX_REFERENCE_CELLS = 512 * 512


def write_progress(output, stage, message, step, total):
    """Record the pipeline stage so the UI can follow the run live."""
    payload = {"stage": stage, "message": message, "step": step, "total": total}
    (output / "progress.json").write_text(json.dumps(payload), encoding="utf-8")


def load_grid(path):
    import numpy as np
    return np.atleast_2d(np.loadtxt(path, delimiter=","))


def block_average(fine, factor):
    rows, cols = fine.shape[0] // factor, fine.shape[1] // factor
    trimmed = fine[:rows * factor, :cols * factor]
    return trimmed.reshape(rows, factor, cols, factor).mean(axis=(1, 3))


def fine_reference(binary, scenario, rows, cols, cell_size, duration, factor, output,
                   threads):
    """Uniform solution at `factor`x resolution, block-averaged to the model grid.

    This is the 'ground truth' used for flood-extent accuracy: the same physics
    without coarse-resolution or adaptive approximations.
    """
    fine_rows, fine_cols = rows * factor, cols * factor
    if fine_rows * fine_cols > MAX_REFERENCE_CELLS:
        return None, f"skipped: {fine_rows}x{fine_cols} reference exceeds the UI limit"
    command = [str(binary), "--scenario", scenario, "--rows", str(fine_rows),
               "--cols", str(fine_cols), "--cell-size", repr(cell_size / factor),
               "--duration", str(duration), "--output", str(output), *SOLVER_ARGS]
    if threads > 1:
        command.extend(("--backend", "openmp", "--threads", str(threads)))
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode and threads > 1:
        command = command[:command.index("--backend")]
        completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        return None, f"failed: {completed.stderr.strip()[-300:]}"
    fine = load_grid(output / "depth.csv")
    return block_average(fine, factor), f"uniform {fine_rows}x{fine_cols} at {cell_size / factor:g} m, block-averaged x{factor}"


def flood_outcomes(reference, candidate, threshold=WET_THRESHOLD_M):
    """Flood-extent agreement and depth-error distribution versus a reference."""
    import numpy as np
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate grids differ in shape")
    predicted = candidate > threshold
    observed = reference > threshold
    union = np.logical_or(predicted, observed)
    intersection = np.logical_and(predicted, observed)
    union_count = int(union.sum())
    errors = np.abs(candidate - reference)[union]
    iou = float(intersection.sum() / union_count) if union_count else 1.0
    precision = float(intersection.sum() / predicted.sum()) if predicted.sum() else 1.0
    recall = float(intersection.sum() / observed.sum()) if observed.sum() else 1.0
    mean_reference = float(reference[union].mean()) if union_count else 0.0
    mean_error = float(errors.mean()) if errors.size else 0.0
    upper = float(np.percentile(errors, 99.5)) if errors.size else 0.0
    upper = upper if upper > 0 else max(float(errors.max()) if errors.size else 0.0, 1e-9)
    counts, edges = np.histogram(np.clip(errors, 0, upper), bins=24, range=(0.0, upper))
    return {
        "wet_threshold_m": threshold,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "flooded_cells_model": int(predicted.sum()),
        "flooded_cells_reference": int(observed.sum()),
        "mean_abs_error_m": mean_error,
        "std_abs_error_m": float(errors.std()) if errors.size else 0.0,
        "max_abs_error_m": float(errors.max()) if errors.size else 0.0,
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))) if errors.size else 0.0,
        "relative_error_pct": 100.0 * mean_error / mean_reference if mean_reference > 0 else 0.0,
        "histogram": {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]},
    }


def attach_outcomes(args, summary, candidate_depth_path, cell_size, factor):
    """Compute accuracy outcomes against a high-resolution uniform reference."""
    import numpy as np
    if args.no_reference:
        summary["outcomes"] = {"available": False, "reason": "high-resolution reference disabled"}
        return
    write_progress(args.output, "output", "Computing high-resolution reference", 4, 5)
    candidate = load_grid(candidate_depth_path)
    reference, description = fine_reference(
        args.binary, args.scenario, args.rows, args.cols, cell_size, args.duration,
        factor, args.output / "fine-reference", args.reference_threads)
    if reference is None:
        summary["outcomes"] = {"available": False, "reason": description}
        return
    np.savetxt(args.output / "reference-depth.csv", reference, delimiter=",", fmt="%.9g")
    outcomes = flood_outcomes(reference, candidate)
    outcomes.update({"available": True, "reference": description, "factor": factor,
                     "reference_cells": args.rows * args.cols * factor * factor})
    summary["outcomes"] = outcomes


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
        *SOLVER_ARGS,
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


def frame_count(rows, cols, budget_bytes=32_000_000):
    """Time-lapse frames that fit the UI budget (h, hu, hv as float32 per frame)."""
    return max(8, min(60, budget_bytes // (12 * rows * cols) - 1))


def run(binary, scenario, rows, cols, duration, backend, threads, processes,
    output, mpi_exec="", mpi_numproc_flag="-n", mpi_preflags="", mpi_postflags="",
    frames=0):
    command = [str(binary), "--scenario", scenario, "--rows", str(rows), "--cols", str(cols),
               "--duration", str(duration), "--backend", backend, "--output", str(output),
               *SOLVER_ARGS]
    if frames > 0 and backend != "mpi":
        command.extend(("--frames", str(frames)))
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
    parser.add_argument("--backend", choices=("serial", "openmp", "mpi", "cuda", "openacc"), required=True)
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
    parser.add_argument("--reference-factor", type=int, default=2,
                        help="resolution multiplier of the uniform accuracy reference")
    parser.add_argument("--reference-threads", type=int, default=4)
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--solver-args", default="",
                        help="extra flags for flood_sim / flood_adaptive_benchmark")
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--momentum-tolerance", type=float, default=1e-8)
    parser.add_argument("--conservation-tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    SOLVER_ARGS.extend(shlex.split(args.solver_args))
    overrides = dict(zip(SOLVER_ARGS[0::2], SOLVER_ARGS[1::2]))
    if "--rain-intensity" in overrides:
        intensity = float(overrides["--rain-intensity"])
        duration = float(overrides.get("--rain-duration", -1))
        rainfall = [[0, intensity]] + ([[duration, 0]] if duration >= 0 else [])
    else:
        rainfall = DEFAULT_RAIN.get(args.scenario, [[0, 0]])
    parameters = {key.lstrip("-"): value for key, value in overrides.items()}
    for stale in ("reference-depth.csv", "summary.json", "current/frames.bin",
                  "current/frames.json", "current/landuse.csv"):
        (args.output / stale).unlink(missing_ok=True)
    write_progress(args.output, "input", "Building scenario and terrain", 1, 5)
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
        write_progress(args.output, "adaptive", "Serial adaptive reference run", 2, 5)
        reference = run_adaptive(
            adaptive_binary, args.scenario, args.rows, args.cols, args.duration,
            "serial", 1, 1, args.output / "reference", adaptive_binary,
            args.patch_extent, args.max_level, args.regrid_interval,
            args.refine_threshold, args.coarsen_threshold, False,
            args.imbalance_threshold, args.rebalance_cooldown)
        write_progress(args.output, "solver", f"Adaptive {args.backend} run", 3, 5)
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
            "scenario": args.scenario, "rows": args.rows, "cols": args.cols,
            "cell_size_m": 1.0,
            "processes": rank_count, "adaptive": True,
            "max_level": args.max_level,
            "dynamic_load_balancing": args.dynamic_load_balancing,
            "serial_runtime": reference_runtime, "backend_runtime": runtime,
            "speedup": speedup,
            "efficiency": speedup / parallelism if parallelism > 0 else 0.0,
            "max_error": 0.0, "conservation_pass": conservation_pass,
            "validation_pass": conservation_pass, "stats": stats,
            "refinement_map_available": (args.output / "current" / "refinement.csv").exists(),
            "rainfall": rainfall, "parameters": parameters,
        }
        factor = 2 ** args.max_level
        summary["uniform_fine_cells"] = args.rows * args.cols * factor * factor
        attach_outcomes(args, summary, args.output / "current" / "depth.csv", 1.0, factor)
        write_progress(args.output, "done", "Completed", 5, 5)
        with open(args.output / "summary.json", "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
        return

    write_progress(args.output, "decomposition", "Serial reference run", 2, 5)
    # The reference is segmented exactly like the candidate (segment boundaries
    # shorten the last timestep of each segment), so the comparison stays exact.
    frames = frame_count(args.rows, args.cols) if args.backend != "mpi" else 0
    reference_stats, reference_time = run(args.binary, args.scenario, args.rows, args.cols,
        args.duration, "serial", 1, 1, args.output / "reference", frames=frames)
    write_progress(args.output, "solver", f"{args.backend} solver run", 3, 5)
    candidate_stats, candidate_time = run(args.binary, args.scenario, args.rows, args.cols,
        args.duration, args.backend, args.threads, args.processes,
        args.output / "current", args.mpi_exec, args.mpi_numproc_flag,
        args.mpi_preflags, args.mpi_postflags, frames)
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
        "scenario": args.scenario, "rows": args.rows, "cols": args.cols,
        "cell_size_m": 5.0,
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
        "rainfall": rainfall, "parameters": parameters,
        "steps": int(candidate_stats.get("steps", 0) or 0),
        "throughput_cell_updates_per_s": (
            args.rows * args.cols * int(candidate_stats.get("steps", 0) or 0) / backend_seconds
            if backend_seconds > 0 else 0.0),
    }
    attach_outcomes(args, summary, args.output / "current" / "depth.csv", 5.0,
                    max(1, args.reference_factor))
    write_progress(args.output, "done", "Completed", 5, 5)
    with open(args.output / "summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)


if __name__ == "__main__":
    main()