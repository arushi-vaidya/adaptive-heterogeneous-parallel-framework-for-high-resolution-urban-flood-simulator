#!/usr/bin/env python3
"""Measure solver runtime against domain size for every runnable backend.

Feeds the UI's "Performance Comparison" chart. Each point is one real run of
the C++ solver; nothing is extrapolated. Results are rewritten after every run
so the UI can draw the curves while the sweep is still in progress.
"""
import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
import time


def solver_seconds(output):
    with open(output / "benchmark.csv", newline="", encoding="utf-8") as stream:
        return float(next(csv.DictReader(stream))["solver_seconds"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--scenario", default="urban-district")
    parser.add_argument("--sizes", default="64,128,192,256,384")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--cell-size", type=float, default=5.0)
    parser.add_argument("--backends", default="serial,openmp,mpi")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--mpi-exec", default="")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflags", default="")
    parser.add_argument("--mpi-postflags", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",") if value]
    backends = [value for value in args.backends.split(",") if value]
    work = args.output / "sweep-runs"
    work.mkdir(parents=True, exist_ok=True)
    result = {
        "scenario": args.scenario, "duration_s": args.duration,
        "cell_size_m": args.cell_size, "threads": args.threads,
        "processes": args.processes, "started": time.time(),
        "complete": False, "points": [], "errors": [],
        "planned": len(sizes) * len(backends),
    }
    target = args.output / "sweep.json"

    def save():
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(target)

    save()
    for size in sizes:
        for backend in backends:
            output = work / f"{backend}-{size}"
            command = [str(args.binary), "--scenario", args.scenario,
                       "--rows", str(size), "--cols", str(size),
                       "--cell-size", str(args.cell_size),
                       "--duration", str(args.duration),
                       "--backend", backend, "--output", str(output)]
            if backend == "openmp":
                command.extend(("--threads", str(args.threads)))
            if backend == "mpi":
                if not args.mpi_exec:
                    result["errors"].append({"backend": backend, "size": size,
                                             "error": "no MPI launcher detected"})
                    save()
                    continue
                command = [args.mpi_exec, args.mpi_numproc_flag, str(args.processes),
                           *shlex.split(args.mpi_preflags), *command,
                           *shlex.split(args.mpi_postflags)]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                result["errors"].append({"backend": backend, "size": size,
                                         "error": completed.stderr.strip()[-300:]})
            else:
                cells = size * size
                result["points"].append({
                    "backend": backend, "rows": size, "cols": size, "cells": cells,
                    "domain_km2": cells * args.cell_size ** 2 / 1e6,
                    "solver_seconds": solver_seconds(output),
                })
            save()
    result["complete"] = True
    result["finished"] = time.time()
    save()


if __name__ == "__main__":
    main()
