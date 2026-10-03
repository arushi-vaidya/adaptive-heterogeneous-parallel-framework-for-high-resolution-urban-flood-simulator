#!/usr/bin/env python3
"""Generate performance plots only for backend measurements present in CSV results."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_results(paths):
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def numeric(row, field):
    try:
        return float(row[field])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def save_bar(rows, field, title, ylabel, path):
    groups = {}
    for row in rows:
        value = numeric(row, field)
        if np.isfinite(value):
            groups.setdefault(row["backend"], []).append(value)
    if not groups:
        return False
    labels = sorted(groups)
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    axis.bar(labels, [np.median(groups[label]) for label in labels], color="#176b83")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def save_runtime_bar(rows, path):
    groups = {}
    for row in rows:
        serial = numeric(row, "serial_runtime")
        backend = numeric(row, "backend_runtime")
        if np.isfinite(serial):
            groups.setdefault("serial", []).append(serial)
        if np.isfinite(backend):
            groups.setdefault(row["backend"], []).append(backend)
    if not groups:
        return False
    labels = sorted(groups)
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    axis.bar(labels, [np.median(groups[label]) for label in labels], color="#176b83")
    axis.set_title("Runtime by backend")
    axis.set_ylabel("Solver runtime (s)")
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def save_conservation_plot(rows, path):
    groups = {"serial": [], "openmp": []}
    for row in rows:
        serial = numeric(row, "serial_conservation_residual_m3")
        candidate = numeric(row, "conservation_residual_m3")
        if np.isfinite(serial):
            groups["serial"].append(serial)
        if np.isfinite(candidate):
            groups.setdefault(row["backend"], []).append(candidate)
    groups = {name: values for name, values in groups.items() if values}
    if not groups:
        return False
    names = sorted(groups)
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    axis.bar(names, [np.median(groups[name]) for name in names], color="#176b83")
    axis.set(title="Conservation residual by backend", ylabel="Absolute mass residual (m^3)")
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", type=Path, default=[Path("benchmarks/**/results*.csv")])
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/plots"))
    args = parser.parse_args()
    paths = []
    for pattern in args.input:
        paths.extend(Path().glob(str(pattern)) if "*" in str(pattern) else [pattern])
    paths = sorted({path for path in paths if path.is_file()})
    if not paths:
        parser.error("no benchmark result CSVs found")
    rows = read_results(paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    produced = []
    if save_runtime_bar(rows, args.output_dir / "runtime_by_backend.png"):
        produced.append("runtime_by_backend.png")
    bar_specs = [
        ("speedup", "Speedup by backend", "Speedup", "speedup_by_backend.png"),
        ("max_error", "Maximum depth error by backend", "Maximum absolute error (m)",
         "numerical_error_by_backend.png"),
    ]
    for field, title, ylabel, filename in bar_specs:
        if save_bar(rows, field, title, ylabel, args.output_dir / filename):
            produced.append(filename)
    if save_conservation_plot(rows, args.output_dir / "conservation_residual.png"):
        produced.append("conservation_residual.png")

    openmp = [row for row in rows if row.get("backend") == "openmp" and numeric(row, "threads") > 0]
    if openmp:
        figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        for size in sorted({(row["rows"], row["cols"]) for row in openmp}):
            subset = [row for row in openmp if (row["rows"], row["cols"]) == size]
            grouped = {}
            for row in subset:
                grouped.setdefault(numeric(row, "threads"), []).append(numeric(row, "backend_runtime"))
            x = sorted(grouped)
            axis.plot(x, [np.median(grouped[value]) for value in x], marker="o",
                      label=f"{size[0]}x{size[1]}")
        axis.set(title="OpenMP scaling", xlabel="Threads", ylabel="Solver runtime (s)")
        axis.grid(alpha=0.2); axis.legend()
        figure.savefig(args.output_dir / "openmp_scaling.png", dpi=180); plt.close(figure)
        produced.append("openmp_scaling.png")
        for field, filename, title, ylabel in (
            ("speedup", "openmp_speedup.png", "OpenMP speedup vs threads", "Speedup"),
            ("efficiency", "openmp_efficiency.png", "OpenMP efficiency vs threads", "Parallel efficiency"),
        ):
            figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
            for size in sorted({(row["rows"], row["cols"]) for row in openmp}):
                subset = [row for row in openmp if (row["rows"], row["cols"]) == size]
                grouped = {}
                for row in subset:
                    grouped.setdefault(numeric(row, "threads"), []).append(numeric(row, field))
                x = sorted(grouped)
                axis.plot(x, [np.median(grouped[value]) for value in x], marker="o",
                          label=f"{size[0]}x{size[1]}")
            axis.set(title=title, xlabel="Threads", ylabel=ylabel)
            axis.grid(alpha=0.2); axis.legend()
            figure.savefig(args.output_dir / filename, dpi=180); plt.close(figure)
            produced.append(filename)

    mpi_rows = [row for row in rows if row.get("backend") == "mpi" and
                numeric(row, "processes") > 0]
    if mpi_rows:
        processes = sorted({numeric(row, "processes") for row in mpi_rows})
        for field, filename, title, ylabel in (
            ("backend_runtime", "mpi_runtime.png", "MPI runtime vs process count", "Runtime (s)"),
            ("speedup", "mpi_speedup.png", "MPI speedup vs process count", "Speedup"),
            ("efficiency", "mpi_efficiency.png", "MPI efficiency vs process count", "Parallel efficiency"),
            ("communication_seconds", "mpi_communication.png", "MPI communication vs process count", "Communication time (s)"),
            ("max_error", "mpi_numerical_error.png", "MPI numerical error", "Maximum depth error (m)"),
        ):
            values_by_process = {count: [numeric(row, field) for row in mpi_rows
                if numeric(row, "processes") == count and np.isfinite(numeric(row, field))]
                for count in processes}
            available = [count for count in processes if values_by_process[count]]
            if not available:
                continue
            figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
            axis.plot(available, [np.median(values_by_process[count]) for count in available], marker="o")
            axis.set(title=title, xlabel="MPI processes", ylabel=ylabel)
            axis.grid(alpha=0.2)
            filename_path = args.output_dir / filename
            figure.savefig(filename_path, dpi=180)
            plt.close(figure)
            produced.append(filename)

        figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        has_series = False
        for field, label in (("computation_seconds", "Computation"),
                             ("communication_seconds", "Communication"),
                             ("synchronization_seconds", "Synchronization")):
            values_by_process = {count: [numeric(row, field) for row in mpi_rows
                if numeric(row, "processes") == count and np.isfinite(numeric(row, field))]
                for count in processes}
            available = [count for count in processes if values_by_process[count]]
            if available:
                axis.plot(available, [np.median(values_by_process[count]) for count in available],
                          marker="o", label=label)
                has_series = True
        if has_series:
            axis.set(title="MPI computation, communication, synchronization",
                     xlabel="Processes", ylabel="Time (s)")
            axis.grid(alpha=0.2); axis.legend()
            figure.savefig(args.output_dir / "mpi_compute_communication.png", dpi=180)
            produced.append("mpi_compute_communication.png")
        else:
            plt.close(figure)


    grid_groups = {}
    for row in rows:
        cells = numeric(row, "rows") * numeric(row, "cols")
        serial_runtime = numeric(row, "serial_runtime")
        backend_runtime = numeric(row, "backend_runtime")
        if np.isfinite(cells) and np.isfinite(serial_runtime):
            grid_groups.setdefault("serial", {}).setdefault(cells, []).append(serial_runtime)
        if np.isfinite(cells) and np.isfinite(backend_runtime):
            grid_groups.setdefault(row["backend"], {}).setdefault(cells, []).append(backend_runtime)
    if grid_groups:
        figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        for backend, values in sorted(grid_groups.items()):
            sizes = sorted(values)
            axis.plot(sizes, [np.median(values[size]) for size in sizes], marker="o", label=backend)
        axis.set(title="Runtime by grid size", xlabel="Number of cells", ylabel="Solver runtime (s)")
        axis.grid(alpha=0.2); axis.legend()
        figure.savefig(args.output_dir / "runtime_by_grid_size.png", dpi=180); plt.close(figure)
        produced.append("runtime_by_grid_size.png")
    print("Generated: " + (", ".join(produced) if produced else "no plots"))
    if any(row.get("backend") == "mpi" for row in rows):
        print("MPI plots use only recorded MPI measurements.")
    else:
        print("Skipped MPI plots: no MPI result rows were provided.")


if __name__ == "__main__":
    main()