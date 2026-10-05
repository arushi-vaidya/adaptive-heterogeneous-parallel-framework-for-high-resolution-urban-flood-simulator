# Adaptive Heterogeneous Parallel Framework for High-Resolution Urban Flood Simulation

This repository contains a research implementation of a two-dimensional urban
flood simulator. It includes a uniform-grid shallow-water solver, adaptive mesh
refinement (AMR), and Serial, OpenMP, and optional MPI backends. Python utilities
support benchmark runs, comparisons, plotting, example-data generation, and a
local web UI. The Python tools orchestrate the C++ solver; simulation physics
remain in C++.

The implementation and validation are complete for the features described
here. This is a research framework, not a calibrated or production flood
forecasting system. Benchmark measurements are host-specific, and passing
regression tests does not establish accuracy against field observations.

## Features

- First-order finite-volume shallow-water solver with hydrostatic bed
  reconstruction, Rusanov face fluxes, rainfall, infiltration, Manning
  friction, wet/dry handling, and closed or outflow boundaries.
- Serial reference backend and an OpenMP backend. MPI is compiled when a C++
  MPI toolchain and launcher are available; unavailable backends fail
  explicitly rather than silently falling back.
- Adaptive grid with regridding and coarse/fine flux handling. MPI adaptive
  runs can optionally use dynamic load balancing.
- Built-in scenarios: `flat-basin`, `slope`, `dam-break`, `rain-drain`, and
  `wet-dry`. The adaptive benchmark also provides `localized-refinement`.
- CSV outputs and diagnostics for water depth, momentum, terrain, mass
  balance, solver timing, and benchmark comparisons.

The conserved state is $U=(h, hu, hv)^T$, with depth $h$ in metres and
depth-averaged velocities $u,v$ in metres per second. Rainfall CSV time is in
seconds and intensity in mm/hour; infiltration is in m/s. Terrain CSV files
contain elevation only, with no header and one row per grid row. See
[config/example.ini](config/example.ini) for the supported configuration-file
format.

## Requirements

- CMake 3.16 or newer and a C++17 compiler.
- OpenMP and MPI are optional. CMake detects available toolchains when their
  options are enabled (both are enabled by default).
- Python 3.8 or newer for the Python tools. The core executable does not
  require Python. NumPy is needed for benchmark/comparison tools; Matplotlib
  is needed to render plots and flood maps. The local UI uses Python's standard
  library.

## Build and test

From the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
./build/flood_sim --capabilities
```

CMake enables OpenMP and probes for MPI by default. If either backend is not
available in the environment, the build still supports the backends that were
found. To request a serial-only build:

```sh
cmake -S . -B build-serial \
  -DCMAKE_BUILD_TYPE=Release \
  -DFLOOD_ENABLE_OPENMP=OFF -DFLOOD_ENABLE_MPI=OFF
cmake --build build-serial -j2
```

On macOS, OpenMP may require Homebrew `libomp`; the CMake configuration checks
the standard Homebrew location when `FindOpenMP` cannot locate it. For MPI,
install an MPI C++ toolchain and launcher, then configure with
`-DFLOOD_ENABLE_MPI=ON`. Check the detected capabilities with
`./build/flood_sim --capabilities` or `./build/flood_sim --list-backends`.

## Run without Python scripts

Run the uniform-grid solver directly:

```sh
./build/flood_sim \
  --scenario dam-break --rows 128 --cols 128 \
  --duration 60 --output output/dam-break
```

The default backend is Serial. Choose another compiled backend explicitly:

```sh
./build/flood_sim --backend openmp --threads 4 \
  --scenario rain-drain --rows 256 --cols 256 \
  --duration 60 --output output/rain-drain-openmp
```

For MPI, launch the executable with the MPI launcher available on your system:

```sh
mpiexec -n 4 ./build/flood_sim --backend mpi \
  --scenario dam-break --rows 256 --cols 256 \
  --duration 60 --output output/dam-break-mpi
```

Configure a uniform-grid run with a key/value file, or override configuration
values on the command line (command-line values take precedence):

```sh
./build/flood_sim --config config/example.ini
./build/flood_sim --config config/example.ini \
  --backend openmp --threads 4 --output output/example-openmp
```

`flood_sim --help` lists its command-line options. These include
`--terrain` and `--rainfall` for input CSVs, `--cell-size`, and physical
parameters such as `--gravity`, `--manning`, `--infiltration`, `--cfl`,
`--max-dt`, `--dry-depth`, and `--boundary closed|outflow`. The executable
writes `terrain.csv`, `depth.csv`, `hu.csv`, `hv.csv`, `stats.csv`, and
`benchmark.csv` in the selected output directory.

Run the adaptive solver directly with the adaptive benchmark executable:

```sh
./build/flood_adaptive_benchmark \
  --backend serial --scenario localized-refinement \
  --rows 64 --cols 64 --patch-extent 8 --max-level 2 \
  --end-time 0.2 --output-dir output/adaptive
```

It prints a CSV summary to standard output and writes adaptive maps to the
requested output directory. Use `--backend openmp --threads N` for OpenMP. For
MPI, launch it with `mpiexec -n N` and `--backend mpi`. Optional MPI dynamic
load balancing is controlled by `--dynamic-load-balancing true`, with
`--imbalance-threshold` and `--rebalance-cooldown` tuning options.

## Run with Python scripts

The scripts below are optional convenience tools; the C++ executable can be
built and run without them. Run commands from the repository root.

Generate example rainfall and terrain inputs:

```sh
python3 scripts/generate_rainfall.py --output output/rainfall.csv
python3 scripts/generate_test_terrain.py \
  --kind bowl --rows 64 --cols 64 --output output/terrain.csv
```

Use those files with the uniform-grid solver:

```sh
./build/flood_sim --terrain output/terrain.csv \
  --rainfall output/rainfall.csv --duration 60 \
  --output output/custom-run
```

Run a small uniform-grid benchmark matrix (requires NumPy). The runner compares
the selected backend runs against Serial, validates output and conservation,
and writes measured CSV results and a metadata sidecar:

```sh
python3 scripts/run_benchmarks.py --binary build/flood_sim \
  --backends serial,openmp --scenarios dam-break \
  --sizes 64x64 --threads 1,2 --repeats 1 \
  --output benchmarks/openmp/readme-example.csv
```

For AMR benchmarks, use `scripts/run_adaptive_benchmarks.py`. This example
runs only Serial, so it does not require an MPI launcher:

```sh
python3 scripts/run_adaptive_benchmarks.py \
  --binary build/flood_adaptive_benchmark \
  --backends serial --scenarios localized-refinement \
  --rows 64 --cols 64 --repeats 1 \
  --output benchmarks/adaptive/readme-example.csv
```

To visualize a completed uniform-grid run (requires Matplotlib):

```sh
python3 scripts/visualize_results.py \
  --terrain output/custom-run/terrain.csv \
  --depth output/custom-run/depth.csv \
  --output output/custom-run/flood_map.png
```

Compare two depth grids with `scripts/compare_outputs.py` (requires NumPy):

```sh
python3 scripts/compare_outputs.py \
  reference/depth.csv candidate/depth.csv --cell-size 5
```

The local UI offers backend and scenario selection, runs comparisons, and
displays measured results. It uses the `build-mpi/` executable paths, so build
that directory first:

```sh
cmake -S . -B build-mpi -DCMAKE_BUILD_TYPE=Release \
  -DFLOOD_ENABLE_MPI=ON
cmake --build build-mpi -j2
python3 scripts/ui.py
```

Then open <http://127.0.0.1:8000>. The UI binds to the local machine; available
backends depend on the detected toolchains and executables in `build-mpi/`.

## Validation and further information

CTest covers solver, backend, adaptive-grid, load-balancing, and configuration
regressions. MPI-specific tests are included when CMake detects MPI. See
[docs/phase2.md](docs/phase2.md) for validation details and the tested
environment; [benchmarks/README.md](benchmarks/README.md) and
[benchmarks/mpi/README.md](benchmarks/mpi/README.md) document recorded uniform
grid measurements. Architecture and backend details are in
[docs/phase2_architecture.md](docs/phase2_architecture.md),
[docs/openmp_design.md](docs/openmp_design.md), and
[docs/mpi_design.md](docs/mpi_design.md).

The mass-balance residual is a diagnostic and the test tolerance is a
regression threshold, not an accuracy specification. The solver uses
first-order reconstruction on a uniform grid for the non-adaptive path; it
does not provide a calibrated inundation forecast. Validate and calibrate
against suitable reference data before using results for operational decisions.
