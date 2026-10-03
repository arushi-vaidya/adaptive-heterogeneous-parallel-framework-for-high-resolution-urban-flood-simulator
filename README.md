# Adaptive Heterogeneous Parallel Framework for High-Resolution Urban Flood Simulation

## Phase 1: Serial Reference Baseline

This repository contains a deterministic, dependency-free C++17 serial shallow-water solver and small Python tools for test data, plotting, output comparison, and a local launch UI. It is a research baseline, not a calibrated urban inundation product. It contains no parallel backend, adaptive mesh refinement, machine learning, or learned prediction.

## Model and Assumptions

The conserved state is $U=(h, hu, hv)^T$, where $h$ is water depth in metres and $u,v$ are depth-averaged velocities in metres per second. The bed elevation $z$ is in metres. The equations are

$$
\partial_t U + \partial_x F(U) + \partial_y G(U) = S,
$$

with $F=(hu, hu^2+gh^2/2, huv)^T$, $G=(hv, huv, hv^2+gh^2/2)^T$. Rain adds $r$ to the depth equation. Infiltration removes water at a configured constant rate, capped by available depth. Manning friction is an implicit local momentum damping, with factor $1/(1+\Delta t g n^2 |V|/h^{4/3})$. Momentum is set to zero at or below the dry-depth threshold.

The solver is first-order finite volume on a uniform Cartesian grid. At faces it uses hydrostatic reconstruction over the higher adjacent bed followed by a Rusanov (local Lax-Friedrichs) flux and side-specific hydrostatic pressure corrections. This balances a lake at rest over a bed step to first order. There is no high-order reconstruction or slope limiting. The timestep obeys the two-dimensional gravity-wave CFL estimate and is clipped to the next rainfall breakpoint and the configured maximum. A cell-based outgoing-water limiter reduces face fluxes when a step could drain more water than the donor stores; it preserves mass but adds diffusion near wet/dry fronts. Floating-point calculations use IEEE double precision and fixed serial iteration order; cross-compiler bitwise identity is not promised.

Rainfall CSV time is seconds and intensity is mm/hour; conversion to m/s is `intensity / 1000 / 3600`. Values are piecewise constant from their timestamp up to the next timestamp; rain before the first timestamp uses the first value. Infiltration is m/s. Grid coordinates are cell-centred in a uniform grid; terrain CSV contains elevation only, without a header, one row per grid row. The outflow boundary uses zero-gradient extrapolation for outward flow and reflected normal momentum to suppress inflow. Closed boundaries have zero normal flux.

Mass accounting reports `initial + rainfall - outflow - infiltration - final` in cubic metres. This is a diagnostic, not a proof of solution accuracy. It does not account for external sources other than rainfall or initial water.

## Architecture

```text
Scenario / CLI / Python UI
          |
Simulation inputs + SolverConfig
          |
SolverBackend interface ---- SerialSolver
          |                     |-- finite-volume face fluxes
CSV I/O + diagnostics          |-- source terms and CFL step
          |                     |-- mass budget
CSV outputs + Python plots
```

`include/grid` owns the structured grid and conserved cell state. `include/physics` owns rainfall representation. `include/solver` defines backend/configuration and the serial numerical method. `include/simulation` builds reproducible initial conditions. `include/io` handles CSV and output. The UI is only a subprocess client; no physics is in Python. Future backends can implement `SolverBackend`, while sharing the same grid and physical parameters; distributed-memory layouts will likely require a later grid-view interface.

## Build and Run

Requirements: CMake 3.16+, C++17 compiler, Python 3.8+. Python plotting/comparison scripts additionally require NumPy and Matplotlib.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/flood_sim --scenario flat-basin --rows 64 --cols 64 --duration 60 --output output/flat
```

The conservation regression tolerance defaults to `1e-8` m³ and can be changed with `-DFLOOD_MASS_TOLERANCE=<value>` at CMake configure time. This is an absolute numerical test threshold, not an accuracy specification.

Scenario names: `flat-basin`, `slope`, `dam-break`, `rain-drain`, `wet-dry`. CLI options include `--terrain path.csv`, `--rainfall path.csv`, `--cell-size metres`, `--duration seconds`, `--output directory`, and physics controls `--gravity`, `--manning`, `--infiltration`, `--cfl`, `--max-dt`, `--dry-depth`, and `--boundary closed|outflow`. `--config config/example.ini` reads key/value configuration; later CLI arguments override matching settings. External terrain and rainfall can be combined with a built-in scenario; terrain CSV dimensions override `--rows`/`--cols`.

Rainfall and terrain examples can be generated with `python3 scripts/generate_rainfall.py` and `python3 scripts/generate_test_terrain.py`. To plot outputs, run `python3 scripts/visualize_results.py --terrain output/flat/terrain.csv --depth output/flat/depth.csv`. Compare depth grids with `python3 scripts/compare_outputs.py reference.csv candidate.csv --cell-size 5`.

Run the local UI after building the selected backend executable with `python3 scripts/ui.py`, then open http://127.0.0.1:8000. The UI offers Serial, OpenMP, and MPI, launches a serial reference and selected-backend run, then presents measured metrics and a depth canvas. MPI selection requires an MPI-enabled binary and launcher. The UI currently offers built-in benchmark scenarios; arbitrary file upload is not implemented.

## Scenarios and Validation

1. **Flat basin:** uniform zero bed, constant 30 mm/hour rain, closed walls. Expected: spatially uniform depth rise; compare stored volume with rainfall input.
2. **Simple slope:** descending bed with a shallow strip of water at the high end, no rain, closed walls. Expected: water moves downslope; verify cells below the release wet and total water remains constant.
3. **Dam break:** one-metre initial water on the left half, dry on the right, no rain, closed walls. Expected: a front propagates into initially dry cells with nonnegative depth.
4. **Rainfall + drainage:** rainfall changes from 60 to 0 mm/hour at 30 s, positive infiltration, sloped bed, open boundaries. Expected: rainfall and infiltration are in the budget and any boundary loss is reported.
5. **Wet/dry transition:** a single shallow moving patch on a dry grid. Expected: finite velocity and nonnegative depths.

CTest runs all five cases, checks volume balance, downhill/front propagation, wet/dry stability, and output generation. These are regression checks, not comparison to analytical benchmarks or field observations. Before publication, add convergence studies, dam-break reference solutions, lake-at-rest equilibrium tests over non-flat beds, independent boundary-flux checks, and calibration/validation against measured data.

Outputs per run: `terrain.csv`, conserved `depth.csv`, `hu.csv`, `hv.csv`, `stats.csv`, and `benchmark.csv`. The benchmark separates scenario/input preparation, solver, and output wall time; it is a local performance record, not a performance claim. `stats.csv` contains final time, steps, volumes, mass residual, maximum depth/velocity, wet cells, flooded area, backend, and thread count. Comparison reports depth errors, relative error, signed volume/flood-area differences, and optional momentum errors under separate configurable tolerances. PNG rendering requires the optional Python packages noted above.

## Phase 2 Status

`SolverBackend` remains the dispatch boundary. The supported backend choices are Serial, OpenMP, and MPI. Serial is the default reference; OpenMP shares the same numerical kernels and passes parity checks at 1/2/4/8/16 threads. MPI is conditionally compiled when an MPI C++ toolchain is found and was built and experimentally validated here with Open MPI 5.0.11. CMake's `FLOOD_ENABLE_MPI` option is optional, and `./build-mpi/flood_sim --capabilities` reports source, toolchain, and runtime status.

Build and run OpenMP with:

```sh
cmake -S . -B build-openmp -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DFLOOD_ENABLE_OPENMP=ON
cmake --build build-openmp -j
ctest --test-dir build-openmp --output-on-failure
./build-openmp/flood_sim --backend openmp --threads 8 --scenario dam-break --rows 512 --cols 512
```

On this Mac, OpenMP uses Homebrew `libomp` if CMake does not discover it through `FindOpenMP`. Use `--backend serial` or `--backend openmp --threads N`; unavailable backends fail explicitly and never fall back. Numerical output comparison uses `-DFLOOD_NUMERICAL_TOLERANCE=1e-8` for depth and `-DFLOOD_MOMENTUM_TOLERANCE=1e-8` for each momentum component. The script equivalents are `--tolerance` and `--momentum-tolerance`.

The final validation table, environment, benchmark measurements, and remaining Phase 2 work are in [docs/phase2.md](docs/phase2.md). Measured OpenMP and MPI data and plotting commands are documented in [benchmarks/README.md](benchmarks/README.md) and [benchmarks/mpi/README.md](benchmarks/mpi/README.md); backend details are in [docs/phase2_architecture.md](docs/phase2_architecture.md), [docs/openmp_design.md](docs/openmp_design.md), and [docs/mpi_design.md](docs/mpi_design.md). No adaptive mesh refinement or dynamic load balancing is included.