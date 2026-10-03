# Phase 2 Validation Report

## Status

Phase 2 is **experimentally complete** on this machine: Serial, OpenMP, and MPI are implemented, numerically validated, and supported by a repeated MPI performance dataset. Adaptive mesh refinement and dynamic load balancing have not been started.

## Backend Validation Table

| Backend | Source exists | Builds here | Runs here | Numerical validation | Benchmark validation |
|---|---|---|---|---|---|
| Serial | YES | YES | YES | PASS; Phase 1 tests and reference comparisons | PASS; baseline included in measured pairs |
| OpenMP | YES | YES | YES | PASS; all five scenarios at 1/2/4/8/16, zero depth/`hu`/`hv` error in tested runs | PASS; 64², 128², 512² plus one 1024²/2048² capacity run |
| MPI | YES | YES (Open MPI 5.0.11) | YES | PASS; 34/34 CTests, including serial comparisons for five scenarios plus lake-at-rest at 1/2/3/4/6 ranks | PASS; 512² five-scenario matrix at 1/2/4 ranks and 1024²/2048² capacity checks |

MPI was compiled and run against Homebrew Open MPI 5.0.11 on Apple M2/macOS. The full MPI CTest matrix passes 34/34. Direct checks also passed for a one-rank 16² dam-break and uneven 100x101 rain-drain cases at three and four ranks; the latter matched serial depth and both momentum fields exactly. The benchmark runner validates state, budgets, and tolerance outcomes for every recorded MPI row. No backend silently falls back.

## Architecture and Parallelization

The common `SolverBackend::run` contract is retained. `SerialSolver` remains the default and reference. `OpenMPSolver` enables OpenMP execution in the shared numerical timestep code; `MpiSolver` is an optional distributed implementation using the extracted common hydrostatic/Rusanov face and ghost kernels. It uses uneven 2D Cartesian blocks, one axial ghost layer, nonblocking four-edge exchange of `(bed,h,hu,hv)`, exchanged positivity limiters, a global CFL max reduction, global budget/statistic reductions, and root-only gather/output. Critical-rank setup, computation, communication, synchronization, and other timings are recorded as disjoint phases. The 1/2/3/4/6-rank CTest matrix passed, including uneven decompositions.

The per-step work is: CFL max reduction; independent X/Y face construction; per-cell gather of outgoing rates and positivity limiters; per-cell gather of four incident face fluxes; source/state update; serial diagnostic accumulation. Distinct faces and cells own their output records, avoiding scatter races/atomics. Barriers between stages are required because each stage consumes the previous stage's arrays. Per-cell face summation order is fixed so measured OpenMP outputs match the serial outputs exactly in the tested cases.

The serial solver source was refactored only to share its timestep implementation with OpenMP. Compensated summation is used for global volume diagnostics to keep roundoff within the unchanged mass tolerance on large grids; cell states and flux computations are unchanged. The saved Phase 1 outputs for all five scenarios were rerun through the refactored serial path and their final depth files matched exactly. The Phase 1 test suite remains in CTest.

## Numerical Comparison

The comparison utility measures max and mean absolute depth error, RMSE, mean relative error, signed water-volume difference, flooded-area difference, and component-wise maximum/RMSE errors for `hu` and `hv`. Depth and momentum tolerances are independently configurable. The benchmark runner fails and stops when either configured state tolerance or the conservation-residual threshold is exceeded. Wet-cell count, max-depth/max-velocity statistics, and conservation residuals are recorded separately.

Final `terrain.csv`, `depth.csv`, `hu.csv`, and `hv.csv` files are generated for every simulation. Tests and benchmark parity checks use final states; intermediate timestep snapshots are not currently emitted or compared.

## Benchmark Method

`scripts/run_benchmarks.py` measures solver time from the executable's high-resolution timer, records process wall time, and computes speedup and efficiency. The existing OpenMP scenario matrices use two repeats and medians. Newly generated MPI data uses one run per case and is an exploratory host-specific measurement, not a publication-grade repeated study. MPI per-rank RSS is not measured; MPI communication and phase timing fields come from solver instrumentation. Plot scripts read result CSVs and do not hard-code measurements.

The refreshed 512² MPI matrix contains five scenarios at 1/2/4 ranks (15 rows). Mean measured speedups across those scenarios were 1.02x, 1.69x, and 2.22x at 1, 2, and 4 ranks. All depth and momentum comparison errors were zero; the largest absolute conservation residual was approximately $5.82\times10^{-11}$ m³. One flat-basin capacity run per size was also recorded: 1024² speedups were 0.97x, 1.69x, and 2.26x at 1/2/4 ranks; 2048² speedups were 1.04x, 1.28x, and 1.28x. These six capacity measurements are single runs, not repeated estimates. The maximum absolute difference between a rounded total runtime and its recorded phase sum is 3 microseconds. See [benchmarks/mpi/README.md](../benchmarks/mpi/README.md).

Recorded files:

- `benchmarks/openmp/results.csv`: 64² and 128²; five scenarios, five thread counts, two repeats.
- `benchmarks/openmp/results-512.csv`: 512²; five scenarios, five thread counts, two repeats.
- `benchmarks/openmp/results-capacity.csv`: one flat-basin run per size, serial and OpenMP(4), for 1024² and 2048².
- `benchmarks/openmp/availability.csv`: machine-detected backend availability; unavailable backends have no benchmark result rows.
- `benchmarks/mpi/availability.csv`: MPI availability from the Open MPI-enabled executable.
- `benchmarks/mpi/results-512.csv` and `results-capacity.csv`: 21 measured MPI rows, with metadata sidecars.
- `benchmarks/plots/`: runtime, speedup, OpenMP and MPI scaling, MPI communication/phase timing, grid-size, conservation, and numerical-error plots derived from recorded CSVs.

Legacy CSVs are unchanged (SHA-256: `results.csv` `7842b70d99aa80508a46cef280486a49d5644cc8a77aeefb095f9df4c00ece5a`; `results-512.csv` `37c6eea7f438def81bcd3335e54900b2455483c40307cb3b74c35c2afd85b085`). Their sidecars contain known host/compiler/repetition metadata and explicitly leave original timestamps unknown. Newly generated rows include timestamp/compiler/OS/CPU/repetition metadata and peak RSS when `/usr/bin/time` supports it. The runner refuses to overwrite result files unless explicitly requested.

Across scenarios at 512², the mean measured speedups were 1.015x, 1.522x, 1.944x, 2.019x, and 1.757x for 1, 2, 4, 8, and 16 threads respectively. Maximum depth and component momentum errors were zero in these runs; the largest recorded serial or OpenMP conservation residual was approximately $5.82\times10^{-11}$ m³. These are two-trial host-specific results, not a claim of linear scaling.

One controlled flat-basin run per larger size also completed: 1024² serial 0.623 s/259.2 MB RSS and OpenMP(4) 0.303 s/259.4 MB; 2048² serial 3.324 s/1030.0 MB and OpenMP(4) 1.193 s/1030.1 MB. Depth and momentum errors were zero and conservation passed. These single-run capacity checks are not repeatable benchmark estimates.

## Environment and Reproduction

Validated environment: Apple M2 (8 logical CPU cores), macOS 26.6.2, Apple Clang 16, CMake 4.3.3, Homebrew `libomp` 23.1.2, Homebrew Open MPI 5.0.11, Python 3.11, NumPy, and Matplotlib. OpenMP uses Homebrew `libomp` when CMake does not discover it through `FindOpenMP`; plain Apple-Clang `-fopenmp` is unsupported. MPI wrappers and launcher are available from `/opt/homebrew/bin`. The MPI-enabled configuration passes 34 CTests.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DFLOOD_ENABLE_OPENMP=ON -DFLOOD_NUMERICAL_TOLERANCE=1e-8 \
  -DFLOOD_MOMENTUM_TOLERANCE=1e-8 -DFLOOD_MASS_TOLERANCE=1e-8 \
  -DFLOOD_ENABLE_MPI=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
./build/flood_sim --capabilities
python3 scripts/run_benchmarks.py --binary build/flood_sim \
  --scenarios flat-basin,slope,dam-break,rain-drain,wet-dry \
  --sizes 512x512 --threads 1,2,4,8,16 --repeats 2 \
  --output benchmarks/openmp/results-512.csv
python3 scripts/plot_benchmarks.py --input benchmarks/openmp/results.csv \
  benchmarks/openmp/results-512.csv
python3 scripts/ui.py
```

Configure MPI with `-DFLOOD_ENABLE_MPI=ON`, build, verify `--capabilities`, and run `mpirun -np 4 ./build-mpi/flood_sim --backend mpi --scenario dam-break --rows 512 --cols 512`. The registered MPI CTests cover 1/2/3/4/6 ranks across five scenarios and lake-at-rest.

The local UI is an orchestration client; it uses the executable capability report, displays NOT RUN/RUNNING/PASS/FAIL/UNAVAILABLE, and never silently falls back. The 75 legacy OpenMP rows are preserved byte-for-byte with partial metadata sidecars; original timestamps were not captured and are not fabricated.

## Remaining Phase 2 Work

Remaining Phase 2 work is the repeated and broader MPI performance study and any project-level acceptance items beyond this implementation validation. The recorded MPI speedups are exploratory because each case was run once. Phase 3 must wait until Phase 2 acceptance is complete.