# Phase 2 Architecture Note

## Phase 1 Summary

The program builds a `Scenario` (row-major `Grid`, piecewise-constant `Rainfall`, and `SolverConfig`), dispatches through `SolverBackend::run`, and writes grid CSVs plus final diagnostics. `SerialSolver` in `src/solver/SerialSolver.cpp` is the reference implementation. A `Cell` is an array-of-structures entry containing four doubles: bed elevation and conserved depth/momenta. The serial solver must remain the behavior oracle; parallel output is compared as numerical fields, not text.

## Computational Hotspots and Dependencies

Each timestep has these full-grid passes:

1. CFL wave-speed scan and maximum reduction.
2. X- and Y-face hydrostatic reconstruction/Rusanov flux calculations. Each face reads the old state of at most two adjacent cells and writes one distinct face record.
3. Outgoing-water calculation by donor cell, then the positivity limiter by cell.
4. Flux divergence into each cell, followed by rain, infiltration, and local Manning momentum damping.
5. Final/periodic diagnostics reductions.

No cell update may begin until all fluxes and outgoing rates for that step are ready. A step must finish before the next CFL scan because the next timestep depends on the updated state. The existing face pass also scatters outgoing amounts to donor cells, and the flux-application pass scatters to both adjacent cells; those writes race if the loops are naively parallelized. Parallel kernels should instead write independent face records and gather the four incident faces per cell in a fixed order. That avoids atomics and retains deterministic per-cell arithmetic order. Global reductions and step transitions remain synchronization points. Rainfall breakpoints are shared scalar events, not spatial data.

## Proposed Backend Plan

Keep `SerialSolver` as the default oracle. Add backend selection at the CLI boundary and leave scenario construction, physical parameters, CSV I/O, and diagnostics contracts common. The first parallel backend is OpenMP: use static work distribution for independent face and cell loops, a maximum reduction for the CFL rate, and explicit barriers between face construction, per-cell outgoing/limiter calculation, and state update. Thread count is a runtime setting. Compare all conserved depth outputs and mass diagnostics to the serial run for every standard scenario before reporting speedup.

MPI uses static 2D domain decomposition and the same face-then-cell dependency schedule, with global timestep synchronization and halo exchange. It does not introduce new flux or source formulas. Backend failures outside tolerance require diagnosis; tolerance must not be widened to conceal them.

## Memory Estimate and Layout

The current host `Cell` array is AoS: 32 bytes per cell. Current timestep scratch includes face arrays, outgoing rates, limiters, and three-component cell deltas. MPI stores a rectangular local interior and one halo layer, while rank 0 gathers the final global state for current CSV output. Larger grids should continue to be sized against measured peak resident memory.

## MPI Decomposition and Halo Exchange

Use a static 2D Cartesian block decomposition, choosing process-grid dimensions close to the global grid aspect ratio. Each rank owns a contiguous rectangular interior plus one ghost layer. The present first-order face flux needs only the neighboring cell's `bed`, `h`, `hu`, and `hv`; exchange those four values on north/south/east/west edges each timestep. The implementation posts nonblocking receives/sends and waits for completion before computing fluxes; communication is not overlapped with computation. Physical boundaries remain owned by edge ranks and use the same wall/outflow treatment as serial.

All ranks need the same timestep: reduce local CFL maxima with `MPI_Allreduce`, then apply the same rainfall breakpoint and end-time clips. Sum rainfall, infiltration, outflow, and stored-volume diagnostics globally. Measure compute, halo communication, reductions/synchronization, and total wall time separately. Static decomposition only; no dynamic balancing or adaptive mesh in this phase.

## Local Validation Constraints

This workspace is Apple M2/arm64. Apple Clang does not accept plain `-fopenmp`, but OpenMP builds successfully with the installed Homebrew `libomp` and explicit `-Xpreprocessor -fopenmp` flags. MPI was compiled with Homebrew Open MPI 5.0.11; the 34-test MPI matrix passed, including uneven process decompositions. See [phase2.md](phase2.md) and [mpi_design.md](mpi_design.md) for measured validation and benchmark results.