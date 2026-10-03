# Benchmarking

`scripts/run_benchmarks.py` queries `flood_sim --capabilities`, writes backend states separately to `availability.csv`, skips unavailable backends without result rows, and compares executable backends with serial. OpenMP uses `--threads`; MPI uses `--processes` (default 1/2/4/8) through the CMake-detected launcher. Use a separate `--output benchmarks/mpi/results.csv` for MPI runs. The default matrix is 512x512, 1024x1024, and 2048x2048 with 1/2/4/8/16 OpenMP threads. Each result row includes actual solver/process wall times, peak RSS where measurable, speedup, efficiency, final max depth, depth max/mean/RMSE/relative error, momentum max errors/RMSEs, velocity-statistic error, stored volume/difference, flooded area/difference, wet-cell agreement, conservation residuals, compute/communication/synchronization timing fields, tolerance outcome, timestamp, compiler/version, OS, CPU, and repeat count. It refuses to overwrite an existing result file unless `--overwrite` is explicit. A JSON metadata sidecar records run configuration and capabilities.

Example:

```sh
python3 scripts/run_benchmarks.py --binary build-openmp/flood_sim \
  --scenarios flat-basin,slope,dam-break,rain-drain,wet-dry \
  --sizes 512x512 --threads 1,2,4,8,16 --repeats 2 \
  --output benchmarks/openmp/new-run.csv
python3 scripts/plot_benchmarks.py \
  --input benchmarks/openmp/results.csv benchmarks/openmp/results-512.csv
```

MPI example, only on a host where `--capabilities` reports MPI `AVAILABLE`:

```sh
python3 scripts/run_benchmarks.py --binary build/flood_sim \
  --backends mpi --processes 1,2,4,8 --sizes 512x512 \
  --output benchmarks/mpi/results.csv
```

`scripts/plot_benchmarks.py` creates plots only for result categories actually present, including runtime, speedup and efficiency versus thread/process count, grid size, numerical error, conservation residual, MPI communication, and MPI compute/communication/synchronization breakdown. The validated MPI datasets and host-specific measurements are documented in [mpi/README.md](mpi/README.md).

The original `results.csv` and `results-512.csv` were preserved byte-for-byte; their sidecars record the known host/configuration and explicitly note that exact original timestamps were not captured. `results-capacity.csv` contains one measured flat-basin run per size: serial and OpenMP at four threads for 1024² and 2048². Both passed state/conservation checks. The 2048² run used about 1.03 GB peak RSS per process on this host. These are capacity checks, not repeated publication benchmarks.