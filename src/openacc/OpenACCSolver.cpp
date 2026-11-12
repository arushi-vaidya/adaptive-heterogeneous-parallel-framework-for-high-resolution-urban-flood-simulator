#include "solver/AcceleratorSolvers.hpp"
#include "solver/detail/FlatKernels.hpp"
#include "solver/detail/FlatSolverCommon.hpp"

#include <chrono>
#include <stdexcept>
#include <vector>

#ifdef _OPENACC
#include <openacc.h>
#endif

namespace flood {

Diagnostics detail::runOpenACCSolver(Grid& grid, const Rainfall& rainfall,
                                     const SolverConfig& config) {
    using namespace detail::flat;
    validateFlatConfig(grid, config);
    const auto start = std::chrono::steady_clock::now();
    const Params p = makeParams(grid, config);
    const long n = p.rows * p.cols;
    const long nx = p.rows * (p.cols + 1);
    const long ny = (p.rows + 1) * p.cols;
    const double area = p.dx * p.dy;

    SoA state = pack(grid);
    std::vector<Face> xFacesStore(static_cast<std::size_t>(nx));
    std::vector<Face> yFacesStore(static_cast<std::size_t>(ny));
    std::vector<double> limiterStore(static_cast<std::size_t>(n), 1.0);
    double* bed = state.bed.data();
    double* h = state.h.data();
    double* hu = state.hu.data();
    double* hv = state.hv.data();
    Face* xf = xFacesStore.data();
    Face* yf = yFacesStore.data();
    double* limiter = limiterStore.data();

    double initialVolume = 0.0;
    for (long i = 0; i < n; ++i) initialVolume += h[i] * area;
    double rainfallVolume = 0.0, outflowVolume = 0.0, infiltrationVolume = 0.0;
    Diagnostics diagnostics;
    double time = config.startTime;
    const auto setupDone = std::chrono::steady_clock::now();

    // State stays resident on the device for the whole run; only scalars
    // (dt, rain rate, reductions) cross the host/device boundary each step.
    #pragma acc data copyin(bed[0:n]) copy(h[0:n], hu[0:n], hv[0:n]) \
                     create(xf[0:nx], yf[0:ny], limiter[0:n])
    {
        while (time < config.endTime) {
            double maxRate = 0.0;
            #pragma acc parallel loop reduction(max:maxRate) present(h, hu, hv)
            for (long i = 0; i < n; ++i) {
                const double rate = cflRate(p, h[i], hu[i], hv[i]);
                maxRate = rate > maxRate ? rate : maxRate;
            }
            const double dt = nextTimestep(maxRate, time, rainfall, config);
            const double rainRate = rainfall.metersPerSecond(time);

            #pragma acc parallel loop collapse(2) present(bed, h, hu, hv, xf)
            for (long row = 0; row < p.rows; ++row)
                for (long faceCol = 0; faceCol <= p.cols; ++faceCol)
                    xFace(p, bed, h, hu, hv, row, faceCol, xf[row * (p.cols + 1) + faceCol]);
            #pragma acc parallel loop collapse(2) present(bed, h, hu, hv, yf)
            for (long faceRow = 0; faceRow <= p.rows; ++faceRow)
                for (long col = 0; col < p.cols; ++col)
                    yFace(p, bed, h, hu, hv, faceRow, col, yf[faceRow * p.cols + col]);

            #pragma acc parallel loop present(h, xf, yf, limiter)
            for (long i = 0; i < n; ++i)
                limiter[i] = limiterFor(h[i], area, outgoingRate(p, xf, yf, i), dt);

            double outflow = 0.0;
            #pragma acc parallel loop reduction(+:outflow) present(xf, limiter)
            for (long f = 0; f < nx; ++f) outflow += faceOutflowVolume(xf[f], limiter, p.dy, dt);
            #pragma acc parallel loop reduction(+:outflow) present(yf, limiter)
            for (long f = 0; f < ny; ++f) outflow += faceOutflowVolume(yf[f], limiter, p.dx, dt);

            double infiltrated = 0.0;
            int invalidCells = 0;
            #pragma acc parallel loop reduction(+:infiltrated, invalidCells) \
                                      present(h, hu, hv, xf, yf, limiter)
            for (long i = 0; i < n; ++i) {
                int invalid = 0;
                infiltrated += updateCell(p, xf, yf, limiter, i, dt, rainRate, h, hu, hv, invalid);
                invalidCells += invalid;
            }
            if (invalidCells > 0)
                throw std::runtime_error("Non-finite or negative state detected; simulation stopped");
            outflowVolume += outflow;
            infiltrationVolume += infiltrated;
            rainfallVolume += rainRate * area * static_cast<double>(n) * dt;
            time += dt;
            ++diagnostics.steps;
        }
    }

    unpack(state, grid);
    finalize(grid, config, time, initialVolume, rainfallVolume, outflowVolume,
             infiltrationVolume, diagnostics);
    const auto end = std::chrono::steady_clock::now();
    diagnostics.solverSeconds = std::chrono::duration<double>(end - start).count();
    diagnostics.setupSeconds = std::chrono::duration<double>(setupDone - start).count();
    diagnostics.computationSeconds = diagnostics.solverSeconds - diagnostics.setupSeconds;
    diagnostics.backend = "openacc";
    diagnostics.threads = 1;
    return diagnostics;
}

Diagnostics OpenACCSolver::run(Grid& grid, const Rainfall& rainfall,
                               const SolverConfig& config) const {
#ifdef FLOOD_HAS_OPENACC
    return detail::runOpenACCSolver(grid, rainfall, config);
#else
    (void)grid;
    (void)rainfall;
    (void)config;
    throw std::runtime_error(
        "OpenACC backend unavailable: the C++ compiler has no OpenACC support "
        "(build with nvc++ -acc or g++ -fopenacc)");
#endif
}

} // namespace flood
