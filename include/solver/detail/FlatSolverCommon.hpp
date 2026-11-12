#pragma once

// Host-side helpers shared by the accelerator backends (OpenACC, CUDA):
// configuration checks, packing the grid into structure-of-arrays buffers,
// and the end-of-run diagnostics that the serial solver also reports.

#include "solver/SerialSolver.hpp"
#include "solver/detail/FlatKernels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace flood::detail::flat {

inline void validateFlatConfig(const Grid& grid, const SolverConfig& config) {
    if (!(config.gravity > 0.0) || config.manningN < 0.0 ||
        config.infiltrationMetersPerSecond < 0.0 || !(config.dryDepth > 0.0) ||
        !(config.cfl > 0.0 && config.cfl <= 1.0) || !(config.maxTimestep > 0.0) ||
        config.endTime < 0.0 || !std::isfinite(config.endTime)) {
        throw std::invalid_argument("Invalid solver configuration");
    }
    for (const auto& cell : grid.cells()) {
        if (cell.h < 0.0 || !std::isfinite(cell.h) || !std::isfinite(cell.bed) ||
            !std::isfinite(cell.hu) || !std::isfinite(cell.hv))
            throw std::invalid_argument("Initial grid contains invalid state values");
    }
}

inline Params makeParams(const Grid& grid, const SolverConfig& config) {
    Params p;
    p.rows = static_cast<long>(grid.rows());
    p.cols = static_cast<long>(grid.cols());
    p.dx = grid.dx();
    p.dy = grid.dy();
    p.gravity = config.gravity;
    p.dryDepth = config.dryDepth;
    p.manningN = config.manningN;
    p.infiltration = config.infiltrationMetersPerSecond;
    p.closed = config.boundary == BoundaryCondition::Closed ? 1 : 0;
    return p;
}

struct SoA {
    std::vector<double> bed, h, hu, hv;
};

inline SoA pack(const Grid& grid) {
    SoA s;
    const auto n = grid.size();
    s.bed.resize(n); s.h.resize(n); s.hu.resize(n); s.hv.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& c = grid.cells()[i];
        s.bed[i] = c.bed; s.h[i] = c.h; s.hu[i] = c.hu; s.hv[i] = c.hv;
    }
    return s;
}

inline void unpack(const SoA& s, Grid& grid) {
    for (std::size_t i = 0; i < grid.size(); ++i) {
        auto& c = grid.cells()[i];
        c.h = s.h[i]; c.hu = s.hu[i]; c.hv = s.hv[i];
    }
}

inline double nextTimestep(double maxRate, double time, const Rainfall& rainfall,
                           const SolverConfig& config) {
    double dt = maxRate > 0.0 ? config.cfl / maxRate : config.maxTimestep;
    dt = std::min({dt, config.maxTimestep, config.endTime - time});
    const double nextRainChange = rainfall.nextChangeAfter(time);
    if (nextRainChange > time) dt = std::min(dt, nextRainChange - time);
    if (!(dt > 0.0) || !std::isfinite(dt)) throw std::runtime_error("CFL timestep became invalid");
    return dt;
}

inline void finalize(const Grid& grid, const SolverConfig& config, double time,
                     double initialVolume, double rainfallVolume, double outflowVolume,
                     double infiltrationVolume, Diagnostics& d) {
    const double area = grid.dx() * grid.dy();
    d.time = time;
    d.minDepth = std::numeric_limits<double>::infinity();
    d.maxDepth = 0.0;
    d.maxVelocity = 0.0;
    d.wetCells = 0;
    double stored = 0.0;
    for (const auto& cell : grid.cells()) {
        stored += cell.h * area;
        d.minDepth = std::min(d.minDepth, cell.h);
        d.maxDepth = std::max(d.maxDepth, cell.h);
        if (cell.h > config.dryDepth) {
            ++d.wetCells;
            d.maxVelocity = std::max(d.maxVelocity, std::hypot(cell.hu, cell.hv) / cell.h);
        }
    }
    d.storedVolume = stored;
    d.rainfallVolume = rainfallVolume;
    d.outflowVolume = outflowVolume;
    d.infiltrationVolume = infiltrationVolume;
    d.floodedArea = static_cast<double>(d.wetCells) * area;
    d.massResidual = initialVolume + rainfallVolume - outflowVolume - infiltrationVolume - stored;
}

} // namespace flood::detail::flat
