#include "solver/SerialSolver.hpp"
#include "solver/detail/SharedSolver.hpp"
#include "solver/detail/FiniteVolumeKernels.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef FLOOD_HAS_OPENMP
#include <omp.h>
#endif

namespace flood {
namespace {

using detail::finite_volume::FaceFlux;
using detail::finite_volume::State;
using detail::finite_volume::ghostForOutflow;
using detail::finite_volume::ghostForWall;
using detail::finite_volume::rusanovFace;

struct CompensatedSum {
    double value = 0.0;
    double correction = 0.0;

    void add(double term) {
        const double adjusted = term - correction;
        const double next = value + adjusted;
        correction = (next - value) - adjusted;
        value = next;
    }
};

void validateConfig(const SolverConfig& config) {
    if (!(config.gravity > 0.0) || config.manningN < 0.0 ||
        config.infiltrationMetersPerSecond < 0.0 || !(config.dryDepth > 0.0) ||
        !(config.cfl > 0.0 && config.cfl <= 1.0) || !(config.maxTimestep > 0.0) ||
        config.endTime < 0.0 || !std::isfinite(config.gravity) ||
        !std::isfinite(config.manningN) ||
        !std::isfinite(config.infiltrationMetersPerSecond) ||
        !std::isfinite(config.dryDepth) || !std::isfinite(config.cfl) ||
        !std::isfinite(config.maxTimestep) || !std::isfinite(config.endTime)) {
        throw std::invalid_argument("Invalid solver configuration");
    }
}

} // namespace

Diagnostics detail::runSharedSolver(Grid& grid, const Rainfall& rainfall,
                                    const SolverConfig& config, bool useOpenMP,
                                    int threads) {
    validateConfig(config);
#ifdef FLOOD_HAS_OPENMP
    if (useOpenMP) {
        if (threads < 1) throw std::invalid_argument("OpenMP thread count must be positive");
        omp_set_num_threads(threads);
    }
#else
    if (useOpenMP) throw std::runtime_error("OpenMP support was not compiled in");
    (void)threads;
#endif
    const auto start = std::chrono::steady_clock::now();
    const std::size_t rows = grid.rows(), cols = grid.cols();
    const double area = grid.dx() * grid.dy();
    const double xFaceLength = grid.dy(), yFaceLength = grid.dx();
    CompensatedSum initialVolumeSum;
    for (const auto& cell : grid.cells()) {
        if (cell.h < 0.0 || !std::isfinite(cell.h) || !std::isfinite(cell.bed) ||
            !std::isfinite(cell.hu) || !std::isfinite(cell.hv)) {
            throw std::invalid_argument("Initial grid contains invalid state values");
        }
        initialVolumeSum.add(cell.h * area);
    }
    const double initialVolume = initialVolumeSum.value;
    CompensatedSum rainfallVolumeSum, outflowVolumeSum, infiltrationVolumeSum;

    Diagnostics diagnostics;
    diagnostics.minDepth = std::numeric_limits<double>::infinity();
    double time = 0.0;
    while (time < config.endTime) {
        double maxRate = 0.0;
        #ifdef FLOOD_HAS_OPENMP
        #pragma omp parallel for reduction(max:maxRate) schedule(static) if(useOpenMP)
        #endif
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(grid.size()); ++index) {
            const auto& cell = grid.cells()[static_cast<std::size_t>(index)];
            if (cell.h <= config.dryDepth) continue;
            const double u = cell.hu / cell.h, v = cell.hv / cell.h;
            const double wave = std::sqrt(config.gravity * cell.h);
            maxRate = std::max(maxRate, (std::abs(u) + wave) / grid.dx() +
                                         (std::abs(v) + wave) / grid.dy());
        }
        double dt = maxRate > 0.0 ? config.cfl / maxRate : config.maxTimestep;
        dt = std::min({dt, config.maxTimestep, config.endTime - time});
        const double nextRainChange = rainfall.nextChangeAfter(time);
        if (nextRainChange > time) dt = std::min(dt, nextRainChange - time);
        if (!(dt > 0.0) || !std::isfinite(dt)) throw std::runtime_error("CFL timestep became invalid");

        std::vector<FaceFlux> xFaces(rows * (cols + 1));
        std::vector<FaceFlux> yFaces((rows + 1) * cols);
        std::vector<double> outgoing(grid.size(), 0.0);
        auto recordFace = [&](FaceFlux& face, bool lowBoundary, bool highBoundary) {
            if (face.massRate > 0.0 && face.left != std::numeric_limits<std::size_t>::max()) {
                face.donor = face.left;
            } else if (face.massRate < 0.0 && face.right != std::numeric_limits<std::size_t>::max()) {
                face.donor = face.right;
            }
            face.boundaryOutflow = (lowBoundary && face.massRate < 0.0) ||
                                   (highBoundary && face.massRate > 0.0);
        };

        #ifdef FLOOD_HAS_OPENMP
        #pragma omp parallel for collapse(2) schedule(static) if(useOpenMP)
        #endif
        for (std::ptrdiff_t rowIndex = 0; rowIndex < static_cast<std::ptrdiff_t>(rows); ++rowIndex) {
            for (std::ptrdiff_t faceColIndex = 0; faceColIndex <= static_cast<std::ptrdiff_t>(cols); ++faceColIndex) {
                const auto row = static_cast<std::size_t>(rowIndex);
                const auto faceCol = static_cast<std::size_t>(faceColIndex);
                const bool low = faceCol == 0, high = faceCol == cols;
                const std::size_t leftIndex = low ? std::numeric_limits<std::size_t>::max()
                                                  : row * cols + faceCol - 1;
                const std::size_t rightIndex = high ? std::numeric_limits<std::size_t>::max()
                                                    : row * cols + faceCol;
                Cell left = low ? (config.boundary == BoundaryCondition::Closed
                    ? ghostForWall(grid.at(row, 0), true)
                    : ghostForOutflow(grid.at(row, 0), true, true))
                                : grid.cells()[leftIndex];
                Cell right = high ? (config.boundary == BoundaryCondition::Closed
                    ? ghostForWall(grid.at(row, cols - 1), true)
                    : ghostForOutflow(grid.at(row, cols - 1), true, false))
                                  : grid.cells()[rightIndex];
                auto& face = xFaces[row * (cols + 1) + faceCol];
                face = rusanovFace(left, right, leftIndex, rightIndex, true,
                                   config.gravity, config.dryDepth);
                recordFace(face, low, high);
            }
        }
        #ifdef FLOOD_HAS_OPENMP
        #pragma omp parallel for collapse(2) schedule(static) if(useOpenMP)
        #endif
        for (std::ptrdiff_t faceRowIndex = 0; faceRowIndex <= static_cast<std::ptrdiff_t>(rows); ++faceRowIndex) {
            for (std::ptrdiff_t colIndex = 0; colIndex < static_cast<std::ptrdiff_t>(cols); ++colIndex) {
                const auto faceRow = static_cast<std::size_t>(faceRowIndex);
                const auto col = static_cast<std::size_t>(colIndex);
                const bool low = faceRow == 0, high = faceRow == rows;
                const std::size_t leftIndex = low ? std::numeric_limits<std::size_t>::max()
                                                  : (faceRow - 1) * cols + col;
                const std::size_t rightIndex = high ? std::numeric_limits<std::size_t>::max()
                                                    : faceRow * cols + col;
                Cell left = low ? (config.boundary == BoundaryCondition::Closed
                    ? ghostForWall(grid.at(0, col), false)
                    : ghostForOutflow(grid.at(0, col), false, true))
                                : grid.cells()[leftIndex];
                Cell right = high ? (config.boundary == BoundaryCondition::Closed
                    ? ghostForWall(grid.at(rows - 1, col), false)
                    : ghostForOutflow(grid.at(rows - 1, col), false, false))
                                  : grid.cells()[rightIndex];
                auto& face = yFaces[faceRow * cols + col];
                face = rusanovFace(left, right, leftIndex, rightIndex, false,
                                   config.gravity, config.dryDepth);
                recordFace(face, low, high);
            }
        }

        const auto addOutgoing = [](const FaceFlux& face, std::size_t cell,
                                    double& total) {
            if (face.donor == cell) total += std::abs(face.massRate);
        };
        #ifdef FLOOD_HAS_OPENMP
        #pragma omp parallel for schedule(static) if(useOpenMP)
        #endif
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(grid.size()); ++index) {
            const std::size_t cell = static_cast<std::size_t>(index);
            const std::size_t row = cell / cols, col = cell % cols;
            double rate = 0.0;
            addOutgoing(xFaces[row * (cols + 1) + col], cell, rate);
            addOutgoing(xFaces[row * (cols + 1) + col + 1], cell, rate);
            addOutgoing(yFaces[row * cols + col], cell, rate);
            addOutgoing(yFaces[(row + 1) * cols + col], cell, rate);
            outgoing[cell] = rate;
        }

        std::vector<double> limiter(grid.size(), 1.0);
        #ifdef FLOOD_HAS_OPENMP
        #pragma omp parallel for schedule(static) if(useOpenMP)
        #endif
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(grid.size()); ++index) {
            const std::size_t i = static_cast<std::size_t>(index);
            const double available = grid.cells()[i].h * area;
            if (outgoing[i] * dt > available && outgoing[i] > 0.0)
                limiter[i] = available / (outgoing[i] * dt);
        }
        std::vector<State> delta(grid.size(), State{0.0, 0.0, 0.0});
        const auto addFaceContribution = [&](const FaceFlux& face, bool cellIsRight,
                                             double length, State& cellDelta) {
            const double scale = face.donor == std::numeric_limits<std::size_t>::max()
                ? 1.0 : limiter[face.donor];
            const State& flux = cellIsRight ? face.rightFlux : face.leftFlux;
            const double sign = cellIsRight ? 1.0 : -1.0;
            for (std::size_t k = 0; k < 3; ++k)
                cellDelta[k] += sign * scale * flux[k] * length / area;
        };
        #ifdef FLOOD_HAS_OPENMP
        #pragma omp parallel for schedule(static) if(useOpenMP)
        #endif
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(grid.size()); ++index) {
            const std::size_t cell = static_cast<std::size_t>(index);
            const std::size_t row = cell / cols, col = cell % cols;
            State cellDelta{0.0, 0.0, 0.0};
            addFaceContribution(xFaces[row * (cols + 1) + col], true, xFaceLength, cellDelta);
            addFaceContribution(xFaces[row * (cols + 1) + col + 1], false, xFaceLength, cellDelta);
            addFaceContribution(yFaces[row * cols + col], true, yFaceLength, cellDelta);
            addFaceContribution(yFaces[(row + 1) * cols + col], false, yFaceLength, cellDelta);
            delta[cell] = cellDelta;
        }
        const auto accumulateOutflow = [&](const std::vector<FaceFlux>& faces, double length) {
            for (const auto& face : faces) {
                if (!face.boundaryOutflow) continue;
                const double scale = face.donor == std::numeric_limits<std::size_t>::max()
                    ? 1.0 : limiter[face.donor];
                outflowVolumeSum.add(scale * std::abs(face.massRate) * length * dt);
            }
        };
        accumulateOutflow(xFaces, xFaceLength);
        accumulateOutflow(yFaces, yFaceLength);

        const double rainRate = rainfall.metersPerSecond(time);
        rainfallVolumeSum.add(rainRate * area * static_cast<double>(grid.size()) * dt);
        std::vector<double> infiltrationByCell(grid.size(), 0.0);
        std::vector<unsigned char> invalidState(grid.size(), 0);
        #ifdef FLOOD_HAS_OPENMP
        #pragma omp parallel for schedule(static) if(useOpenMP)
        #endif
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(grid.size()); ++index) {
            const std::size_t i = static_cast<std::size_t>(index);
            auto& cell = grid.cells()[i];
            cell.h += dt * (delta[i][0] + rainRate);
            cell.hu += dt * delta[i][1];
            cell.hv += dt * delta[i][2];
            if (cell.h < 0.0 && cell.h > -1e-12) cell.h = 0.0;
            if (cell.h < 0.0 || !std::isfinite(cell.h) || !std::isfinite(cell.hu) ||
                !std::isfinite(cell.hv)) {
                invalidState[i] = 1;
                continue;
            }
            const double infiltrated = std::min(cell.h, config.infiltrationMetersPerSecond * dt);
            cell.h -= infiltrated;
            infiltrationByCell[i] = infiltrated * area;
            if (cell.h <= config.dryDepth) {
                cell.hu = 0.0;
                cell.hv = 0.0;
            } else if (config.manningN > 0.0) {
                const double u = cell.hu / cell.h, v = cell.hv / cell.h;
                const double speed = std::hypot(u, v);
                const double coefficient = config.gravity * config.manningN * config.manningN *
                                           speed / std::pow(cell.h, 4.0 / 3.0);
                const double frictionScale = 1.0 / (1.0 + dt * coefficient);
                cell.hu *= frictionScale;
                cell.hv *= frictionScale;
            }
        }
        for (std::size_t i = 0; i < grid.size(); ++i) {
            if (invalidState[i])
                throw std::runtime_error("Non-finite or negative state detected; simulation stopped");
            infiltrationVolumeSum.add(infiltrationByCell[i]);
        }
        time += dt;
        ++diagnostics.steps;
        if (diagnostics.steps % 100 == 0) {
            double maxDepth = 0.0, maxVelocity = 0.0;
            std::size_t wetCells = 0;
            for (const auto& cell : grid.cells()) {
                maxDepth = std::max(maxDepth, cell.h);
                if (cell.h > config.dryDepth) {
                    ++wetCells;
                    maxVelocity = std::max(maxVelocity,
                        std::hypot(cell.hu, cell.hv) / cell.h);
                }
            }
            std::clog << "Step " << diagnostics.steps << " time=" << time
                      << " s dt=" << dt << " s max_depth=" << maxDepth
                      << " m max_velocity=" << maxVelocity << " m/s wet_cells="
                      << wetCells << '\n';
        }
    }

    diagnostics.time = time;
    CompensatedSum storedVolumeSum;
    diagnostics.minDepth = std::numeric_limits<double>::infinity();
    diagnostics.maxDepth = 0.0;
    diagnostics.maxVelocity = 0.0;
    diagnostics.wetCells = 0;
    for (const auto& cell : grid.cells()) {
        storedVolumeSum.add(cell.h * area);
        diagnostics.minDepth = std::min(diagnostics.minDepth, cell.h);
        diagnostics.maxDepth = std::max(diagnostics.maxDepth, cell.h);
        if (cell.h > config.dryDepth) {
            ++diagnostics.wetCells;
            diagnostics.maxVelocity = std::max(diagnostics.maxVelocity,
                std::hypot(cell.hu, cell.hv) / cell.h);
        }
    }
    diagnostics.storedVolume = storedVolumeSum.value;
    diagnostics.rainfallVolume = rainfallVolumeSum.value;
    diagnostics.outflowVolume = outflowVolumeSum.value;
    diagnostics.infiltrationVolume = infiltrationVolumeSum.value;
    diagnostics.floodedArea = static_cast<double>(diagnostics.wetCells) * area;
    diagnostics.massResidual = initialVolume + diagnostics.rainfallVolume -
        diagnostics.outflowVolume - diagnostics.infiltrationVolume - diagnostics.storedVolume;
    diagnostics.solverSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return diagnostics;
}

Diagnostics SerialSolver::run(Grid& grid, const Rainfall& rainfall,
                              const SolverConfig& config) const {
    return detail::runSharedSolver(grid, rainfall, config, false, 1);
}

std::string toString(BoundaryCondition boundary) {
    return boundary == BoundaryCondition::Closed ? "closed" : "outflow";
}

} // namespace flood