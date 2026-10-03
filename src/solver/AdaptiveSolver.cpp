#include "solver/AdaptiveSolver.hpp"

#include "solver/detail/FiniteVolumeKernels.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace flood {
namespace {

using detail::finite_volume::FaceFlux;
using detail::finite_volume::State;
using detail::finite_volume::ghostForOutflow;
using detail::finite_volume::ghostForWall;
using detail::finite_volume::rusanovFace;

constexpr std::size_t noCell = std::numeric_limits<std::size_t>::max();
using CellKey = std::tuple<PatchId, std::size_t, std::size_t>;

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

struct ConservedIntegrals {
    long double water = 0.0L;
    long double hu = 0.0L;
    long double hv = 0.0L;
};

struct CellRef {
    AdaptiveCellId id;
    Cell* state;
    std::size_t level;
    double dx;
    double dy;
    double area;
    std::size_t rowBegin;
    std::size_t rowEnd;
    std::size_t colBegin;
    std::size_t colEnd;
};

struct EdgeRef {
    std::size_t begin;
    std::size_t end;
    std::size_t cell;
    bool isLeft;
};

struct FaceSegment {
    std::size_t left = noCell;
    std::size_t right = noCell;
    double length = 0.0;
    bool xDirection = true;
    bool lowBoundary = false;
    bool highBoundary = false;
    FaceFlux flux;
};

struct Mesh {
    std::vector<CellRef> cells;
    std::vector<FaceSegment> faces;
    double areaSum = 0.0;
};

std::size_t finestCellScale(const AdaptiveGrid& grid, std::size_t level) {
    std::size_t scale = 1;
    for (std::size_t current = level; current < grid.config().maxRefinementLevel; ++current) {
        if (scale > std::numeric_limits<std::size_t>::max() /
                        grid.config().refinementRatio)
            throw std::overflow_error("Adaptive solver coordinate scale overflows");
        scale *= grid.config().refinementRatio;
    }
    return scale;
}

CellKey keyFor(const AdaptiveCellId& id) {
    return {id.patchId, id.row, id.col};
}

void validateConfig(const SolverConfig& config,
                    const AdaptiveSolverOptions& options,
                    const AdaptiveGrid& grid) {
    if (!(config.gravity > 0.0) || config.manningN < 0.0 ||
        config.infiltrationMetersPerSecond < 0.0 || !(config.dryDepth > 0.0) ||
        !(config.cfl > 0.0 && config.cfl <= 1.0) || !(config.maxTimestep > 0.0) ||
        config.endTime < 0.0 || !std::isfinite(config.gravity) ||
        !std::isfinite(config.manningN) ||
        !std::isfinite(config.infiltrationMetersPerSecond) ||
        !std::isfinite(config.dryDepth) || !std::isfinite(config.cfl) ||
        !std::isfinite(config.maxTimestep) || !std::isfinite(config.endTime))
        throw std::invalid_argument("Invalid adaptive solver configuration");
    if (grid.config().maxRefinementLevel > 2)
        throw std::invalid_argument("Adaptive solver currently supports refinement levels 0, 1, and 2");
    if (!std::isfinite(options.refineThreshold) ||
        !std::isfinite(options.coarsenThreshold) ||
        options.coarsenThreshold < 0.0 ||
        options.refineThreshold <= options.coarsenThreshold ||
        options.coarsenPersistence == 0 || options.regridIntervalSteps == 0)
        throw std::invalid_argument("Invalid adaptive refinement/coarsening options");
}

Mesh buildMesh(AdaptiveGrid& grid) {
    Mesh mesh;
    std::size_t finestScale = 1;
    for (std::size_t level = 0; level < grid.config().maxRefinementLevel; ++level) {
        if (finestScale > std::numeric_limits<std::size_t>::max() /
                              grid.config().refinementRatio)
            throw std::overflow_error("Adaptive solver coordinate scale overflows");
        finestScale *= grid.config().refinementRatio;
    }
    const std::size_t domainRows = grid.rows() * finestScale;
    const std::size_t domainCols = grid.cols() * finestScale;
    const double finestDy = grid.dy0() / static_cast<double>(finestScale);
    const double finestDx = grid.dx0() / static_cast<double>(finestScale);

    std::map<std::size_t, std::vector<EdgeRef>> verticalEdges;
    std::map<std::size_t, std::vector<EdgeRef>> horizontalEdges;
    mesh.cells.reserve(grid.activeCellCount());
    for (const PatchId patchId : grid.activePatches()) {
        AdaptivePatch& patch = grid.patch(patchId);
        const std::size_t scale = finestCellScale(grid, patch.level());
        const std::size_t originRow = patch.rowOriginFineUnits();
        const std::size_t originCol = patch.colOriginFineUnits();
        const double dx = patch.grid().dx();
        const double dy = patch.grid().dy();
        for (std::size_t row = 0; row < patch.grid().rows(); ++row) {
            for (std::size_t col = 0; col < patch.grid().cols(); ++col) {
                const std::size_t index = mesh.cells.size();
                const std::size_t rowBegin = originRow + row * scale;
                const std::size_t colBegin = originCol + col * scale;
                mesh.cells.push_back({{patchId, row, col}, &patch.grid().at(row, col),
                    patch.level(), dx, dy, dx * dy, rowBegin, rowBegin + scale,
                    colBegin, colBegin + scale});
                const double area = dx * dy;
                if (!std::isfinite(area) || !(area > 0.0))
                    throw std::invalid_argument("Adaptive cell has invalid physical area");
                mesh.areaSum += area;
                verticalEdges[colBegin].push_back(
                    {rowBegin, rowBegin + scale, index, false});
                verticalEdges[colBegin + scale].push_back(
                    {rowBegin, rowBegin + scale, index, true});
                horizontalEdges[rowBegin].push_back(
                    {colBegin, colBegin + scale, index, false});
                horizontalEdges[rowBegin + scale].push_back(
                    {colBegin, colBegin + scale, index, true});
            }
        }
    }

    const auto compareEdges = [](const EdgeRef& left, const EdgeRef& right) {
        if (left.begin != right.begin) return left.begin < right.begin;
        if (left.end != right.end) return left.end < right.end;
        return left.cell < right.cell;
    };
    const auto addAxisFaces = [&](std::map<std::size_t, std::vector<EdgeRef>>& edges,
                                  bool xDirection, std::size_t domainExtent,
                                  double finestLength) {
        for (auto& entry : edges) {
            const std::size_t coordinate = entry.first;
            auto& all = entry.second;
            std::vector<EdgeRef> left, right;
            for (const EdgeRef& edge : all)
                (edge.isLeft ? left : right).push_back(edge);
            std::sort(left.begin(), left.end(), compareEdges);
            std::sort(right.begin(), right.end(), compareEdges);
            if (coordinate != 0 && coordinate != domainExtent &&
                (left.empty() || right.empty()))
                throw std::logic_error("Adaptive mesh has an unmatched interior face");

            if (coordinate == 0 || coordinate == domainExtent) {
                const bool low = coordinate == 0;
                const auto& interiorEdges = low ? right : left;
                for (const EdgeRef& edge : interiorEdges) {
                    FaceSegment face;
                    face.left = low ? noCell : edge.cell;
                    face.right = low ? edge.cell : noCell;
                    face.length = static_cast<double>(edge.end - edge.begin) * finestLength;
                    face.xDirection = xDirection;
                    face.lowBoundary = low;
                    face.highBoundary = !low;
                    mesh.faces.push_back(face);
                }
                continue;
            }

            std::size_t leftIndex = 0, rightIndex = 0;
            std::size_t coveredLength = 0;
            while (leftIndex < left.size() && rightIndex < right.size()) {
                const std::size_t begin = std::max(left[leftIndex].begin,
                                                   right[rightIndex].begin);
                const std::size_t end = std::min(left[leftIndex].end,
                                                 right[rightIndex].end);
                if (begin < end) {
                    FaceSegment face;
                    face.left = left[leftIndex].cell;
                    face.right = right[rightIndex].cell;
                    face.length = static_cast<double>(end - begin) * finestLength;
                    face.xDirection = xDirection;
                    mesh.faces.push_back(face);
                    coveredLength += end - begin;
                }
                const std::size_t leftEnd = left[leftIndex].end;
                const std::size_t rightEnd = right[rightIndex].end;
                if (leftEnd <= rightEnd) ++leftIndex;
                if (rightEnd <= leftEnd) ++rightIndex;
            }
            std::size_t leftLength = 0, rightLength = 0;
            for (const EdgeRef& edge : left) leftLength += edge.end - edge.begin;
            for (const EdgeRef& edge : right) rightLength += edge.end - edge.begin;
            if (coveredLength != leftLength || coveredLength != rightLength)
                throw std::logic_error("Adaptive interior face coverage is inconsistent");
        }
    };
    addAxisFaces(verticalEdges, true, domainCols, finestDy);
    addAxisFaces(horizontalEdges, false, domainRows, finestDx);

    const double expectedArea = grid.dx0() * static_cast<double>(grid.rows()) *
                                grid.dy0() * static_cast<double>(grid.cols());
    const double areaTolerance = 256.0 * std::numeric_limits<double>::epsilon() *
                                 std::max(expectedArea, 1.0);
    if (std::abs(mesh.areaSum - expectedArea) > areaTolerance)
        throw std::logic_error("Active adaptive leaves do not cover the base domain exactly");
    return mesh;
}

void applyBoundaryAndComputeFluxes(Mesh& mesh, BoundaryCondition boundary,
                                   double gravity, double dryDepth) {
    for (FaceSegment& face : mesh.faces) {
        const Cell* leftCell = face.left == noCell ? nullptr : mesh.cells[face.left].state;
        const Cell* rightCell = face.right == noCell ? nullptr : mesh.cells[face.right].state;
        Cell leftGhost, rightGhost;
        if (leftCell == nullptr) {
            const Cell& interior = *rightCell;
            leftGhost = boundary == BoundaryCondition::Closed
                ? ghostForWall(interior, face.xDirection)
                : ghostForOutflow(interior, face.xDirection, true);
            leftCell = &leftGhost;
        }
        if (rightCell == nullptr) {
            const Cell& interior = *leftCell;
            rightGhost = boundary == BoundaryCondition::Closed
                ? ghostForWall(interior, face.xDirection)
                : ghostForOutflow(interior, face.xDirection, false);
            rightCell = &rightGhost;
        }
        face.flux = rusanovFace(*leftCell, *rightCell, face.left, face.right,
                                face.xDirection, gravity, dryDepth);
        if (face.flux.massRate > 0.0 && face.left != noCell)
            face.flux.donor = face.left;
        else if (face.flux.massRate < 0.0 && face.right != noCell)
            face.flux.donor = face.right;
        face.flux.boundaryOutflow =
            (face.lowBoundary && face.flux.massRate < 0.0) ||
            (face.highBoundary && face.flux.massRate > 0.0);
    }
}

std::map<PatchId, double> activityByPatch(
    const AdaptiveGrid& grid, const Mesh& mesh,
    const std::map<CellKey, double>& previousDepth,
    double gravity, double dryDepth) {
    std::map<PatchId, double> result;
    for (const CellRef& cell : mesh.cells) {
        const Cell& state = *cell.state;
        const double speed = state.h > dryDepth
            ? std::hypot(state.hu, state.hv) / state.h : 0.0;
        const double waveDepth = std::max(state.h, dryDepth);
        const double froude = speed / std::sqrt(gravity * waveDepth);
        const auto previous = previousDepth.find(keyFor(cell.id));
        const double oldDepth = previous == previousDepth.end() ? state.h : previous->second;
        const double temporal = std::abs(state.h - oldDepth) /
            std::max({state.h, oldDepth, dryDepth});
        result[cell.id.patchId] = std::max(result[cell.id.patchId],
                                           std::max(froude, temporal));
    }
    for (const FaceSegment& face : mesh.faces) {
        if (face.left == noCell || face.right == noCell) continue;
        const CellRef& left = mesh.cells[face.left];
        const CellRef& right = mesh.cells[face.right];
        const double denominator = std::max({left.state->h, right.state->h, dryDepth});
        const double jump = std::abs(left.state->h - right.state->h) / denominator;
        result[left.id.patchId] = std::max(result[left.id.patchId], jump);
        result[right.id.patchId] = std::max(result[right.id.patchId], jump);
    }
    for (const PatchId id : grid.activePatches())
        result.try_emplace(id, 0.0);
    return result;
}

void recordPreviousDepth(const AdaptiveGrid& grid,
                         std::map<CellKey, double>& previousDepth) {
    previousDepth.clear();
    grid.forEachActiveCell([&](AdaptiveCellId id, const Cell& cell) {
        previousDepth.emplace(keyFor(id), cell.h);
    });
}

ConservedIntegrals activeIntegrals(const AdaptiveGrid& grid) {
    ConservedIntegrals result;
    grid.forEachActiveCell([&](AdaptiveCellId id, const Cell& cell) {
        const AdaptivePatch& patch = grid.patch(id.patchId);
        const long double area = static_cast<long double>(patch.grid().dx()) *
                                 patch.grid().dy();
        result.water += static_cast<long double>(cell.h) * area;
        result.hu += static_cast<long double>(cell.hu) * area;
        result.hv += static_cast<long double>(cell.hv) * area;
    });
    return result;
}

void recordRegridEvent(AdaptiveDiagnostics& diagnostics,
                       AdaptiveRegridOperation operation, PatchId patchId,
                       double time, const ConservedIntegrals& before,
                       const AdaptiveGrid& grid) {
    const ConservedIntegrals after = activeIntegrals(grid);
    const double beforeWater = static_cast<double>(before.water);
    const double afterWater = static_cast<double>(after.water);
    const double beforeHu = static_cast<double>(before.hu);
    const double afterHu = static_cast<double>(after.hu);
    const double beforeHv = static_cast<double>(before.hv);
    const double afterHv = static_cast<double>(after.hv);
    const double waterDelta = afterWater - beforeWater;
    diagnostics.maximumRegridVolumeDelta = std::max(
        diagnostics.maximumRegridVolumeDelta, std::abs(waterDelta));
    diagnostics.regridEvents.push_back({operation, patchId, time, beforeWater,
        afterWater, waterDelta, beforeHu, afterHu, afterHu - beforeHu,
        beforeHv, afterHv, afterHv - beforeHv});
}

void refineBalanced(AdaptiveGrid& grid, PatchId patchId,
                    std::size_t targetLevel, std::size_t& refinementCount,
                    AdaptiveDiagnostics& diagnostics, double time) {
    AdaptivePatch& candidate = grid.patch(patchId);
    if (!candidate.isActiveLeaf() || candidate.level() >= targetLevel) return;
    const std::size_t currentLevel = candidate.level();
    std::vector<PatchId> lowerNeighbors;
    for (const PatchNeighbor& neighbor : candidate.neighbors()) {
        if (neighbor.patchId && grid.patch(*neighbor.patchId).level() < currentLevel)
            lowerNeighbors.push_back(*neighbor.patchId);
    }
    std::sort(lowerNeighbors.begin(), lowerNeighbors.end());
    lowerNeighbors.erase(std::unique(lowerNeighbors.begin(), lowerNeighbors.end()),
                         lowerNeighbors.end());
    for (const PatchId neighborId : lowerNeighbors)
        refineBalanced(grid, neighborId, currentLevel, refinementCount,
                       diagnostics, time);
    if (!grid.patch(patchId).isActiveLeaf()) return;
    const ConservedIntegrals before = activeIntegrals(grid);
    grid.refine(patchId);
    recordRegridEvent(diagnostics, AdaptiveRegridOperation::Refine,
                      patchId, time, before, grid);
    ++refinementCount;
    refineBalanced(grid, patchId, targetLevel, refinementCount,
                   diagnostics, time);
}

void refineCandidateAndBuffer(AdaptiveGrid& grid, PatchId candidateId,
                              std::size_t& refinementCount,
                              AdaptiveDiagnostics& diagnostics, double time) {
    if (!grid.patch(candidateId).isActiveLeaf() ||
        grid.patch(candidateId).level() >= grid.config().maxRefinementLevel)
        return;
    const std::size_t level = grid.patch(candidateId).level();
    std::vector<PatchId> buffered{candidateId};
    for (const PatchNeighbor& neighbor : grid.patch(candidateId).neighbors()) {
        if (neighbor.patchId && grid.patch(*neighbor.patchId).level() == level)
            buffered.push_back(*neighbor.patchId);
    }
    std::sort(buffered.begin(), buffered.end());
    buffered.erase(std::unique(buffered.begin(), buffered.end()), buffered.end());
    for (const PatchId patchId : buffered)
        refineBalanced(grid, patchId, level + 1, refinementCount,
                       diagnostics, time);
}

std::size_t activeLeafCellCount(const AdaptiveGrid& grid, std::size_t level) {
    std::size_t count = 0;
    for (const PatchId patchId : grid.activePatches()) {
        const AdaptivePatch& patch = grid.patch(patchId);
        if (patch.level() == level) count += patch.grid().size();
    }
    return count;
}

} // namespace

AdaptiveSolver::AdaptiveSolver(AdaptiveSolverOptions options) : options_(options) {}

AdaptiveDiagnostics AdaptiveSolver::run(AdaptiveGrid& grid, const Rainfall& rainfall,
                                        const SolverConfig& config) const {
    validateConfig(config, options_, grid);
    const auto runtimeStart = std::chrono::steady_clock::now();
    AdaptiveDiagnostics diagnostics;
    CompensatedSum initialVolume, rainfallVolume, infiltrationVolume, outflowVolume;
    std::map<CellKey, double> previousDepth;
    grid.forEachActiveCell([&](AdaptiveCellId id, const Cell& cell) {
        if (cell.h < 0.0 || !std::isfinite(cell.bed) || !std::isfinite(cell.h) ||
            !std::isfinite(cell.hu) || !std::isfinite(cell.hv))
            throw std::invalid_argument("Initial adaptive state contains invalid values");
        const AdaptivePatch& patch = grid.patch(id.patchId);
        initialVolume.add(cell.h * patch.grid().dx() * patch.grid().dy());
    });
    diagnostics.initialWaterVolume = initialVolume.value;
    recordPreviousDepth(grid, previousDepth);

    double time = 0.0;
    double lastDt = 0.0;
    std::map<PatchId, std::size_t> coarseningPersistence;
    while (time < config.endTime) {
        const auto computeStart = std::chrono::steady_clock::now();
        Mesh mesh = buildMesh(grid);
        double maximumRate = 0.0;
        for (const CellRef& cell : mesh.cells) {
            const Cell& state = *cell.state;
            if (state.h <= config.dryDepth) continue;
            const double u = state.hu / state.h;
            const double v = state.hv / state.h;
            const double wave = std::sqrt(config.gravity * state.h);
            maximumRate = std::max(maximumRate,
                (std::abs(u) + wave) / cell.dx + (std::abs(v) + wave) / cell.dy);
        }
        double dt = maximumRate > 0.0 ? config.cfl / maximumRate : config.maxTimestep;
        dt = std::min({dt, config.maxTimestep, config.endTime - time});
        const double nextRainChange = rainfall.nextChangeAfter(time);
        if (nextRainChange > time) dt = std::min(dt, nextRainChange - time);
        if (!(dt > 0.0) || !std::isfinite(dt))
            throw std::runtime_error("Adaptive CFL timestep became invalid");

        applyBoundaryAndComputeFluxes(mesh, config.boundary, config.gravity, config.dryDepth);
        std::vector<double> outgoing(mesh.cells.size(), 0.0);
        for (const FaceSegment& face : mesh.faces) {
            if (face.flux.donor != noCell)
                outgoing[face.flux.donor] += std::abs(face.flux.massRate) * face.length;
        }
        std::vector<double> limiter(mesh.cells.size(), 1.0);
        for (std::size_t index = 0; index < mesh.cells.size(); ++index) {
            const double available = mesh.cells[index].state->h * mesh.cells[index].area;
            if (outgoing[index] * dt > available && outgoing[index] > 0.0)
                limiter[index] = available / (outgoing[index] * dt);
        }

        std::vector<State> delta(mesh.cells.size(), State{0.0, 0.0, 0.0});
        for (const FaceSegment& face : mesh.faces) {
            const double scale = face.flux.donor == noCell ? 1.0 :
                                 limiter[face.flux.donor];
            if (face.left != noCell) {
                for (std::size_t component = 0; component < 3; ++component)
                    delta[face.left][component] -= scale *
                        face.flux.leftFlux[component] * face.length /
                        mesh.cells[face.left].area;
            }
            if (face.right != noCell) {
                for (std::size_t component = 0; component < 3; ++component)
                    delta[face.right][component] += scale *
                        face.flux.rightFlux[component] * face.length /
                        mesh.cells[face.right].area;
            }
            if (face.left != noCell && face.right != noCell) {
                if (mesh.cells[face.left].level != mesh.cells[face.right].level) {
                    const double leftMassContribution =
                        -scale * face.flux.leftFlux[0] * face.length;
                    const double rightMassContribution =
                        scale * face.flux.rightFlux[0] * face.length;
                    diagnostics.maximumInterfaceMassFluxResidual = std::max(
                        diagnostics.maximumInterfaceMassFluxResidual,
                        std::abs(leftMassContribution + rightMassContribution));
                    ++diagnostics.coarseFineInterfaceSegments;
                    diagnostics.coarseFineIntegratedMassFlux +=
                        std::abs(scale * face.flux.massRate * face.length) * dt;
                }
            }
            if (face.flux.boundaryOutflow) {
                const double outflowScale = face.flux.donor == noCell ? 1.0 :
                                            limiter[face.flux.donor];
                outflowVolume.add(outflowScale * std::abs(face.flux.massRate) *
                                  face.length * dt);
            }
        }

        const double rainRate = rainfall.metersPerSecond(time);
        rainfallVolume.add(rainRate * mesh.areaSum * dt);
        for (std::size_t index = 0; index < mesh.cells.size(); ++index) {
            Cell& cell = *mesh.cells[index].state;
            cell.h += dt * (delta[index][0] + rainRate);
            cell.hu += dt * delta[index][1];
            cell.hv += dt * delta[index][2];
            if (cell.h < 0.0 && cell.h > -1e-12) cell.h = 0.0;
            if (cell.h < 0.0 || !std::isfinite(cell.h) ||
                !std::isfinite(cell.hu) || !std::isfinite(cell.hv))
                throw std::runtime_error(
                    "Adaptive timestep produced an invalid state; simulation stopped");
            const double infiltrated = std::min(
                cell.h, config.infiltrationMetersPerSecond * dt);
            cell.h -= infiltrated;
            infiltrationVolume.add(infiltrated * mesh.cells[index].area);
            if (cell.h <= config.dryDepth) {
                cell.hu = 0.0;
                cell.hv = 0.0;
            } else if (config.manningN > 0.0) {
                const double u = cell.hu / cell.h;
                const double v = cell.hv / cell.h;
                const double speed = std::hypot(u, v);
                const double coefficient = config.gravity * config.manningN *
                    config.manningN * speed / std::pow(cell.h, 4.0 / 3.0);
                const double frictionScale = 1.0 / (1.0 + dt * coefficient);
                cell.hu *= frictionScale;
                cell.hv *= frictionScale;
            }
        }
        diagnostics.computeSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - computeStart).count();
        const auto activityStart = std::chrono::steady_clock::now();
        const auto activity = activityByPatch(
            grid, mesh, previousDepth, config.gravity, config.dryDepth);
        for (const auto& entry : activity)
            diagnostics.maximumActivity = std::max(diagnostics.maximumActivity,
                                                    entry.second);
        diagnostics.computeSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - activityStart).count();
        time += dt;
        lastDt = dt;
        ++diagnostics.steps;

        if (diagnostics.steps % options_.regridIntervalSteps == 0 &&
            time < config.endTime && grid.config().maxRefinementLevel > 0) {
            const auto regridStart = std::chrono::steady_clock::now();
            std::vector<PatchId> candidates;
            for (const PatchId patchId : grid.activePatches()) {
                const AdaptivePatch& patch = grid.patch(patchId);
                const auto found = activity.find(patchId);
                if (patch.level() < grid.config().maxRefinementLevel &&
                    found != activity.end() &&
                    found->second > options_.refineThreshold)
                    candidates.push_back(patchId);
            }
            std::sort(candidates.begin(), candidates.end(),
                [&grid](PatchId left, PatchId right) {
                    if (grid.patch(left).level() != grid.patch(right).level())
                        return grid.patch(left).level() < grid.patch(right).level();
                    return left < right;
                });
            for (const PatchId patchId : candidates)
                refineCandidateAndBuffer(grid, patchId, diagnostics.refinedPatches,
                                         diagnostics, time);

            std::vector<PatchId> possibleCoarsen;
            for (PatchId parentId = 0; parentId < grid.patchCount(); ++parentId) {
                const AdaptivePatch& parent = grid.patch(parentId);
                if (!parent.isCovered() || parent.childIds().empty()) continue;
                bool eligible = true;
                for (const PatchId childId : parent.childIds()) {
                    const AdaptivePatch& child = grid.patch(childId);
                    const auto found = activity.find(childId);
                    if (!child.isActiveLeaf() || found == activity.end() ||
                        found->second >= options_.coarsenThreshold) {
                        eligible = false;
                        break;
                    }
                }
                if (eligible) {
                    ++coarseningPersistence[parentId];
                    if (coarseningPersistence[parentId] >= options_.coarsenPersistence)
                        possibleCoarsen.push_back(parentId);
                } else {
                    coarseningPersistence[parentId] = 0;
                }
            }
            std::sort(possibleCoarsen.begin(), possibleCoarsen.end(),
                [&grid](PatchId left, PatchId right) {
                    if (grid.patch(left).level() != grid.patch(right).level())
                        return grid.patch(left).level() > grid.patch(right).level();
                    return left < right;
                });
            for (const PatchId parentId : possibleCoarsen) {
                const AdaptivePatch& parent = grid.patch(parentId);
                if (!parent.isCovered()) continue;
                bool violatesBalance = false;
                for (const PatchId childId : parent.childIds()) {
                    for (const PatchNeighbor& neighbor : grid.patch(childId).neighbors()) {
                        if (neighbor.patchId &&
                            grid.patch(*neighbor.patchId).level() > parent.level() + 1)
                            violatesBalance = true;
                    }
                }
                if (violatesBalance) continue;
                const ConservedIntegrals before = activeIntegrals(grid);
                grid.coarsen(parentId);
                recordRegridEvent(diagnostics, AdaptiveRegridOperation::Coarsen,
                                  parentId, time, before, grid);
                ++diagnostics.coarsenedPatches;
                coarseningPersistence[parentId] = 0;
            }
            recordPreviousDepth(grid, previousDepth);
            diagnostics.regriddingSeconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - regridStart).count();
        }
    }

    diagnostics.time = time;
    diagnostics.currentTimestep = lastDt;
    diagnostics.activeLeafCells = grid.activeCellCount();
    diagnostics.level0Cells = activeLeafCellCount(grid, 0);
    diagnostics.level1Cells = activeLeafCellCount(grid, 1);
    diagnostics.level2Cells = activeLeafCellCount(grid, 2);
    const std::size_t fineCellCount = diagnostics.level1Cells + diagnostics.level2Cells;
    diagnostics.fineCellPercentage = diagnostics.activeLeafCells == 0 ? 0.0 :
        100.0 * static_cast<double>(fineCellCount) /
        static_cast<double>(diagnostics.activeLeafCells);
    diagnostics.rainfallVolume = rainfallVolume.value;
    diagnostics.infiltrationVolume = infiltrationVolume.value;
    diagnostics.outflowVolume = outflowVolume.value;
    CompensatedSum finalVolume;
    diagnostics.minimumDepth = std::numeric_limits<double>::infinity();
    for (const PatchId patchId : grid.activePatches()) {
        const AdaptivePatch& patch = grid.patch(patchId);
        const double area = patch.grid().dx() * patch.grid().dy();
        for (const Cell& cell : patch.grid().cells()) {
            finalVolume.add(cell.h * area);
            diagnostics.minimumDepth = std::min(diagnostics.minimumDepth, cell.h);
            diagnostics.maximumDepth = std::max(diagnostics.maximumDepth, cell.h);
            if (cell.h > config.dryDepth) {
                ++diagnostics.wetCells;
                diagnostics.maximumVelocity = std::max(
                    diagnostics.maximumVelocity,
                    std::hypot(cell.hu, cell.hv) / cell.h);
            }
        }
    }
    diagnostics.finalWaterVolume = finalVolume.value;
    diagnostics.massBalanceResidual = diagnostics.initialWaterVolume +
        diagnostics.rainfallVolume - diagnostics.outflowVolume -
        diagnostics.infiltrationVolume - diagnostics.finalWaterVolume;
    diagnostics.runtimeSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - runtimeStart).count();
    return diagnostics;
}

} // namespace flood
