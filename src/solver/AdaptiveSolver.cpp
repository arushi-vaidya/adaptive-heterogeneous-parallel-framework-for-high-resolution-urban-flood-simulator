#include "solver/AdaptiveSolver.hpp"

#include "solver/detail/FiniteVolumeKernels.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#ifdef FLOOD_HAS_MPI
#include <mpi.h>
#endif

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

std::vector<std::vector<std::size_t>> facesByCell(const Mesh& mesh) {
    std::vector<std::vector<std::size_t>> result(mesh.cells.size());
    for (std::size_t faceIndex = 0; faceIndex < mesh.faces.size(); ++faceIndex) {
        const FaceSegment& face = mesh.faces[faceIndex];
        if (face.left != noCell) result[face.left].push_back(faceIndex);
        if (face.right != noCell) result[face.right].push_back(faceIndex);
    }
    return result;
}

#ifdef FLOOD_HAS_MPI
void checkAdaptiveMpi(int status, const char* operation) {
    if (status == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, message, &length);
    throw std::runtime_error(std::string(operation) + " failed: " +
        std::string(message, static_cast<std::size_t>(length)));
}

int adaptiveMpiCount(std::size_t count) {
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Adaptive MPI message exceeds the supported count range");
    return static_cast<int>(count);
}

class AdaptiveMpiTopology {
public:
    AdaptiveMpiTopology(const AdaptiveGrid& grid, MPI_Comm communicator)
        : comm(communicator) {
        checkAdaptiveMpi(MPI_Comm_size(comm, &size), "MPI_Comm_size adaptive");
        checkAdaptiveMpi(MPI_Comm_rank(comm, &rank), "MPI_Comm_rank adaptive");
        checkAdaptiveMpi(MPI_Topo_test(comm, &topology), "MPI_Topo_test adaptive");
        if (topology != MPI_CART)
            throw std::invalid_argument("Adaptive MPI solver requires a Cartesian communicator");
        checkAdaptiveMpi(MPI_Cart_get(comm, 2, dims, periods, coords),
                         "MPI_Cart_get adaptive");
        refresh(grid);
    }

    void refresh(const AdaptiveGrid& grid) {
        patchOwners.resize(grid.patchCount(), 0);
        std::size_t finestScale = 1;
        for (std::size_t level = 0; level < grid.config().maxRefinementLevel; ++level)
            finestScale *= grid.config().refinementRatio;
        for (PatchId id = 0; id < grid.patchCount(); ++id) {
            const AdaptivePatch& patch = grid.patch(id);
            if (patch.parentId()) {
                // Static ownership: descendants remain on their level-0 patch's rank.
                patchOwners[static_cast<std::size_t>(id)] =
                    patchOwners[static_cast<std::size_t>(*patch.parentId())];
                continue;
            }
            const std::size_t row = patch.rowOriginFineUnits() / finestScale +
                                    patch.grid().rows() / 2;
            const std::size_t col = patch.colOriginFineUnits() / finestScale +
                                    patch.grid().cols() / 2;
            const int patchCoords[2] = {
                coordinateFor(row, grid.rows(), dims[0]),
                coordinateFor(col, grid.cols(), dims[1])};
            int owner = 0;
            checkAdaptiveMpi(MPI_Cart_rank(comm, patchCoords, &owner),
                             "MPI_Cart_rank adaptive patch owner");
            patchOwners[static_cast<std::size_t>(id)] = owner;
        }
    }

    std::vector<int> cellOwners(const Mesh& mesh) const {
        std::vector<int> result;
        result.reserve(mesh.cells.size());
        for (const CellRef& cell : mesh.cells)
            result.push_back(patchOwners[static_cast<std::size_t>(cell.id.patchId)]);
        return result;
    }

    int ownerOfPatch(PatchId id) const {
        return patchOwners.at(static_cast<std::size_t>(id));
    }

    MPI_Comm comm;
    int rank = 0;
    int size = 1;
    int topology = MPI_UNDEFINED;
    int dims[2]{};
    int periods[2]{};
    int coords[2]{};

private:
    static int coordinateFor(std::size_t cell, std::size_t extent, int parts) {
        for (int coordinate = 0; coordinate < parts; ++coordinate) {
            const std::size_t begin = extent * static_cast<std::size_t>(coordinate) /
                                      static_cast<std::size_t>(parts);
            const std::size_t end = extent * static_cast<std::size_t>(coordinate + 1) /
                                    static_cast<std::size_t>(parts);
            if (cell >= begin && cell < end) return coordinate;
        }
        return parts - 1;
    }

    std::vector<int> patchOwners;
};

void completePackedExchange(const std::vector<double>& local,
                            std::vector<double>& global,
                            const AdaptiveMpiTopology& topology,
                            const char* operation) {
    MPI_Request request = MPI_REQUEST_NULL;
    checkAdaptiveMpi(MPI_Iallreduce(local.data(), global.data(),
        adaptiveMpiCount(global.size()), MPI_DOUBLE, MPI_SUM, topology.comm,
        &request), operation);
    checkAdaptiveMpi(MPI_Wait(&request, MPI_STATUS_IGNORE),
                     "MPI_Wait adaptive packed exchange");
}

std::vector<std::size_t> remoteFaces(const Mesh& mesh,
                                     const std::vector<int>& cellOwners) {
    std::vector<std::size_t> result;
    for (std::size_t index = 0; index < mesh.faces.size(); ++index) {
        const FaceSegment& face = mesh.faces[index];
        if (face.left != noCell && face.right != noCell &&
            cellOwners[face.left] != cellOwners[face.right])
            result.push_back(index);
    }
    return result;
}

int faceOwner(const FaceSegment& face, const std::vector<int>& cellOwners) {
    if (face.left != noCell) return cellOwners[face.left];
    return cellOwners[face.right];
}

void exchangeRemoteStates(Mesh& mesh, const std::vector<int>& cellOwners,
                          const std::vector<std::size_t>& remoteFaceIndices,
                          const AdaptiveMpiTopology& topology) {
    std::vector<std::size_t> cells;
    for (const std::size_t index : remoteFaceIndices) {
        const FaceSegment& face = mesh.faces[index];
        cells.push_back(face.left);
        cells.push_back(face.right);
    }
    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    std::vector<double> local(cells.size() * 4, 0.0), global(local.size(), 0.0);
    for (std::size_t index = 0; index < cells.size(); ++index) {
        if (cellOwners[cells[index]] != topology.rank) continue;
        const Cell& cell = *mesh.cells[cells[index]].state;
        local[index * 4] = cell.bed;
        local[index * 4 + 1] = cell.h;
        local[index * 4 + 2] = cell.hu;
        local[index * 4 + 3] = cell.hv;
    }
    completePackedExchange(local, global, topology,
                           "MPI_Iallreduce adaptive interface states");
    for (std::size_t index = 0; index < cells.size(); ++index) {
        Cell& cell = *mesh.cells[cells[index]].state;
        cell.bed = global[index * 4];
        cell.h = global[index * 4 + 1];
        cell.hu = global[index * 4 + 2];
        cell.hv = global[index * 4 + 3];
    }
}

void synchronizeOwnedStates(AdaptiveGrid& grid,
                            const AdaptiveMpiTopology& topology) {
    std::vector<double> local(grid.activeCellCount() * 4, 0.0);
    std::vector<double> global(local.size(), 0.0);
    std::size_t index = 0;
    grid.forEachActiveCell([&](AdaptiveCellId id, const Cell& cell) {
        if (topology.ownerOfPatch(id.patchId) == topology.rank) {
            local[index * 4] = cell.bed;
            local[index * 4 + 1] = cell.h;
            local[index * 4 + 2] = cell.hu;
            local[index * 4 + 3] = cell.hv;
        }
        ++index;
    });
    completePackedExchange(local, global, topology,
                           "MPI_Iallreduce adaptive final state");
    index = 0;
    grid.forEachActiveCell([&](AdaptiveCellId id, Cell& cell) {
        (void)id;
        cell.bed = global[index * 4];
        cell.h = global[index * 4 + 1];
        cell.hu = global[index * 4 + 2];
        cell.hv = global[index * 4 + 3];
        ++index;
    });
}

void exchangeRemoteFluxes(Mesh& mesh,
                          const std::vector<std::size_t>& remoteFaceIndices,
                          const std::vector<int>& cellOwners,
                          const AdaptiveMpiTopology& topology) {
    constexpr std::size_t fluxWidth = 6;
    std::vector<double> local(remoteFaceIndices.size() * fluxWidth, 0.0);
    std::vector<double> global(local.size(), 0.0);
    for (std::size_t entry = 0; entry < remoteFaceIndices.size(); ++entry) {
        FaceSegment& face = mesh.faces[remoteFaceIndices[entry]];
        if (faceOwner(face, cellOwners) != topology.rank) continue;
        for (std::size_t component = 0; component < 3; ++component) {
            local[entry * fluxWidth + component] = face.flux.leftFlux[component];
            local[entry * fluxWidth + 3 + component] = face.flux.rightFlux[component];
        }
    }
    completePackedExchange(local, global, topology,
                           "MPI_Iallreduce adaptive interface fluxes");
    for (std::size_t entry = 0; entry < remoteFaceIndices.size(); ++entry) {
        FaceSegment& face = mesh.faces[remoteFaceIndices[entry]];
        for (std::size_t component = 0; component < 3; ++component) {
            face.flux.leftFlux[component] = global[entry * fluxWidth + component];
            face.flux.rightFlux[component] = global[entry * fluxWidth + 3 + component];
        }
        face.flux.massRate = face.flux.leftFlux[0];
        if (face.flux.massRate > 0.0) face.flux.donor = face.left;
        else if (face.flux.massRate < 0.0) face.flux.donor = face.right;
    }
}

void exchangeRemoteLimiters(const Mesh& mesh,
                            const std::vector<std::size_t>& remoteFaceIndices,
                            const std::vector<int>& cellOwners,
                            const std::vector<double>& limiters,
                            const AdaptiveMpiTopology& topology,
                            std::vector<double>& faceScales) {
    std::vector<double> local(remoteFaceIndices.size(), 0.0);
    std::vector<double> global(local.size(), 0.0);
    for (std::size_t entry = 0; entry < remoteFaceIndices.size(); ++entry) {
        const FaceSegment& face = mesh.faces[remoteFaceIndices[entry]];
        const std::size_t donor = face.flux.donor;
        if (donor == noCell && faceOwner(face, cellOwners) == topology.rank)
            local[entry] = 1.0;
        else if (donor != noCell && cellOwners[donor] == topology.rank)
            local[entry] = limiters[donor];
    }
    completePackedExchange(local, global, topology,
                           "MPI_Iallreduce adaptive donor limiters");
    for (std::size_t entry = 0; entry < remoteFaceIndices.size(); ++entry)
        faceScales[remoteFaceIndices[entry]] = global[entry];
}

ConservedIntegrals activeIntegralsOwned(const AdaptiveGrid& grid,
                                        const AdaptiveMpiTopology& topology) {
    ConservedIntegrals local;
    for (const PatchId patchId : grid.activePatches()) {
        if (topology.ownerOfPatch(patchId) != topology.rank) continue;
        const AdaptivePatch& patch = grid.patch(patchId);
        const long double area = static_cast<long double>(patch.grid().dx()) *
                                 patch.grid().dy();
        for (const Cell& cell : patch.grid().cells()) {
            local.water += static_cast<long double>(cell.h) * area;
            local.hu += static_cast<long double>(cell.hu) * area;
            local.hv += static_cast<long double>(cell.hv) * area;
        }
    }
    const double localValues[3] = {static_cast<double>(local.water),
                                   static_cast<double>(local.hu),
                                   static_cast<double>(local.hv)};
    double globalValues[3]{};
    checkAdaptiveMpi(MPI_Allreduce(localValues, globalValues, 3, MPI_DOUBLE,
                                   MPI_SUM, topology.comm),
                     "MPI_Allreduce adaptive conserved integrals");
    return {globalValues[0], globalValues[1], globalValues[2]};
}
#endif

void applyBoundaryAndComputeFluxes(Mesh& mesh, BoundaryCondition boundary,
                                   double gravity, double dryDepth,
                                   bool useOpenMP, int openMPThreads
#ifdef FLOOD_HAS_MPI
                                   , const std::vector<int>* cellOwners = nullptr,
                                   int mpiRank = -1
#endif
                                   ) {
#ifndef FLOOD_HAS_OPENMP
    (void)useOpenMP;
    (void)openMPThreads;
#endif
#ifdef FLOOD_HAS_OPENMP
    #pragma omp parallel for schedule(static) num_threads(openMPThreads) if(useOpenMP)
#endif
    for (std::ptrdiff_t faceIndex = 0;
         faceIndex < static_cast<std::ptrdiff_t>(mesh.faces.size()); ++faceIndex) {
        FaceSegment& face = mesh.faces[static_cast<std::size_t>(faceIndex)];
#ifdef FLOOD_HAS_MPI
        if (cellOwners != nullptr && faceOwner(face, *cellOwners) != mpiRank) continue;
#endif
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
                       const AdaptiveGrid& grid,
                       const std::function<ConservedIntegrals(const AdaptiveGrid&)>&
                           integrate = activeIntegrals) {
    const ConservedIntegrals after = integrate(grid);
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
                    AdaptiveDiagnostics& diagnostics, double time,
                    const std::function<ConservedIntegrals(const AdaptiveGrid&)>&
                        integrate = activeIntegrals) {
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
                       diagnostics, time, integrate);
    if (!grid.patch(patchId).isActiveLeaf()) return;
    const ConservedIntegrals before = integrate(grid);
    grid.refine(patchId);
    recordRegridEvent(diagnostics, AdaptiveRegridOperation::Refine,
                      patchId, time, before, grid, integrate);
    ++refinementCount;
    refineBalanced(grid, patchId, targetLevel, refinementCount,
                   diagnostics, time, integrate);
}

void refineCandidateAndBuffer(AdaptiveGrid& grid, PatchId candidateId,
                              std::size_t& refinementCount,
                              AdaptiveDiagnostics& diagnostics, double time,
                              const std::function<ConservedIntegrals(const AdaptiveGrid&)>&
                                  integrate = activeIntegrals) {
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
                       diagnostics, time, integrate);
}

#ifdef FLOOD_HAS_MPI
std::map<PatchId, double> activityByPatchOwned(
    const AdaptiveGrid& grid, const Mesh& mesh,
    const std::map<CellKey, double>& previousDepth,
    double gravity, double dryDepth,
    const std::vector<int>& cellOwners,
    const AdaptiveMpiTopology& topology) {
    std::vector<double> local(grid.patchCount(), 0.0);
    for (std::size_t index = 0; index < mesh.cells.size(); ++index) {
        if (cellOwners[index] != topology.rank) continue;
        const CellRef& cell = mesh.cells[index];
        const Cell& state = *cell.state;
        const double speed = state.h > dryDepth
            ? std::hypot(state.hu, state.hv) / state.h : 0.0;
        const double froude = speed /
            std::sqrt(gravity * std::max(state.h, dryDepth));
        const auto old = previousDepth.find(keyFor(cell.id));
        const double oldDepth = old == previousDepth.end() ? state.h : old->second;
        const double temporal = std::abs(state.h - oldDepth) /
            std::max({state.h, oldDepth, dryDepth});
        local[static_cast<std::size_t>(cell.id.patchId)] = std::max(
            local[static_cast<std::size_t>(cell.id.patchId)],
            std::max(froude, temporal));
    }
    for (const FaceSegment& face : mesh.faces) {
        if (face.left == noCell || face.right == noCell ||
            faceOwner(face, cellOwners) != topology.rank)
            continue;
        const CellRef& left = mesh.cells[face.left];
        const CellRef& right = mesh.cells[face.right];
        const double denominator = std::max({left.state->h, right.state->h, dryDepth});
        const double jump = std::abs(left.state->h - right.state->h) / denominator;
        local[static_cast<std::size_t>(left.id.patchId)] = std::max(
            local[static_cast<std::size_t>(left.id.patchId)], jump);
        local[static_cast<std::size_t>(right.id.patchId)] = std::max(
            local[static_cast<std::size_t>(right.id.patchId)], jump);
    }
    std::vector<double> global(local.size(), 0.0);
    checkAdaptiveMpi(MPI_Allreduce(local.data(), global.data(),
        adaptiveMpiCount(local.size()), MPI_DOUBLE, MPI_MAX, topology.comm),
        "MPI_Allreduce adaptive patch activity");
    std::map<PatchId, double> result;
    for (PatchId id = 0; id < grid.patchCount(); ++id)
        result.emplace(id, global[static_cast<std::size_t>(id)]);
    return result;
}

void reduceAdaptiveSum(double& value, const AdaptiveMpiTopology& topology,
                       const char* operation) {
    double global = 0.0;
    checkAdaptiveMpi(MPI_Allreduce(&value, &global, 1, MPI_DOUBLE, MPI_SUM,
                                   topology.comm), operation);
    value = global;
}

void reduceAdaptiveMax(double& value, const AdaptiveMpiTopology& topology,
                       const char* operation) {
    double global = 0.0;
    checkAdaptiveMpi(MPI_Allreduce(&value, &global, 1, MPI_DOUBLE, MPI_MAX,
                                   topology.comm), operation);
    value = global;
}

void reduceAdaptiveMin(double& value, const AdaptiveMpiTopology& topology,
                       const char* operation) {
    double global = 0.0;
    checkAdaptiveMpi(MPI_Allreduce(&value, &global, 1, MPI_DOUBLE, MPI_MIN,
                                   topology.comm), operation);
    value = global;
}

void reduceAdaptiveCount(std::size_t& value, const AdaptiveMpiTopology& topology,
                         const char* operation) {
    const unsigned long long local = static_cast<unsigned long long>(value);
    unsigned long long global = 0;
    checkAdaptiveMpi(MPI_Allreduce(&local, &global, 1, MPI_UNSIGNED_LONG_LONG,
                                   MPI_SUM, topology.comm), operation);
    value = static_cast<std::size_t>(global);
}
#endif

} // namespace

AdaptiveSolver::AdaptiveSolver(AdaptiveSolverOptions options) : options_(options) {}

AdaptiveDiagnostics AdaptiveSolver::run(AdaptiveGrid& grid, const Rainfall& rainfall,
                                        const SolverConfig& config) const {
    return runImpl(grid, rainfall, config, false, 1
#ifdef FLOOD_HAS_MPI
                   , MPI_COMM_NULL
#endif
                   );
}

AdaptiveDiagnostics AdaptiveSolver::runImpl(AdaptiveGrid& grid,
                                            const Rainfall& rainfall,
                                            const SolverConfig& config,
                                            bool useOpenMP,
                                            int openMPThreads
#ifdef FLOOD_HAS_MPI
                                            , MPI_Comm mpiComm
#endif
                                            ) const {
#ifndef FLOOD_HAS_OPENMP
    (void)openMPThreads;
    if (useOpenMP)
        throw std::runtime_error(
            "Adaptive OpenMP backend is unavailable; configure with FLOOD_ENABLE_OPENMP=ON");
#endif
    validateConfig(config, options_, grid);
#ifdef FLOOD_HAS_MPI
    std::unique_ptr<AdaptiveMpiTopology> mpiTopology;
    if (mpiComm != MPI_COMM_NULL)
        mpiTopology = std::make_unique<AdaptiveMpiTopology>(grid, mpiComm);
    const bool useMpi = static_cast<bool>(mpiTopology);
    const std::function<ConservedIntegrals(const AdaptiveGrid&)> integrate =
        [&mpiTopology](const AdaptiveGrid& value) {
            if (mpiTopology) mpiTopology->refresh(value);
            return mpiTopology ? activeIntegralsOwned(value, *mpiTopology)
                               : activeIntegrals(value);
        };
#else
    const std::function<ConservedIntegrals(const AdaptiveGrid&)> integrate =
        [](const AdaptiveGrid& value) { return activeIntegrals(value); };
#endif
    const auto runtimeStart = std::chrono::steady_clock::now();
    AdaptiveDiagnostics diagnostics;
    CompensatedSum initialVolume, rainfallVolume, infiltrationVolume, outflowVolume;
    std::map<CellKey, double> previousDepth;
    grid.forEachActiveCell([&](AdaptiveCellId id, const Cell& cell) {
        if (cell.h < 0.0 || !std::isfinite(cell.bed) || !std::isfinite(cell.h) ||
            !std::isfinite(cell.hu) || !std::isfinite(cell.hv))
            throw std::invalid_argument("Initial adaptive state contains invalid values");
        const AdaptivePatch& patch = grid.patch(id.patchId);
#ifdef FLOOD_HAS_MPI
        if (useMpi && mpiTopology->ownerOfPatch(id.patchId) != mpiTopology->rank) return;
#endif
        initialVolume.add(cell.h * patch.grid().dx() * patch.grid().dy());
    });
    diagnostics.initialWaterVolume = initialVolume.value;
#ifdef FLOOD_HAS_MPI
    if (useMpi)
        reduceAdaptiveSum(diagnostics.initialWaterVolume, *mpiTopology,
                          "MPI_Allreduce adaptive initial volume");
#endif
    recordPreviousDepth(grid, previousDepth);

    double time = 0.0;
    double lastDt = 0.0;
    std::map<PatchId, std::size_t> coarseningPersistence;
    while (time < config.endTime) {
        const auto computeStart = std::chrono::steady_clock::now();
        Mesh mesh = buildMesh(grid);
        const auto cellFaces = facesByCell(mesh);
#ifdef FLOOD_HAS_MPI
        if (useMpi) mpiTopology->refresh(grid);
        const std::vector<int> cellOwners = useMpi
            ? mpiTopology->cellOwners(mesh) : std::vector<int>();
        const std::vector<std::size_t> remoteFaceIndices = useMpi
            ? remoteFaces(mesh, cellOwners) : std::vector<std::size_t>();
        if (useMpi)
            exchangeRemoteStates(mesh, cellOwners, remoteFaceIndices, *mpiTopology);
#endif
        double maximumRate = 0.0;
        const auto cellCount = static_cast<std::ptrdiff_t>(mesh.cells.size());
#ifdef FLOOD_HAS_OPENMP
        #pragma omp parallel for reduction(max:maximumRate) schedule(static) \
            num_threads(openMPThreads) if(useOpenMP)
#endif
        for (std::ptrdiff_t cellIndex = 0; cellIndex < cellCount; ++cellIndex) {
#ifdef FLOOD_HAS_MPI
            if (useMpi &&
                cellOwners[static_cast<std::size_t>(cellIndex)] != mpiTopology->rank)
                continue;
#endif
            const CellRef& cell = mesh.cells[static_cast<std::size_t>(cellIndex)];
            const Cell& state = *cell.state;
            if (state.h <= config.dryDepth) continue;
            const double u = state.hu / state.h;
            const double v = state.hv / state.h;
            const double wave = std::sqrt(config.gravity * state.h);
            maximumRate = std::max(maximumRate,
                (std::abs(u) + wave) / cell.dx + (std::abs(v) + wave) / cell.dy);
        }
#ifdef FLOOD_HAS_MPI
        if (useMpi)
            reduceAdaptiveMax(maximumRate, *mpiTopology,
                              "MPI_Allreduce adaptive CFL rate");
#endif
        double dt = maximumRate > 0.0 ? config.cfl / maximumRate : config.maxTimestep;
        dt = std::min({dt, config.maxTimestep, config.endTime - time});
        const double nextRainChange = rainfall.nextChangeAfter(time);
        if (nextRainChange > time) dt = std::min(dt, nextRainChange - time);
        if (!(dt > 0.0) || !std::isfinite(dt))
            throw std::runtime_error("Adaptive CFL timestep became invalid");

        applyBoundaryAndComputeFluxes(mesh, config.boundary, config.gravity,
                                      config.dryDepth, useOpenMP, openMPThreads
#ifdef FLOOD_HAS_MPI
                                      , useMpi ? &cellOwners : nullptr,
                                      useMpi ? mpiTopology->rank : -1
#endif
                                      );
#ifdef FLOOD_HAS_MPI
        if (useMpi)
            exchangeRemoteFluxes(mesh, remoteFaceIndices, cellOwners, *mpiTopology);
#endif
        std::vector<double> outgoing(mesh.cells.size(), 0.0);
#ifdef FLOOD_HAS_OPENMP
        #pragma omp parallel for schedule(static) num_threads(openMPThreads) if(useOpenMP)
#endif
        for (std::ptrdiff_t cellIndex = 0; cellIndex < cellCount; ++cellIndex) {
            const std::size_t index = static_cast<std::size_t>(cellIndex);
#ifdef FLOOD_HAS_MPI
            if (useMpi && cellOwners[index] != mpiTopology->rank) continue;
#endif
            if (useOpenMP || (
#ifdef FLOOD_HAS_MPI
                useMpi
#else
                false
#endif
                )) {
                for (const std::size_t faceIndex : cellFaces[index]) {
                    const FaceSegment& face = mesh.faces[faceIndex];
                    if (face.flux.donor == index)
                        outgoing[index] += std::abs(face.flux.massRate) * face.length;
                }
            } else {
                for (const FaceSegment& face : mesh.faces) {
                    if (face.flux.donor == index)
                        outgoing[index] += std::abs(face.flux.massRate) * face.length;
                }
            }
        }
        std::vector<double> limiter(mesh.cells.size(), 1.0);
#ifdef FLOOD_HAS_OPENMP
        #pragma omp parallel for schedule(static) num_threads(openMPThreads) if(useOpenMP)
#endif
        for (std::ptrdiff_t cellIndex = 0; cellIndex < cellCount; ++cellIndex) {
            const std::size_t index = static_cast<std::size_t>(cellIndex);
#ifdef FLOOD_HAS_MPI
            if (useMpi && cellOwners[index] != mpiTopology->rank) continue;
#endif
            const double available = mesh.cells[index].state->h * mesh.cells[index].area;
            if (outgoing[index] * dt > available && outgoing[index] > 0.0)
                limiter[index] = available / (outgoing[index] * dt);
        }

        std::vector<double> faceScales(mesh.faces.size(), 1.0);
        for (std::size_t faceIndex = 0; faceIndex < mesh.faces.size(); ++faceIndex) {
            const FaceSegment& face = mesh.faces[faceIndex];
            if (face.flux.donor != noCell)
                faceScales[faceIndex] = limiter[face.flux.donor];
        }
#ifdef FLOOD_HAS_MPI
        if (useMpi)
            exchangeRemoteLimiters(mesh, remoteFaceIndices, cellOwners, limiter,
                                   *mpiTopology, faceScales);
#endif
        std::vector<State> delta(mesh.cells.size(), State{0.0, 0.0, 0.0});
        std::vector<double> infiltrationByCell(mesh.cells.size(), 0.0);
        std::vector<unsigned char> invalidState(mesh.cells.size(), 0);
        for (const FaceSegment& face : mesh.faces) {
            const double scale = faceScales[static_cast<std::size_t>(&face - mesh.faces.data())];
            if (!useOpenMP
#ifdef FLOOD_HAS_MPI
                && !useMpi
#endif
                && face.left != noCell) {
                for (std::size_t component = 0; component < 3; ++component)
                    delta[face.left][component] -= scale *
                        face.flux.leftFlux[component] * face.length /
                        mesh.cells[face.left].area;
            }
            if (!useOpenMP
#ifdef FLOOD_HAS_MPI
                && !useMpi
#endif
                && face.right != noCell) {
                for (std::size_t component = 0; component < 3; ++component)
                    delta[face.right][component] += scale *
                        face.flux.rightFlux[component] * face.length /
                        mesh.cells[face.right].area;
            }
            if ((
#ifdef FLOOD_HAS_MPI
                 !useMpi || faceOwner(face, cellOwners) == mpiTopology->rank
#else
                 true
#endif
                ) && face.left != noCell && face.right != noCell) {
                if (mesh.cells[face.left].level != mesh.cells[face.right].level) {
                    const double leftMassContribution =
                        -scale * face.flux.leftFlux[0] * face.length;
                    const double rightMassContribution =
                        scale * face.flux.rightFlux[0] * face.length;
                    diagnostics.maximumInterfaceMassFluxResidual = std::max(
                        diagnostics.maximumInterfaceMassFluxResidual,
                        std::abs(leftMassContribution + rightMassContribution));
                    ++diagnostics.coarseFineInterfaceSegments;
#ifdef FLOOD_HAS_MPI
                    if (useMpi &&
                        cellOwners[face.left] != cellOwners[face.right] &&
                        faceOwner(face, cellOwners) == mpiTopology->rank) {
                        ++diagnostics.crossRankCoarseFineInterfaceSegments;
                        if (face.xDirection)
                            ++diagnostics.crossRankCoarseFineXSegments;
                        else
                            ++diagnostics.crossRankCoarseFineYSegments;
                    }
#endif
                    diagnostics.coarseFineIntegratedMassFlux +=
                        std::abs(scale * face.flux.massRate * face.length) * dt;
                }
            }
            if (face.flux.boundaryOutflow
#ifdef FLOOD_HAS_MPI
                && (!useMpi || faceOwner(face, cellOwners) == mpiTopology->rank)
#endif
                ) {
                const double outflowScale = faceScales[
                    static_cast<std::size_t>(&face - mesh.faces.data())];
                outflowVolume.add(outflowScale * std::abs(face.flux.massRate) *
                                  face.length * dt);
            }
        }

        const double rainRate = rainfall.metersPerSecond(time);
        double localArea = mesh.areaSum;
#ifdef FLOOD_HAS_MPI
        if (useMpi) {
            localArea = 0.0;
            for (const CellRef& cell : mesh.cells)
                if (cellOwners[&cell - mesh.cells.data()] == mpiTopology->rank)
                    localArea += cell.area;
        }
#endif
        rainfallVolume.add(rainRate * localArea * dt);
#ifdef FLOOD_HAS_OPENMP
        #pragma omp parallel for schedule(static) num_threads(openMPThreads) if(useOpenMP)
#endif
        for (std::ptrdiff_t cellIndex = 0; cellIndex < cellCount; ++cellIndex) {
            const std::size_t index = static_cast<std::size_t>(cellIndex);
#ifdef FLOOD_HAS_MPI
            if (useMpi && cellOwners[index] != mpiTopology->rank) continue;
#endif
            if (useOpenMP
#ifdef FLOOD_HAS_MPI
                || useMpi
#endif
                ) {
                for (const std::size_t faceIndex : cellFaces[index]) {
                    const FaceSegment& face = mesh.faces[faceIndex];
                    const double scale = faceScales[faceIndex];
                    if (face.left == index) {
                        for (std::size_t component = 0; component < 3; ++component)
                            delta[index][component] -= scale *
                                face.flux.leftFlux[component] * face.length /
                                mesh.cells[index].area;
                    } else if (face.right == index) {
                        for (std::size_t component = 0; component < 3; ++component)
                            delta[index][component] += scale *
                                face.flux.rightFlux[component] * face.length /
                                mesh.cells[index].area;
                    }
                }
            }
            Cell& cell = *mesh.cells[index].state;
            cell.h += dt * (delta[index][0] + rainRate);
            cell.hu += dt * delta[index][1];
            cell.hv += dt * delta[index][2];
            if (cell.h < 0.0 && cell.h > -1e-12) cell.h = 0.0;
            if (cell.h < 0.0 || !std::isfinite(cell.h) ||
                !std::isfinite(cell.hu) || !std::isfinite(cell.hv)) {
                invalidState[index] = 1;
                continue;
            }
            const double infiltrated = std::min(
                cell.h, config.infiltrationMetersPerSecond * dt);
            cell.h -= infiltrated;
            infiltrationByCell[index] = infiltrated * mesh.cells[index].area;
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
        for (std::size_t index = 0; index < mesh.cells.size(); ++index) {
#ifdef FLOOD_HAS_MPI
            if (useMpi && cellOwners[index] != mpiTopology->rank) continue;
#endif
            if (invalidState[index])
                throw std::runtime_error(
                    "Adaptive timestep produced an invalid state; simulation stopped");
            infiltrationVolume.add(infiltrationByCell[index]);
        }
#ifdef FLOOD_HAS_MPI
        if (useMpi)
            exchangeRemoteStates(mesh, cellOwners, remoteFaceIndices, *mpiTopology);
#endif
        diagnostics.computeSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - computeStart).count();
        const auto activityStart = std::chrono::steady_clock::now();
#ifdef FLOOD_HAS_MPI
        const auto activity = useMpi
            ? activityByPatchOwned(grid, mesh, previousDepth, config.gravity,
                                   config.dryDepth, cellOwners, *mpiTopology)
            : activityByPatch(grid, mesh, previousDepth, config.gravity,
                              config.dryDepth);
#else
        const auto activity = activityByPatch(
            grid, mesh, previousDepth, config.gravity, config.dryDepth);
#endif
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
                                         diagnostics, time, integrate);

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
                const ConservedIntegrals before = integrate(grid);
                grid.coarsen(parentId);
                recordRegridEvent(diagnostics, AdaptiveRegridOperation::Coarsen,
                                  parentId, time, before, grid, integrate);
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
#ifdef FLOOD_HAS_MPI
    if (useMpi) synchronizeOwnedStates(grid, *mpiTopology);
#endif
    diagnostics.activeLeafCells = 0;
    diagnostics.level0Cells = 0;
    diagnostics.level1Cells = 0;
    diagnostics.level2Cells = 0;
    for (const PatchId patchId : grid.activePatches()) {
#ifdef FLOOD_HAS_MPI
        if (useMpi && mpiTopology->ownerOfPatch(patchId) != mpiTopology->rank)
            continue;
#endif
        const std::size_t count = grid.patch(patchId).grid().size();
        diagnostics.activeLeafCells += count;
        const std::size_t level = grid.patch(patchId).level();
        if (level == 0) diagnostics.level0Cells += count;
        else if (level == 1) diagnostics.level1Cells += count;
        else if (level == 2) diagnostics.level2Cells += count;
    }
#ifdef FLOOD_HAS_MPI
    if (useMpi) {
        reduceAdaptiveCount(diagnostics.activeLeafCells, *mpiTopology,
                            "MPI_Allreduce adaptive active cells");
        reduceAdaptiveCount(diagnostics.level0Cells, *mpiTopology,
                            "MPI_Allreduce adaptive level-0 cells");
        reduceAdaptiveCount(diagnostics.level1Cells, *mpiTopology,
                            "MPI_Allreduce adaptive level-1 cells");
        reduceAdaptiveCount(diagnostics.level2Cells, *mpiTopology,
                            "MPI_Allreduce adaptive level-2 cells");
    }
#endif
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
#ifdef FLOOD_HAS_MPI
        if (useMpi && mpiTopology->ownerOfPatch(patchId) != mpiTopology->rank)
            continue;
#endif
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
#ifdef FLOOD_HAS_MPI
    if (useMpi) {
        reduceAdaptiveSum(diagnostics.rainfallVolume, *mpiTopology,
                          "MPI_Allreduce adaptive rainfall volume");
        reduceAdaptiveSum(diagnostics.infiltrationVolume, *mpiTopology,
                          "MPI_Allreduce adaptive infiltration volume");
        reduceAdaptiveSum(diagnostics.outflowVolume, *mpiTopology,
                          "MPI_Allreduce adaptive outflow volume");
        reduceAdaptiveSum(diagnostics.finalWaterVolume, *mpiTopology,
                          "MPI_Allreduce adaptive final volume");
        reduceAdaptiveMin(diagnostics.minimumDepth, *mpiTopology,
                          "MPI_Allreduce adaptive minimum depth");
        reduceAdaptiveMax(diagnostics.maximumDepth, *mpiTopology,
                          "MPI_Allreduce adaptive maximum depth");
        reduceAdaptiveMax(diagnostics.maximumVelocity, *mpiTopology,
                          "MPI_Allreduce adaptive maximum velocity");
        reduceAdaptiveCount(diagnostics.wetCells, *mpiTopology,
                            "MPI_Allreduce adaptive wet cells");
        reduceAdaptiveMax(diagnostics.maximumInterfaceMassFluxResidual,
                          *mpiTopology, "MPI_Allreduce adaptive interface residual");
        reduceAdaptiveCount(diagnostics.coarseFineInterfaceSegments, *mpiTopology,
                            "MPI_Allreduce adaptive interface segments");
        reduceAdaptiveCount(diagnostics.crossRankCoarseFineInterfaceSegments,
                            *mpiTopology,
                            "MPI_Allreduce adaptive cross-rank interface segments");
        reduceAdaptiveCount(diagnostics.crossRankCoarseFineXSegments, *mpiTopology,
                            "MPI_Allreduce adaptive cross-rank x segments");
        reduceAdaptiveCount(diagnostics.crossRankCoarseFineYSegments, *mpiTopology,
                            "MPI_Allreduce adaptive cross-rank y segments");
        reduceAdaptiveSum(diagnostics.coarseFineIntegratedMassFlux, *mpiTopology,
                          "MPI_Allreduce adaptive integrated interface flux");
        reduceAdaptiveMax(diagnostics.maximumRegridVolumeDelta, *mpiTopology,
                          "MPI_Allreduce adaptive regrid volume delta");
    }
#endif
    diagnostics.massBalanceResidual = diagnostics.initialWaterVolume +
        diagnostics.rainfallVolume - diagnostics.outflowVolume -
        diagnostics.infiltrationVolume - diagnostics.finalWaterVolume;
    diagnostics.runtimeSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - runtimeStart).count();
    return diagnostics;
}

} // namespace flood
