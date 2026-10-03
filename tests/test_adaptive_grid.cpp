#include "grid/AdaptiveGrid.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void requireNear(double actual, double expected, double tolerance,
                 const std::string& message) {
    if (std::abs(actual - expected) > tolerance)
        throw std::runtime_error(message + ": expected " + std::to_string(expected) +
                                 ", got " + std::to_string(actual));
}

std::array<long double, 3> activeIntegrals(const flood::AdaptiveGrid& grid) {
    std::array<long double, 3> sums{};
    grid.forEachActiveCell([&](flood::AdaptiveCellId id, const flood::Cell& cell) {
        const auto& patch = grid.patch(id.patchId);
        const long double area =
            static_cast<long double>(patch.grid().dx()) * patch.grid().dy();
        sums[0] += static_cast<long double>(cell.h) * area;
        sums[1] += static_cast<long double>(cell.hu) * area;
        sums[2] += static_cast<long double>(cell.hv) * area;
    });
    return sums;
}

void baseGridTest() {
    flood::Grid base(5, 7, 2.0, 3.0);
    base.at(2, 4) = {7.0, 0.3, 0.4, -0.2};
    flood::AdaptiveGrid adaptive(base);
    require(adaptive.rows() == 5 && adaptive.cols() == 7,
            "Base grid dimensions were not retained");
    require(adaptive.patchCount() == 1 && adaptive.leafCount() == 1,
            "Base grid should start as one active patch");
    require(adaptive.activeCellCount() == 35, "Base active-cell count is wrong");
    const auto& patch = adaptive.patch(adaptive.activePatches().front());
    require(patch.level() == 0 && patch.grid().rows() == 5 &&
            patch.grid().cols() == 7, "Base patch shape is wrong");
    requireNear(patch.grid().dx(), 2.0, 0.0, "Base dx is wrong");
    requireNear(patch.grid().dy(), 3.0, 0.0, "Base dy is wrong");
    require(patch.neighbors().size() == 4,
            "Single base patch should expose all physical boundaries");
    for (const auto& neighbor : patch.neighbors())
        require(neighbor.type == flood::PatchNeighborType::PhysicalBoundary &&
                !neighbor.patchId,
                "Physical boundary topology was classified incorrectly");
    require(patch.grid().at(2, 4).bed == 7.0 &&
            patch.grid().at(2, 4).h == 0.3 &&
            patch.grid().at(2, 4).hu == 0.4 &&
            patch.grid().at(2, 4).hv == -0.2,
            "Base state values were not copied");
}

void refinementAndConservationTest() {
    flood::Grid base(4, 6, 2.0, 3.0);
    for (std::size_t row = 0; row < base.rows(); ++row) {
        for (std::size_t col = 0; col < base.cols(); ++col) {
            base.at(row, col) = {0.01 * static_cast<double>(row + col),
                                 0.2 + 0.01 * static_cast<double>(row),
                                 -0.3 + 0.02 * static_cast<double>(col),
                                 0.4 + 0.01 * static_cast<double>(row + col)};
        }
    }
    flood::AdaptiveGridConfig config;
    config.patchRows = 4;
    config.patchCols = 6;
    flood::AdaptiveGrid adaptive(base, config);
    const auto before = activeIntegrals(adaptive);
    const auto children = adaptive.refine(0);
    require(children.size() == 4, "Refinement did not create four children");
    require(adaptive.patch(0).isCovered() && !adaptive.patch(0).isActiveLeaf(),
            "Refined parent was not made covered and inactive");
    require(adaptive.leafCount() == 4 && adaptive.activeCellCount() == 96,
            "Refined active leaf set is incorrect");
    for (const flood::PatchId childId : children) {
        const auto& child = adaptive.patch(childId);
        require(child.isActiveLeaf() && child.level() == 1 &&
                child.parentId() == std::optional<flood::PatchId>(0),
                "Child hierarchy metadata is incorrect");
        require(child.grid().rows() == 4 && child.grid().cols() == 6,
                "Child patch dimensions are incorrect");
        requireNear(child.grid().dx(), 1.0, 0.0, "Refined dx did not halve");
        requireNear(child.grid().dy(), 1.5, 0.0, "Refined dy did not halve");
    }
    const auto after = activeIntegrals(adaptive);
    for (std::size_t component = 0; component < before.size(); ++component)
        require(std::abs(before[component] - after[component]) < 1e-12L,
                "Refinement did not preserve a conserved integral");
}

void coarseningTest() {
    flood::Grid base(4, 4, 1.0, 2.0);
    flood::AdaptiveGridConfig config;
    config.patchRows = 4;
    config.patchCols = 4;
    flood::AdaptiveGrid adaptive(base, config);
    const auto children = adaptive.refine(0);
    for (std::size_t childIndex = 0; childIndex < children.size(); ++childIndex) {
        auto& grid = adaptive.patch(children[childIndex]).grid();
        const double value = static_cast<double>(childIndex + 1);
        for (flood::Cell& cell : grid.cells())
            cell = {10.0 + value, value, -2.0 * value, value * 0.5};
    }
    adaptive.coarsen(0);
    const auto& parent = adaptive.patch(0);
    require(parent.isActiveLeaf() && !parent.isCovered(),
            "Coarsened parent did not become active");
    require(adaptive.leafCount() == 1 && adaptive.activeCellCount() == 16,
            "Coarsening did not deactivate child patches");
    const std::array<std::array<double, 2>, 2> expected = {{{1.0, 2.0}, {3.0, 4.0}}};
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t col = 0; col < 4; ++col) {
            const double value = expected[row / 2][col / 2];
            const auto& cell = parent.grid().at(row, col);
            requireNear(cell.h, value, 1e-14, "Coarsened depth average is incorrect");
            requireNear(cell.hu, -2.0 * value, 1e-14,
                        "Coarsened hu average is incorrect");
            requireNear(cell.hv, value * 0.5, 1e-14,
                        "Coarsened hv average is incorrect");
            requireNear(cell.bed, 10.0 + value, 1e-14,
                        "Coarsened bed average is incorrect");
        }
    }
    for (const flood::PatchId childId : children)
        require(!adaptive.patch(childId).isActiveLeaf(),
                "Coarsened child remained active");
}

void deterministicTraversalTest() {
    flood::Grid base(8, 8, 1.0, 1.0);
    flood::AdaptiveGridConfig config;
    config.patchRows = 4;
    config.patchCols = 4;
    flood::AdaptiveGrid first(base, config);
    flood::AdaptiveGrid second(base, config);
    const auto firstChildren = first.refine(0);
    const auto secondChildren = second.refine(0);
    first.refine(firstChildren[0]);
    second.refine(secondChildren[0]);
    require(first.patchCount() == second.patchCount(),
            "Repeated refinement produced different patch counts");
    require(first.activePatches() == second.activePatches(),
            "Repeated refinement produced different active ordering");
    std::vector<flood::AdaptiveCellId> firstCellIds, secondCellIds;
    first.forEachActiveCell([&](flood::AdaptiveCellId id, const flood::Cell&) {
        firstCellIds.push_back(id);
    });
    second.forEachActiveCell([&](flood::AdaptiveCellId id, const flood::Cell&) {
        secondCellIds.push_back(id);
    });
    require(firstCellIds == secondCellIds,
            "Repeated refinement produced different stable cell identifiers");
    for (std::size_t index = 0; index < first.patchCount(); ++index) {
        const auto& left = first.patch(static_cast<flood::PatchId>(index));
        const auto& right = second.patch(static_cast<flood::PatchId>(index));
        require(left.id() == right.id() && left.level() == right.level() &&
                left.parentId() == right.parentId() &&
                left.childIds() == right.childIds() &&
                left.isActiveLeaf() == right.isActiveLeaf() &&
                left.isCovered() == right.isCovered(),
                "Repeated refinement produced different hierarchy metadata");
    }
}

void topologyAndBalanceTest() {
    flood::Grid base(8, 4, 1.0, 1.0);
    flood::AdaptiveGridConfig config;
    config.patchRows = 4;
    config.patchCols = 4;
    flood::AdaptiveGrid adaptive(base, config);
    require(adaptive.patchCount() == 2, "Expected two base patches");
    const auto& northPatch = adaptive.patch(0);
    require(!northPatch.neighbors().empty(), "Patch topology is missing");
    const auto children = adaptive.refine(0);
    bool seesFineNeighbor = false;
    for (const auto& neighbor : adaptive.patch(1).neighbors()) {
        if (neighbor.patchId && neighbor.type == flood::PatchNeighborType::Finer)
            seesFineNeighbor = true;
    }
    require(seesFineNeighbor, "Coarse/fine neighbor relation was not recorded");
    bool rejected = false;
    try {
        adaptive.refine(children[2]);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    require(rejected, "Refinement violating the 2:1 rule was not rejected");
    adaptive.coarsen(0);
}

void invalidCoarseningTest() {
    flood::AdaptiveGridConfig config;
    config.patchRows = 4;
    config.patchCols = 4;
    flood::AdaptiveGrid adaptive(flood::Grid(8, 8, 1.0, 1.0), config);
    bool rejected = false;
    try {
        adaptive.coarsen(0);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    require(rejected, "Unrefined patch was incorrectly coarsened");
    const auto children = adaptive.refine(0);
    adaptive.refine(children.front());
    rejected = false;
    try {
        adaptive.coarsen(0);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    require(rejected, "A parent with a covered child was incorrectly coarsened");
    rejected = false;
    try {
        adaptive.coarsen(children[1]);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    require(rejected, "A child with no child set was incorrectly coarsened");
}

} // namespace

int main() {
    try {
        baseGridTest();
        refinementAndConservationTest();
        coarseningTest();
        deterministicTraversalTest();
        topologyAndBalanceTest();
        invalidCoarseningTest();
        std::cout << "adaptive grid transfer, topology, and determinism checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
