#include "grid/AdaptiveGrid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace flood {
namespace {

std::size_t checkedMultiply(std::size_t left, std::size_t right,
                            const char* description) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
        throw std::overflow_error(description);
    return left * right;
}

std::size_t checkedAdd(std::size_t left, std::size_t right,
                       const char* description) {
    if (left > std::numeric_limits<std::size_t>::max() - right)
        throw std::overflow_error(description);
    return left + right;
}

std::pair<std::size_t, std::size_t> partition(std::size_t extent,
                                               std::size_t parts,
                                               std::size_t index) {
    const std::size_t quotient = extent / parts;
    const std::size_t remainder = extent % parts;
    const std::size_t start = index * quotient + std::min(index, remainder);
    return {start, quotient + (index < remainder ? 1 : 0)};
}

struct StateIntegrals {
    std::array<long double, 4> values{};
    std::array<long double, 4> magnitudes{};

    void add(const Cell& cell, long double area) {
        const std::array<double, 4> state = {cell.h, cell.hu, cell.hv, cell.bed};
        for (std::size_t component = 0; component < state.size(); ++component) {
            const long double contribution = static_cast<long double>(state[component]) * area;
            values[component] += contribution;
            magnitudes[component] += std::abs(contribution);
        }
    }
};

StateIntegrals integrals(const Grid& grid) {
    StateIntegrals result;
    const long double area = static_cast<long double>(grid.dx()) * grid.dy();
    for (const Cell& cell : grid.cells()) {
        if (cell.h < 0.0 || !std::isfinite(cell.h) || !std::isfinite(cell.hu) ||
            !std::isfinite(cell.hv) || !std::isfinite(cell.bed))
            throw std::invalid_argument("Adaptive grid state must be finite with nonnegative depth");
        result.add(cell, area);
    }
    return result;
}

StateIntegrals integrals(const std::vector<const AdaptivePatch*>& patches) {
    StateIntegrals result;
    for (const AdaptivePatch* patch : patches) {
        const long double area =
            static_cast<long double>(patch->grid().dx()) * patch->grid().dy();
        for (const Cell& cell : patch->grid().cells()) {
            if (cell.h < 0.0 || !std::isfinite(cell.h) || !std::isfinite(cell.hu) ||
                !std::isfinite(cell.hv) || !std::isfinite(cell.bed))
                throw std::invalid_argument("Adaptive grid state must be finite with nonnegative depth");
            result.add(cell, area);
        }
    }
    return result;
}

void requireSameIntegrals(const StateIntegrals& expected,
                          const StateIntegrals& actual,
                          const char* operation) {
    constexpr long double factor = 128.0L * std::numeric_limits<double>::epsilon();
    for (std::size_t component = 0; component < expected.values.size(); ++component) {
        const long double scale = std::max(expected.magnitudes[component],
                                           actual.magnitudes[component]);
        const long double tolerance = factor *
            std::max(scale, static_cast<long double>(std::numeric_limits<double>::min()));
        if (std::abs(expected.values[component] - actual.values[component]) > tolerance)
            throw std::logic_error(std::string("Adaptive state-transfer invariant failed during ") +
                                   operation);
    }
}

PatchNeighborType neighborType(std::size_t ownLevel, std::size_t neighborLevel) {
    if (ownLevel == neighborLevel) return PatchNeighborType::SameLevel;
    return neighborLevel < ownLevel ? PatchNeighborType::Coarser
                                    : PatchNeighborType::Finer;
}

} // namespace

AdaptivePatch::AdaptivePatch(PatchId id, std::size_t level,
                             std::optional<PatchId> parentId, Grid grid,
                             std::size_t rowOriginFineUnits,
                             std::size_t colOriginFineUnits, bool activeLeaf)
    : id_(id), level_(level), parentId_(parentId), grid_(std::move(grid)),
      rowOriginFineUnits_(rowOriginFineUnits),
      colOriginFineUnits_(colOriginFineUnits), activeLeaf_(activeLeaf) {}

AdaptiveGrid::AdaptiveGrid(const Grid& baseGrid, AdaptiveGridConfig config)
    : config_(config), baseRows_(baseGrid.rows()), baseCols_(baseGrid.cols()),
      dx0_(baseGrid.dx()), dy0_(baseGrid.dy()), domainRowsFineUnits_(0),
      domainColsFineUnits_(0) {
    if (config_.patchRows == 0 || config_.patchCols == 0 ||
        config_.refinementRatio != 2)
        throw std::invalid_argument(
            "Adaptive patches require positive extents and a refinement ratio of 2");
    if (config_.maxRefinementLevel == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Maximum refinement level is too large");

    std::size_t finestScale = 1;
    for (std::size_t level = 0; level < config_.maxRefinementLevel; ++level)
        finestScale = checkedMultiply(finestScale, config_.refinementRatio,
                                      "Adaptive refinement depth overflows coordinates");
    cellScales_.resize(config_.maxRefinementLevel + 1);
    cellScales_[config_.maxRefinementLevel] = 1;
    for (std::size_t level = config_.maxRefinementLevel; level > 0; --level)
        cellScales_[level - 1] =
            checkedMultiply(cellScales_[level], config_.refinementRatio,
                            "Adaptive refinement depth overflows coordinates");
    domainRowsFineUnits_ = checkedMultiply(baseRows_, finestScale,
                                           "Adaptive row coordinates overflow");
    domainColsFineUnits_ = checkedMultiply(baseCols_, finestScale,
                                           "Adaptive column coordinates overflow");

    for (std::size_t rowStart = 0; rowStart < baseRows_;) {
        const std::size_t patchRows = std::min(config_.patchRows, baseRows_ - rowStart);
        for (std::size_t colStart = 0; colStart < baseCols_;) {
            const std::size_t patchCols = std::min(config_.patchCols, baseCols_ - colStart);
            Grid patchGrid(patchRows, patchCols, dx0_, dy0_);
            for (std::size_t row = 0; row < patchRows; ++row)
                for (std::size_t col = 0; col < patchCols; ++col)
                    patchGrid.at(row, col) = baseGrid.at(rowStart + row, colStart + col);
            if (patches_.size() > std::numeric_limits<PatchId>::max())
                throw std::overflow_error("Adaptive patch identifier space exhausted");
            const PatchId id = static_cast<PatchId>(patches_.size());
            patches_.push_back(AdaptivePatch(
                id, 0, std::nullopt, std::move(patchGrid),
                checkedMultiply(rowStart, finestScale, "Adaptive row origin overflows"),
                checkedMultiply(colStart, finestScale, "Adaptive column origin overflows"),
                true));
            colStart += patchCols;
        }
        rowStart += patchRows;
    }
    rebuildActiveIndex();
    rebuildTopology();
}

std::size_t AdaptiveGrid::cellScale(std::size_t level) const {
    if (level >= cellScales_.size())
        throw std::out_of_range("Adaptive refinement level is out of range");
    return cellScales_[level];
}

void AdaptiveGrid::validatePatchId(PatchId id) const {
    if (id >= patches_.size() || patches_[static_cast<std::size_t>(id)].id_ != id)
        throw std::out_of_range("Adaptive patch identifier is invalid");
}

const AdaptivePatch& AdaptiveGrid::patch(PatchId id) const {
    validatePatchId(id);
    return patches_[static_cast<std::size_t>(id)];
}

AdaptivePatch& AdaptiveGrid::patch(PatchId id) {
    validatePatchId(id);
    return patches_[static_cast<std::size_t>(id)];
}

std::vector<PatchId> AdaptiveGrid::refine(PatchId parentId) {
    validatePatchId(parentId);
    const AdaptivePatch& initialParent = patch(parentId);
    if (!initialParent.activeLeaf_ || initialParent.covered_)
        throw std::logic_error("Only an active leaf patch can be refined");
    if (initialParent.level_ >= config_.maxRefinementLevel)
        throw std::logic_error("Patch is already at the maximum refinement level");
    if (initialParent.grid_.rows() < config_.refinementRatio ||
        initialParent.grid_.cols() < config_.refinementRatio)
        throw std::logic_error("Patch is too small to create all child patches");
    for (const PatchNeighbor& neighbor : initialParent.neighbors_) {
        if (neighbor.patchId && patch(*neighbor.patchId).level_ < initialParent.level_)
            throw std::logic_error("Refinement would violate the 2:1 neighboring-level rule");
    }

    const std::size_t ratio = config_.refinementRatio;
    const std::size_t parentLevel = initialParent.level_;
    const std::size_t parentRows = initialParent.grid_.rows();
    const std::size_t parentCols = initialParent.grid_.cols();
    const std::size_t parentRowOrigin = initialParent.rowOriginFineUnits_;
    const std::size_t parentColOrigin = initialParent.colOriginFineUnits_;
    const double childDx = initialParent.grid_.dx() / static_cast<double>(ratio);
    const double childDy = initialParent.grid_.dy() / static_cast<double>(ratio);
    const Grid parentGrid = initialParent.grid_;
    const StateIntegrals expected = integrals(parentGrid);

    std::vector<PatchId> childIds;
    std::vector<AdaptivePatch> newChildren;
    const bool creatingChildren = initialParent.childIds_.empty();
    if (creatingChildren) {
        const std::size_t childCount = checkedMultiply(ratio, ratio,
                                                       "Adaptive child count overflows");
        childIds.reserve(childCount);
        newChildren.reserve(childCount);
        const std::size_t firstId = patches_.size();
        if (childCount > std::numeric_limits<PatchId>::max() - firstId)
            throw std::overflow_error("Adaptive patch identifier space exhausted");

        for (std::size_t partRow = 0; partRow < ratio; ++partRow) {
            const auto [parentRowStart, parentPartRows] =
                partition(parentRows, ratio, partRow);
            for (std::size_t partCol = 0; partCol < ratio; ++partCol) {
                const auto [parentColStart, parentPartCols] =
                    partition(parentCols, ratio, partCol);
                const std::size_t childRows = checkedMultiply(
                    parentPartRows, ratio, "Adaptive child patch dimensions overflow");
                const std::size_t childCols = checkedMultiply(
                    parentPartCols, ratio, "Adaptive child patch dimensions overflow");
                Grid childGrid(childRows, childCols, childDx, childDy);
                for (std::size_t row = 0; row < childRows; ++row) {
                    for (std::size_t col = 0; col < childCols; ++col) {
                        // Piecewise-constant prolongation copies each parent's conserved averages to its children.
                        childGrid.at(row, col) =
                            parentGrid.at(parentRowStart + row / ratio,
                                          parentColStart + col / ratio);
                    }
                }
                const PatchId id = static_cast<PatchId>(firstId + childIds.size());
                const std::size_t rowOffset = checkedMultiply(
                    parentRowStart, cellScale(parentLevel),
                    "Adaptive child row origin overflows");
                const std::size_t colOffset = checkedMultiply(
                    parentColStart, cellScale(parentLevel),
                    "Adaptive child column origin overflows");
                childIds.push_back(id);
                newChildren.push_back(AdaptivePatch(
                    id, parentLevel + 1, parentId, std::move(childGrid),
                    checkedAdd(parentRowOrigin, rowOffset,
                               "Adaptive child row origin overflows"),
                    checkedAdd(parentColOrigin, colOffset,
                               "Adaptive child column origin overflows"), true));
            }
        }
    } else {
        childIds = initialParent.childIds_;
        if (childIds.size() != checkedMultiply(ratio, ratio,
                                               "Adaptive child count overflows"))
            throw std::logic_error("Refined patch has an incomplete child set");
        for (const PatchId id : childIds) {
            const AdaptivePatch& child = patch(id);
            if (child.activeLeaf_ || child.covered_ || child.parentId_ != parentId)
                throw std::logic_error("Cannot reactivate an inconsistent child patch set");
        }
        std::size_t childIndex = 0;
        for (std::size_t partRow = 0; partRow < ratio; ++partRow) {
            const auto [parentRowStart, parentPartRows] =
                partition(parentRows, ratio, partRow);
            for (std::size_t partCol = 0; partCol < ratio; ++partCol, ++childIndex) {
                const auto [parentColStart, parentPartCols] =
                    partition(parentCols, ratio, partCol);
                const PatchId id = childIds[childIndex];
                const AdaptivePatch& child = patch(id);
                Grid childGrid(checkedMultiply(parentPartRows, ratio,
                                               "Adaptive child patch dimensions overflow"),
                               checkedMultiply(parentPartCols, ratio,
                                               "Adaptive child patch dimensions overflow"),
                               childDx, childDy);
                if (child.grid_.rows() != childGrid.rows() ||
                    child.grid_.cols() != childGrid.cols())
                    throw std::logic_error("Stored child patch geometry does not match its parent");
                for (std::size_t row = 0; row < childGrid.rows(); ++row)
                    for (std::size_t col = 0; col < childGrid.cols(); ++col)
                        childGrid.at(row, col) =
                            parentGrid.at(parentRowStart + row / ratio,
                                          parentColStart + col / ratio);
                newChildren.push_back(AdaptivePatch(
                    id, parentLevel + 1, parentId, std::move(childGrid),
                    child.rowOriginFineUnits_, child.colOriginFineUnits_, true));
            }
        }
    }

    std::vector<const AdaptivePatch*> childPointers;
    childPointers.reserve(newChildren.size());
    for (const AdaptivePatch& child : newChildren) childPointers.push_back(&child);
    requireSameIntegrals(expected, integrals(childPointers), "refinement");

    if (creatingChildren) {
        patches_.reserve(patches_.size() + newChildren.size());
        for (AdaptivePatch& child : newChildren) patches_.push_back(std::move(child));
        patch(parentId).childIds_ = childIds;
    } else {
        for (std::size_t index = 0; index < childIds.size(); ++index) {
            AdaptivePatch& child = patch(childIds[index]);
            child.grid_ = std::move(newChildren[index].grid_);
            child.activeLeaf_ = true;
            child.covered_ = false;
        }
    }
    AdaptivePatch& parent = patch(parentId);
    parent.activeLeaf_ = false;
    parent.covered_ = true;
    rebuildActiveIndex();
    rebuildTopology();
    validateTwoToOne();
    return childIds;
}

void AdaptiveGrid::coarsen(PatchId parentId) {
    validatePatchId(parentId);
    AdaptivePatch& parent = patch(parentId);
    const std::size_t ratio = config_.refinementRatio;
    const std::size_t expectedChildCount = checkedMultiply(
        ratio, ratio, "Adaptive child count overflows");
    if (parent.activeLeaf_ || !parent.covered_ ||
        parent.childIds_.size() != expectedChildCount)
        throw std::logic_error("Patch cannot be coarsened without a complete child set");

    for (const PatchId childId : parent.childIds_) {
        const AdaptivePatch& child = patch(childId);
        if (child.parentId_ != parentId || child.level_ != parent.level_ + 1 ||
            !child.activeLeaf_ || child.covered_)
            throw std::logic_error("Coarsening requires all direct sibling patches to be active leaves");
    }

    Grid coarseGrid = parent.grid_;
    const std::size_t parentRows = coarseGrid.rows();
    const std::size_t parentCols = coarseGrid.cols();
    std::size_t childIndex = 0;
    for (std::size_t partRow = 0; partRow < ratio; ++partRow) {
        const auto [parentRowStart, parentPartRows] =
            partition(parentRows, ratio, partRow);
        for (std::size_t partCol = 0; partCol < ratio; ++partCol, ++childIndex) {
            const auto [parentColStart, parentPartCols] =
                partition(parentCols, ratio, partCol);
            const AdaptivePatch& child = patch(parent.childIds_[childIndex]);
            if (child.grid_.rows() != parentPartRows * ratio ||
                child.grid_.cols() != parentPartCols * ratio)
                throw std::logic_error("Child patch dimensions do not match their parent");
            for (std::size_t row = 0; row < parentPartRows; ++row) {
                for (std::size_t col = 0; col < parentPartCols; ++col) {
                    std::array<long double, 4> sum{};
                    for (std::size_t subRow = 0; subRow < ratio; ++subRow) {
                        for (std::size_t subCol = 0; subCol < ratio; ++subCol) {
                            const Cell& fine = child.grid_.at(row * ratio + subRow,
                                                             col * ratio + subCol);
                            if (fine.h < 0.0 || !std::isfinite(fine.h) ||
                                !std::isfinite(fine.hu) || !std::isfinite(fine.hv) ||
                                !std::isfinite(fine.bed))
                                throw std::invalid_argument(
                                    "Adaptive grid state must be finite with nonnegative depth");
                            sum[0] += fine.h;
                            sum[1] += fine.hu;
                            sum[2] += fine.hv;
                            sum[3] += fine.bed;
                        }
                    }
                    const long double inverseAreaCount =
                        1.0L / static_cast<long double>(ratio * ratio);
                    Cell& coarse = coarseGrid.at(parentRowStart + row,
                                                 parentColStart + col);
                    coarse.h = static_cast<double>(sum[0] * inverseAreaCount);
                    coarse.hu = static_cast<double>(sum[1] * inverseAreaCount);
                    coarse.hv = static_cast<double>(sum[2] * inverseAreaCount);
                    coarse.bed = static_cast<double>(sum[3] * inverseAreaCount);
                }
            }
        }
    }

    std::vector<const AdaptivePatch*> childPointers;
    childPointers.reserve(parent.childIds_.size());
    for (const PatchId childId : parent.childIds_)
        childPointers.push_back(&patch(childId));
    requireSameIntegrals(integrals(coarseGrid), integrals(childPointers), "coarsening");

    parent.grid_ = std::move(coarseGrid);
    parent.activeLeaf_ = true;
    parent.covered_ = false;
    for (const PatchId childId : parent.childIds_) {
        AdaptivePatch& child = patch(childId);
        child.activeLeaf_ = false;
        child.covered_ = false;
    }
    rebuildActiveIndex();
    rebuildTopology();
    validateTwoToOne();
}

void AdaptiveGrid::rebuildActiveIndex() {
    activePatchIds_.clear();
    activeCellCount_ = 0;
    for (const AdaptivePatch& patchValue : patches_) {
        if (!patchValue.activeLeaf_) continue;
        activePatchIds_.push_back(patchValue.id_);
        activeCellCount_ = checkedAdd(
            activeCellCount_, patchValue.grid_.size(),
            "Adaptive active-cell count overflows");
    }
}

void AdaptiveGrid::rebuildTopology() {
    for (AdaptivePatch& patchValue : patches_) patchValue.neighbors_.clear();
    const auto addNeighbor = [](AdaptivePatch& patchValue, PatchSide side,
                                PatchNeighborType type,
                                std::optional<PatchId> neighborId) {
        patchValue.neighbors_.push_back({side, type, neighborId});
    };
    const auto rowEnd = [this](const AdaptivePatch& patchValue) {
        return checkedAdd(patchValue.rowOriginFineUnits_,
            checkedMultiply(patchValue.grid_.rows(), cellScale(patchValue.level_),
                            "Adaptive patch row extent overflows"),
            "Adaptive patch row bound overflows");
    };
    const auto colEnd = [this](const AdaptivePatch& patchValue) {
        return checkedAdd(patchValue.colOriginFineUnits_,
            checkedMultiply(patchValue.grid_.cols(), cellScale(patchValue.level_),
                            "Adaptive patch column extent overflows"),
            "Adaptive patch column bound overflows");
    };

    for (const PatchId id : activePatchIds_) {
        AdaptivePatch& patchValue = patch(id);
        const std::size_t rEnd = rowEnd(patchValue);
        const std::size_t cEnd = colEnd(patchValue);
        if (patchValue.rowOriginFineUnits_ == 0)
            addNeighbor(patchValue, PatchSide::North,
                        PatchNeighborType::PhysicalBoundary, std::nullopt);
        if (rEnd == domainRowsFineUnits_)
            addNeighbor(patchValue, PatchSide::South,
                        PatchNeighborType::PhysicalBoundary, std::nullopt);
        if (patchValue.colOriginFineUnits_ == 0)
            addNeighbor(patchValue, PatchSide::West,
                        PatchNeighborType::PhysicalBoundary, std::nullopt);
        if (cEnd == domainColsFineUnits_)
            addNeighbor(patchValue, PatchSide::East,
                        PatchNeighborType::PhysicalBoundary, std::nullopt);
    }

    for (std::size_t firstIndex = 0; firstIndex < activePatchIds_.size(); ++firstIndex) {
        AdaptivePatch& first = patch(activePatchIds_[firstIndex]);
        const std::size_t firstRowEnd = rowEnd(first);
        const std::size_t firstColEnd = colEnd(first);
        for (std::size_t secondIndex = firstIndex + 1;
             secondIndex < activePatchIds_.size(); ++secondIndex) {
            AdaptivePatch& second = patch(activePatchIds_[secondIndex]);
            const std::size_t secondRowEnd = rowEnd(second);
            const std::size_t secondColEnd = colEnd(second);
            const bool rowOverlap = std::max(first.rowOriginFineUnits_,
                second.rowOriginFineUnits_) < std::min(firstRowEnd, secondRowEnd);
            const bool colOverlap = std::max(first.colOriginFineUnits_,
                second.colOriginFineUnits_) < std::min(firstColEnd, secondColEnd);
            const PatchNeighborType firstType = neighborType(first.level_, second.level_);
            const PatchNeighborType secondType = neighborType(second.level_, first.level_);
            if (first.rowOriginFineUnits_ == secondRowEnd && colOverlap) {
                addNeighbor(first, PatchSide::North, firstType, second.id_);
                addNeighbor(second, PatchSide::South, secondType, first.id_);
            }
            if (firstRowEnd == second.rowOriginFineUnits_ && colOverlap) {
                addNeighbor(first, PatchSide::South, firstType, second.id_);
                addNeighbor(second, PatchSide::North, secondType, first.id_);
            }
            if (first.colOriginFineUnits_ == secondColEnd && rowOverlap) {
                addNeighbor(first, PatchSide::West, firstType, second.id_);
                addNeighbor(second, PatchSide::East, secondType, first.id_);
            }
            if (firstColEnd == second.colOriginFineUnits_ && rowOverlap) {
                addNeighbor(first, PatchSide::East, firstType, second.id_);
                addNeighbor(second, PatchSide::West, secondType, first.id_);
            }
        }
    }

    for (const PatchId id : activePatchIds_) {
        auto& neighbors = patch(id).neighbors_;
        std::sort(neighbors.begin(), neighbors.end(),
            [](const PatchNeighbor& left, const PatchNeighbor& right) {
                if (left.side != right.side) return left.side < right.side;
                return left.patchId.value_or(0) < right.patchId.value_or(0);
            });
    }
}

void AdaptiveGrid::validateTwoToOne() const {
    for (const PatchId id : activePatchIds_) {
        const AdaptivePatch& patchValue = patch(id);
        for (const PatchNeighbor& neighbor : patchValue.neighbors_) {
            if (!neighbor.patchId) continue;
            const std::size_t neighborLevel = patch(*neighbor.patchId).level_;
            const std::size_t difference = patchValue.level_ > neighborLevel
                ? patchValue.level_ - neighborLevel
                : neighborLevel - patchValue.level_;
            if (difference > 1)
                throw std::logic_error("Adaptive topology violates the 2:1 neighboring-level rule");
        }
    }
}

} // namespace flood
