#pragma once

#include "grid/Grid.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace flood {

using PatchId = std::uint64_t;

struct AdaptiveCellId {
    PatchId patchId;
    std::size_t row;
    std::size_t col;

    bool operator==(const AdaptiveCellId& other) const noexcept {
        return patchId == other.patchId && row == other.row && col == other.col;
    }
};

struct AdaptiveGridConfig {
    std::size_t patchRows = 16;
    std::size_t patchCols = 16;
    std::size_t refinementRatio = 2;
    std::size_t maxRefinementLevel = 2;
};

enum class PatchSide { North, South, West, East };
enum class PatchNeighborType { SameLevel, Coarser, Finer, PhysicalBoundary };

struct PatchNeighbor {
    PatchSide side;
    PatchNeighborType type;
    std::optional<PatchId> patchId;
};

class AdaptivePatch {
public:
    PatchId id() const noexcept { return id_; }
    std::size_t level() const noexcept { return level_; }
    std::optional<PatchId> parentId() const noexcept { return parentId_; }
    const std::vector<PatchId>& childIds() const noexcept { return childIds_; }
    bool isActiveLeaf() const noexcept { return activeLeaf_; }
    bool isCovered() const noexcept { return covered_; }
    const Grid& grid() const noexcept { return grid_; }
    Grid& grid() noexcept { return grid_; }
    const std::vector<PatchNeighbor>& neighbors() const noexcept { return neighbors_; }
    std::size_t rowOriginFineUnits() const noexcept { return rowOriginFineUnits_; }
    std::size_t colOriginFineUnits() const noexcept { return colOriginFineUnits_; }

private:
    friend class AdaptiveGrid;

    AdaptivePatch(PatchId id, std::size_t level, std::optional<PatchId> parentId,
                  Grid grid, std::size_t rowOriginFineUnits,
                  std::size_t colOriginFineUnits, bool activeLeaf);

    PatchId id_;
    std::size_t level_;
    std::optional<PatchId> parentId_;
    std::vector<PatchId> childIds_;
    Grid grid_;
    std::size_t rowOriginFineUnits_;
    std::size_t colOriginFineUnits_;
    bool activeLeaf_;
    bool covered_ = false;
    std::vector<PatchNeighbor> neighbors_;
};

class AdaptiveGrid {
public:
    explicit AdaptiveGrid(const Grid& baseGrid, AdaptiveGridConfig config = {});

    const AdaptiveGridConfig& config() const noexcept { return config_; }
    std::size_t rows() const noexcept { return baseRows_; }
    std::size_t cols() const noexcept { return baseCols_; }
    double dx0() const noexcept { return dx0_; }
    double dy0() const noexcept { return dy0_; }
    std::size_t patchCount() const noexcept { return patches_.size(); }
    std::size_t leafCount() const noexcept { return activePatchIds_.size(); }
    std::size_t activeCellCount() const noexcept { return activeCellCount_; }

    const std::vector<PatchId>& activePatches() const noexcept { return activePatchIds_; }
    const AdaptivePatch& patch(PatchId id) const;
    AdaptivePatch& patch(PatchId id);

    std::vector<PatchId> refine(PatchId parentId);
    void coarsen(PatchId parentId);

    template <typename Function>
    void forEachActiveCell(Function&& function) {
        for (const PatchId id : activePatchIds_) {
            auto& patchValue = patch(id);
            auto& grid = patchValue.grid_;
            for (std::size_t row = 0; row < grid.rows(); ++row)
                for (std::size_t col = 0; col < grid.cols(); ++col)
                    function(AdaptiveCellId{id, row, col}, grid.at(row, col));
        }
    }

    template <typename Function>
    void forEachActiveCell(Function&& function) const {
        for (const PatchId id : activePatchIds_) {
            const auto& patchValue = patch(id);
            const auto& grid = patchValue.grid_;
            for (std::size_t row = 0; row < grid.rows(); ++row)
                for (std::size_t col = 0; col < grid.cols(); ++col)
                    function(AdaptiveCellId{id, row, col}, grid.at(row, col));
        }
    }

private:
    std::size_t cellScale(std::size_t level) const;
    void validatePatchId(PatchId id) const;
    void rebuildTopology();
    void rebuildActiveIndex();
    void validateTwoToOne() const;

    AdaptiveGridConfig config_;
    std::size_t baseRows_;
    std::size_t baseCols_;
    double dx0_;
    double dy0_;
    std::size_t domainRowsFineUnits_;
    std::size_t domainColsFineUnits_;
    std::vector<std::size_t> cellScales_;
    std::vector<AdaptivePatch> patches_;
    std::vector<PatchId> activePatchIds_;
    std::size_t activeCellCount_ = 0;
};

} // namespace flood
