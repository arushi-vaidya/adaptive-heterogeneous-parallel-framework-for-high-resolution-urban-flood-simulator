#pragma once

#include "grid/AdaptiveGrid.hpp"

#include <cstddef>
#include <optional>
#include <vector>

namespace flood {

struct AdaptiveRootWorkload {
    PatchId rootPatchId = 0;
    std::size_t activePatches = 0;
    std::size_t activeLeafCells = 0;
    int currentOwnerRank = 0;
};

struct AdaptiveOwnershipPlan {
    std::vector<PatchId> rootPatchIds;
    std::vector<int> ownerRanks;
    double projectedImbalance = 1.0;
};

class AdaptivePatchPartitioner {
public:
    AdaptiveOwnershipPlan propose(
        const std::vector<AdaptiveRootWorkload>& workload, int rankCount) const;

    bool shouldRebalance(double imbalance, std::size_t timestep,
                         const std::optional<std::size_t>& lastRebalanceTimestep,
                         double threshold, std::size_t cooldownSteps) const;
};

} // namespace flood
