#include "solver/AdaptiveLoadBalance.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace flood {

AdaptiveOwnershipPlan AdaptivePatchPartitioner::propose(
    const std::vector<AdaptiveRootWorkload>& workload, int rankCount) const {
    if (rankCount <= 0)
        throw std::invalid_argument("Adaptive partitioning requires at least one rank");

    std::vector<AdaptiveRootWorkload> roots = workload;
    std::sort(roots.begin(), roots.end(),
              [](const AdaptiveRootWorkload& left,
                 const AdaptiveRootWorkload& right) {
                  return left.rootPatchId < right.rootPatchId;
              });
    for (std::size_t index = 1; index < roots.size(); ++index) {
        if (roots[index - 1].rootPatchId == roots[index].rootPatchId)
            throw std::invalid_argument("Adaptive root workload contains duplicate IDs");
    }

    AdaptiveOwnershipPlan plan;
    plan.rootPatchIds.reserve(roots.size());
    plan.ownerRanks.resize(roots.size(), 0);
    for (const AdaptiveRootWorkload& root : roots)
        plan.rootPatchIds.push_back(root.rootPatchId);
    if (roots.empty()) return plan;

    const std::size_t participatingRanks = std::min(
        roots.size(), static_cast<std::size_t>(rankCount));
    std::vector<long double> prefixWork(roots.size() + 1, 0.0L);
    for (std::size_t index = 0; index < roots.size(); ++index) {
        prefixWork[index + 1] = prefixWork[index] +
            static_cast<long double>(roots[index].activeLeafCells);
    }
    const bool noWork = prefixWork.back() == 0.0L;
    if (noWork) {
        for (std::size_t index = 0; index < roots.size(); ++index)
            prefixWork[index + 1] = static_cast<long double>(index + 1);
    }

    std::size_t begin = 0;
    for (std::size_t rank = 0; rank < participatingRanks; ++rank) {
        const std::size_t minimumEnd = begin + 1;
        const std::size_t maximumEnd =
            roots.size() - (participatingRanks - rank - 1);
        const long double target = prefixWork.back() *
            static_cast<long double>(rank + 1) /
            static_cast<long double>(participatingRanks);
        std::size_t bestEnd = minimumEnd;
        long double bestDistance = std::abs(prefixWork[bestEnd] - target);
        for (std::size_t end = minimumEnd + 1; end <= maximumEnd; ++end) {
            const long double distance = std::abs(prefixWork[end] - target);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestEnd = end;
            }
        }
        for (std::size_t index = begin; index < bestEnd; ++index)
            plan.ownerRanks[index] = static_cast<int>(rank);
        begin = bestEnd;
    }

    std::vector<long double> rankWork(static_cast<std::size_t>(rankCount), 0.0L);
    for (std::size_t index = 0; index < roots.size(); ++index)
        rankWork[static_cast<std::size_t>(plan.ownerRanks[index])] +=
            static_cast<long double>(roots[index].activeLeafCells);
    const long double totalWork = rankWork.empty() ? 0.0L :
        std::accumulate(rankWork.begin(), rankWork.end(), 0.0L);
    if (totalWork > 0.0L) {
        const long double mean = totalWork / static_cast<long double>(rankCount);
        const long double maximum =
            *std::max_element(rankWork.begin(), rankWork.end());
        plan.projectedImbalance = static_cast<double>(maximum / mean);
    }
    return plan;
}

bool AdaptivePatchPartitioner::shouldRebalance(
    double imbalance, std::size_t timestep,
    const std::optional<std::size_t>& lastRebalanceTimestep,
    double threshold, std::size_t cooldownSteps) const {
    if (!std::isfinite(imbalance) || !std::isfinite(threshold) ||
        threshold < 1.0)
        throw std::invalid_argument("Invalid adaptive load-balance trigger");
    if (imbalance < threshold) return false;
    if (!lastRebalanceTimestep) return true;
    return timestep >= *lastRebalanceTimestep &&
        timestep - *lastRebalanceTimestep >= cooldownSteps;
}

} // namespace flood
