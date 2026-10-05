#include "solver/AdaptiveLoadBalance.hpp"

#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

double imbalance(const std::vector<flood::AdaptiveRootWorkload>& roots,
                 const flood::AdaptiveOwnershipPlan& plan, int rankCount) {
    std::vector<double> work(static_cast<std::size_t>(rankCount), 0.0);
    for (std::size_t index = 0; index < roots.size(); ++index)
        work[static_cast<std::size_t>(plan.ownerRanks[index])] +=
            static_cast<double>(roots[index].activeLeafCells);
    double total = 0.0;
    double maximum = 0.0;
    for (const double value : work) {
        total += value;
        if (value > maximum) maximum = value;
    }
    const double mean = total / static_cast<double>(rankCount);
    return mean > 0.0 ? maximum / mean : 1.0;
}

} // namespace

int main() {
    try {
        const flood::AdaptivePatchPartitioner partitioner;
        check(!partitioner.shouldRebalance(1.0, 1, std::nullopt, 1.20, 10),
              "Balanced work must not trigger migration");
        check(!partitioner.shouldRebalance(1.19, 1, std::nullopt, 1.20, 10),
              "Below-threshold work must not trigger migration");
        check(partitioner.shouldRebalance(1.20, 1, std::nullopt, 1.20, 10),
              "Threshold equality must trigger migration");
        check(!partitioner.shouldRebalance(1.50, 19, 10, 1.20, 10),
              "Cooldown must suppress migration before ten steps");
        check(partitioner.shouldRebalance(1.50, 20, 10, 1.20, 10),
              "Cooldown must expire at ten steps");

        std::vector<flood::AdaptiveRootWorkload> concentrated;
        for (flood::PatchId id = 0; id < 8; ++id) {
            concentrated.push_back({id, 1, id == 0 ? 100U : 10U,
                                    id < 4 ? 0 : 1});
        }
        const auto plan = partitioner.propose(concentrated, 2);
        const auto repeated = partitioner.propose(concentrated, 2);
        check(plan.rootPatchIds == repeated.rootPatchIds &&
              plan.ownerRanks == repeated.ownerRanks,
              "Identical workloads must produce identical ownership");
        check(imbalance(concentrated, plan, 2) < 1.20,
              "Concentrated refinement must be redistributed below threshold");
        check(plan.ownerRanks[1] != concentrated[1].currentOwnerRank,
              "Balancing must move ownership when static assignment is uneven");

        std::vector<flood::AdaptiveRootWorkload> fewRoots = {
            {3, 1, 0, 0}, {9, 1, 0, 0}};
        const auto fewRootPlan = partitioner.propose(fewRoots, 4);
        check(fewRootPlan.rootPatchIds.size() == 2 &&
              fewRootPlan.ownerRanks.size() == 2,
              "Fewer roots than ranks must retain a valid complete ownership map");
        for (const int owner : fewRootPlan.ownerRanks)
            check(owner >= 0 && owner < 4,
                  "Every root owner must be a valid MPI rank");
        check(fewRootPlan.projectedImbalance == 1.0,
              "Zero-work roots must have a finite neutral imbalance");

        std::vector<flood::AdaptiveRootWorkload> equalWork;
        for (flood::PatchId id = 0; id < 12; ++id)
            equalWork.push_back({id, 1, 16, 0});
        const auto equalPlan = partitioner.propose(equalWork, 4);
        check(std::abs(equalPlan.projectedImbalance - 1.0) < 1e-12,
              "Equal roots should partition evenly");

        const auto zeroRankPlan = partitioner.propose(
            {{0, 1, 64, 0}, {1, 1, 0, 0}, {2, 1, 0, 0},
             {3, 1, 0, 0}, {4, 1, 0, 0}}, 4);
        check(zeroRankPlan.ownerRanks.size() == 5 &&
              std::isfinite(zeroRankPlan.projectedImbalance),
              "Zero/low-work roots must yield a valid finite partition");

        std::cout << "Adaptive load-balance partition and trigger tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
