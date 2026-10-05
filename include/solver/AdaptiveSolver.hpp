#pragma once

#include "grid/AdaptiveGrid.hpp"
#include "physics/Rainfall.hpp"
#include "solver/SerialSolver.hpp"

#include <cstddef>
#include <vector>

#ifdef FLOOD_HAS_MPI
#include <mpi.h>
#endif

namespace flood {

class AdaptiveOpenMPSolver;
class AdaptiveMpiSolver;

struct AdaptiveSolverOptions {
    double refineThreshold = 0.10;
    double coarsenThreshold = 0.05;
    std::size_t coarsenPersistence = 3;
    std::size_t regridIntervalSteps = 10;
    bool collectWorkloadSnapshots = false;
    bool dynamicLoadBalancing = false;
    double loadBalanceImbalanceThreshold = 1.20;
    std::size_t loadBalanceCooldownSteps = 10;
};

enum class AdaptiveRegridOperation { Refine, Coarsen };

struct AdaptiveRegridEvent {
    AdaptiveRegridOperation operation;
    PatchId patchId;
    double time;
    double waterVolumeBefore;
    double waterVolumeAfter;
    double waterVolumeDelta;
    double huIntegralBefore;
    double huIntegralAfter;
    double huIntegralDelta;
    double hvIntegralBefore;
    double hvIntegralAfter;
    double hvIntegralDelta;
};

struct AdaptiveRankWorkload {
    int rank = 0;
    std::size_t activeLeafCells = 0;
    std::size_t activePatches = 0;
    std::size_t level0Cells = 0;
    std::size_t level1Cells = 0;
    std::size_t level2Cells = 0;
    double estimatedWork = 0.0;
    double computeSeconds = 0.0;
    double communicationSeconds = 0.0;
};

struct AdaptiveWorkloadSnapshot {
    std::size_t step = 0;
    double time = 0.0;
    std::vector<AdaptiveRankWorkload> ranks;
    double minimumRankWork = 0.0;
    double maximumRankWork = 0.0;
    double meanRankWork = 0.0;
    double loadImbalance = 1.0;
};

struct AdaptiveLoadBalanceEvent {
    std::size_t timestep = 0;
    std::size_t migratedPatchCount = 0;
    std::size_t migratedCellCount = 0;
    double migrationTimeSeconds = 0.0;
    double preRebalanceImbalance = 1.0;
    double postRebalanceImbalance = 1.0;
    std::vector<AdaptiveRankWorkload> preRankWorkloads;
    std::vector<AdaptiveRankWorkload> postRankWorkloads;
    std::vector<PatchId> rootPatchIds;
    std::vector<int> ownerRanks;
};

struct AdaptiveDiagnostics {
    double time = 0.0;
    double currentTimestep = 0.0;
    std::size_t steps = 0;
    std::size_t activeLeafCells = 0;
    std::size_t level0Cells = 0;
    std::size_t level1Cells = 0;
    std::size_t level2Cells = 0;
    double fineCellPercentage = 0.0;
    std::size_t refinedPatches = 0;
    std::size_t coarsenedPatches = 0;
    std::vector<AdaptiveRegridEvent> regridEvents;
    double maximumActivity = 0.0;
    double initialWaterVolume = 0.0;
    double rainfallVolume = 0.0;
    double infiltrationVolume = 0.0;
    double outflowVolume = 0.0;
    double finalWaterVolume = 0.0;
    double massBalanceResidual = 0.0;
    double maximumInterfaceMassFluxResidual = 0.0;
    std::size_t coarseFineInterfaceSegments = 0;
    std::size_t crossRankCoarseFineInterfaceSegments = 0;
    std::size_t crossRankCoarseFineXSegments = 0;
    std::size_t crossRankCoarseFineYSegments = 0;
    double coarseFineIntegratedMassFlux = 0.0;
    double maximumRegridVolumeDelta = 0.0;
    double runtimeSeconds = 0.0;
    double computeSeconds = 0.0;
    double meshSeconds = 0.0;
    double cflSeconds = 0.0;
    double interfaceSeconds = 0.0;
    double updateSeconds = 0.0;
    double activitySeconds = 0.0;
    double regriddingSeconds = 0.0;
    double communicationSeconds = 0.0;
    double diagnosticsSeconds = 0.0;
    std::vector<AdaptiveRankWorkload> rankWorkloads;
    std::vector<AdaptiveWorkloadSnapshot> workloadSnapshots;
    std::size_t migrationCount = 0;
    std::size_t migratedPatchCount = 0;
    std::size_t migratedCellCount = 0;
    double migrationTimeSeconds = 0.0;
    double preRebalanceImbalance = 1.0;
    double postRebalanceImbalance = 1.0;
    std::size_t lastRebalanceTimestep = 0;
    std::vector<AdaptiveLoadBalanceEvent> loadBalanceEvents;
    double maximumDepth = 0.0;
    double minimumDepth = 0.0;
    double maximumVelocity = 0.0;
    std::size_t wetCells = 0;
};

class AdaptiveSolver {
public:
    explicit AdaptiveSolver(AdaptiveSolverOptions options = {});

    AdaptiveDiagnostics run(AdaptiveGrid& grid, const Rainfall& rainfall,
                            const SolverConfig& config) const;

private:
    friend class AdaptiveOpenMPSolver;
    friend class AdaptiveMpiSolver;

    AdaptiveDiagnostics runImpl(AdaptiveGrid& grid, const Rainfall& rainfall,
                               const SolverConfig& config, bool useOpenMP,
                               int openMPThreads
#ifdef FLOOD_HAS_MPI
                               , MPI_Comm mpiComm
#endif
                               ) const;

    AdaptiveSolverOptions options_;
};

} // namespace flood
