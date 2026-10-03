#pragma once

#include "grid/AdaptiveGrid.hpp"
#include "physics/Rainfall.hpp"
#include "solver/SerialSolver.hpp"

#include <cstddef>

namespace flood {

struct AdaptiveSolverOptions {
    double refineThreshold = 0.10;
    double coarsenThreshold = 0.05;
    std::size_t coarsenPersistence = 3;
    std::size_t regridIntervalSteps = 10;
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
    double maximumActivity = 0.0;
    double initialWaterVolume = 0.0;
    double rainfallVolume = 0.0;
    double infiltrationVolume = 0.0;
    double outflowVolume = 0.0;
    double finalWaterVolume = 0.0;
    double massBalanceResidual = 0.0;
    double maximumInterfaceMassFluxResidual = 0.0;
    double runtimeSeconds = 0.0;
    double computeSeconds = 0.0;
    double regriddingSeconds = 0.0;
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
    AdaptiveSolverOptions options_;
};

} // namespace flood
