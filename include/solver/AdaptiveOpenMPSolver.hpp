#pragma once

#include "solver/AdaptiveSolver.hpp"

namespace flood {

class AdaptiveOpenMPSolver {
public:
    explicit AdaptiveOpenMPSolver(AdaptiveSolverOptions options = {},
                                  int threads = 0);

    AdaptiveDiagnostics run(AdaptiveGrid& grid, const Rainfall& rainfall,
                            const SolverConfig& config) const;

private:
    AdaptiveSolverOptions options_;
    int threads_;
};

} // namespace flood
