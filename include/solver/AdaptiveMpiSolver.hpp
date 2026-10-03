#pragma once

#include "solver/AdaptiveSolver.hpp"

#ifdef FLOOD_HAS_MPI
#include <mpi.h>
#endif

namespace flood {

class AdaptiveMpiSolver {
public:
    explicit AdaptiveMpiSolver(AdaptiveSolverOptions options = {});

#ifdef FLOOD_HAS_MPI
    AdaptiveDiagnostics run(AdaptiveGrid& grid, const Rainfall& rainfall,
                            const SolverConfig& config,
                            MPI_Comm communicator = MPI_COMM_WORLD) const;
#else
    AdaptiveDiagnostics run(AdaptiveGrid& grid, const Rainfall& rainfall,
                            const SolverConfig& config) const;
#endif

private:
    AdaptiveSolverOptions options_;
};

} // namespace flood
