#include "solver/OpenMPSolver.hpp"
#include "solver/detail/SharedSolver.hpp"

#include <stdexcept>

#ifdef FLOOD_HAS_OPENMP
#include <omp.h>
#endif

namespace flood {

OpenMPSolver::OpenMPSolver(int threads) : threads_(threads) {
    if (threads < 0) throw std::invalid_argument("OpenMP thread count cannot be negative");
}

Diagnostics OpenMPSolver::run(Grid& grid, const Rainfall& rainfall,
                              const SolverConfig& config) const {
#ifdef FLOOD_HAS_OPENMP
    const int activeThreads = threads_ == 0 ? omp_get_max_threads() : threads_;
    auto diagnostics = detail::runSharedSolver(grid, rainfall, config, true, activeThreads);
    diagnostics.backend = "openmp";
    diagnostics.threads = activeThreads;
    return diagnostics;
#else
    (void)grid;
    (void)rainfall;
    (void)config;
    throw std::runtime_error("OpenMP backend is unavailable; configure with FLOOD_ENABLE_OPENMP=ON");
#endif
}

} // namespace flood