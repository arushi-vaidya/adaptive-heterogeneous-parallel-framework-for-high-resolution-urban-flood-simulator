#include "solver/AdaptiveOpenMPSolver.hpp"

#include <stdexcept>

#ifdef FLOOD_HAS_OPENMP
#include <omp.h>
#endif

namespace flood {

AdaptiveOpenMPSolver::AdaptiveOpenMPSolver(AdaptiveSolverOptions options,
                                           int threads)
    : options_(options), threads_(threads) {
    if (threads < 0)
        throw std::invalid_argument("OpenMP thread count cannot be negative");
}

AdaptiveDiagnostics AdaptiveOpenMPSolver::run(
    AdaptiveGrid& grid, const Rainfall& rainfall,
    const SolverConfig& config) const {
#ifdef FLOOD_HAS_OPENMP
    const int activeThreads = threads_ == 0 ? omp_get_max_threads() : threads_;
    return AdaptiveSolver(options_).runImpl(
        grid, rainfall, config, true, activeThreads);
#else
    (void)grid;
    (void)rainfall;
    (void)config;
    throw std::runtime_error(
        "Adaptive OpenMP backend is unavailable; configure with FLOOD_ENABLE_OPENMP=ON");
#endif
}

} // namespace flood
