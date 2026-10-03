#pragma once

#include "solver/SerialSolver.hpp"

namespace flood::detail {

Diagnostics runSharedSolver(Grid& grid, const Rainfall& rainfall,
                            const SolverConfig& config, bool useOpenMP,
                            int threads);

} // namespace flood::detail