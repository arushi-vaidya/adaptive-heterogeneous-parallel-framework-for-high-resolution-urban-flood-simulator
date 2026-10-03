#pragma once

#include "solver/SerialSolver.hpp"

namespace flood {

class MpiSolver final : public SolverBackend {
public:
    Diagnostics run(Grid& grid, const Rainfall& rainfall,
                    const SolverConfig& config) const override;
};

} // namespace flood