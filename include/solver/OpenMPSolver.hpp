#pragma once

#include "solver/SerialSolver.hpp"

namespace flood {

class OpenMPSolver final : public SolverBackend {
public:
    explicit OpenMPSolver(int threads = 0);
    Diagnostics run(Grid& grid, const Rainfall& rainfall,
                    const SolverConfig& config) const override;

private:
    int threads_;
};

} // namespace flood