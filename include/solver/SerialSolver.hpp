#pragma once

#include "grid/Grid.hpp"
#include "physics/Rainfall.hpp"

#include <cstddef>
#include <string>

namespace flood {

enum class BoundaryCondition { Closed, Outflow };

struct SolverConfig {
    double gravity = 9.81;
    double manningN = 0.03;
    double infiltrationMetersPerSecond = 0.0;
    double dryDepth = 1e-6;
    double cfl = 0.4;
    double maxTimestep = 10.0;
    double endTime = 60.0;
    BoundaryCondition boundary = BoundaryCondition::Closed;
};

struct Diagnostics {
    double time = 0.0;
    double minDepth = 0.0;
    double maxDepth = 0.0;
    double maxVelocity = 0.0;
    double floodedArea = 0.0;
    double storedVolume = 0.0;
    double rainfallVolume = 0.0;
    double outflowVolume = 0.0;
    double infiltrationVolume = 0.0;
    double massResidual = 0.0;
    std::size_t wetCells = 0;
    std::size_t steps = 0;
    double solverSeconds = 0.0;
    std::string backend = "serial";
    int threads = 1;
    int processes = 1;
    bool isRoot = true;
    double setupSeconds = 0.0;
    double computationSeconds = 0.0;
    double communicationSeconds = 0.0;
    double synchronizationSeconds = 0.0;
    double otherSeconds = 0.0;
};

class SolverBackend {
public:
    virtual ~SolverBackend() = default;
    virtual Diagnostics run(Grid& grid, const Rainfall& rainfall,
                            const SolverConfig& config) const = 0;
};

class SerialSolver final : public SolverBackend {
public:
    Diagnostics run(Grid& grid, const Rainfall& rainfall,
                    const SolverConfig& config) const override;
};

std::string toString(BoundaryCondition boundary);

} // namespace flood