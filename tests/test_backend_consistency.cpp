#include "simulation/Scenarios.hpp"
#include "solver/OpenMPSolver.hpp"
#include "solver/SerialSolver.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef FLOOD_NUMERICAL_TOLERANCE
#define FLOOD_NUMERICAL_TOLERANCE 1e-8
#endif
#ifndef FLOOD_MOMENTUM_TOLERANCE
#define FLOOD_MOMENTUM_TOLERANCE 1e-8
#endif

int main() {
    try {
        const char* scenarios[] = {"flat-basin", "slope", "dam-break", "rain-drain", "wet-dry"};
        const int threadCounts[] = {1, 2, 4, 8, 16};
        for (const auto* scenarioName : scenarios) {
            auto serialCase = flood::makeScenario(scenarioName, 16, 16, 5.0);
            const auto serial = flood::SerialSolver().run(
                serialCase.grid, serialCase.rainfall, serialCase.config);
            for (const int threads : threadCounts) {
                auto parallelCase = flood::makeScenario(scenarioName, 16, 16, 5.0);
                const auto parallel = flood::OpenMPSolver(threads).run(
                    parallelCase.grid, parallelCase.rainfall, parallelCase.config);
                double maxDepthError = 0.0;
                double maxVelocityError = 0.0;
                double maxMomentumError = 0.0;
                for (std::size_t i = 0; i < parallelCase.grid.size(); ++i) {
                    const auto& expected = serialCase.grid.cells()[i];
                    const auto& actual = parallelCase.grid.cells()[i];
                    maxDepthError = std::max(maxDepthError, std::abs(expected.h - actual.h));
                    maxMomentumError = std::max(maxMomentumError,
                        std::max(std::abs(expected.hu - actual.hu), std::abs(expected.hv - actual.hv)));
                    if (expected.h > serialCase.config.dryDepth &&
                        actual.h > serialCase.config.dryDepth) {
                        maxVelocityError = std::max(maxVelocityError,
                            std::hypot(expected.hu / expected.h - actual.hu / actual.h,
                                       expected.hv / expected.h - actual.hv / actual.h));
                    }
                }
                if (maxDepthError > FLOOD_NUMERICAL_TOLERANCE ||
                    maxMomentumError > FLOOD_MOMENTUM_TOLERANCE ||
                    maxVelocityError > FLOOD_NUMERICAL_TOLERANCE ||
                    parallel.wetCells != serial.wetCells ||
                    std::abs(parallel.storedVolume - serial.storedVolume) > 1e-8 ||
                    std::abs(parallel.massResidual - serial.massResidual) > 1e-8) {
                    throw std::runtime_error(std::string(scenarioName) +
                        " failed serial/OpenMP consistency at " + std::to_string(threads) + " threads");
                }
                std::cout << scenarioName << " threads=" << threads
                          << " max_depth_error=" << maxDepthError
                          << " max_momentum_error=" << maxMomentumError
                          << " max_velocity_error=" << maxVelocityError << '\n';
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}