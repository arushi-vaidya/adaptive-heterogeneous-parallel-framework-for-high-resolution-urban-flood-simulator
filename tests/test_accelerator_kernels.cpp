// Regression test for the flat accelerator kernels shared by the OpenACC and
// CUDA backends: the OpenACC loop nest, executed on the host, must reproduce
// the serial reference solver on every built-in scenario.

#include "simulation/Scenarios.hpp"
#include "solver/AcceleratorSolvers.hpp"
#include "solver/SerialSolver.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    const double fieldTolerance = 1e-10;
    const double massTolerance = 1e-6;
    for (const std::string name : {"flat-basin", "slope", "dam-break", "rain-drain", "wet-dry",
                                   "localized-refinement", "urban-district"}) {
        auto reference = flood::makeScenario(name, 40, 36, 5.0);
        auto candidate = flood::makeScenario(name, 40, 36, 5.0);
        if (name == "urban-district") {
            reference.config.endTime = 20.0;
            candidate.config.endTime = 20.0;
        }
        const auto serial = flood::SerialSolver().run(reference.grid, reference.rainfall,
                                                      reference.config);
        const auto flat = flood::detail::runOpenACCSolver(candidate.grid, candidate.rainfall,
                                                          candidate.config);
        double maxError = 0.0;
        for (std::size_t i = 0; i < reference.grid.size(); ++i) {
            const auto& a = reference.grid.cells()[i];
            const auto& b = candidate.grid.cells()[i];
            maxError = std::max({maxError, std::abs(a.h - b.h), std::abs(a.hu - b.hu),
                                 std::abs(a.hv - b.hv)});
        }
        if (serial.steps != flat.steps)
            throw std::runtime_error(name + ": step count differs from serial reference");
        if (maxError > fieldTolerance)
            throw std::runtime_error(name + ": accelerator kernels diverge from serial, max error " +
                                     std::to_string(maxError));
        if (std::abs(flat.massResidual) > massTolerance)
            throw std::runtime_error(name + ": accelerator mass residual too large");
        std::cout << name << " steps=" << flat.steps << " max_error=" << maxError
                  << " mass_residual=" << flat.massResidual << '\n';
    }
    return 0;
}
