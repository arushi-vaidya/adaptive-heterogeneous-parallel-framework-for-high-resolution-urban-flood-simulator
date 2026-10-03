#pragma once

#include "grid/Grid.hpp"
#include "physics/Rainfall.hpp"
#include "solver/SerialSolver.hpp"

#include <string>

namespace flood {

struct Scenario {
    Grid grid;
    Rainfall rainfall;
    SolverConfig config;
    std::string name;
};

Scenario makeScenario(const std::string& name, std::size_t rows, std::size_t cols,
                      double cellSize);

} // namespace flood