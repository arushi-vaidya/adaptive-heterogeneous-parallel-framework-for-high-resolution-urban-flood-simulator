#include "simulation/Scenarios.hpp"

#include <stdexcept>

namespace flood {

Scenario makeScenario(const std::string& name, std::size_t rows, std::size_t cols,
                      double cellSize) {
    Scenario scenario{Grid(rows, cols, cellSize, cellSize), Rainfall(), SolverConfig(), name};
    scenario.config.endTime = 60.0;
    scenario.config.boundary = BoundaryCondition::Closed;

    if (name == "flat-basin") {
        scenario.rainfall = Rainfall({{0.0, 30.0}});
    } else if (name == "slope") {
        scenario.rainfall = Rainfall({{0.0, 0.0}});
        scenario.config.endTime = 20.0;
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                scenario.grid.at(row, col).bed = 0.02 * static_cast<double>(cols - col);
            }
        }
        for (std::size_t row = 0; row < rows; ++row) scenario.grid.at(row, 0).h = 0.03;
    } else if (name == "dam-break") {
        scenario.rainfall = Rainfall({{0.0, 0.0}});
        scenario.config.endTime = 10.0;
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols / 2; ++col) scenario.grid.at(row, col).h = 1.0;
        }
    } else if (name == "rain-drain") {
        scenario.rainfall = Rainfall({{0.0, 60.0}, {30.0, 0.0}});
        scenario.config.endTime = 60.0;
        scenario.config.infiltrationMetersPerSecond = 1.0 / 3600.0;
        scenario.config.boundary = BoundaryCondition::Outflow;
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                scenario.grid.at(row, col).bed = 0.001 * static_cast<double>(col);
                scenario.grid.at(row, col).h = 0.03;
            }
        }
    } else if (name == "wet-dry") {
        scenario.rainfall = Rainfall({{0.0, 0.0}});
        scenario.config.endTime = 15.0;
        scenario.grid.at(rows / 2, cols / 2).h = 2e-6;
        scenario.grid.at(rows / 2, cols / 2).hu = 1e-7;
    } else {
        throw std::invalid_argument("Unknown scenario: " + name);
    }
    return scenario;
}

} // namespace flood