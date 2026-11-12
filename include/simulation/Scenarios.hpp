#pragma once

#include "grid/Grid.hpp"
#include "physics/Rainfall.hpp"
#include "solver/SerialSolver.hpp"

#include <string>
#include <vector>

namespace flood {

// Land-use classes exported for visualisation (landuse.csv).
namespace LandUse {
constexpr unsigned char Pavement = 0;
constexpr unsigned char Road = 1;
constexpr unsigned char Building = 2;
constexpr unsigned char Grass = 3;
constexpr unsigned char River = 4;   // river channel, reservoir or pond bed
constexpr unsigned char Farmland = 6;
constexpr unsigned char Dam = 7;
} // namespace LandUse

// Optional scenario controls; a negative value keeps the built-in default.
struct ScenarioOptions {
    double surgeHeight = -1.0;    // urban-district: extra river level in the upstream reach (m)
    double reservoirDepth = -1.0; // dam-break: initial depth behind the dam (m)
    double pondDepth = -1.0;      // localized-refinement: initial pond depth (m)
    double breachWidth = -1.0;    // dam-breach: breach width as a fraction of the dam length
};

struct Scenario {
    Grid grid;
    Rainfall rainfall;
    SolverConfig config;
    std::string name;
    std::vector<unsigned char> landUse = {}; // empty when the scenario has none
};

Scenario makeScenario(const std::string& name, std::size_t rows, std::size_t cols,
                      double cellSize, const ScenarioOptions& options = {});

// Rainfall override: constant intensity (mm/h) from t=0 for `durationSeconds`
// (negative duration = for the whole run).
Rainfall constantStorm(double intensityMmPerHour, double durationSeconds);

} // namespace flood