#include "simulation/Scenarios.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace flood {
namespace {

// Deterministic per-block hash so building heights do not depend on resolution.
double blockNoise(int bx, int by, int salt) {
    std::uint32_t h = static_cast<std::uint32_t>(bx) * 73856093u ^
                      static_cast<std::uint32_t>(by) * 19349663u ^
                      static_cast<std::uint32_t>(salt) * 83492791u;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return static_cast<double>(h & 0xffffu) / 65535.0;
}

// Synthetic urban district defined in normalised coordinates, so the same
// city is produced at every grid resolution (needed for fine-grid references).
// Features: a meandering river channel, a surge in its upstream reach, a street
// grid, building footprints that block flow, parks, and a storm hyetograph.
void buildUrbanDistrict(Scenario& scenario, double surgeHeight) {
    auto& grid = scenario.grid;
    const std::size_t rows = grid.rows(), cols = grid.cols();
    const double pi = 3.14159265358979323846;
    const double blocks = 9.0;
    const double roadHalfWidth = 0.16;   // fraction of a block
    const double riverHalfWidth = 0.045; // normalised domain units
    scenario.rainfall = Rainfall({{0.0, 120.0}, {60.0, 40.0}, {150.0, 0.0}});
    scenario.config.endTime = 120.0;
    scenario.config.manningN = 0.035;
    scenario.config.infiltrationMetersPerSecond = 2e-6;
    scenario.config.boundary = BoundaryCondition::Outflow;
    scenario.landUse.assign(rows * cols, LandUse::Pavement);
    for (std::size_t row = 0; row < rows; ++row) {
        const double y = (static_cast<double>(row) + 0.5) / static_cast<double>(rows);
        const double riverX = 0.34 + 0.22 * y + 0.07 * std::sin(2.0 * pi * 1.3 * y);
        for (std::size_t col = 0; col < cols; ++col) {
            const double x = (static_cast<double>(col) + 0.5) / static_cast<double>(cols);
            const double d = std::abs(x - riverX);
            Cell& cell = grid.at(row, col);
            unsigned char& use = scenario.landUse[row * cols + col];
            // Floodplain rising away from the river, with a gentle valley slope.
            double ground = 0.35 + 2.6 * std::pow(d, 1.15) + 0.6 * (1.0 - y) +
                            0.25 * std::sin(5.0 * x + 1.7) * std::cos(4.0 * y);
            if (d < riverHalfWidth) {
                const double t = d / riverHalfWidth;
                cell.bed = -1.6 * (1.0 - t * t) + 0.6 * (1.0 - y);
                // River at bank-full; upstream reach carries a flood surge.
                const double surface = 0.65 + 0.6 * (1.0 - y) + (y < 0.3 ? surgeHeight : 0.0);
                cell.h = std::max(0.0, surface - cell.bed);
                use = LandUse::River;
                continue;
            }
            const double bxF = x * blocks, byF = y * blocks;
            const int bx = static_cast<int>(bxF), by = static_cast<int>(byF);
            const double fx = bxF - bx, fy = byF - by;
            const bool road = fx < roadHalfWidth || fx > 1.0 - roadHalfWidth ||
                              fy < roadHalfWidth || fy > 1.0 - roadHalfWidth;
            const bool riverside = d < riverHalfWidth + 0.035;
            if (road) ground -= 0.18; // streets sit below kerbs and act as channels
            cell.bed = ground;
            if (road) { use = LandUse::Road; continue; }
            if (riverside) { use = LandUse::Grass; continue; }
            const double kind = blockNoise(bx, by, 1);
            if (kind < 0.16) { cell.bed = ground - 0.08; use = LandUse::Grass; continue; } // park
            // Split the block into 1, 2 or 4 buildings separated by alleys.
            const int split = kind < 0.45 ? 1 : (kind < 0.75 ? 2 : 4);
            const double ux = (fx - roadHalfWidth) / (1.0 - 2.0 * roadHalfWidth);
            const double uy = (fy - roadHalfWidth) / (1.0 - 2.0 * roadHalfWidth);
            int sx = 0, sy = 0;
            bool alley = false;
            if (split >= 2) {
                sx = ux < 0.5 ? 0 : 1;
                alley = alley || std::abs(ux - 0.5) < 0.06;
            }
            if (split == 4) {
                sy = uy < 0.5 ? 0 : 1;
                alley = alley || std::abs(uy - 0.5) < 0.06;
            }
            const bool setback = ux < 0.06 || ux > 0.94 || uy < 0.06 || uy > 0.94;
            if (alley || setback) continue;
            const double height = 6.0 + 16.0 * blockNoise(bx * 2 + sx, by * 2 + sy, 7);
            cell.bed = ground + height;
            use = LandUse::Building;
        }
    }
}

// Reservoir behind a concrete dam with a partial breach, draining into a
// valley with a stream, a crossing road, farmland and a village.
void buildDamBreach(Scenario& scenario, double depth, double breachWidth) {
    auto& grid = scenario.grid;
    const std::size_t rows = grid.rows(), cols = grid.cols();
    const double damX = 0.26, damWidth = std::max(0.03, 1.5 / static_cast<double>(cols));
    scenario.rainfall = Rainfall({{0.0, 0.0}});
    scenario.config.endTime = 60.0;
    scenario.config.manningN = 0.04;
    scenario.config.boundary = BoundaryCondition::Outflow;
    scenario.landUse.assign(rows * cols, LandUse::Farmland);
    for (std::size_t row = 0; row < rows; ++row) {
        const double y = (static_cast<double>(row) + 0.5) / static_cast<double>(rows);
        for (std::size_t col = 0; col < cols; ++col) {
            const double x = (static_cast<double>(col) + 0.5) / static_cast<double>(cols);
            Cell& cell = grid.at(row, col);
            unsigned char& use = scenario.landUse[row * cols + col];
            if (x < damX) { // reservoir, deepening towards the dam
                cell.bed = -0.6 * (x / damX) + 0.25 * std::pow(std::abs(y - 0.5) * 2.0, 2.0);
                cell.h = std::max(0.0, depth - cell.bed);
                use = LandUse::River;
                continue;
            }
            if (x < damX + damWidth) {
                if (std::abs(y - 0.5) < breachWidth * 0.5) { cell.bed = -0.6; use = LandUse::River; }
                else { cell.bed = depth + 1.5; use = LandUse::Dam; }
                continue;
            }
            const double t = (x - damX - damWidth) / (1.0 - damX - damWidth);
            const double across = std::abs(y - 0.5) * 2.0;
            double ground = -0.7 - 1.4 * t + 2.4 * std::pow(across, 1.6) +
                            0.18 * std::sin(9.0 * x + 1.3) * std::cos(7.0 * y);
            const double streamY = 0.5 + 0.06 * std::sin(6.0 * t);
            if (std::abs(y - streamY) < 0.022) {
                ground -= 0.5; use = LandUse::River;
                cell.h = 0.15; // base flow
            }
            const bool road = std::abs(t - 0.38) < 0.012;
            if (road && use != LandUse::River) { ground -= 0.05; use = LandUse::Road; }
            cell.bed = ground;
            // village: house lots on both valley sides
            const double lot = 0.05;
            if (!road && use == LandUse::Farmland && t > 0.45 && t < 0.8 && across > 0.18 && across < 0.72) {
                const int lx = static_cast<int>(x / lot), ly = static_cast<int>(y / lot);
                const double fx = x / lot - lx, fy = y / lot - ly;
                if (blockNoise(lx, ly, 3) < 0.7 && fx > 0.22 && fx < 0.78 && fy > 0.22 && fy < 0.78) {
                    cell.bed = ground + 5.0 + 3.0 * blockNoise(lx, ly, 5);
                    use = LandUse::Building;
                } else if (fx < 0.12 || fy < 0.12) use = LandUse::Pavement;
            }
        }
    }
}

// Hilly rural catchment: storm runoff concentrates in valleys and a stream
// that leaves the domain at its southern edge, passing a small village.
void buildCatchment(Scenario& scenario) {
    auto& grid = scenario.grid;
    const std::size_t rows = grid.rows(), cols = grid.cols();
    const double pi = 3.14159265358979323846;
    scenario.rainfall = Rainfall({{0.0, 200.0}, {600.0, 0.0}});
    scenario.config.endTime = 900.0;
    scenario.config.manningN = 0.05;
    scenario.config.infiltrationMetersPerSecond = 5.0 / 1000.0 / 3600.0;
    scenario.config.boundary = BoundaryCondition::Outflow;
    scenario.landUse.assign(rows * cols, LandUse::Farmland);
    for (std::size_t row = 0; row < rows; ++row) {
        const double y = (static_cast<double>(row) + 0.5) / static_cast<double>(rows);
        const double streamX = 0.5 + 0.12 * std::sin(pi * 1.4 * y + 0.4);
        for (std::size_t col = 0; col < cols; ++col) {
            const double x = (static_cast<double>(col) + 0.5) / static_cast<double>(cols);
            Cell& cell = grid.at(row, col);
            unsigned char& use = scenario.landUse[row * cols + col];
            const double d = std::abs(x - streamX);
            double ground = 3.5 * (1.0 - y) + 6.0 * std::pow(d, 1.4) +
                            1.1 * std::sin(7.0 * x + 1.0) * std::sin(5.0 * y + 2.0) +
                            0.5 * std::sin(13.0 * x + 0.5) * std::cos(11.0 * y);
            if (d < 0.018) { ground -= 0.5; use = LandUse::River; cell.h = 0.3; }
            else if (ground > 4.5 + 1.5 * y) use = LandUse::Grass; // upland woodland and pasture
            const bool road = std::abs(y - 0.82) < 0.01 && use != LandUse::River;
            if (road) { ground -= 0.05; use = LandUse::Road; }
            cell.bed = ground;
            const double lot = 0.045;
            if (!road && use == LandUse::Farmland && y > 0.7 && y < 0.95 && d > 0.05 && d < 0.26) {
                const int lx = static_cast<int>(x / lot), ly = static_cast<int>(y / lot);
                const double fx = x / lot - lx, fy = y / lot - ly;
                if (blockNoise(lx, ly, 11) < 0.6 && fx > 0.2 && fx < 0.8 && fy > 0.2 && fy < 0.8) {
                    cell.bed = ground + 4.5 + 3.0 * blockNoise(lx, ly, 13);
                    use = LandUse::Building;
                } else if (fx < 0.1 || fy < 0.1) use = LandUse::Pavement;
            }
        }
    }
}

} // namespace

Rainfall constantStorm(double intensityMmPerHour, double durationSeconds) {
    if (!(intensityMmPerHour >= 0.0) || !std::isfinite(intensityMmPerHour))
        throw std::invalid_argument("Rain intensity must be a non-negative number (mm/h)");
    if (durationSeconds < 0.0) return Rainfall({{0.0, intensityMmPerHour}});
    return Rainfall({{0.0, intensityMmPerHour}, {durationSeconds, 0.0}});
}

Scenario makeScenario(const std::string& name, std::size_t rows, std::size_t cols,
                      double cellSize, const ScenarioOptions& options) {
    const auto pick = [](double value, double fallback) { return value >= 0.0 ? value : fallback; };
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
        const double depth = pick(options.reservoirDepth, 1.0);
        // Land use is for visualisation only: reservoir upstream, farmland downstream.
        scenario.landUse.assign(rows * cols, LandUse::Farmland);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols / 2; ++col) {
                scenario.grid.at(row, col).h = depth;
                scenario.landUse[row * cols + col] = LandUse::River;
            }
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
        scenario.landUse.assign(rows * cols, LandUse::Farmland); // sloping agricultural catchment
    } else if (name == "wet-dry") {
        scenario.rainfall = Rainfall({{0.0, 0.0}});
        scenario.config.endTime = 15.0;
        scenario.grid.at(rows / 2, cols / 2).h = 2e-6;
        scenario.grid.at(rows / 2, cols / 2).hu = 1e-7;
    } else if (name == "localized-refinement") {
        scenario.rainfall = Rainfall({{0.0, 0.0}});
        scenario.config.endTime = 10.0;
        const std::size_t rowBegin = rows / 3;
        const std::size_t rowEnd = std::min(rows, rowBegin + rows / 5);
        const std::size_t colBegin = cols / 3;
        const std::size_t colEnd = std::min(cols, colBegin + cols / 5);
        const double depth = pick(options.pondDepth, 0.5);
        scenario.landUse.assign(rows * cols, LandUse::Grass); // pond in a park
        for (std::size_t row = rowBegin; row < rowEnd; ++row)
            for (std::size_t col = colBegin; col < colEnd; ++col) {
                scenario.grid.at(row, col).h = depth;
                scenario.landUse[row * cols + col] = LandUse::River;
            }
    } else if (name == "urban-district") {
        buildUrbanDistrict(scenario, pick(options.surgeHeight, 3.4));
    } else if (name == "dam-breach") {
        buildDamBreach(scenario, pick(options.reservoirDepth, 3.0),
                       std::min(1.0, pick(options.breachWidth, 0.2)));
    } else if (name == "catchment") {
        buildCatchment(scenario);
    } else {
        throw std::invalid_argument("Unknown scenario: " + name);
    }
    return scenario;
}

} // namespace flood