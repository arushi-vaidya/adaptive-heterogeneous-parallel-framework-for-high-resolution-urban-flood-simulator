#include "simulation/Scenarios.hpp"
#include "solver/SerialSolver.hpp"
#include "io/Output.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

#ifndef FLOOD_MASS_TOLERANCE
#define FLOOD_MASS_TOLERANCE 1e-8
#endif

int main() {
    try {
        auto scenario = flood::makeScenario("flat-basin", 8, 9, 2.0);
        const auto result = flood::SerialSolver().run(
            scenario.grid, scenario.rainfall, scenario.config);
        const double expected = (30.0 / 1000.0 / 3600.0) * result.time * 8.0 * 9.0 * 4.0;
        if (std::abs(result.storedVolume - expected) > 1e-9) {
            throw std::runtime_error("Flat-basin rainfall volume mismatch");
        }
        if (std::abs(result.massResidual) > FLOOD_MASS_TOLERANCE) {
            throw std::runtime_error("Flat-basin mass balance failed");
        }
        const auto loadedTerrain = flood::readTerrainCsv("tests/data/terrain.csv", 2.0, 2.0);
        if (loadedTerrain.rows() != 2 || loadedTerrain.cols() != 3 ||
            std::abs(loadedTerrain.at(1, 2).bed - 5.0) > 1e-12) {
            throw std::runtime_error("Terrain CSV loading failed");
        }
        const auto loadedRainfall = flood::readRainfallCsv("tests/data/rainfall.csv");
        if (std::abs(loadedRainfall.metersPerSecond(0.0) - 1e-5) > 1e-12 ||
            std::abs(loadedRainfall.metersPerSecond(60.0) - 2e-5) > 1e-12) {
            throw std::runtime_error("Rainfall CSV loading or unit conversion failed");
        }
        {
            std::ofstream crlf("test-crlf.csv", std::ios::binary);
            crlf << "0.25\r\n";
        }
        const auto crlfTerrain = flood::readTerrainCsv("test-crlf.csv", 1.0, 1.0);
        std::filesystem::remove("test-crlf.csv");
        if (std::abs(crlfTerrain.at(0, 0).bed - 0.25) > 1e-12) {
            throw std::runtime_error("CRLF terrain CSV loading failed");
        }
        auto slope = flood::makeScenario("slope", 6, 10, 1.0);
        const double initialLeftDepth = slope.grid.at(2, 0).h;
        const auto slopeResult = flood::SerialSolver().run(
            slope.grid, slope.rainfall, slope.config);
        if (slope.grid.at(2, 1).h <= 0.0 || slope.grid.at(2, 0).h >= initialLeftDepth ||
            std::abs(slopeResult.massResidual) > FLOOD_MASS_TOLERANCE) {
            throw std::runtime_error("Slope propagation or mass balance failed");
        }
        flood::Grid equilibrium(4, 7, 1.0, 1.0);
        for (std::size_t row = 0; row < equilibrium.rows(); ++row) {
            for (std::size_t col = 0; col < equilibrium.cols(); ++col) {
                equilibrium.at(row, col).bed = col < 3 ? 0.0 : 0.5;
                equilibrium.at(row, col).h = 1.0 - equilibrium.at(row, col).bed;
            }
        }
        flood::SolverConfig equilibriumConfig;
        equilibriumConfig.endTime = 1.0;
        const auto equilibriumMetrics = flood::SerialSolver().run(
            equilibrium, flood::Rainfall(), equilibriumConfig);
        if (equilibriumMetrics.maxVelocity > 1e-10 || equilibriumMetrics.maxDepth < 0.5) {
            throw std::runtime_error("Lake-at-rest equilibrium over bed step failed");
        }
        auto wetDry = flood::makeScenario("wet-dry", 4, 4, 1.0);
        const auto wetDryResult = flood::SerialSolver().run(
            wetDry.grid, wetDry.rainfall, wetDry.config);
        if (wetDryResult.minDepth < 0.0 || !std::isfinite(wetDryResult.maxVelocity)) {
            throw std::runtime_error("Wet/dry handling failed");
        }
        for (const auto& name : {"dam-break", "rain-drain"}) {
            auto caseRun = flood::makeScenario(name, 8, 10, 1.0);
            const auto metrics = flood::SerialSolver().run(
                caseRun.grid, caseRun.rainfall, caseRun.config);
            if (metrics.minDepth < 0.0 || !std::isfinite(metrics.maxVelocity) ||
                std::abs(metrics.massResidual) > FLOOD_MASS_TOLERANCE) {
                throw std::runtime_error(std::string(name) + " stability/mass check failed");
            }
            if (std::string(name) == "dam-break" && caseRun.grid.at(4, 5).h <= 0.0) {
                throw std::runtime_error("Dam-break did not propagate");
            }
            if (std::string(name) == "rain-drain" &&
                (metrics.rainfallVolume <= 0.0 || metrics.infiltrationVolume <= 0.0 ||
                 metrics.outflowVolume <= 0.0)) {
                throw std::runtime_error("Rain-drain sources/outflow were not accounted for");
            }
        }
        flood::Grid outputGrid(1, 1, 1.0, 1.0);
        outputGrid.at(0, 0).h = 0.25;
        flood::Diagnostics outputMetrics;
        flood::writeOutputs(outputGrid, outputMetrics, "test-output");
        if (!std::filesystem::exists("test-output/hu.csv") ||
            !std::filesystem::exists("test-output/hv.csv")) {
            throw std::runtime_error("Momentum output generation failed");
        }
        std::filesystem::remove_all("test-output");
        std::cout << "five scenarios, conservation, wet/dry, and output checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}