#include "grid/AdaptiveGrid.hpp"
#include "simulation/Scenarios.hpp"
#include "solver/AdaptiveSolver.hpp"
#include "solver/SerialSolver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef FLOOD_MASS_TOLERANCE
#define FLOOD_MASS_TOLERANCE 1e-8
#endif

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void requireNear(double actual, double expected, double tolerance,
                 const std::string& message) {
    if (std::abs(actual - expected) > tolerance)
        throw std::runtime_error(message + ": expected " + std::to_string(expected) +
                                 ", got " + std::to_string(actual));
}

std::array<long double, 4> integrals(const flood::AdaptiveGrid& grid) {
    std::array<long double, 4> result{};
    grid.forEachActiveCell([&](flood::AdaptiveCellId id, const flood::Cell& cell) {
        const auto& patch = grid.patch(id.patchId);
        const long double area =
            static_cast<long double>(patch.grid().dx()) * patch.grid().dy();
        result[0] += cell.h * area;
        result[1] += cell.hu * area;
        result[2] += cell.hv * area;
        result[3] += cell.bed * area;
    });
    return result;
}

void noRefinementMatchesSerial() {
    auto referenceScenario = flood::makeScenario("dam-break", 8, 10, 1.0);
    referenceScenario.config.endTime = 0.4;
    const auto referenceDiagnostics = flood::SerialSolver().run(
        referenceScenario.grid, referenceScenario.rainfall, referenceScenario.config);

    auto adaptiveScenario = flood::makeScenario("dam-break", 8, 10, 1.0);
    adaptiveScenario.config.endTime = 0.4;
    flood::AdaptiveGridConfig gridConfig;
    gridConfig.patchRows = 8;
    gridConfig.patchCols = 10;
    gridConfig.maxRefinementLevel = 0;
    flood::AdaptiveGrid adaptiveGrid(adaptiveScenario.grid, gridConfig);
    const auto adaptiveDiagnostics = flood::AdaptiveSolver().run(
        adaptiveGrid, adaptiveScenario.rainfall, adaptiveScenario.config);
    require(adaptiveDiagnostics.refinedPatches == 0,
            "Maximum refinement level zero must disable refinement");
    require(adaptiveDiagnostics.activeLeafCells == referenceScenario.grid.size(),
            "No-refinement adaptive active cell count differs from serial");
    double maximumDepthError = 0.0;
    double maximumMomentumError = 0.0;
    adaptiveGrid.forEachActiveCell([&](flood::AdaptiveCellId id, const flood::Cell& cell) {
        const auto& expected = referenceScenario.grid.at(id.row, id.col);
        maximumDepthError = std::max(maximumDepthError, std::abs(expected.h - cell.h));
        maximumMomentumError = std::max(maximumMomentumError,
            std::max(std::abs(expected.hu - cell.hu), std::abs(expected.hv - cell.hv)));
    });
    require(maximumDepthError < 2e-12 && maximumMomentumError < 2e-12,
            "No-refinement adaptive solution differs from SerialSolver");
    requireNear(adaptiveDiagnostics.finalWaterVolume,
                referenceDiagnostics.storedVolume, 1e-10,
                "No-refinement volume differs from SerialSolver");
}

flood::Scenario lakeAtRestScenario() {
    flood::Scenario scenario{flood::Grid(8, 8, 1.0, 1.0), flood::Rainfall(),
                             flood::SolverConfig(), "lake-at-rest-adaptive"};
    scenario.config.endTime = 0.3;
    scenario.config.manningN = 0.0;
    for (std::size_t row = 0; row < scenario.grid.rows(); ++row) {
        for (std::size_t col = 0; col < scenario.grid.cols(); ++col) {
            scenario.grid.at(row, col).bed = col < 4 ? 0.0 : 0.5;
            scenario.grid.at(row, col).h = 1.0 - scenario.grid.at(row, col).bed;
        }
    }
    return scenario;
}

void forcedRefinementAndLakeAtRest() {
    auto scenario = lakeAtRestScenario();
    flood::AdaptiveGridConfig gridConfig;
    gridConfig.patchRows = 4;
    gridConfig.patchCols = 4;
    gridConfig.maxRefinementLevel = 2;
    flood::AdaptiveGrid adaptiveGrid(scenario.grid, gridConfig);
    flood::AdaptiveSolverOptions options;
    options.refineThreshold = 0.10;
    options.coarsenThreshold = 0.05;
    options.regridIntervalSteps = 1;
    const auto diagnostics = flood::AdaptiveSolver(options).run(
        adaptiveGrid, scenario.rainfall, scenario.config);
    require(diagnostics.refinedPatches > 0,
            "Lake-at-rest depth jump did not trigger refinement");
    require(diagnostics.activeLeafCells > scenario.grid.size(),
            "Refinement did not increase the active leaf cell count");
    require(diagnostics.level1Cells > 0, "No level-1 leaf cells were created");
    require(diagnostics.level2Cells > 0, "No level-2 leaf cells were created");
    require(diagnostics.maximumDepth < 1.0 + 1e-12 &&
            diagnostics.minimumDepth >= 0.0,
            "Adaptive lake-at-rest produced an invalid depth");
    require(diagnostics.maximumVelocity < 1e-10,
            "Adaptive refinement introduced spurious lake-at-rest motion");
    require(std::abs(diagnostics.massBalanceResidual) < FLOOD_MASS_TOLERANCE,
            "Adaptive lake-at-rest conservation failed");
}

void coarseningAndRefinementConservation() {
    flood::Grid base(8, 8, 2.0, 3.0);
    for (auto& cell : base.cells()) cell = {0.5, 0.0, 0.0, 0.5};
    flood::AdaptiveGridConfig gridConfig;
    gridConfig.patchRows = 4;
    gridConfig.patchCols = 4;
    gridConfig.maxRefinementLevel = 1;
    flood::AdaptiveGrid adaptiveGrid(base, gridConfig);
    const auto before = integrals(adaptiveGrid);
    adaptiveGrid.refine(0);
    const auto refined = integrals(adaptiveGrid);
    for (std::size_t component = 0; component < before.size(); ++component)
        require(std::abs(before[component] - refined[component]) < 1e-11L,
                "Refinement changed a conserved domain integral");

    flood::SolverConfig solverConfig;
    solverConfig.endTime = 0.4;
    solverConfig.maxTimestep = 0.05;
    flood::AdaptiveSolverOptions options;
    options.refineThreshold = 10.0;
    options.coarsenThreshold = 0.05;
    options.coarsenPersistence = 1;
    options.regridIntervalSteps = 1;
    const auto diagnostics = flood::AdaptiveSolver(options).run(
        adaptiveGrid, flood::Rainfall(), solverConfig);
    require(diagnostics.coarsenedPatches > 0,
            "Quiescent refined region did not coarsen");
    require(adaptiveGrid.patch(0).isActiveLeaf(),
            "Coarsening did not reactivate the parent patch");
    require(std::abs(diagnostics.massBalanceResidual) < FLOOD_MASS_TOLERANCE,
            "Refinement/coarsening changed water volume");
}

void wetDryStability() {
    auto scenario = flood::makeScenario("wet-dry", 8, 8, 1.0);
    scenario.config.endTime = 0.5;
    flood::AdaptiveGridConfig gridConfig;
    gridConfig.patchRows = 4;
    gridConfig.patchCols = 4;
    gridConfig.maxRefinementLevel = 1;
    flood::AdaptiveGrid adaptiveGrid(scenario.grid, gridConfig);
    flood::AdaptiveSolverOptions options;
    options.refineThreshold = 0.01;
    options.coarsenThreshold = 0.005;
    options.regridIntervalSteps = 1;
    const auto diagnostics = flood::AdaptiveSolver(options).run(
        adaptiveGrid, scenario.rainfall, scenario.config);
    require(diagnostics.minimumDepth >= -1e-12,
            "Wet/dry adaptive run produced negative depth");
    require(std::isfinite(diagnostics.maximumVelocity) &&
            std::isfinite(diagnostics.massBalanceResidual),
            "Wet/dry adaptive run produced non-finite diagnostics");
    require(std::abs(diagnostics.massBalanceResidual) < FLOOD_MASS_TOLERANCE,
            "Wet/dry adaptive mass balance failed");
    adaptiveGrid.forEachActiveCell([](flood::AdaptiveCellId, const flood::Cell& cell) {
        require(std::isfinite(cell.h) && std::isfinite(cell.hu) &&
                std::isfinite(cell.hv), "Wet/dry adaptive state contains NaNs");
        if (cell.h <= 1e-6)
            require(cell.hu == 0.0 && cell.hv == 0.0,
                    "Dry-cell momentum was not cleared");
    });
}

void rainfallDrainageBudget() {
    auto scenario = flood::makeScenario("rain-drain", 8, 8, 1.0);
    scenario.config.endTime = 1.0;
    flood::AdaptiveGridConfig gridConfig;
    gridConfig.patchRows = 4;
    gridConfig.patchCols = 4;
    gridConfig.maxRefinementLevel = 1;
    flood::AdaptiveGrid adaptiveGrid(scenario.grid, gridConfig);
    const auto children = adaptiveGrid.refine(0);
    (void)children;
    flood::AdaptiveSolverOptions options;
    options.refineThreshold = 10.0;
    options.regridIntervalSteps = 1000;
    const auto diagnostics = flood::AdaptiveSolver(options).run(
        adaptiveGrid, scenario.rainfall, scenario.config);
    require(diagnostics.rainfallVolume > 0.0 &&
            diagnostics.infiltrationVolume > 0.0,
            "Adaptive rainfall/infiltration volume was not tracked");
    require(diagnostics.outflowVolume >= 0.0,
            "Adaptive outflow volume is invalid");
    require(std::abs(diagnostics.massBalanceResidual) < FLOOD_MASS_TOLERANCE,
            "Adaptive rainfall/drainage mass budget failed");
}

void coarseFineMassFluxAndDeterminism() {
    flood::Grid base(8, 8, 1.0, 1.0);
    for (std::size_t row = 0; row < base.rows(); ++row) {
        for (std::size_t col = 0; col < base.cols(); ++col) {
            auto& cell = base.at(row, col);
            cell.h = col < 4 ? 0.5 : 0.25;
            cell.hu = col < 4 ? 0.03 : 0.01;
        }
    }
    flood::AdaptiveGridConfig gridConfig;
    gridConfig.patchRows = 4;
    gridConfig.patchCols = 4;
    gridConfig.maxRefinementLevel = 1;
    flood::AdaptiveGrid first(base, gridConfig);
    flood::AdaptiveGrid second(base, gridConfig);
    first.refine(0);
    second.refine(0);
    flood::SolverConfig solverConfig;
    solverConfig.endTime = 0.05;
    solverConfig.manningN = 0.0;
    flood::AdaptiveSolverOptions options;
    options.refineThreshold = 10.0;
    options.regridIntervalSteps = 100;
    const auto firstDiagnostics = flood::AdaptiveSolver(options).run(
        first, flood::Rainfall(), solverConfig);
    const auto secondDiagnostics = flood::AdaptiveSolver(options).run(
        second, flood::Rainfall(), solverConfig);
    require(firstDiagnostics.maximumInterfaceMassFluxResidual == 0.0,
            "Coarse/fine interface mass flux was not equal-and-opposite");
    require(std::abs(firstDiagnostics.massBalanceResidual) < FLOOD_MASS_TOLERANCE,
            "Coarse/fine interface did not preserve total water");
    require(first.activePatches() == second.activePatches() &&
            firstDiagnostics.refinedPatches == secondDiagnostics.refinedPatches &&
            firstDiagnostics.coarsenedPatches == secondDiagnostics.coarsenedPatches,
            "Repeated adaptive runs produced different refinement histories");
    std::vector<std::array<double, 3>> firstState, secondState;
    first.forEachActiveCell([&](flood::AdaptiveCellId, const flood::Cell& cell) {
        firstState.push_back({cell.h, cell.hu, cell.hv});
    });
    second.forEachActiveCell([&](flood::AdaptiveCellId, const flood::Cell& cell) {
        secondState.push_back({cell.h, cell.hu, cell.hv});
    });
    require(firstState.size() == secondState.size(),
            "Repeated adaptive runs produced different leaf counts");
    for (std::size_t index = 0; index < firstState.size(); ++index)
        for (std::size_t component = 0; component < firstState[index].size(); ++component)
            requireNear(firstState[index][component], secondState[index][component],
                        1e-14, "Repeated adaptive state is not deterministic");
}

} // namespace

int main() {
    try {
        noRefinementMatchesSerial();
        forcedRefinementAndLakeAtRest();
        coarseningAndRefinementConservation();
        wetDryStability();
        rainfallDrainageBudget();
        coarseFineMassFluxAndDeterminism();
        std::cout << "adaptive solver conservation, balance, stability, and determinism checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
