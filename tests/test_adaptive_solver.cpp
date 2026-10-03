#include "grid/AdaptiveGrid.hpp"
#include "simulation/Scenarios.hpp"
#include "solver/AdaptiveSolver.hpp"
#include "solver/SerialSolver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef FLOOD_MASS_TOLERANCE
#define FLOOD_MASS_TOLERANCE 1e-8
#endif

namespace {

double maximumRegridVolumeDelta = 0.0;
double maximumInterfaceConservationError = 0.0;
double maximumMassBalanceResidual = 0.0;
double maximumDepthError = 0.0;
double maximumMomentumDifference = 0.0;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void recordDiagnostics(const flood::AdaptiveDiagnostics& diagnostics) {
    maximumRegridVolumeDelta = std::max(maximumRegridVolumeDelta,
                                       diagnostics.maximumRegridVolumeDelta);
    maximumInterfaceConservationError = std::max(
        maximumInterfaceConservationError,
        diagnostics.maximumInterfaceMassFluxResidual);
    maximumMassBalanceResidual = std::max(
        maximumMassBalanceResidual, std::abs(diagnostics.massBalanceResidual));
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
    for (const char* name : {"flat-basin", "slope", "dam-break",
                             "rain-drain", "wet-dry"}) {
        auto referenceScenario = flood::makeScenario(name, 8, 10, 1.0);
        referenceScenario.config.endTime = std::min(referenceScenario.config.endTime, 1.0);
        const auto referenceDiagnostics = flood::SerialSolver().run(
            referenceScenario.grid, referenceScenario.rainfall, referenceScenario.config);

        auto adaptiveScenario = flood::makeScenario(name, 8, 10, 1.0);
        adaptiveScenario.config.endTime = referenceScenario.config.endTime;
        flood::AdaptiveGridConfig gridConfig;
        gridConfig.patchRows = 8;
        gridConfig.patchCols = 10;
        gridConfig.maxRefinementLevel = 0;
        flood::AdaptiveGrid adaptiveGrid(adaptiveScenario.grid, gridConfig);
        const auto adaptiveDiagnostics = flood::AdaptiveSolver().run(
            adaptiveGrid, adaptiveScenario.rainfall, adaptiveScenario.config);
        recordDiagnostics(adaptiveDiagnostics);
        require(adaptiveDiagnostics.refinedPatches == 0,
                std::string(name) + ": max_level=0 must disable refinement");
        require(adaptiveDiagnostics.activeLeafCells == referenceScenario.grid.size(),
                std::string(name) + ": active cell count differs from serial");
        double scenarioDepthError = 0.0;
        double maximumHuDifference = 0.0;
        double maximumHvDifference = 0.0;
        adaptiveGrid.forEachActiveCell([&](flood::AdaptiveCellId id, const flood::Cell& cell) {
            const auto& expected = referenceScenario.grid.at(id.row, id.col);
            scenarioDepthError = std::max(scenarioDepthError, std::abs(expected.h - cell.h));
            maximumHuDifference = std::max(maximumHuDifference,
                                           std::abs(expected.hu - cell.hu));
            maximumHvDifference = std::max(maximumHvDifference,
                                           std::abs(expected.hv - cell.hv));
        });
        maximumDepthError = std::max(maximumDepthError, scenarioDepthError);
        maximumMomentumDifference = std::max(
            maximumMomentumDifference,
            std::max(maximumHuDifference, maximumHvDifference));
        require(scenarioDepthError < 2e-12 && maximumHuDifference < 2e-12 &&
                maximumHvDifference < 2e-12,
                std::string(name) + ": no-refinement solution differs from SerialSolver");
        requireNear(adaptiveDiagnostics.finalWaterVolume,
                    referenceDiagnostics.storedVolume, 1e-10,
                    std::string(name) + ": no-refinement volume differs from SerialSolver");
        requireNear(adaptiveDiagnostics.massBalanceResidual,
                    referenceDiagnostics.massResidual, 1e-10,
                    std::string(name) + ": no-refinement residual differs from SerialSolver");
    }
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
    auto levelZeroScenario = lakeAtRestScenario();
    flood::AdaptiveGridConfig levelZeroConfig;
    levelZeroConfig.patchRows = 4;
    levelZeroConfig.patchCols = 4;
    levelZeroConfig.maxRefinementLevel = 0;
    flood::AdaptiveGrid levelZeroGrid(levelZeroScenario.grid, levelZeroConfig);
    const auto levelZeroDiagnostics = flood::AdaptiveSolver().run(
        levelZeroGrid, levelZeroScenario.rainfall, levelZeroScenario.config);
    recordDiagnostics(levelZeroDiagnostics);
    require(levelZeroDiagnostics.maximumVelocity < 1e-10,
            "Level-0 lake-at-rest generated spurious motion");

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
    recordDiagnostics(diagnostics);
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
    require(!diagnostics.regridEvents.empty(),
            "Adaptive lake-at-rest did not record regrid transitions");
    for (const auto& event : diagnostics.regridEvents) {
        require(std::abs(event.waterVolumeDelta) < 1e-11 &&
                std::abs(event.huIntegralDelta) < 1e-11 &&
                std::abs(event.hvIntegralDelta) < 1e-11,
                "Lake-at-rest regrid transition changed a conserved integral");
    }
    require(diagnostics.maximumRegridVolumeDelta < 1e-11,
            "Maximum regrid volume delta exceeds tolerance");
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
    recordDiagnostics(diagnostics);
    require(diagnostics.coarsenedPatches > 0,
            "Quiescent refined region did not coarsen");
    require(adaptiveGrid.patch(0).isActiveLeaf(),
            "Coarsening did not reactivate the parent patch");
    require(std::abs(diagnostics.massBalanceResidual) < FLOOD_MASS_TOLERANCE,
            "Refinement/coarsening changed water volume");
    for (const auto& event : diagnostics.regridEvents)
        require(std::abs(event.waterVolumeDelta) < 1e-11 &&
                std::abs(event.huIntegralDelta) < 1e-11 &&
                std::abs(event.hvIntegralDelta) < 1e-11,
                "Solver regrid event changed water/momentum integrals");
}

void nestedNonuniformTransferSequence() {
    flood::Grid base(4, 4, 1.0, 1.0);
    for (std::size_t row = 0; row < base.rows(); ++row)
        for (std::size_t col = 0; col < base.cols(); ++col)
            base.at(row, col) = {0.1 * static_cast<double>(row + col),
                0.2 + 0.01 * static_cast<double>(row * col),
                -0.3 + 0.02 * static_cast<double>(row + col),
                0.4 + 0.03 * static_cast<double>(row + col)};
    flood::AdaptiveGridConfig config;
    config.patchRows = 4;
    config.patchCols = 4;
    config.maxRefinementLevel = 2;
    flood::AdaptiveGrid adaptive(base, config);
    const auto expected = integrals(adaptive);
    const auto level1 = adaptive.refine(0);
    require(adaptive.activeCellCount() == 64 && adaptive.patch(0).isCovered(),
            "Level-0 to level-1 transition double-counted or omitted cells");
    const auto level2 = adaptive.refine(level1.front());
    require(adaptive.activeCellCount() == 112 &&
            adaptive.patch(level1.front()).isCovered(),
            "Level-1 to level-2 transition double-counted or omitted cells");
    for (std::size_t child = 0; child < level2.size(); ++child) {
        auto& cells = adaptive.patch(level2[child]).grid().cells();
        for (std::size_t index = 0; index < cells.size(); ++index) {
            const double perturbation = 0.001 * static_cast<double>(child + index + 1);
            cells[index].h += perturbation;
            cells[index].hu -= 2.0 * perturbation;
            cells[index].hv += 3.0 * perturbation;
            cells[index].bed += 0.5 * perturbation;
        }
    }
    const auto perturbed = integrals(adaptive);
    adaptive.coarsen(level1.front());
    require(adaptive.activeCellCount() == 64 &&
            adaptive.patch(level1.front()).isActiveLeaf(),
            "Level-2 to level-1 transition did not restore leaf coverage");
    const auto afterToLevel1 = integrals(adaptive);
    for (std::size_t component = 0; component < expected.size(); ++component)
        require(std::abs(afterToLevel1[component] - perturbed[component]) < 1e-11L,
                "Level-2 to level-1 coarsening changed a conserved integral");
    require(adaptive.patch(level1.front()).isActiveLeaf(),
            "Level-1 parent did not reactivate after coarsening");
    adaptive.coarsen(0);
    const auto afterToLevel0 = integrals(adaptive);
    require(adaptive.activeCellCount() == 16 && adaptive.patch(0).isActiveLeaf(),
            "Level-1 to level-0 transition did not restore leaf coverage");
    for (std::size_t component = 0; component < expected.size(); ++component)
        require(std::abs(afterToLevel0[component] - perturbed[component]) < 1e-11L,
                "Level-1 to level-0 coarsening changed a conserved integral");
    require(adaptive.leafCount() == 1 && adaptive.activeCellCount() == 16,
            "Nested coarsening did not restore one level-0 leaf patch");
}

void lakeAtRestRefineThenCoarsen() {
    flood::Grid base(8, 8, 1.0, 1.0);
    for (std::size_t row = 0; row < base.rows(); ++row)
        for (std::size_t col = 0; col < base.cols(); ++col) {
            base.at(row, col).bed = 0.001 * static_cast<double>(col);
            base.at(row, col).h = 0.5 - base.at(row, col).bed;
        }
    flood::AdaptiveGridConfig gridConfig;
    gridConfig.patchRows = 4;
    gridConfig.patchCols = 4;
    gridConfig.maxRefinementLevel = 1;
    flood::AdaptiveGrid adaptive(base, gridConfig);
    adaptive.refine(0);
    flood::SolverConfig solverConfig;
    solverConfig.endTime = 0.2;
    solverConfig.maxTimestep = 0.02;
    solverConfig.manningN = 0.0;
    flood::AdaptiveSolverOptions options;
    options.refineThreshold = 10.0;
    options.coarsenThreshold = 0.05;
    options.coarsenPersistence = 1;
    options.regridIntervalSteps = 1;
    const auto diagnostics = flood::AdaptiveSolver(options).run(
        adaptive, flood::Rainfall(), solverConfig);
    recordDiagnostics(diagnostics);
    require(diagnostics.coarsenedPatches > 0 && adaptive.patch(0).isActiveLeaf(),
            "Lake-at-rest refinement did not coarsen back to its parent");
    require(diagnostics.maximumVelocity < 1e-10 &&
            std::abs(diagnostics.massBalanceResidual) < FLOOD_MASS_TOLERANCE,
            "Refine/coarsen lake-at-rest generated motion or lost mass");
}

void shorelineLakeAtRestStress() {
    flood::Scenario scenario{flood::Grid(12, 12, 1.0, 1.0), flood::Rainfall(),
                             flood::SolverConfig(), "shoreline-lake-at-rest"};
    scenario.config.endTime = 0.25;
    scenario.config.manningN = 0.0;
    const double surface = 1.0;
    for (std::size_t row = 0; row < scenario.grid.rows(); ++row) {
        for (std::size_t col = 0; col < scenario.grid.cols(); ++col) {
            const double bed = 0.1 * static_cast<double>(col) +
                               0.03 * static_cast<double>(row);
            scenario.grid.at(row, col).bed = bed;
            scenario.grid.at(row, col).h = std::max(0.0, surface - bed);
        }
    }
    flood::AdaptiveGridConfig config;
    config.patchRows = 4;
    config.patchCols = 4;
    config.maxRefinementLevel = 2;
    flood::AdaptiveGrid adaptive(scenario.grid, config);
    flood::AdaptiveSolverOptions options;
    options.refineThreshold = 0.05;
    options.coarsenThreshold = 0.01;
    options.coarsenPersistence = 1;
    options.regridIntervalSteps = 1;
    const auto diagnostics = flood::AdaptiveSolver(options).run(
        adaptive, scenario.rainfall, scenario.config);
    recordDiagnostics(diagnostics);
    require(diagnostics.refinedPatches > 1,
            "Shoreline stress case did not create multiple refinement patches");
    require(diagnostics.minimumDepth >= 0.0 &&
            diagnostics.maximumVelocity < 1e-9,
            "Shoreline lake-at-rest produced invalid depth or spurious motion");
    require(std::abs(diagnostics.massBalanceResidual) < FLOOD_MASS_TOLERANCE,
            "Shoreline lake-at-rest conservation failed");
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
    recordDiagnostics(diagnostics);
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
    scenario.config.maxTimestep = 0.2;
    flood::AdaptiveGridConfig gridConfig;
    gridConfig.patchRows = 4;
    gridConfig.patchCols = 4;
    gridConfig.maxRefinementLevel = 1;
    flood::AdaptiveGrid adaptiveGrid(scenario.grid, gridConfig);
    const auto children = adaptiveGrid.refine(0);
    (void)children;
    flood::AdaptiveSolverOptions options;
    options.refineThreshold = 0.01;
    options.coarsenThreshold = 0.005;
    options.regridIntervalSteps = 10;
    const auto diagnostics = flood::AdaptiveSolver(options).run(
        adaptiveGrid, scenario.rainfall, scenario.config);
    recordDiagnostics(diagnostics);
    require(diagnostics.rainfallVolume > 0.0 &&
            diagnostics.infiltrationVolume > 0.0,
            "Adaptive rainfall/infiltration volume was not tracked");
    require(diagnostics.outflowVolume > 0.0,
            "Adaptive outflow boundary did not record any discharge");
    require(diagnostics.refinedPatches > 0,
            "Rainfall/drainage case did not exercise regridding");
    const double expectedFinalVolume = diagnostics.initialWaterVolume +
        diagnostics.rainfallVolume - diagnostics.infiltrationVolume -
        diagnostics.outflowVolume;
    requireNear(diagnostics.finalWaterVolume, expectedFinalVolume,
                FLOOD_MASS_TOLERANCE,
                "Rainfall/infiltration/outflow mass budget is inconsistent");
    require(std::abs(diagnostics.massBalanceResidual) < FLOOD_MASS_TOLERANCE,
            "Adaptive rainfall/drainage mass budget failed");
    for (const auto& event : diagnostics.regridEvents)
        require(std::abs(event.waterVolumeDelta) < 1e-10,
                "Rain/drain regridding changed water volume");
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
    recordDiagnostics(firstDiagnostics);
    recordDiagnostics(secondDiagnostics);
    require(firstDiagnostics.maximumInterfaceMassFluxResidual == 0.0,
            "Coarse/fine interface mass flux was not equal-and-opposite");
    require(firstDiagnostics.coarseFineInterfaceSegments > 0 &&
            firstDiagnostics.coarseFineIntegratedMassFlux > 0.0,
            "Coarse/fine interface was not exercised by the test");
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

void coarseFineOrientationAndBoundaryStress() {
    const auto runCase = [](flood::PatchId refinedPatch,
                            const std::string& label,
                            flood::BoundaryCondition boundary =
                                flood::BoundaryCondition::Closed) {
        flood::Grid base(8, 8, 1.0, 1.0);
        for (std::size_t row = 0; row < base.rows(); ++row) {
            for (std::size_t col = 0; col < base.cols(); ++col) {
                auto& cell = base.at(row, col);
                cell.h = 0.15 + 0.01 * static_cast<double>(row + 2 * col);
                cell.hu = 0.02 + 0.001 * static_cast<double>(row);
                cell.hv = -0.015 + 0.001 * static_cast<double>(col);
            }
        }
        flood::AdaptiveGridConfig gridConfig;
        gridConfig.patchRows = 4;
        gridConfig.patchCols = 4;
        gridConfig.maxRefinementLevel = 1;
        flood::AdaptiveGrid adaptive(base, gridConfig);
        adaptive.refine(refinedPatch);
        flood::SolverConfig solverConfig;
        solverConfig.endTime = 0.02;
        solverConfig.boundary = boundary;
        flood::AdaptiveSolverOptions options;
        options.refineThreshold = 10.0;
        options.regridIntervalSteps = 10;
        const auto diagnostics = flood::AdaptiveSolver(options).run(
            adaptive, flood::Rainfall(), solverConfig);
        recordDiagnostics(diagnostics);
        require(diagnostics.coarseFineInterfaceSegments > 0,
                label + ": no coarse/fine interface segments were measured");
        require(diagnostics.maximumInterfaceMassFluxResidual == 0.0,
                label + ": interface contributions were not equal-and-opposite");
        require(diagnostics.minimumDepth >= -1e-12,
                label + ": negative water depth at coarse/fine interface");
        require(std::abs(diagnostics.massBalanceResidual) < FLOOD_MASS_TOLERANCE,
                label + ": mass budget failed");
        return diagnostics;
    };
    const auto fineCoarseX = runCase(0, "fine|coarse x-interface");
    const auto coarseFineX = runCase(1, "coarse|fine x-interface");
    const auto fineCoarseY = runCase(0, "fine|coarse y-interface");
    const auto coarseFineY = runCase(2, "coarse|fine y-interface");
    const auto corner = runCase(0, "refined patch corner");
    const auto physicalBoundary = runCase(0, "refined physical boundary",
                                          flood::BoundaryCondition::Outflow);
    require(fineCoarseX.coarseFineIntegratedMassFlux > 0.0 &&
            coarseFineX.coarseFineIntegratedMassFlux > 0.0 &&
            fineCoarseY.coarseFineIntegratedMassFlux > 0.0 &&
            coarseFineY.coarseFineIntegratedMassFlux > 0.0 &&
            corner.coarseFineIntegratedMassFlux > 0.0 &&
            physicalBoundary.coarseFineIntegratedMassFlux > 0.0,
            "At least one orientation/corner/boundary case had no integrated interface flux");
}

void shallowInterfacePositivityAndMultiPatch() {
    flood::Grid base(12, 12, 1.0, 1.0);
    for (std::size_t row = 0; row < base.rows(); ++row) {
        for (std::size_t col = 0; col < base.cols(); ++col) {
            auto& cell = base.at(row, col);
            cell.h = ((row >= 2 && row <= 3 && col >= 2 && col <= 3) ||
                      (row >= 8 && row <= 9 && col >= 8 && col <= 9)) ? 2e-5 : 0.0;
            cell.hu = cell.h > 0.0 ? 1e-5 : 0.0;
            cell.hv = cell.h > 0.0 ? -5e-6 : 0.0;
        }
    }
    flood::AdaptiveGridConfig config;
    config.patchRows = 4;
    config.patchCols = 4;
    config.maxRefinementLevel = 1;
    flood::AdaptiveGrid adaptive(base, config);
    flood::SolverConfig solverConfig;
    solverConfig.endTime = 0.04;
    solverConfig.maxTimestep = 0.001;
    solverConfig.manningN = 0.0;
    flood::AdaptiveSolverOptions options;
    options.refineThreshold = 0.001;
    options.coarsenThreshold = 0.0005;
    options.regridIntervalSteps = 1;
    const auto diagnostics = flood::AdaptiveSolver(options).run(
        adaptive, flood::Rainfall(), solverConfig);
    recordDiagnostics(diagnostics);
    require(diagnostics.refinedPatches >= 2,
            "Separated wet regions did not produce multiple refined patches (count=" +
                std::to_string(diagnostics.refinedPatches) + ")");
    require(diagnostics.minimumDepth >= -1e-12,
            "Shallow interface case produced negative water depth");
    require(std::abs(diagnostics.massBalanceResidual) < FLOOD_MASS_TOLERANCE,
            "Shallow interface limiter broke conservation");
    std::size_t wet = 0;
    adaptive.forEachActiveCell([&](flood::AdaptiveCellId, const flood::Cell& cell) {
        require(std::isfinite(cell.h) && std::isfinite(cell.hu) &&
                std::isfinite(cell.hv), "Multi-patch state contains NaNs");
        if (cell.h > 1e-6) ++wet;
        else require(cell.hu == 0.0 && cell.hv == 0.0,
                     "Momentum remains in a dry adaptive cell");
    });
    require(wet > 0, "Initially wet areas disappeared unexpectedly");
    std::set<flood::PatchId> refinedBasePatches;
    for (const auto& event : diagnostics.regridEvents) {
        if (event.operation == flood::AdaptiveRegridOperation::Refine &&
            adaptive.patch(event.patchId).level() == 0)
            refinedBasePatches.insert(event.patchId);
    }
    require(refinedBasePatches.size() >= 2,
            "Spatially separated wet regions did not refine distinct base patches");
    for (const flood::PatchId id : adaptive.activePatches()) {
        for (const auto& neighbor : adaptive.patch(id).neighbors()) {
            if (!neighbor.patchId) continue;
            const auto leftLevel = adaptive.patch(id).level();
            const auto rightLevel = adaptive.patch(*neighbor.patchId).level();
            require(leftLevel > rightLevel ? leftLevel - rightLevel <= 1
                                           : rightLevel - leftLevel <= 1,
                    "Multi-patch topology violates 2:1 balance");
        }
    }
}

void damBreakDryToWetFront() {
    auto scenario = flood::makeScenario("dam-break", 8, 16, 1.0);
    scenario.config.endTime = 0.5;
    flood::AdaptiveGridConfig config;
    config.patchRows = 4;
    config.patchCols = 4;
    config.maxRefinementLevel = 1;
    flood::AdaptiveGrid adaptive(scenario.grid, config);
    flood::AdaptiveSolverOptions options;
    options.refineThreshold = 0.05;
    options.coarsenThreshold = 0.02;
    options.regridIntervalSteps = 1;
    const auto diagnostics = flood::AdaptiveSolver(options).run(
        adaptive, scenario.rainfall, scenario.config);
    recordDiagnostics(diagnostics);
    std::size_t wetRightCells = 0;
    adaptive.forEachActiveCell([&](flood::AdaptiveCellId id, const flood::Cell& cell) {
        const auto& patch = adaptive.patch(id.patchId);
        double scale = 1.0;
        for (std::size_t level = patch.level();
             level < adaptive.config().maxRefinementLevel; ++level)
            scale *= static_cast<double>(adaptive.config().refinementRatio);
        const double centerX =
            (static_cast<double>(patch.colOriginFineUnits()) +
             (static_cast<double>(id.col) + 0.5) * scale) *
            adaptive.dx0() /
            std::pow(static_cast<double>(adaptive.config().refinementRatio),
                     static_cast<double>(adaptive.config().maxRefinementLevel));
        if (centerX > 8.0 && cell.h > scenario.config.dryDepth) ++wetRightCells;
        require(cell.h >= -1e-12 && std::isfinite(cell.h),
                "Dam-break adaptive front produced invalid depth");
    });
    require(wetRightCells > 0,
            "Adaptive dam-break did not propagate water into initially dry cells");
    require(std::abs(diagnostics.massBalanceResidual) < FLOOD_MASS_TOLERANCE,
            "Adaptive dam-break front failed conservation");
}

} // namespace

int main() {
    try {
        noRefinementMatchesSerial();
        forcedRefinementAndLakeAtRest();
        coarseningAndRefinementConservation();
        nestedNonuniformTransferSequence();
        shorelineLakeAtRestStress();
        lakeAtRestRefineThenCoarsen();
        wetDryStability();
        rainfallDrainageBudget();
        coarseFineMassFluxAndDeterminism();
        coarseFineOrientationAndBoundaryStress();
        shallowInterfacePositivityAndMultiPatch();
        damBreakDryToWetFront();
        std::cout << std::setprecision(17)
                  << "adaptive solver conservation, balance, stability, and determinism checks passed\n"
                  << "maximum_regrid_volume_delta=" << maximumRegridVolumeDelta << '\n'
                  << "maximum_interface_conservation_error="
                  << maximumInterfaceConservationError << '\n'
                  << "maximum_mass_balance_residual=" << maximumMassBalanceResidual << '\n'
                  << "maximum_uniform_reference_depth_error=" << maximumDepthError << '\n'
                  << "maximum_uniform_reference_momentum_difference="
                  << maximumMomentumDifference << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
