#include "grid/AdaptiveGrid.hpp"
#include "simulation/Scenarios.hpp"
#include "solver/AdaptiveOpenMPSolver.hpp"
#include "solver/AdaptiveSolver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef FLOOD_MASS_TOLERANCE
#define FLOOD_MASS_TOLERANCE 1e-8
#endif

namespace {

constexpr double stateTolerance = 2e-11;
constexpr double diagnosticTolerance = 2e-11;

struct TestCase {
    std::string name;
    flood::Scenario scenario;
    flood::AdaptiveGridConfig gridConfig;
    flood::AdaptiveSolverOptions options;
    std::size_t nestedRefinementLevel = 0;
    std::vector<flood::PatchId> additionalRefinements;
};

struct Outcome {
    flood::AdaptiveDiagnostics diagnostics;
    std::vector<flood::AdaptiveCellId> ids;
    std::vector<std::array<double, 3>> state;
    std::vector<flood::PatchId> activePatches;
};

struct MaximumDifferences {
    double h = 0.0;
    double hu = 0.0;
    double hv = 0.0;
    double waterVolume = 0.0;
    double massResidual = 0.0;
    double maximumMassBalanceResidual = 0.0;
    double maximumVelocity = 0.0;
    double regridVolumeDelta = 0.0;
    double interfaceResidual = 0.0;
    double maximumInterfaceMassFluxResidual = 0.0;
    double integratedInterfaceFlux = 0.0;
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

TestCase makeCase(const std::string& name, const std::string& scenarioName,
                  std::size_t maxLevel = 2) {
    auto scenario = flood::makeScenario(scenarioName, 8, 8, 1.0);
    scenario.config.endTime = std::min(scenario.config.endTime, 0.3);
    scenario.config.maxTimestep = std::min(scenario.config.maxTimestep, 0.02);
    flood::AdaptiveGridConfig gridConfig;
    gridConfig.patchRows = 4;
    gridConfig.patchCols = 4;
    gridConfig.maxRefinementLevel = maxLevel;
    return {name, std::move(scenario), gridConfig, {}, 0, {}};
}

TestCase lakeAtRestCase() {
    flood::Scenario scenario{flood::Grid(8, 8, 1.0, 1.0), flood::Rainfall(),
                             flood::SolverConfig(), "openmp-lake-at-rest"};
    scenario.config.endTime = 0.2;
    scenario.config.maxTimestep = 0.02;
    scenario.config.manningN = 0.0;
    for (std::size_t row = 0; row < scenario.grid.rows(); ++row)
        for (std::size_t col = 0; col < scenario.grid.cols(); ++col) {
            scenario.grid.at(row, col).bed = col < 4 ? 0.0 : 0.5;
            scenario.grid.at(row, col).h = 1.0 - scenario.grid.at(row, col).bed;
        }
    flood::AdaptiveGridConfig gridConfig;
    gridConfig.patchRows = 4;
    gridConfig.patchCols = 4;
    gridConfig.maxRefinementLevel = 2;
    flood::AdaptiveSolverOptions options;
    options.refineThreshold = 0.10;
    options.coarsenThreshold = 0.05;
    options.regridIntervalSteps = 1;
    return {"lake-at-rest", std::move(scenario), gridConfig, options, 0, {}};
}

Outcome runCase(const TestCase& testCase, int threads, bool parallel) {
    flood::AdaptiveGrid grid(testCase.scenario.grid, testCase.gridConfig);
    for (std::size_t level = 0; level < testCase.nestedRefinementLevel; ++level) {
        const auto children = grid.refine(level == 0 ? 0 : grid.patch(0).childIds().front());
        require(!children.empty(), testCase.name + ": test setup refinement had no children");
    }
    for (const flood::PatchId patchId : testCase.additionalRefinements)
        grid.refine(patchId);

    const auto diagnostics = parallel
        ? flood::AdaptiveOpenMPSolver(testCase.options, threads).run(
              grid, testCase.scenario.rainfall, testCase.scenario.config)
        : flood::AdaptiveSolver(testCase.options).run(
              grid, testCase.scenario.rainfall, testCase.scenario.config);
    Outcome outcome{diagnostics, {}, {}, grid.activePatches()};
    grid.forEachActiveCell([&](flood::AdaptiveCellId id, const flood::Cell& cell) {
        outcome.ids.push_back(id);
        outcome.state.push_back({cell.h, cell.hu, cell.hv});
    });
    return outcome;
}

void compareOutcomes(const TestCase& testCase, const Outcome& reference,
                     const Outcome& actual, MaximumDifferences& maxima) {
    require(reference.ids == actual.ids && reference.activePatches == actual.activePatches,
            testCase.name + ": active leaf ordering/topology differs");
    require(reference.state.size() == actual.state.size(),
            testCase.name + ": active cell count differs");
    require(reference.diagnostics.refinedPatches == actual.diagnostics.refinedPatches &&
            reference.diagnostics.coarsenedPatches == actual.diagnostics.coarsenedPatches,
            testCase.name + ": refinement/coarsening counts differ");
    require(reference.diagnostics.regridEvents.size() ==
                actual.diagnostics.regridEvents.size(),
            testCase.name + ": regrid event count differs");
    require(reference.diagnostics.coarseFineInterfaceSegments ==
                actual.diagnostics.coarseFineInterfaceSegments,
            testCase.name + ": coarse/fine interface segment count differs");

    for (std::size_t index = 0; index < reference.state.size(); ++index) {
        for (const double value : actual.state[index])
            require(std::isfinite(value), testCase.name + ": non-finite active state");
        require(actual.state[index][0] >= -1e-12,
                testCase.name + ": negative active-cell depth");
        maxima.h = std::max(maxima.h,
            std::abs(reference.state[index][0] - actual.state[index][0]));
        maxima.hu = std::max(maxima.hu,
            std::abs(reference.state[index][1] - actual.state[index][1]));
        maxima.hv = std::max(maxima.hv,
            std::abs(reference.state[index][2] - actual.state[index][2]));
    }
    maxima.waterVolume = std::max(maxima.waterVolume,
        std::abs(reference.diagnostics.finalWaterVolume -
                 actual.diagnostics.finalWaterVolume));
    maxima.massResidual = std::max(maxima.massResidual,
        std::abs(reference.diagnostics.massBalanceResidual -
                 actual.diagnostics.massBalanceResidual));
    maxima.maximumMassBalanceResidual = std::max(
        maxima.maximumMassBalanceResidual,
        std::max(std::abs(reference.diagnostics.massBalanceResidual),
                 std::abs(actual.diagnostics.massBalanceResidual)));
    maxima.maximumVelocity = std::max(maxima.maximumVelocity,
        std::abs(reference.diagnostics.maximumVelocity -
                 actual.diagnostics.maximumVelocity));
    maxima.regridVolumeDelta = std::max(maxima.regridVolumeDelta,
        std::abs(reference.diagnostics.maximumRegridVolumeDelta -
                 actual.diagnostics.maximumRegridVolumeDelta));
    maxima.interfaceResidual = std::max(maxima.interfaceResidual,
        std::abs(reference.diagnostics.maximumInterfaceMassFluxResidual -
                 actual.diagnostics.maximumInterfaceMassFluxResidual));
    maxima.maximumInterfaceMassFluxResidual = std::max(
        maxima.maximumInterfaceMassFluxResidual,
        std::max(reference.diagnostics.maximumInterfaceMassFluxResidual,
                 actual.diagnostics.maximumInterfaceMassFluxResidual));
    maxima.integratedInterfaceFlux = std::max(maxima.integratedInterfaceFlux,
        std::abs(reference.diagnostics.coarseFineIntegratedMassFlux -
                 actual.diagnostics.coarseFineIntegratedMassFlux));

    require(maxima.h <= stateTolerance && maxima.hu <= stateTolerance &&
            maxima.hv <= stateTolerance,
            testCase.name + ": adaptive OpenMP state differs from serial reference");
    require(maxima.waterVolume <= diagnosticTolerance &&
            maxima.massResidual <= diagnosticTolerance &&
            maxima.maximumVelocity <= diagnosticTolerance,
            testCase.name + ": adaptive OpenMP diagnostics differ from serial reference");
    require(maxima.regridVolumeDelta <= diagnosticTolerance &&
            maxima.interfaceResidual <= diagnosticTolerance &&
            maxima.integratedInterfaceFlux <= diagnosticTolerance,
            testCase.name + ": adaptive OpenMP conservation diagnostics differ");
    require(actual.diagnostics.maximumRegridVolumeDelta <= 1e-11 &&
            actual.diagnostics.maximumInterfaceMassFluxResidual == 0.0,
            testCase.name + ": adaptive conservation diagnostic exceeded its gate");

    for (std::size_t index = 0;
         index < reference.diagnostics.regridEvents.size(); ++index) {
        const auto& expected = reference.diagnostics.regridEvents[index];
        const auto& observed = actual.diagnostics.regridEvents[index];
        require(expected.operation == observed.operation &&
                expected.patchId == observed.patchId &&
                std::abs(expected.time - observed.time) <= diagnosticTolerance,
                testCase.name + ": regrid history differs");
        require(std::abs(expected.waterVolumeDelta - observed.waterVolumeDelta) <=
                    diagnosticTolerance &&
                std::abs(expected.huIntegralDelta - observed.huIntegralDelta) <=
                    diagnosticTolerance &&
                std::abs(expected.hvIntegralDelta - observed.hvIntegralDelta) <=
                    diagnosticTolerance,
                testCase.name + ": regrid conservation event differs");
    }
    require(actual.diagnostics.maximumInterfaceMassFluxResidual == 0.0,
            testCase.name + ": interface mass flux is not equal and opposite");
    require(std::abs(actual.diagnostics.massBalanceResidual) <=
                FLOOD_MASS_TOLERANCE,
            testCase.name + ": mass balance tolerance exceeded");
}

} // namespace

int main() {
    try {
        std::vector<TestCase> cases;
        for (const std::string scenario :
             {"flat-basin", "slope", "dam-break", "rain-drain", "wet-dry"})
            cases.push_back(makeCase(scenario, scenario));

        auto level1 = makeCase("forced-level-1", "slope", 1);
        level1.nestedRefinementLevel = 1;
        level1.options.refineThreshold = 1e9;
        level1.options.coarsenThreshold = 0.0;
        cases.push_back(level1);

        auto level2 = makeCase("forced-level-2", "dam-break", 2);
        level2.nestedRefinementLevel = 2;
        level2.options.refineThreshold = 1e9;
        level2.options.coarsenThreshold = 0.0;
        cases.push_back(level2);

        auto multiPatch = makeCase("multiple-patches", "wet-dry", 1);
        multiPatch.additionalRefinements = {0, 3};
        multiPatch.options.refineThreshold = 1e9;
        multiPatch.options.coarsenThreshold = 0.0;
        cases.push_back(multiPatch);

        auto coarsening = makeCase("regrid-coarsening", "flat-basin", 2);
        coarsening.nestedRefinementLevel = 2;
        coarsening.options.refineThreshold = 1e9;
        coarsening.options.coarsenThreshold = 0.1;
        coarsening.options.coarsenPersistence = 1;
        coarsening.options.regridIntervalSteps = 1;
        cases.push_back(coarsening);
        cases.push_back(lakeAtRestCase());

        MaximumDifferences maxima;
        for (const TestCase& testCase : cases) {
            const Outcome serial = runCase(testCase, 1, false);
            Outcome firstOpenMP;
            bool hasFirst = false;
            for (const int threads : {1, 2, 4}) {
                const Outcome parallel = runCase(testCase, threads, true);
                compareOutcomes(testCase, serial, parallel, maxima);
                if (hasFirst) compareOutcomes(testCase, firstOpenMP, parallel, maxima);
                else {
                    firstOpenMP = parallel;
                    hasFirst = true;
                }
                require(parallel.diagnostics.minimumDepth >= -1e-12,
                        testCase.name + ": negative water depth");
                if (testCase.name == "lake-at-rest")
                    require(parallel.diagnostics.maximumVelocity < 1e-10,
                            "Adaptive OpenMP lake-at-rest generated motion");
            }
            if (testCase.name == "forced-level-1")
                require(serial.diagnostics.coarseFineInterfaceSegments > 0,
                        "Forced level-1 case did not exercise a coarse/fine interface");
            if (testCase.name == "forced-level-2")
                require(serial.diagnostics.level2Cells > 0,
                        "Forced level-2 case did not retain level-2 leaves");
            if (testCase.name == "multiple-patches")
                require(serial.activePatches.size() > 4,
                        "Multiple-patch case did not retain separated refined regions");
            if (testCase.name == "regrid-coarsening")
                require(serial.diagnostics.coarsenedPatches > 0,
                        "Regrid/coarsening case did not coarsen");
        }

        const TestCase deterministicCase = lakeAtRestCase();
        const Outcome first = runCase(deterministicCase, 4, true);
        const Outcome second = runCase(deterministicCase, 4, true);
        compareOutcomes(deterministicCase, first, second, maxima);

        std::cout << std::setprecision(17)
                  << "adaptive OpenMP equivalence checks passed\n"
                  << "maximum_depth_difference=" << maxima.h << '\n'
                  << "maximum_hu_difference=" << maxima.hu << '\n'
                  << "maximum_hv_difference=" << maxima.hv << '\n'
                  << "maximum_water_volume_difference=" << maxima.waterVolume << '\n'
                  << "maximum_mass_balance_residual="
                  << maxima.maximumMassBalanceResidual << '\n'
                  << "maximum_mass_balance_residual_difference="
                  << maxima.massResidual << '\n'
                  << "maximum_velocity_difference=" << maxima.maximumVelocity << '\n'
                  << "maximum_regrid_volume_delta_difference="
                  << maxima.regridVolumeDelta << '\n'
                  << "maximum_interface_conservation_error="
                  << maxima.maximumInterfaceMassFluxResidual << '\n'
                  << "maximum_integrated_interface_flux_difference="
                  << maxima.integratedInterfaceFlux << '\n'
                  << "thread_counts=1,2,4\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
