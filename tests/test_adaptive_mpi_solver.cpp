#include "grid/AdaptiveGrid.hpp"
#include "simulation/Scenarios.hpp"
#include "solver/AdaptiveMpiSolver.hpp"
#include "solver/AdaptiveSolver.hpp"
#ifdef FLOOD_HAS_OPENMP
#include "solver/AdaptiveOpenMPSolver.hpp"
#endif

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef FLOOD_MASS_TOLERANCE
#define FLOOD_MASS_TOLERANCE 1e-8
#endif

namespace {

constexpr double stateTolerance = 2e-10;
constexpr double diagnosticTolerance = 2e-10;

struct TestCase {
    std::string name;
    flood::Scenario scenario;
    flood::AdaptiveGridConfig gridConfig;
    flood::AdaptiveSolverOptions options;
    flood::PatchId rootPatchToRefine = 0;
    std::size_t nestedRefinementLevel = 0;
    std::vector<flood::PatchId> additionalRefinements;
    bool requireCrossRankInterface = false;
    int requiredRemoteOrientation = 0;
};

struct Outcome {
    flood::AdaptiveDiagnostics diagnostics;
    std::vector<flood::AdaptiveCellId> ids;
    std::vector<std::array<double, 3>> states;
    std::vector<flood::PatchId> activePatches;
};

struct Differences {
    double h = 0.0;
    double hu = 0.0;
    double hv = 0.0;
    double volume = 0.0;
    double massResidual = 0.0;
    double maxVelocity = 0.0;
    double interfaceError = 0.0;
    double regridError = 0.0;
};

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

TestCase makeCase(const std::string& label, const std::string& scenarioName,
                  std::size_t rows = 8, std::size_t cols = 8,
                  std::size_t patchExtent = 4, std::size_t maxLevel = 2) {
    auto scenario = flood::makeScenario(scenarioName, rows, cols, 1.0);
    scenario.config.endTime = std::min(scenario.config.endTime, 0.25);
    scenario.config.maxTimestep = std::min(scenario.config.maxTimestep, 0.02);
    flood::AdaptiveGridConfig gridConfig;
    gridConfig.patchRows = patchExtent;
    gridConfig.patchCols = patchExtent;
    gridConfig.maxRefinementLevel = maxLevel;
    return {label, std::move(scenario), gridConfig, {}, 0, 0, {}, false, 0};
}

TestCase lakeAtRestCase() {
    flood::Scenario scenario{flood::Grid(8, 8, 1.0, 1.0), flood::Rainfall(),
                             flood::SolverConfig(), "adaptive-mpi-lake-at-rest"};
    scenario.config.endTime = 0.2;
    scenario.config.maxTimestep = 0.02;
    scenario.config.manningN = 0.0;
    for (std::size_t row = 0; row < scenario.grid.rows(); ++row)
        for (std::size_t col = 0; col < scenario.grid.cols(); ++col) {
            scenario.grid.at(row, col).bed = col < 4 ? 0.0 : 0.5;
            scenario.grid.at(row, col).h = 1.0 - scenario.grid.at(row, col).bed;
        }
    flood::AdaptiveGridConfig config;
    config.patchRows = 4;
    config.patchCols = 4;
    config.maxRefinementLevel = 2;
    flood::AdaptiveSolverOptions options;
    options.refineThreshold = 0.1;
    options.coarsenThreshold = 0.05;
    options.regridIntervalSteps = 1;
    return {"lake-at-rest", std::move(scenario), config, options, 0, 0, {}, false, 0};
}

void prepareGrid(flood::AdaptiveGrid& grid, const TestCase& testCase) {
    for (std::size_t level = 0; level < testCase.nestedRefinementLevel; ++level) {
        const flood::PatchId parent = level == 0
            ? testCase.rootPatchToRefine : grid.patch(testCase.rootPatchToRefine).childIds().front();
        check(!grid.refine(parent).empty(),
              testCase.name + ": initial refinement created no children");
    }
    for (const flood::PatchId patchId : testCase.additionalRefinements)
        check(!grid.refine(patchId).empty(),
              testCase.name + ": patch refinement created no children");
}

template <typename RunSolver>
Outcome runCase(const TestCase& testCase, RunSolver&& runSolver) {
    flood::AdaptiveGrid grid(testCase.scenario.grid, testCase.gridConfig);
    prepareGrid(grid, testCase);
    const auto diagnostics = runSolver(grid);
    Outcome outcome{diagnostics, {}, {}, grid.activePatches()};
    grid.forEachActiveCell([&](flood::AdaptiveCellId id, const flood::Cell& cell) {
        outcome.ids.push_back(id);
        outcome.states.push_back({cell.h, cell.hu, cell.hv});
    });
    return outcome;
}

void compare(const TestCase& testCase, const Outcome& reference,
             const Outcome& candidate, Differences& differences,
             const std::string& backend) {
    check(reference.ids == candidate.ids &&
          reference.activePatches == candidate.activePatches,
          testCase.name + ": active leaf topology/order differs");
    check(reference.states.size() == candidate.states.size(),
          testCase.name + ": active leaf count differs");
    const auto& expectedDiagnostics = reference.diagnostics;
    const auto& actualDiagnostics = candidate.diagnostics;
    check(expectedDiagnostics.steps == actualDiagnostics.steps &&
          expectedDiagnostics.refinedPatches == actualDiagnostics.refinedPatches &&
          expectedDiagnostics.coarsenedPatches == actualDiagnostics.coarsenedPatches &&
          expectedDiagnostics.wetCells == actualDiagnostics.wetCells,
          testCase.name + ": step or regrid counts differ");
    check(std::abs(expectedDiagnostics.time - actualDiagnostics.time) <=
              diagnosticTolerance &&
          std::abs(expectedDiagnostics.currentTimestep -
                   actualDiagnostics.currentTimestep) <= diagnosticTolerance,
          testCase.name + ": global timestep history differs");
    check(expectedDiagnostics.regridEvents.size() ==
              actualDiagnostics.regridEvents.size(),
          testCase.name + ": regrid event count differs");
    check(expectedDiagnostics.coarseFineInterfaceSegments ==
              actualDiagnostics.coarseFineInterfaceSegments,
          testCase.name + ": coarse/fine interface counts differ");

    for (std::size_t index = 0; index < reference.states.size(); ++index) {
        const auto& expected = reference.states[index];
        const auto& actual = candidate.states[index];
        for (const double value : actual)
            check(std::isfinite(value), testCase.name + ": non-finite cell state");
        check(actual[0] >= -1e-12, testCase.name + ": negative water depth");
        differences.h = std::max(differences.h, std::abs(expected[0] - actual[0]));
        differences.hu = std::max(differences.hu, std::abs(expected[1] - actual[1]));
        differences.hv = std::max(differences.hv, std::abs(expected[2] - actual[2]));
    }
    differences.volume = std::max(differences.volume,
        std::abs(expectedDiagnostics.finalWaterVolume -
                 actualDiagnostics.finalWaterVolume));
    differences.massResidual = std::max(differences.massResidual,
        std::abs(actualDiagnostics.massBalanceResidual));
    differences.maxVelocity = std::max(differences.maxVelocity,
        std::abs(expectedDiagnostics.maximumVelocity -
                 actualDiagnostics.maximumVelocity));
    differences.interfaceError = std::max(differences.interfaceError,
        actualDiagnostics.maximumInterfaceMassFluxResidual);
    differences.regridError = std::max(differences.regridError,
        std::abs(expectedDiagnostics.maximumRegridVolumeDelta -
                 actualDiagnostics.maximumRegridVolumeDelta));
    check(std::abs(expectedDiagnostics.massBalanceResidual -
                   actualDiagnostics.massBalanceResidual) <= diagnosticTolerance,
          testCase.name + ": mass-balance residual differs");
    check(std::abs(expectedDiagnostics.rainfallVolume -
                   actualDiagnostics.rainfallVolume) <= diagnosticTolerance &&
          std::abs(expectedDiagnostics.infiltrationVolume -
                   actualDiagnostics.infiltrationVolume) <= diagnosticTolerance &&
          std::abs(expectedDiagnostics.outflowVolume -
                   actualDiagnostics.outflowVolume) <= diagnosticTolerance,
          testCase.name + ": source/sink budget differs");
    check(std::abs(expectedDiagnostics.coarseFineIntegratedMassFlux -
                   actualDiagnostics.coarseFineIntegratedMassFlux) <=
              diagnosticTolerance,
          testCase.name + ": integrated interface flux differs");
    check(std::abs(expectedDiagnostics.maximumRegridVolumeDelta -
                   actualDiagnostics.maximumRegridVolumeDelta) <=
              diagnosticTolerance,
          testCase.name + ": regrid conservation diagnostic differs");
    for (std::size_t index = 0; index < expectedDiagnostics.regridEvents.size(); ++index) {
        const auto& expected = expectedDiagnostics.regridEvents[index];
        const auto& actual = actualDiagnostics.regridEvents[index];
        check(expected.operation == actual.operation &&
              expected.patchId == actual.patchId &&
              std::abs(expected.time - actual.time) <= diagnosticTolerance &&
              std::abs(expected.waterVolumeDelta - actual.waterVolumeDelta) <=
                  diagnosticTolerance &&
              std::abs(expected.huIntegralDelta - actual.huIntegralDelta) <=
                  diagnosticTolerance &&
              std::abs(expected.hvIntegralDelta - actual.hvIntegralDelta) <=
                  diagnosticTolerance,
              testCase.name + ": regrid event data differs");
    }
    std::ostringstream differenceMessage;
    differenceMessage << std::setprecision(17) << testCase.name
        << " vs " << backend
        << ": state or diagnostic comparison exceeded tolerance (h="
        << differences.h << ", hu=" << differences.hu
        << ", hv=" << differences.hv << ", volume=" << differences.volume
        << ", velocity=" << differences.maxVelocity
        << ", regrid=" << differences.regridError << ")";
    std::size_t worstCell = 0;
    double worstCellError = 0.0;
    for (std::size_t index = 0; index < reference.states.size(); ++index) {
        for (std::size_t component = 0; component < 3; ++component) {
            const double error = std::abs(reference.states[index][component] -
                                          candidate.states[index][component]);
            if (error > worstCellError) {
                worstCellError = error;
                worstCell = index;
            }
        }
    }
    if (worstCellError > stateTolerance) {
        differenceMessage << " worst_cell_patch="
            << candidate.ids[worstCell].patchId << " row="
            << candidate.ids[worstCell].row << " col="
            << candidate.ids[worstCell].col << " serial_h="
            << reference.states[worstCell][0] << " mpi_h="
            << candidate.states[worstCell][0] << " serial_hu="
            << reference.states[worstCell][1] << " mpi_hu="
            << candidate.states[worstCell][1] << " serial_hv="
            << reference.states[worstCell][2] << " mpi_hv="
            << candidate.states[worstCell][2];
    }
    check(differences.h <= stateTolerance &&
          differences.hu <= stateTolerance &&
          differences.hv <= stateTolerance &&
          differences.volume <= diagnosticTolerance &&
          differences.maxVelocity <= diagnosticTolerance &&
          differences.regridError <= diagnosticTolerance,
          differenceMessage.str());
    check(std::abs(actualDiagnostics.massBalanceResidual) <=
              FLOOD_MASS_TOLERANCE,
          testCase.name + ": mass-balance residual exceeded tolerance");
    check(actualDiagnostics.maximumInterfaceMassFluxResidual == 0.0,
          testCase.name + ": coarse/fine interface is not equal-and-opposite");
}

} // namespace

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 2;
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    try {
        std::vector<TestCase> cases;
        for (const std::string scenario :
             {"flat-basin", "slope", "dam-break", "rain-drain", "wet-dry"})
            cases.push_back(makeCase(scenario, scenario));

        auto level1 = makeCase("forced-level-1", "slope", 8, 8, 4, 1);
        level1.nestedRefinementLevel = 1;
        level1.options.refineThreshold = 1e9;
        level1.options.coarsenThreshold = 0.0;
        cases.push_back(level1);

        auto level2 = makeCase("forced-level-2", "dam-break", 8, 8, 4, 2);
        level2.nestedRefinementLevel = 2;
        level2.options.refineThreshold = 1e9;
        level2.options.coarsenThreshold = 0.0;
        cases.push_back(level2);

        auto multiPatch = makeCase("uneven-patch-ownership", "wet-dry",
                                   12, 12, 4, 1);
        multiPatch.additionalRefinements = {0, 8};
        multiPatch.options.refineThreshold = 1e9;
        multiPatch.options.coarsenThreshold = 0.0;
        cases.push_back(multiPatch);

        auto remote = makeCase("remote-coarse-fine-interface", "slope",
                               8, 8, 4, 1);
        remote.nestedRefinementLevel = 1;
        remote.options.refineThreshold = 1e9;
        remote.options.coarsenThreshold = 0.0;
        remote.requireCrossRankInterface = true;
        cases.push_back(remote);

        int processDims[2] = {0, 0};
        MPI_Dims_create(size, 2, processDims);
        if (processDims[1] > 1) {
            auto vertical = makeCase("rank-0-coarse-rank-1-fine-vertical",
                                     "dam-break", 8, 8, 4, 1);
            vertical.rootPatchToRefine = 1;
            vertical.nestedRefinementLevel = 1;
            vertical.options.refineThreshold = 1e9;
            vertical.options.coarsenThreshold = 0.0;
            vertical.requireCrossRankInterface = true;
            vertical.requiredRemoteOrientation = 1;
            cases.push_back(vertical);
        }
        if (processDims[0] > 1) {
            auto horizontal = makeCase("rank-0-coarse-rank-1-row-fine-horizontal",
                                       "dam-break", 8, 8, 4, 1);
            horizontal.rootPatchToRefine = 2;
            horizontal.nestedRefinementLevel = 1;
            horizontal.options.refineThreshold = 1e9;
            horizontal.options.coarsenThreshold = 0.0;
            horizontal.requireCrossRankInterface = true;
            horizontal.requiredRemoteOrientation = 2;
            cases.push_back(horizontal);
        }

        auto emptyRanks = makeCase("low-work-ranks", "flat-basin",
                                   8, 8, 8, 1);
        emptyRanks.options.refineThreshold = 1e9;
        emptyRanks.options.coarsenThreshold = 0.0;
        cases.push_back(emptyRanks);

        auto coarsening = makeCase("coarsening-and-regridding", "flat-basin");
        coarsening.nestedRefinementLevel = 2;
        coarsening.options.refineThreshold = 1e9;
        coarsening.options.coarsenThreshold = 0.1;
        coarsening.options.coarsenPersistence = 1;
        coarsening.options.regridIntervalSteps = 1;
        cases.push_back(coarsening);
        cases.push_back(lakeAtRestCase());

        Differences maxDifferences;
        std::size_t maxInterfaceSegments = 0;
        for (const TestCase& testCase : cases) {
            const Outcome serial = runCase(testCase, [&](flood::AdaptiveGrid& grid) {
                return flood::AdaptiveSolver(testCase.options).run(
                    grid, testCase.scenario.rainfall, testCase.scenario.config);
            });
#ifdef FLOOD_HAS_OPENMP
            const Outcome openmp = runCase(testCase, [&](flood::AdaptiveGrid& grid) {
                return flood::AdaptiveOpenMPSolver(testCase.options, 2).run(
                    grid, testCase.scenario.rainfall, testCase.scenario.config);
            });
            compare(testCase, serial, openmp, maxDifferences, "OpenMP");
#endif
            const Outcome mpi = runCase(testCase, [&](flood::AdaptiveGrid& grid) {
                return flood::AdaptiveMpiSolver(testCase.options).run(
                    grid, testCase.scenario.rainfall, testCase.scenario.config);
            });
            compare(testCase, serial, mpi, maxDifferences, "MPI");
            maxInterfaceSegments = std::max(
                maxInterfaceSegments,
                mpi.diagnostics.crossRankCoarseFineInterfaceSegments);
            if (testCase.requireCrossRankInterface && size > 1)
                check(mpi.diagnostics.crossRankCoarseFineInterfaceSegments >= 2,
                      testCase.name + ": no remote coarse/fine face was exercised");
            if (testCase.requiredRemoteOrientation == 1)
                check(mpi.diagnostics.crossRankCoarseFineXSegments > 0,
                      testCase.name + ": no remote vertical coarse/fine face was exercised");
            if (testCase.requiredRemoteOrientation == 2)
                check(mpi.diagnostics.crossRankCoarseFineYSegments > 0,
                      testCase.name + ": no remote horizontal coarse/fine face was exercised");
            if (testCase.name == "forced-level-2")
                check(mpi.diagnostics.level2Cells > 0,
                      "Level-2 refinement did not produce level-2 leaves");
            if (testCase.name == "coarsening-and-regridding")
                check(mpi.diagnostics.coarsenedPatches > 0,
                      "Coarsening case did not coarsen");
            if (testCase.name == "lake-at-rest")
                check(mpi.diagnostics.maximumVelocity < 1e-10,
                      "MPI lake-at-rest generated spurious motion");
        }

        auto deterministic = lakeAtRestCase();
        const Outcome first = runCase(deterministic, [&](flood::AdaptiveGrid& grid) {
            return flood::AdaptiveMpiSolver(deterministic.options).run(
                grid, deterministic.scenario.rainfall, deterministic.scenario.config);
        });
        const Outcome second = runCase(deterministic, [&](flood::AdaptiveGrid& grid) {
            return flood::AdaptiveMpiSolver(deterministic.options).run(
                grid, deterministic.scenario.rainfall, deterministic.scenario.config);
        });
        compare(deterministic, first, second, maxDifferences, "repeated MPI");

        if (rank == 0) {
            std::cout << std::setprecision(17)
                      << "adaptive MPI consistency passed ranks=" << size << '\n'
                      << "maximum_depth_difference=" << maxDifferences.h << '\n'
                      << "maximum_hu_difference=" << maxDifferences.hu << '\n'
                      << "maximum_hv_difference=" << maxDifferences.hv << '\n'
                      << "maximum_volume_difference=" << maxDifferences.volume << '\n'
                      << "maximum_mass_balance_residual=" << maxDifferences.massResidual << '\n'
                      << "maximum_velocity_difference=" << maxDifferences.maxVelocity << '\n'
                      << "maximum_interface_conservation_error="
                      << maxDifferences.interfaceError << '\n'
                      << "maximum_regrid_diagnostic_difference="
                      << maxDifferences.regridError << '\n'
                      << "cross_rank_coarse_fine_segments=" << maxInterfaceSegments << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "adaptive MPI test rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return 0;
}
