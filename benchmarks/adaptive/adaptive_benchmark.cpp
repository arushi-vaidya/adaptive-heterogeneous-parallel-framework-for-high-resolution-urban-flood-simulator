#include "simulation/Scenarios.hpp"
#include "solver/AdaptiveSolver.hpp"
#ifdef FLOOD_HAS_OPENMP
#include "solver/AdaptiveOpenMPSolver.hpp"
#endif
#ifdef FLOOD_HAS_MPI
#include "solver/AdaptiveMpiSolver.hpp"
#include <mpi.h>
#endif

#include <iomanip>
#include <iostream>
#include <limits>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Arguments {
    std::string backend;
    std::string scenarioName;
    std::size_t rows = 64;
    std::size_t cols = 64;
    std::size_t patchExtent = 8;
    std::size_t maxLevel = 2;
    int threads = 1;
    double cellSize = 1.0;
    double endTime = 0.2;
    bool dynamicLoadBalancing = false;
    double imbalanceThreshold = 1.20;
    std::size_t rebalanceCooldown = 10;
    std::size_t regridInterval = 2;
    double refineThreshold = 0.10;
    double coarsenThreshold = 0.05;
    std::string outputDirectory;
};

bool parseBool(const char* value, const char* name) {
    const std::string text(value);
    if (text == "true") return true;
    if (text == "false") return false;
    throw std::invalid_argument(std::string("Invalid ") + name + ": " + text);
}

std::size_t parseSize(const char* value, const char* name, bool allowZero = false) {
    const std::string text(value);
    std::size_t parsed = 0;
    const unsigned long long result = std::stoull(text, &parsed);
    if (parsed != text.size() || (!allowZero && result == 0))
        throw std::invalid_argument(std::string("Invalid ") + name + ": " + text);
    return static_cast<std::size_t>(result);
}

Arguments parseArguments(int argc, char** argv) {
    Arguments args;
    for (int index = 1; index < argc; ++index) {
        const std::string key(argv[index]);
        if (index + 1 >= argc)
            throw std::invalid_argument("Missing value for " + key);
        const char* value = argv[++index];
        if (key == "--backend") args.backend = value;
        else if (key == "--scenario") args.scenarioName = value;
        else if (key == "--rows") args.rows = parseSize(value, "row count");
        else if (key == "--cols") args.cols = parseSize(value, "column count");
        else if (key == "--patch-extent")
            args.patchExtent = parseSize(value, "patch extent");
        else if (key == "--max-level")
            args.maxLevel = parseSize(value, "maximum level", true);
        else if (key == "--threads") {
            const std::size_t count = parseSize(value, "thread count");
            if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("Thread count is too large");
            args.threads = static_cast<int>(count);
        } else if (key == "--cell-size") args.cellSize = std::stod(value);
        else if (key == "--end-time") args.endTime = std::stod(value);
        else if (key == "--dynamic-load-balancing")
            args.dynamicLoadBalancing = parseBool(value, "dynamic load-balancing flag");
        else if (key == "--imbalance-threshold")
            args.imbalanceThreshold = std::stod(value);
        else if (key == "--rebalance-cooldown")
            args.rebalanceCooldown = parseSize(value, "rebalance cooldown", true);
        else if (key == "--output-dir") args.outputDirectory = value;
        else if (key == "--regrid-interval")
            args.regridInterval = parseSize(value, "regrid interval");
        else if (key == "--refine-threshold")
            args.refineThreshold = std::stod(value);
        else if (key == "--coarsen-threshold")
            args.coarsenThreshold = std::stod(value);
        else throw std::invalid_argument("Unknown argument: " + key);
    }
    if (args.backend.empty() || args.scenarioName.empty() ||
        args.rows < 2 || args.cols < 2 ||
        !(args.cellSize > 0.0) || !(args.endTime > 0.0) ||
        !std::isfinite(args.imbalanceThreshold) ||
        args.imbalanceThreshold < 1.0 ||
        !std::isfinite(args.refineThreshold) ||
        !std::isfinite(args.coarsenThreshold) ||
        args.coarsenThreshold < 0.0 ||
        args.refineThreshold <= args.coarsenThreshold ||
        (args.dynamicLoadBalancing && args.backend != "mpi") ||
        args.maxLevel > 2)
        throw std::invalid_argument("Invalid adaptive benchmark arguments");
    return args;
}

void printHeader() {
    std::cout
        << "scenario,backend,rows,cols,base_cells,adaptive_leaf_cells,"
           "cell_count_ratio,level0_cells,level1_cells,level2_cells,"
           "active_patches,maximum_active_level,refined_patches,coarsened_patches,"
           "steps,final_time,threads,rank_count,dynamic_load_balancing,migration_count,"
           "migrated_patches,migrated_cells,migration_time_seconds,"
           "pre_rebalance_imbalance,post_rebalance_imbalance,last_rebalance_timestep,"
           "max_depth,maximum_velocity,wet_cells,coarse_fine_segments,"
           "runtime_seconds,compute_seconds,mesh_seconds,"
           "cfl_seconds,interface_seconds,update_seconds,activity_seconds,regridding_seconds,"
           "communication_seconds,diagnostics_seconds,minimum_rank_work,maximum_rank_work,"
           "mean_rank_work,load_imbalance,rank_workload_distribution,workload_snapshots,"
           "initial_volume,"
           "final_volume,rainfall_volume,infiltration_volume,outflow_volume,"
           "mass_balance_residual,maximum_velocity,maximum_interface_mass_flux_residual,"
           "maximum_regrid_volume_delta\n";
}

void printRow(const Arguments& args, const flood::AdaptiveGrid& grid,
              const flood::AdaptiveDiagnostics& diagnostics) {
    const flood::AdaptiveWorkloadSnapshot& workload =
        diagnostics.workloadSnapshots.back();
    std::size_t maximumActiveLevel = 0;
    for (const flood::PatchId patchId : grid.activePatches())
        maximumActiveLevel = std::max(maximumActiveLevel, grid.patch(patchId).level());
    std::cout << std::setprecision(17)
              << args.scenarioName << ',' << args.backend << ','
              << args.rows << ',' << args.cols << ','
              << args.rows * args.cols << ','
              << diagnostics.activeLeafCells << ','
              << static_cast<double>(diagnostics.activeLeafCells) /
                     static_cast<double>(args.rows * args.cols) << ','
              << diagnostics.level0Cells << ',' << diagnostics.level1Cells << ','
              << diagnostics.level2Cells << ',' << grid.activePatches().size() << ','
              << maximumActiveLevel << ','
              << diagnostics.refinedPatches << ',' << diagnostics.coarsenedPatches << ','
              << diagnostics.steps << ',' << diagnostics.time << ','
              << (args.backend == "openmp" ? args.threads : 1) << ','
              << workload.ranks.size() << ','
              << (args.dynamicLoadBalancing ? "true" : "false") << ','
              << diagnostics.migrationCount << ','
              << diagnostics.migratedPatchCount << ','
              << diagnostics.migratedCellCount << ','
              << diagnostics.migrationTimeSeconds << ','
              << diagnostics.preRebalanceImbalance << ','
              << diagnostics.postRebalanceImbalance << ','
              << diagnostics.lastRebalanceTimestep << ','
              << diagnostics.maximumDepth << ','
              << diagnostics.maximumVelocity << ','
              << diagnostics.wetCells << ','
              << diagnostics.coarseFineInterfaceSegments << ','
              << diagnostics.runtimeSeconds << ','
              << diagnostics.computeSeconds << ',' << diagnostics.meshSeconds << ','
              << diagnostics.cflSeconds << ',' << diagnostics.interfaceSeconds << ','
              << diagnostics.updateSeconds << ',' << diagnostics.activitySeconds << ','
              << diagnostics.regriddingSeconds << ','
              << diagnostics.communicationSeconds << ','
              << diagnostics.diagnosticsSeconds << ','
              << workload.minimumRankWork << ',' << workload.maximumRankWork << ','
              << workload.meanRankWork << ',' << workload.loadImbalance << ",\"";
    for (std::size_t index = 0; index < workload.ranks.size(); ++index) {
        if (index != 0) std::cout << ';';
        const auto& rank = workload.ranks[index];
        std::cout << rank.rank << ':' << rank.activeLeafCells << ':'
                  << rank.activePatches << ':' << rank.estimatedWork << ':'
                  << rank.computeSeconds << ':' << rank.communicationSeconds;
    }
    std::cout << "\",\"";
    for (std::size_t sampleIndex = 0;
         sampleIndex < diagnostics.workloadSnapshots.size(); ++sampleIndex) {
        if (sampleIndex != 0) std::cout << '|';
        const auto& sample = diagnostics.workloadSnapshots[sampleIndex];
        std::cout << sample.step << '@' << sample.time << '@'
                  << sample.minimumRankWork << '@' << sample.maximumRankWork << '@'
                  << sample.meanRankWork << '@' << sample.loadImbalance << '[';
        for (std::size_t rankIndex = 0; rankIndex < sample.ranks.size(); ++rankIndex) {
            if (rankIndex != 0) std::cout << ';';
            const auto& rank = sample.ranks[rankIndex];
            std::cout << rank.rank << ':' << rank.activeLeafCells << ':'
                      << rank.activePatches << ':' << rank.estimatedWork << ':'
                      << rank.computeSeconds << ':' << rank.communicationSeconds;
        }
        std::cout << ']';
    }
    std::cout << "\"," << diagnostics.initialWaterVolume << ','
              << diagnostics.finalWaterVolume << ',' << diagnostics.rainfallVolume << ','
              << diagnostics.infiltrationVolume << ',' << diagnostics.outflowVolume << ','
              << diagnostics.massBalanceResidual << ',' << diagnostics.maximumVelocity << ','
              << diagnostics.maximumInterfaceMassFluxResidual << ','
              << diagnostics.maximumRegridVolumeDelta << '\n';
}

void writeAdaptiveMaps(const Arguments& args, const flood::AdaptiveGrid& grid) {
    if (args.outputDirectory.empty()) return;
    std::filesystem::create_directories(args.outputDirectory);
    std::size_t finestScale = 1;
    for (std::size_t level = 0; level < args.maxLevel; ++level)
        finestScale *= grid.config().refinementRatio;
    const std::size_t rows = grid.rows();
    const std::size_t cols = grid.cols();
    std::vector<long double> weightedDepth(rows * cols, 0.0L);
    std::vector<std::size_t> representedArea(rows * cols, 0);
    std::vector<std::size_t> refinement(rows * cols, 0);
    for (const flood::PatchId patchId : grid.activePatches()) {
        const flood::AdaptivePatch& patch = grid.patch(patchId);
        std::size_t scale = 1;
        for (std::size_t level = patch.level(); level < args.maxLevel; ++level)
            scale *= grid.config().refinementRatio;
        for (std::size_t row = 0; row < patch.grid().rows(); ++row) {
            for (std::size_t col = 0; col < patch.grid().cols(); ++col) {
                const flood::Cell& cell = patch.grid().at(row, col);
                const std::size_t rowBegin =
                    patch.rowOriginFineUnits() + row * scale;
                const std::size_t colBegin =
                    patch.colOriginFineUnits() + col * scale;
                const std::size_t index =
                    (rowBegin / finestScale) * cols + colBegin / finestScale;
                const std::size_t areaWeight = scale * scale;
                weightedDepth[index] +=
                    static_cast<long double>(cell.h) * areaWeight;
                representedArea[index] += areaWeight;
                refinement[index] = std::max(refinement[index], patch.level());
            }
        }
    }
    std::vector<double> depth(rows * cols, 0.0);
    for (std::size_t index = 0; index < depth.size(); ++index) {
        if (representedArea[index] > 0)
            depth[index] = static_cast<double>(
                weightedDepth[index] / representedArea[index]);
    }
    const auto writeMap = [rows, cols, &args](
        const std::string& filename, const auto& values) {
        std::ofstream output(
            std::filesystem::path(args.outputDirectory) / filename);
        if (!output)
            throw std::runtime_error("Cannot open adaptive output map: " + filename);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                if (col != 0) output << ',';
                output << values[row * cols + col];
            }
            output << '\n';
        }
        if (!output)
            throw std::runtime_error("Failed writing adaptive output map: " + filename);
    };
    writeMap("depth.csv", depth);
    writeMap("refinement.csv", refinement);
}

flood::Scenario makeBenchmarkScenario(const Arguments& args) {
    if (args.scenarioName != "localized-refinement") {
        flood::Scenario scenario = flood::makeScenario(
            args.scenarioName, args.rows, args.cols, args.cellSize);
        scenario.config.endTime = args.endTime;
        scenario.config.maxTimestep = std::min(scenario.config.maxTimestep, 0.02);
        return scenario;
    }
    flood::Scenario scenario{
        flood::Grid(args.rows, args.cols, args.cellSize, args.cellSize),
        flood::Rainfall({{0.0, 0.0}}), flood::SolverConfig(),
        args.scenarioName};
    scenario.config.endTime = args.endTime;
    scenario.config.maxTimestep = 0.02;
    const std::size_t rowBegin = args.rows / 3;
    const std::size_t rowEnd = std::min(args.rows, rowBegin + args.rows / 5);
    const std::size_t colBegin = args.cols / 3;
    const std::size_t colEnd = std::min(args.cols, colBegin + args.cols / 5);
    for (std::size_t row = rowBegin; row < rowEnd; ++row)
        for (std::size_t col = colBegin; col < colEnd; ++col)
            scenario.grid.at(row, col).h = 0.5;
    return scenario;
}

int execute(const Arguments& args, int rank) {
    flood::Scenario scenario = makeBenchmarkScenario(args);
    flood::AdaptiveGridConfig gridConfig;
    gridConfig.patchRows = args.patchExtent;
    gridConfig.patchCols = args.patchExtent;
    gridConfig.maxRefinementLevel = args.maxLevel;
    flood::AdaptiveGrid grid(scenario.grid, gridConfig);
    flood::AdaptiveSolverOptions options;
    options.regridIntervalSteps = args.regridInterval;
    options.refineThreshold = args.refineThreshold;
    options.coarsenThreshold = args.coarsenThreshold;
    options.collectWorkloadSnapshots = true;
    options.dynamicLoadBalancing = args.dynamicLoadBalancing;
    options.loadBalanceImbalanceThreshold = args.imbalanceThreshold;
    options.loadBalanceCooldownSteps = args.rebalanceCooldown;
    flood::AdaptiveDiagnostics diagnostics;
    if (args.backend == "serial") {
        diagnostics = flood::AdaptiveSolver(options).run(
            grid, scenario.rainfall, scenario.config);
    } else if (args.backend == "openmp") {
#ifdef FLOOD_HAS_OPENMP
        diagnostics = flood::AdaptiveOpenMPSolver(options, args.threads).run(
            grid, scenario.rainfall, scenario.config);
#else
        throw std::runtime_error("OpenMP backend is unavailable in this build");
#endif
    } else if (args.backend == "mpi") {
#ifdef FLOOD_HAS_MPI
        diagnostics = flood::AdaptiveMpiSolver(options).run(
            grid, scenario.rainfall, scenario.config);
#else
        throw std::runtime_error("MPI backend is unavailable in this build");
#endif
    } else {
        throw std::invalid_argument("Backend must be serial, openmp, or mpi");
    }
    if (rank == 0) {
        writeAdaptiveMaps(args, grid);
        printRow(args, grid, diagnostics);
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    int rank = 0;
#ifdef FLOOD_HAS_MPI
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
        return 2;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    int status = 0;
    try {
        const Arguments args = parseArguments(argc, argv);
        if (rank == 0) printHeader();
        status = execute(args, rank);
    } catch (const std::exception& error) {
        if (rank == 0) std::cerr << "adaptive benchmark failed: " << error.what() << '\n';
#ifdef FLOOD_HAS_MPI
        MPI_Abort(MPI_COMM_WORLD, 2);
#endif
        status = 2;
    }
#ifdef FLOOD_HAS_MPI
    MPI_Finalize();
#endif
    return status;
}
