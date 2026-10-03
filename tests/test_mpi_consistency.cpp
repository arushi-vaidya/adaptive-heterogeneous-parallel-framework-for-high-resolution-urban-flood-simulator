#include "simulation/Scenarios.hpp"
#include "solver/MpiSolver.hpp"
#include "solver/SerialSolver.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#ifndef FLOOD_NUMERICAL_TOLERANCE
#define FLOOD_NUMERICAL_TOLERANCE 1e-8
#endif
#ifndef FLOOD_MOMENTUM_TOLERANCE
#define FLOOD_MOMENTUM_TOLERANCE 1e-8
#endif
#ifndef FLOOD_MASS_TOLERANCE
#define FLOOD_MASS_TOLERANCE 1e-8
#endif

namespace {

flood::Scenario makeCase(const std::string& name) {
    if (name != "lake-at-rest") return flood::makeScenario(name, 16, 16, 5.0);
    flood::Scenario scenario{flood::Grid(16, 16, 5.0, 5.0), flood::Rainfall(),
                             flood::SolverConfig(), name};
    scenario.config.endTime = 1.0;
    for (std::size_t row = 0; row < scenario.grid.rows(); ++row) {
        for (std::size_t col = 0; col < scenario.grid.cols(); ++col) {
            scenario.grid.at(row, col).bed = col < 8 ? 0.0 : 0.5;
            scenario.grid.at(row, col).h = 1.0 - scenario.grid.at(row, col).bed;
        }
    }
    return scenario;
}

void compareRoot(const flood::Grid& reference, const flood::Grid& candidate,
                 const flood::Diagnostics& referenceDiagnostics,
                 const flood::Diagnostics& candidateDiagnostics,
                 const std::string& name) {
    double maxDepthError = 0.0, maxHuError = 0.0, maxHvError = 0.0;
    double absoluteDepthSum = 0.0, squaredDepthSum = 0.0, referenceDepthSum = 0.0;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        const auto& expected = reference.cells()[index];
        const auto& actual = candidate.cells()[index];
        maxDepthError = std::max(maxDepthError, std::abs(expected.h - actual.h));
        absoluteDepthSum += std::abs(expected.h - actual.h);
        squaredDepthSum += (expected.h - actual.h) * (expected.h - actual.h);
        referenceDepthSum += std::abs(expected.h);
        maxHuError = std::max(maxHuError, std::abs(expected.hu - actual.hu));
        maxHvError = std::max(maxHvError, std::abs(expected.hv - actual.hv));
    }
    const double volumeError = std::abs(referenceDiagnostics.storedVolume -
                                        candidateDiagnostics.storedVolume);
    const double areaError = std::abs(referenceDiagnostics.floodedArea -
                                      candidateDiagnostics.floodedArea);
    const double conservationError = std::abs(candidateDiagnostics.massResidual);
    const double timeError = std::abs(referenceDiagnostics.time - candidateDiagnostics.time);
    const double meanDepthError = absoluteDepthSum / static_cast<double>(reference.size());
    const double rmse = std::sqrt(squaredDepthSum / static_cast<double>(reference.size()));
    const double relativeError = referenceDepthSum > 0.0
        ? absoluteDepthSum / referenceDepthSum
        : (absoluteDepthSum == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
    const double maxDepthDifference = std::abs(referenceDiagnostics.maxDepth - candidateDiagnostics.maxDepth);
    if (maxDepthError > FLOOD_NUMERICAL_TOLERANCE ||
        maxHuError > FLOOD_MOMENTUM_TOLERANCE || maxHvError > FLOOD_MOMENTUM_TOLERANCE ||
        volumeError > FLOOD_MASS_TOLERANCE || areaError > 0.0 ||
        maxDepthDifference > FLOOD_NUMERICAL_TOLERANCE ||
        referenceDiagnostics.steps != candidateDiagnostics.steps || timeError > 1e-12 ||
        conservationError > FLOOD_MASS_TOLERANCE ||
        std::abs(referenceDiagnostics.massResidual) > FLOOD_MASS_TOLERANCE) {
        throw std::runtime_error(name + " failed MPI/serial comparison: max_h=" +
            std::to_string(maxDepthError) + " max_hu=" + std::to_string(maxHuError) +
            " max_hv=" + std::to_string(maxHvError) + " volume=" +
            std::to_string(volumeError) + " conservation=" + std::to_string(conservationError));
    }
    std::cout << name << " processes=" << candidateDiagnostics.processes
              << " max_h=" << maxDepthError << " max_hu=" << maxHuError
              << " max_hv=" << maxHvError << " volume_difference=" << volumeError
              << " mean_h=" << meanDepthError << " rmse=" << rmse
              << " relative_h=" << relativeError
              << " max_depth_difference=" << maxDepthDifference
              << " steps=" << candidateDiagnostics.steps
              << " time_difference=" << timeError
              << " conservation_residual=" << conservationError << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 2;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        const std::string name = argv[1];
        auto referenceCase = makeCase(name);
        flood::Diagnostics referenceDiagnostics;
        if (rank == 0) {
            referenceDiagnostics = flood::SerialSolver().run(
                referenceCase.grid, referenceCase.rainfall, referenceCase.config);
        }
        auto mpiCase = makeCase(name);
        const auto mpiDiagnostics = flood::MpiSolver().run(
            mpiCase.grid, mpiCase.rainfall, mpiCase.config);
        if (mpiDiagnostics.isRoot)
            compareRoot(referenceCase.grid, mpiCase.grid, referenceDiagnostics,
                        mpiDiagnostics, name);
    } catch (const std::exception& error) {
        std::cerr << "MPI consistency test: " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return 0;
}