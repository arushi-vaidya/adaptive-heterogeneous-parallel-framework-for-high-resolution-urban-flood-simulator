#include "io/Output.hpp"
#include "simulation/Scenarios.hpp"
#include "solver/BackendCapabilities.hpp"
#include "solver/SerialSolver.hpp"

#ifdef FLOOD_HAS_OPENMP
#include "solver/OpenMPSolver.hpp"
#endif
#ifdef FLOOD_HAS_MPI
#include "solver/MpiSolver.hpp"
#endif

#include <cstdlib>
#include <chrono>
#include <fstream>
#include <map>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        std::string scenarioName = "flat-basin";
        std::string output = "output/run";
        std::string backendName = "serial";
        std::string terrainPath, rainfallPath;
        std::size_t rows = 32, cols = 32;
        double cellSize = 5.0, duration = -1.0;
        int threads = 0;
        std::map<std::string, std::string> physicsOptions;
        auto applyConfigValue = [&](const std::string& key, const std::string& value) {
            if (key == "scenario") scenarioName = value;
            else if (key == "rows") rows = std::stoul(value);
            else if (key == "cols") cols = std::stoul(value);
            else if (key == "cell-size") cellSize = std::stod(value);
            else if (key == "duration") duration = std::stod(value);
            else if (key == "output") output = value;
            else if (key == "terrain") terrainPath = value;
            else if (key == "rainfall") rainfallPath = value;
            else if (key == "backend") backendName = value;
            else if (key == "threads") threads = std::stoi(value);
            else if (key == "gravity" || key == "manning" || key == "infiltration" ||
                     key == "cfl" || key == "max-dt" || key == "dry-depth" ||
                     key == "boundary") physicsOptions[key] = value;
            else throw std::invalid_argument("Unknown configuration key: " + key);
        };
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--capabilities") {
                std::cout << flood::backendCapabilitiesJson() << '\n';
                return 0;
            }
            if (arg == "--list-backends") {
                for (const auto& capability : flood::backendCapabilities())
                    std::cout << capability.name << ' ' << capability.status << " - "
                              << capability.detail << '\n';
                return 0;
            }
            if (arg == "--config" && i + 1 < argc) {
                std::ifstream configFile(argv[++i]);
                if (!configFile) throw std::runtime_error("Cannot open configuration file");
                std::string line;
                while (std::getline(configFile, line)) {
                    const auto comment = line.find('#');
                    if (comment != std::string::npos) line.erase(comment);
                    const auto first = line.find_first_not_of(" \t\r\n");
                    if (first == std::string::npos) continue;
                    const auto equal = line.find('=', first);
                    if (equal == std::string::npos) throw std::runtime_error("Expected key=value in config");
                    const auto keyEnd = line.find_last_not_of(" \t", equal - 1);
                    const auto valueStart = line.find_first_not_of(" \t", equal + 1);
                    const auto valueEnd = line.find_last_not_of(" \t\r\n");
                    const std::string key = line.substr(first, keyEnd - first + 1);
                    const std::string value = valueStart == std::string::npos ? ""
                        : line.substr(valueStart, valueEnd - valueStart + 1);
                    applyConfigValue(key, value);
                }
            }
            else if (arg == "--scenario" && i + 1 < argc) scenarioName = argv[++i];
            else if (arg == "--output" && i + 1 < argc) output = argv[++i];
            else if (arg == "--terrain" && i + 1 < argc) terrainPath = argv[++i];
            else if (arg == "--rainfall" && i + 1 < argc) rainfallPath = argv[++i];
            else if (arg == "--backend" && i + 1 < argc) backendName = argv[++i];
            else if (arg == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
            else if (arg == "--rows" && i + 1 < argc) rows = std::stoul(argv[++i]);
            else if (arg == "--cols" && i + 1 < argc) cols = std::stoul(argv[++i]);
            else if (arg == "--cell-size" && i + 1 < argc) cellSize = std::stod(argv[++i]);
            else if (arg == "--duration" && i + 1 < argc) duration = std::stod(argv[++i]);
            else if ((arg == "--gravity" || arg == "--manning" || arg == "--infiltration" ||
                      arg == "--cfl" || arg == "--max-dt" || arg == "--dry-depth" ||
                      arg == "--boundary") && i + 1 < argc) {
                const std::string key = arg.substr(2);
                physicsOptions[key] = argv[++i];
            }
            else if (arg == "--help") {
                std::cout << "Usage: flood_sim [--config FILE] [--backend serial|openmp|mpi --threads N] "
                             "[--scenario NAME --rows N --cols N "
                             "--cell-size M --duration SEC --output DIR] [--terrain CSV "
                             "--rainfall CSV] [--gravity G --manning N --infiltration MPS "
                             "--cfl C --max-dt SEC --dry-depth M --boundary closed|outflow]\n";
                return 0;
            } else throw std::invalid_argument("Unknown or incomplete argument: " + arg);
        }
        const auto preprocessingStart = std::chrono::steady_clock::now();
        auto scenario = flood::makeScenario(scenarioName, rows, cols, cellSize);
        for (const auto& option : physicsOptions) {
            if (option.first == "gravity") scenario.config.gravity = std::stod(option.second);
            else if (option.first == "manning") scenario.config.manningN = std::stod(option.second);
            else if (option.first == "infiltration")
                scenario.config.infiltrationMetersPerSecond = std::stod(option.second);
            else if (option.first == "cfl") scenario.config.cfl = std::stod(option.second);
            else if (option.first == "max-dt") scenario.config.maxTimestep = std::stod(option.second);
            else if (option.first == "dry-depth") scenario.config.dryDepth = std::stod(option.second);
            else if (option.first == "boundary") {
                if (option.second == "closed") scenario.config.boundary = flood::BoundaryCondition::Closed;
                else if (option.second == "outflow") scenario.config.boundary = flood::BoundaryCondition::Outflow;
                else throw std::invalid_argument("Boundary must be closed or outflow");
            }
        }
        if (!terrainPath.empty()) {
            scenario.grid = flood::readTerrainCsv(terrainPath, cellSize, cellSize);
            rows = scenario.grid.rows();
            cols = scenario.grid.cols();
        }
        if (!rainfallPath.empty()) scenario.rainfall = flood::readRainfallCsv(rainfallPath);
        if (duration >= 0.0) scenario.config.endTime = duration;
        const double preprocessingSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - preprocessingStart).count();
        std::cout << "Simulation started\nGrid: " << rows << " x " << cols
                  << "\nCell size: " << cellSize << " m\nBoundary: "
                  << flood::toString(scenario.config.boundary) << "\nBackend: "
                  << backendName << '\n';
        flood::Diagnostics diagnostics;
        if (backendName == "serial") {
            diagnostics = flood::SerialSolver().run(
                scenario.grid, scenario.rainfall, scenario.config);
        } else if (backendName == "openmp") {
#ifdef FLOOD_HAS_OPENMP
            diagnostics = flood::OpenMPSolver(threads).run(
                scenario.grid, scenario.rainfall, scenario.config);
#else
            throw std::runtime_error("OpenMP backend unavailable in this build; configure with libomp");
#endif
    } else if (backendName == "mpi") {
#if defined(FLOOD_HAS_MPI) && defined(FLOOD_MPI_RUNTIME_FOUND)
        diagnostics = flood::MpiSolver().run(
        scenario.grid, scenario.rainfall, scenario.config);
#elif defined(FLOOD_HAS_MPI)
            throw std::runtime_error("MPI solver is compiled, but no MPI launcher/runtime was detected");
#else
        throw std::runtime_error(flood::backendUnavailableReason(backendName));
#endif
        } else {
            throw std::runtime_error(flood::backendUnavailableReason(backendName));
        }
    if (!diagnostics.isRoot) return EXIT_SUCCESS;
        const auto outputStart = std::chrono::steady_clock::now();
        flood::writeOutputs(scenario.grid, diagnostics, output);
        const double outputSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - outputStart).count();
        std::ofstream benchmark(std::filesystem::path(output) / "benchmark.csv");
        benchmark << "backend,threads,processes,cells,steps,preprocessing_seconds,solver_seconds,"
                 "setup_seconds,computation_seconds,communication_seconds,synchronization_seconds,other_seconds,output_seconds,total_seconds\n"
              << diagnostics.backend << ',' << diagnostics.threads << ',' << diagnostics.processes
              << ',' << scenario.grid.size() << ',' << diagnostics.steps
              << ',' << preprocessingSeconds
                  << ',' << diagnostics.solverSeconds << ',' << diagnostics.setupSeconds
                  << ',' << diagnostics.computationSeconds
              << ',' << diagnostics.communicationSeconds << ',' << diagnostics.synchronizationSeconds
              << ',' << diagnostics.otherSeconds
              << ',' << outputSeconds << ','
                  << preprocessingSeconds + diagnostics.solverSeconds + outputSeconds << '\n';
        std::cout << "Simulation completed\nTime: " << diagnostics.time << " s\nSteps: "
                  << diagnostics.steps << "\nMax depth: " << diagnostics.maxDepth
                  << " m\nWet cells: " << diagnostics.wetCells << "\nStored volume: "
                  << diagnostics.storedVolume << " m3\nFlooded area: "
                  << diagnostics.floodedArea << " m2\nMass residual: "
                  << diagnostics.massResidual << " m3\nSolver runtime: "
                  << diagnostics.solverSeconds << " s\nBackend used: " << diagnostics.backend
                  << " (threads=" << diagnostics.threads << ", processes="
                  << diagnostics.processes << ")\nOutput: " << output << '\n';
    } catch (const std::exception& error) {
        std::cerr << "flood_sim: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return 0;
}