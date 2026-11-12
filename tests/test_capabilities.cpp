#include "solver/BackendCapabilities.hpp"

#include <iostream>
#include <stdexcept>
#include <string_view>

int main() {
    const auto capabilities = flood::backendCapabilities();
    if (capabilities.size() != 5 || capabilities[0].name != "serial" ||
        capabilities[0].status != "AVAILABLE" || !capabilities[0].implemented) {
        throw std::runtime_error("Serial capability must be available and implemented");
    }
#ifdef FLOOD_HAS_OPENMP
    if (capabilities[1].status != "AVAILABLE" || !capabilities[1].implemented) {
        throw std::runtime_error("Compiled OpenMP backend was not reported available");
    }
#else
    if (capabilities[1].status != "UNAVAILABLE" || capabilities[1].implemented) {
        throw std::runtime_error("Missing OpenMP backend was reported incorrectly");
    }
#endif
#ifdef FLOOD_HAS_MPI
    if (!capabilities[2].implemented) {
        throw std::runtime_error("Compiled MPI backend was not reported implemented");
    }
#ifdef FLOOD_MPI_RUNTIME_FOUND
    if (capabilities[2].status != "AVAILABLE")
        throw std::runtime_error("MPI with detected runtime was not reported available");
#else
    if (capabilities[2].status != "UNAVAILABLE")
        throw std::runtime_error("MPI without a detected launcher must be unavailable");
#endif
#else
#ifdef FLOOD_MPI_SOURCE_AVAILABLE
    if (capabilities[2].status != "UNAVAILABLE" || !capabilities[2].implemented ||
        capabilities[2].toolchainDetected || capabilities[2].runtimeDetected)
        throw std::runtime_error("MPI source/toolchain availability was reported incorrectly");
#else
    if (capabilities[2].status != "UNAVAILABLE" || capabilities[2].implemented)
        throw std::runtime_error("MPI without its source was reported runnable");
#endif
#endif
#ifdef FLOOD_HAS_CUDA
    if (capabilities[3].status != "AVAILABLE")
        throw std::runtime_error("Compiled CUDA backend was not reported available");
#else
    if (capabilities[3].status != "UNAVAILABLE" || capabilities[3].toolchainDetected)
        throw std::runtime_error("CUDA without a toolchain was reported runnable");
#endif
#ifdef FLOOD_HAS_OPENACC
    if (capabilities[4].status != "AVAILABLE")
        throw std::runtime_error("Compiled OpenACC backend was not reported available");
#else
    if (capabilities[4].status != "UNAVAILABLE" || capabilities[4].toolchainDetected)
        throw std::runtime_error("OpenACC without compiler support was reported runnable");
#endif
    const auto json = flood::backendCapabilitiesJson();
    for (const auto name : {"serial", "openmp", "mpi", "cuda", "openacc"}) {
        if (json.find(std::string("\"") + name + "\":{") == std::string::npos) {
            throw std::runtime_error("Capability JSON is missing a backend");
        }
    }
    std::cout << json << '\n';
    return 0;
}
