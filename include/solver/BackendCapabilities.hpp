#pragma once

#include <array>
#include <string>
#include <string_view>

namespace flood {

struct BackendCapability {
    std::string_view name;
    std::string_view status;
    bool implemented;
    bool toolchainDetected;
    bool runtimeDetected;
    std::string_view detail;
};

inline std::array<BackendCapability, 3> backendCapabilities() {
#ifdef FLOOD_HAS_OPENMP
    constexpr BackendCapability openmp{"openmp", "AVAILABLE", true, true, true,
                                       "OpenMP solver is compiled"};
#else
    constexpr BackendCapability openmp{"openmp", "UNAVAILABLE", false, false, false,
                                       "OpenMP solver was not compiled"};
#endif
#ifdef FLOOD_MPI_TOOLCHAIN_FOUND
#ifdef FLOOD_MPI_RUNTIME_FOUND
#ifdef FLOOD_HAS_MPI
    constexpr BackendCapability mpi{"mpi", "AVAILABLE", true, true, true,
                                     "MPI solver is compiled"};
#else
    constexpr BackendCapability mpi{"mpi", "UNAVAILABLE", false, true, true,
                                     "MPI toolchain/runtime detected; MPI solver was not compiled"};
#endif
#else
#ifdef FLOOD_HAS_MPI
    constexpr BackendCapability mpi{"mpi", "UNAVAILABLE", true, true, false,
                                     "MPI solver is compiled but no launcher/runtime was detected"};
#else
    constexpr BackendCapability mpi{"mpi", "UNAVAILABLE", false, true, false,
                                     "MPI toolchain detected; MPI solver was not compiled"};
#endif
#endif
#else
#ifdef FLOOD_MPI_SOURCE_AVAILABLE
    constexpr BackendCapability mpi{"mpi", "UNAVAILABLE", true, false, false,
                                     "MPI solver source exists but MPI compiler/runtime is unavailable"};
#else
    constexpr BackendCapability mpi{"mpi", "UNAVAILABLE", false, false, false,
                                     "MPI solver source is not present and MPI compiler/runtime unavailable"};
#endif
#endif
    return {{{"serial", "AVAILABLE", true, true, true, "serial reference solver is compiled"},
             openmp, mpi}};
}

inline std::string backendCapabilitiesJson() {
    std::string json = "{";
    bool first = true;
    for (const auto& capability : backendCapabilities()) {
        if (!first) json += ',';
        first = false;
        json += "\"" + std::string(capability.name) + "\":{";
        json += "\"status\":\"" + std::string(capability.status) + "\",";
        json += "\"implemented\":" + std::string(capability.implemented ? "true" : "false") + ",";
        json += "\"toolchain_detected\":" + std::string(capability.toolchainDetected ? "true" : "false") + ",";
        json += "\"runtime_detected\":" + std::string(capability.runtimeDetected ? "true" : "false") + ",";
        json += "\"detail\":\"" + std::string(capability.detail) + "\"}";
    }
    return json + "}";
}

inline std::string backendUnavailableReason(std::string_view name) {
    for (const auto& capability : backendCapabilities()) {
        if (capability.name == name) return std::string(capability.detail);
    }
    return "unknown backend: " + std::string(name);
}

} // namespace flood