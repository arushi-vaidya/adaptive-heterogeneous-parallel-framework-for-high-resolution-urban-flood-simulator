// Compiled instead of CudaSolver.cu when CMake finds no CUDA toolchain.
#include "solver/AcceleratorSolvers.hpp"

#include <stdexcept>

namespace flood {

Diagnostics CudaSolver::run(Grid& grid, const Rainfall& rainfall,
                            const SolverConfig& config) const {
    (void)grid;
    (void)rainfall;
    (void)config;
    (void)device_;
    throw std::runtime_error(
        "CUDA backend unavailable: no CUDA toolchain (nvcc) was found at configure time");
}

} // namespace flood
