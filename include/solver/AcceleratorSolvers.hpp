#pragma once

#include "solver/SerialSolver.hpp"

namespace flood {

// OpenACC backend: directive-based offload of the flat finite-volume kernels.
// Compiled with OpenACC when the compiler supports it (FLOOD_HAS_OPENACC);
// otherwise run() reports the backend as unavailable.
class OpenACCSolver final : public SolverBackend {
public:
    Diagnostics run(Grid& grid, const Rainfall& rainfall,
                    const SolverConfig& config) const override;
};

// CUDA backend: one GPU thread per cell/face, device-resident state and
// Thrust reductions. Built only when CMake finds a CUDA toolchain.
class CudaSolver final : public SolverBackend {
public:
    explicit CudaSolver(int device = 0) : device_(device) {}
    Diagnostics run(Grid& grid, const Rainfall& rainfall,
                    const SolverConfig& config) const override;

private:
    int device_;
};

namespace detail {
// The OpenACC loop nest. Without -fopenacc the directives are ignored and it
// executes on the host, which is how the flat kernels are regression-tested
// against the serial reference on machines without an accelerator compiler.
Diagnostics runOpenACCSolver(Grid& grid, const Rainfall& rainfall,
                             const SolverConfig& config);
} // namespace detail

} // namespace flood
