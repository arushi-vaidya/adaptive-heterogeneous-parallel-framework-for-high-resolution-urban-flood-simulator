// CUDA backend for the uniform-grid shallow-water solver.
// State lives on the device for the whole run. Each timestep launches:
//   1. a CFL kernel + Thrust max-reduction for dt,
//   2. x- and y-face Rusanov flux kernels (one thread per face),
//   3. a positivity-limiter kernel (one thread per cell),
//   4. a boundary-outflow kernel + Thrust sum,
//   5. a cell-update kernel (fluxes, rain, infiltration, Manning friction).
// The per-thread bodies are the shared FlatKernels used by OpenACC, which are
// regression-tested on the host against the serial reference solver.

#include "solver/AcceleratorSolvers.hpp"
#include "solver/detail/FlatKernels.hpp"
#include "solver/detail/FlatSolverCommon.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <chrono>
#include <stdexcept>
#include <string>

namespace flood {
namespace {

using detail::flat::Face;
using detail::flat::Params;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(status));
}

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    explicit DeviceBuffer(std::size_t count) { check(cudaMalloc(&ptr, count * sizeof(T)), "malloc"); }
    ~DeviceBuffer() { cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__global__ void cflKernel(Params p, const double* h, const double* hu, const double* hv,
                          double* rate) {
    const long i = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    if (i < p.rows * p.cols) rate[i] = detail::flat::cflRate(p, h[i], hu[i], hv[i]);
}

__global__ void xFaceKernel(Params p, const double* bed, const double* h, const double* hu,
                            const double* hv, Face* xf) {
    const long f = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    if (f >= p.rows * (p.cols + 1)) return;
    const long row = f / (p.cols + 1), faceCol = f % (p.cols + 1);
    detail::flat::xFace(p, bed, h, hu, hv, row, faceCol, xf[f]);
}

__global__ void yFaceKernel(Params p, const double* bed, const double* h, const double* hu,
                            const double* hv, Face* yf) {
    const long f = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    if (f >= (p.rows + 1) * p.cols) return;
    const long faceRow = f / p.cols, col = f % p.cols;
    detail::flat::yFace(p, bed, h, hu, hv, faceRow, col, yf[f]);
}

__global__ void limiterKernel(Params p, const double* h, const Face* xf, const Face* yf,
                              double dt, double* limiter) {
    const long i = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    if (i >= p.rows * p.cols) return;
    limiter[i] = detail::flat::limiterFor(h[i], p.dx * p.dy,
                                          detail::flat::outgoingRate(p, xf, yf, i), dt);
}

__global__ void outflowKernel(long count, const Face* faces, const double* limiter,
                              double length, double dt, double* volume) {
    const long f = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    if (f < count) volume[f] = detail::flat::faceOutflowVolume(faces[f], limiter, length, dt);
}

__global__ void updateKernel(Params p, const Face* xf, const Face* yf, const double* limiter,
                             double dt, double rainRate, double* h, double* hu, double* hv,
                             double* infiltrated, int* invalid) {
    const long i = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    if (i >= p.rows * p.cols) return;
    int bad = 0;
    infiltrated[i] = detail::flat::updateCell(p, xf, yf, limiter, i, dt, rainRate, h, hu, hv, bad);
    invalid[i] = bad;
}

constexpr int kBlock = 256;
inline unsigned blocksFor(long count) { return static_cast<unsigned>((count + kBlock - 1) / kBlock); }

} // namespace

Diagnostics CudaSolver::run(Grid& grid, const Rainfall& rainfall,
                            const SolverConfig& config) const {
    using namespace detail::flat;
    validateFlatConfig(grid, config);
    const auto start = std::chrono::steady_clock::now();
    int devices = 0;
    check(cudaGetDeviceCount(&devices), "device query");
    if (devices == 0) throw std::runtime_error("CUDA backend: no CUDA-capable GPU detected");
    check(cudaSetDevice(device_ % devices), "set device");

    const Params p = makeParams(grid, config);
    const long n = p.rows * p.cols;
    const long nx = p.rows * (p.cols + 1);
    const long ny = (p.rows + 1) * p.cols;
    const double area = p.dx * p.dy;
    SoA state = pack(grid);
    double initialVolume = 0.0;
    for (long i = 0; i < n; ++i) initialVolume += state.h[i] * area;

    DeviceBuffer<double> bed(n), h(n), hu(n), hv(n), limiter(n), scratch(n), infiltrated(n);
    DeviceBuffer<double> faceVolume(nx > ny ? nx : ny);
    DeviceBuffer<int> invalid(n);
    DeviceBuffer<Face> xf(nx), yf(ny);
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(double);
    check(cudaMemcpy(bed.ptr, state.bed.data(), bytes, cudaMemcpyHostToDevice), "copy bed");
    check(cudaMemcpy(h.ptr, state.h.data(), bytes, cudaMemcpyHostToDevice), "copy h");
    check(cudaMemcpy(hu.ptr, state.hu.data(), bytes, cudaMemcpyHostToDevice), "copy hu");
    check(cudaMemcpy(hv.ptr, state.hv.data(), bytes, cudaMemcpyHostToDevice), "copy hv");
    const auto setupDone = std::chrono::steady_clock::now();

    auto sum = [](double* ptr, long count) {
        thrust::device_ptr<double> d(ptr);
        return thrust::reduce(d, d + count, 0.0, thrust::plus<double>());
    };
    double rainfallVolume = 0.0, outflowVolume = 0.0, infiltrationVolume = 0.0, time = config.startTime;
    Diagnostics diagnostics;
    while (time < config.endTime) {
        cflKernel<<<blocksFor(n), kBlock>>>(p, h.ptr, hu.ptr, hv.ptr, scratch.ptr);
        thrust::device_ptr<double> rates(scratch.ptr);
        const double maxRate = thrust::reduce(rates, rates + n, 0.0, thrust::maximum<double>());
        const double dt = nextTimestep(maxRate, time, rainfall, config);
        const double rainRate = rainfall.metersPerSecond(time);

        xFaceKernel<<<blocksFor(nx), kBlock>>>(p, bed.ptr, h.ptr, hu.ptr, hv.ptr, xf.ptr);
        yFaceKernel<<<blocksFor(ny), kBlock>>>(p, bed.ptr, h.ptr, hu.ptr, hv.ptr, yf.ptr);
        limiterKernel<<<blocksFor(n), kBlock>>>(p, h.ptr, xf.ptr, yf.ptr, dt, limiter.ptr);

        outflowKernel<<<blocksFor(nx), kBlock>>>(nx, xf.ptr, limiter.ptr, p.dy, dt, faceVolume.ptr);
        double outflow = sum(faceVolume.ptr, nx);
        outflowKernel<<<blocksFor(ny), kBlock>>>(ny, yf.ptr, limiter.ptr, p.dx, dt, faceVolume.ptr);
        outflow += sum(faceVolume.ptr, ny);

        updateKernel<<<blocksFor(n), kBlock>>>(p, xf.ptr, yf.ptr, limiter.ptr, dt, rainRate,
                                               h.ptr, hu.ptr, hv.ptr, infiltrated.ptr, invalid.ptr);
        check(cudaGetLastError(), "kernel launch");
        thrust::device_ptr<int> bad(invalid.ptr);
        if (thrust::reduce(bad, bad + n, 0, thrust::plus<int>()) > 0)
            throw std::runtime_error("Non-finite or negative state detected; simulation stopped");
        infiltrationVolume += sum(infiltrated.ptr, n);
        outflowVolume += outflow;
        rainfallVolume += rainRate * area * static_cast<double>(n) * dt;
        time += dt;
        ++diagnostics.steps;
    }
    check(cudaDeviceSynchronize(), "synchronize");
    check(cudaMemcpy(state.h.data(), h.ptr, bytes, cudaMemcpyDeviceToHost), "copy back h");
    check(cudaMemcpy(state.hu.data(), hu.ptr, bytes, cudaMemcpyDeviceToHost), "copy back hu");
    check(cudaMemcpy(state.hv.data(), hv.ptr, bytes, cudaMemcpyDeviceToHost), "copy back hv");
    unpack(state, grid);
    finalize(grid, config, time, initialVolume, rainfallVolume, outflowVolume,
             infiltrationVolume, diagnostics);
    const auto end = std::chrono::steady_clock::now();
    diagnostics.solverSeconds = std::chrono::duration<double>(end - start).count();
    diagnostics.setupSeconds = std::chrono::duration<double>(setupDone - start).count();
    diagnostics.computationSeconds = diagnostics.solverSeconds - diagnostics.setupSeconds;
    diagnostics.backend = "cuda";
    diagnostics.threads = 1;
    return diagnostics;
}

} // namespace flood
