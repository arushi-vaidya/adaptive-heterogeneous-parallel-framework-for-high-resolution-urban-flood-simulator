#include "solver/AdaptiveMpiSolver.hpp"

#include <stdexcept>

#ifdef FLOOD_HAS_MPI
#include <mpi.h>

namespace {

void checkMpi(int status, const char* operation) {
    if (status == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, message, &length);
    throw std::runtime_error(std::string(operation) + " failed: " +
        std::string(message, static_cast<std::size_t>(length)));
}

class CartesianCommunicator {
public:
    explicit CartesianCommunicator(MPI_Comm parent) {
        int initialized = 0;
        checkMpi(MPI_Initialized(&initialized), "MPI_Initialized");
        if (!initialized)
            throw std::runtime_error("Adaptive MPI solver requires MPI_Init first");
        int size = 0;
        checkMpi(MPI_Comm_size(parent, &size), "MPI_Comm_size adaptive");
        int dimensions[2] = {0, 0};
        checkMpi(MPI_Dims_create(size, 2, dimensions), "MPI_Dims_create adaptive");
        int periods[2] = {0, 0};
        checkMpi(MPI_Cart_create(parent, 2, dimensions, periods, 0, &comm_),
                 "MPI_Cart_create adaptive");
        if (comm_ == MPI_COMM_NULL)
            throw std::runtime_error("Adaptive MPI Cartesian communicator creation failed");
        checkMpi(MPI_Comm_set_errhandler(comm_, MPI_ERRORS_RETURN),
                 "MPI_Comm_set_errhandler adaptive");
    }

    ~CartesianCommunicator() {
        if (comm_ != MPI_COMM_NULL) MPI_Comm_free(&comm_);
    }

    MPI_Comm get() const noexcept { return comm_; }

    CartesianCommunicator(const CartesianCommunicator&) = delete;
    CartesianCommunicator& operator=(const CartesianCommunicator&) = delete;

private:
    MPI_Comm comm_ = MPI_COMM_NULL;
};

} // namespace
#endif

namespace flood {

AdaptiveMpiSolver::AdaptiveMpiSolver(AdaptiveSolverOptions options)
    : options_(options) {}

#ifdef FLOOD_HAS_MPI
AdaptiveDiagnostics AdaptiveMpiSolver::run(
    AdaptiveGrid& grid, const Rainfall& rainfall, const SolverConfig& config,
    MPI_Comm communicator) const {
    CartesianCommunicator cartesian(communicator);
    return AdaptiveSolver(options_).runImpl(
        grid, rainfall, config, false, 1, cartesian.get());
}
#else
AdaptiveDiagnostics AdaptiveMpiSolver::run(
    AdaptiveGrid& grid, const Rainfall& rainfall,
    const SolverConfig& config) const {
    (void)grid;
    (void)rainfall;
    (void)config;
    throw std::runtime_error(
        "Adaptive MPI backend is unavailable; configure with FLOOD_ENABLE_MPI=ON");
}
#endif

} // namespace flood
