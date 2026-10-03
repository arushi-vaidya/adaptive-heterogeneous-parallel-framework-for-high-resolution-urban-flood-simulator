#include "solver/MpiSolver.hpp"

#include "solver/detail/FiniteVolumeKernels.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace flood {
namespace {

using detail::finite_volume::FaceFlux;
using detail::finite_volume::State;
using detail::finite_volume::ghostForOutflow;
using detail::finite_volume::ghostForWall;
using detail::finite_volume::rusanovFace;

constexpr int stateWidth = 4;
constexpr int tagNorth = 101;
constexpr int tagSouth = 102;
constexpr int tagWest = 103;
constexpr int tagEast = 104;

struct Block {
    std::size_t rowStart;
    std::size_t colStart;
    std::size_t rows;
    std::size_t cols;
};

struct CompensatedSum {
    double value = 0.0;
    double correction = 0.0;

    void add(double term) {
        const double adjusted = term - correction;
        const double next = value + adjusted;
        correction = (next - value) - adjusted;
        value = next;
    }
};

class MpiSession {
public:
    MpiSession() {
        int initialized = 0;
        const int queryStatus = MPI_Initialized(&initialized);
        if (queryStatus != MPI_SUCCESS) throw std::runtime_error("MPI_Initialized failed");
        if (!initialized) {
            int argc = 0;
            char** argv = nullptr;
            const int initStatus = MPI_Init(&argc, &argv);
            if (initStatus != MPI_SUCCESS) throw std::runtime_error("MPI_Init failed");
            owns_ = true;
        }
    }

    ~MpiSession() {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (owns_ && !finalized) MPI_Finalize();
    }

    MpiSession(const MpiSession&) = delete;
    MpiSession& operator=(const MpiSession&) = delete;

private:
    bool owns_ = false;
};

void checkMpi(int status, const char* operation) {
    if (status == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, message, &length);
    throw std::runtime_error(std::string(operation) + " failed: " +
                             std::string(message, static_cast<std::size_t>(length)));
}

int checkedMpiCount(std::size_t count) {
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("MPI message exceeds the supported count range");
    return static_cast<int>(count);
}

Block blockFor(std::size_t rows, std::size_t cols, const int coords[2], const int dims[2]) {
    const std::size_t rowStart = rows * static_cast<std::size_t>(coords[0]) /
                                 static_cast<std::size_t>(dims[0]);
    const std::size_t rowEnd = rows * static_cast<std::size_t>(coords[0] + 1) /
                               static_cast<std::size_t>(dims[0]);
    const std::size_t colStart = cols * static_cast<std::size_t>(coords[1]) /
                                 static_cast<std::size_t>(dims[1]);
    const std::size_t colEnd = cols * static_cast<std::size_t>(coords[1] + 1) /
                               static_cast<std::size_t>(dims[1]);
    return {rowStart, colStart, rowEnd - rowStart, colEnd - colStart};
}

std::size_t cellIndex(std::size_t row, std::size_t col, std::size_t stride) {
    return row * stride + col;
}

std::array<double, stateWidth> packCell(const Cell& cell) {
    return {cell.bed, cell.h, cell.hu, cell.hv};
}

Cell unpackCell(const double* values) {
    return {values[0], values[1], values[2], values[3]};
}

void writeCell(double* values, const Cell& cell) {
    values[0] = cell.bed;
    values[1] = cell.h;
    values[2] = cell.hu;
    values[3] = cell.hv;
}

void fillPhysicalGhosts(std::vector<Cell>& cells, const Block& block,
                        std::size_t stride, const int coords[2], const int dims[2],
                        BoundaryCondition boundary) {
    if (coords[0] == 0) {
        for (std::size_t col = 1; col <= block.cols; ++col) {
            const Cell interior = cells[cellIndex(1, col, stride)];
            cells[cellIndex(0, col, stride)] = boundary == BoundaryCondition::Closed
                ? ghostForWall(interior, false) : ghostForOutflow(interior, false, true);
        }
    }
    if (coords[0] == dims[0] - 1) {
        for (std::size_t col = 1; col <= block.cols; ++col) {
            const Cell interior = cells[cellIndex(block.rows, col, stride)];
            cells[cellIndex(block.rows + 1, col, stride)] = boundary == BoundaryCondition::Closed
                ? ghostForWall(interior, false) : ghostForOutflow(interior, false, false);
        }
    }
    if (coords[1] == 0) {
        for (std::size_t row = 1; row <= block.rows; ++row) {
            const Cell interior = cells[cellIndex(row, 1, stride)];
            cells[cellIndex(row, 0, stride)] = boundary == BoundaryCondition::Closed
                ? ghostForWall(interior, true) : ghostForOutflow(interior, true, true);
        }
    }
    if (coords[1] == dims[1] - 1) {
        for (std::size_t row = 1; row <= block.rows; ++row) {
            const Cell interior = cells[cellIndex(row, block.cols, stride)];
            cells[cellIndex(row, block.cols + 1, stride)] = boundary == BoundaryCondition::Closed
                ? ghostForWall(interior, true) : ghostForOutflow(interior, true, false);
        }
    }
}

void exchangeStateHalos(std::vector<Cell>& cells, const Block& block,
                        std::size_t stride, int north, int south, int west, int east,
                        MPI_Comm comm, double& communicationSeconds) {
    std::vector<double> sendNorth(block.cols * stateWidth), receiveNorth(block.cols * stateWidth);
    std::vector<double> sendSouth(block.cols * stateWidth), receiveSouth(block.cols * stateWidth);
    std::vector<double> sendWest(block.rows * stateWidth), receiveWest(block.rows * stateWidth);
    std::vector<double> sendEast(block.rows * stateWidth), receiveEast(block.rows * stateWidth);
    for (std::size_t col = 1; col <= block.cols; ++col) {
        writeCell(sendNorth.data() + (col - 1) * stateWidth, cells[cellIndex(1, col, stride)]);
        writeCell(sendSouth.data() + (col - 1) * stateWidth,
                  cells[cellIndex(block.rows, col, stride)]);
    }
    for (std::size_t row = 1; row <= block.rows; ++row) {
        writeCell(sendWest.data() + (row - 1) * stateWidth, cells[cellIndex(row, 1, stride)]);
        writeCell(sendEast.data() + (row - 1) * stateWidth,
                  cells[cellIndex(row, block.cols, stride)]);
    }

    std::array<MPI_Request, 8> requests{};
    int requestCount = 0;
    const double start = MPI_Wtime();
    if (north != MPI_PROC_NULL) {
        checkMpi(MPI_Irecv(receiveNorth.data(), checkedMpiCount(receiveNorth.size()), MPI_DOUBLE,
                           north, tagSouth, comm, &requests[requestCount++]), "MPI_Irecv north");
        checkMpi(MPI_Isend(sendNorth.data(), checkedMpiCount(sendNorth.size()), MPI_DOUBLE,
                           north, tagNorth, comm, &requests[requestCount++]), "MPI_Isend north");
    }
    if (south != MPI_PROC_NULL) {
        checkMpi(MPI_Irecv(receiveSouth.data(), checkedMpiCount(receiveSouth.size()), MPI_DOUBLE,
                           south, tagNorth, comm, &requests[requestCount++]), "MPI_Irecv south");
        checkMpi(MPI_Isend(sendSouth.data(), checkedMpiCount(sendSouth.size()), MPI_DOUBLE,
                           south, tagSouth, comm, &requests[requestCount++]), "MPI_Isend south");
    }
    if (west != MPI_PROC_NULL) {
        checkMpi(MPI_Irecv(receiveWest.data(), checkedMpiCount(receiveWest.size()), MPI_DOUBLE,
                           west, tagEast, comm, &requests[requestCount++]), "MPI_Irecv west");
        checkMpi(MPI_Isend(sendWest.data(), checkedMpiCount(sendWest.size()), MPI_DOUBLE,
                           west, tagWest, comm, &requests[requestCount++]), "MPI_Isend west");
    }
    if (east != MPI_PROC_NULL) {
        checkMpi(MPI_Irecv(receiveEast.data(), checkedMpiCount(receiveEast.size()), MPI_DOUBLE,
                           east, tagWest, comm, &requests[requestCount++]), "MPI_Irecv east");
        checkMpi(MPI_Isend(sendEast.data(), checkedMpiCount(sendEast.size()), MPI_DOUBLE,
                           east, tagEast, comm, &requests[requestCount++]), "MPI_Isend east");
    }
    if (requestCount > 0)
        checkMpi(MPI_Waitall(requestCount, requests.data(), MPI_STATUSES_IGNORE), "MPI_Waitall halos");
    communicationSeconds += MPI_Wtime() - start;

    if (north != MPI_PROC_NULL)
        for (std::size_t col = 1; col <= block.cols; ++col)
            cells[cellIndex(0, col, stride)] = unpackCell(receiveNorth.data() + (col - 1) * stateWidth);
    if (south != MPI_PROC_NULL)
        for (std::size_t col = 1; col <= block.cols; ++col)
            cells[cellIndex(block.rows + 1, col, stride)] = unpackCell(receiveSouth.data() + (col - 1) * stateWidth);
    if (west != MPI_PROC_NULL)
        for (std::size_t row = 1; row <= block.rows; ++row)
            cells[cellIndex(row, 0, stride)] = unpackCell(receiveWest.data() + (row - 1) * stateWidth);
    if (east != MPI_PROC_NULL)
        for (std::size_t row = 1; row <= block.rows; ++row)
            cells[cellIndex(row, block.cols + 1, stride)] = unpackCell(receiveEast.data() + (row - 1) * stateWidth);
}

void exchangeScalarHalos(std::vector<double>& field, const Block& block,
                         std::size_t stride, int north, int south, int west, int east,
                         MPI_Comm comm, double& communicationSeconds) {
    std::vector<double> sendNorth(block.cols), receiveNorth(block.cols);
    std::vector<double> sendSouth(block.cols), receiveSouth(block.cols);
    std::vector<double> sendWest(block.rows), receiveWest(block.rows);
    std::vector<double> sendEast(block.rows), receiveEast(block.rows);
    for (std::size_t col = 1; col <= block.cols; ++col) {
        sendNorth[col - 1] = field[cellIndex(1, col, stride)];
        sendSouth[col - 1] = field[cellIndex(block.rows, col, stride)];
    }
    for (std::size_t row = 1; row <= block.rows; ++row) {
        sendWest[row - 1] = field[cellIndex(row, 1, stride)];
        sendEast[row - 1] = field[cellIndex(row, block.cols, stride)];
    }
    std::array<MPI_Request, 8> requests{};
    int requestCount = 0;
    const double start = MPI_Wtime();
    if (north != MPI_PROC_NULL) {
        checkMpi(MPI_Irecv(receiveNorth.data(), checkedMpiCount(receiveNorth.size()), MPI_DOUBLE,
                           north, tagSouth, comm, &requests[requestCount++]), "MPI_Irecv north limiter");
        checkMpi(MPI_Isend(sendNorth.data(), checkedMpiCount(sendNorth.size()), MPI_DOUBLE,
                           north, tagNorth, comm, &requests[requestCount++]), "MPI_Isend north limiter");
    }
    if (south != MPI_PROC_NULL) {
        checkMpi(MPI_Irecv(receiveSouth.data(), checkedMpiCount(receiveSouth.size()), MPI_DOUBLE,
                           south, tagNorth, comm, &requests[requestCount++]), "MPI_Irecv south limiter");
        checkMpi(MPI_Isend(sendSouth.data(), checkedMpiCount(sendSouth.size()), MPI_DOUBLE,
                           south, tagSouth, comm, &requests[requestCount++]), "MPI_Isend south limiter");
    }
    if (west != MPI_PROC_NULL) {
        checkMpi(MPI_Irecv(receiveWest.data(), checkedMpiCount(receiveWest.size()), MPI_DOUBLE,
                           west, tagEast, comm, &requests[requestCount++]), "MPI_Irecv west limiter");
        checkMpi(MPI_Isend(sendWest.data(), checkedMpiCount(sendWest.size()), MPI_DOUBLE,
                           west, tagWest, comm, &requests[requestCount++]), "MPI_Isend west limiter");
    }
    if (east != MPI_PROC_NULL) {
        checkMpi(MPI_Irecv(receiveEast.data(), checkedMpiCount(receiveEast.size()), MPI_DOUBLE,
                           east, tagWest, comm, &requests[requestCount++]), "MPI_Irecv east limiter");
        checkMpi(MPI_Isend(sendEast.data(), checkedMpiCount(sendEast.size()), MPI_DOUBLE,
                           east, tagEast, comm, &requests[requestCount++]), "MPI_Isend east limiter");
    }
    if (requestCount > 0)
        checkMpi(MPI_Waitall(requestCount, requests.data(), MPI_STATUSES_IGNORE), "MPI_Waitall limiters");
    communicationSeconds += MPI_Wtime() - start;
    if (north != MPI_PROC_NULL)
        for (std::size_t col = 1; col <= block.cols; ++col) field[cellIndex(0, col, stride)] = receiveNorth[col - 1];
    if (south != MPI_PROC_NULL)
        for (std::size_t col = 1; col <= block.cols; ++col) field[cellIndex(block.rows + 1, col, stride)] = receiveSouth[col - 1];
    if (west != MPI_PROC_NULL)
        for (std::size_t row = 1; row <= block.rows; ++row) field[cellIndex(row, 0, stride)] = receiveWest[row - 1];
    if (east != MPI_PROC_NULL)
        for (std::size_t row = 1; row <= block.rows; ++row) field[cellIndex(row, block.cols + 1, stride)] = receiveEast[row - 1];
}

void recordFace(FaceFlux& face, bool lowPhysical, bool highPhysical) {
    if (face.massRate > 0.0) face.donor = face.left;
    else if (face.massRate < 0.0) face.donor = face.right;
    face.boundaryOutflow = (lowPhysical && face.massRate < 0.0) ||
                           (highPhysical && face.massRate > 0.0);
}

void validateConfig(const SolverConfig& config) {
    if (!(config.gravity > 0.0) || config.manningN < 0.0 ||
        config.infiltrationMetersPerSecond < 0.0 || !(config.dryDepth > 0.0) ||
        !(config.cfl > 0.0 && config.cfl <= 1.0) || !(config.maxTimestep > 0.0) ||
        config.endTime < 0.0 || !std::isfinite(config.gravity) ||
        !std::isfinite(config.manningN) || !std::isfinite(config.infiltrationMetersPerSecond) ||
        !std::isfinite(config.dryDepth) || !std::isfinite(config.cfl) ||
        !std::isfinite(config.maxTimestep) || !std::isfinite(config.endTime))
        throw std::invalid_argument("Invalid MPI solver configuration");
}

} // namespace

Diagnostics MpiSolver::run(Grid& grid, const Rainfall& rainfall,
                           const SolverConfig& config) const {
    MpiSession session;
    validateConfig(config);
    const double totalStart = MPI_Wtime();
    const double setupStart = totalStart;
    checkMpi(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN), "MPI_Comm_set_errhandler");
    int worldSize = 0;
    checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &worldSize), "MPI_Comm_size");

    const std::size_t globalRows = grid.rows(), globalCols = grid.cols();
    const double dx = grid.dx(), dy = grid.dy();
    int dims[2] = {0, 0};
    checkMpi(MPI_Dims_create(worldSize, 2, dims), "MPI_Dims_create");
    int periods[2] = {0, 0};
    MPI_Comm cart = MPI_COMM_NULL;
    checkMpi(MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart), "MPI_Cart_create");
    if (cart == MPI_COMM_NULL) throw std::runtime_error("MPI Cartesian communicator creation failed");
    checkMpi(MPI_Comm_set_errhandler(cart, MPI_ERRORS_RETURN), "MPI_Comm_set_errhandler cart");
    int rank = 0, coords[2] = {0, 0};
    checkMpi(MPI_Comm_rank(cart, &rank), "MPI_Comm_rank cart");
    checkMpi(MPI_Cart_coords(cart, rank, 2, coords), "MPI_Cart_coords");
    int north = MPI_PROC_NULL, south = MPI_PROC_NULL, west = MPI_PROC_NULL, east = MPI_PROC_NULL;
    checkMpi(MPI_Cart_shift(cart, 0, 1, &north, &south), "MPI_Cart_shift rows");
    checkMpi(MPI_Cart_shift(cart, 1, 1, &west, &east), "MPI_Cart_shift columns");

    const Block block = blockFor(globalRows, globalCols, coords, dims);
    const int localValid = block.rows > 0 && block.cols > 0 ? 1 : 0;
    int allValid = 0;
    checkMpi(MPI_Allreduce(&localValid, &allValid, 1, MPI_INT, MPI_MIN, cart),
             "MPI_Allreduce domain validity");
    if (!allValid) {
        MPI_Comm_free(&cart);
        throw std::invalid_argument("MPI decomposition creates an empty subdomain; reduce process count");
    }

    const bool globalCountFits = globalRows <= std::numeric_limits<std::size_t>::max() / globalCols &&
        globalRows * globalCols <= static_cast<std::size_t>(std::numeric_limits<int>::max()) / stateWidth;
    const int localCountFits = globalCountFits ? 1 : 0;
    int allCountsFit = 0;
    checkMpi(MPI_Allreduce(&localCountFits, &allCountsFit, 1, MPI_INT, MPI_MIN, cart),
             "MPI_Allreduce message size validation");
    if (!allCountsFit) {
        MPI_Comm_free(&cart);
        throw std::invalid_argument("Global grid exceeds MPI collective count range");
    }

    std::vector<int> counts(static_cast<std::size_t>(worldSize));
    std::vector<int> displacements(static_cast<std::size_t>(worldSize));
    std::vector<double> rootPacked;
    int totalPacked = 0;
    if (rank == 0) {
        for (int process = 0; process < worldSize; ++process) {
            int processCoords[2] = {0, 0};
            checkMpi(MPI_Cart_coords(cart, process, 2, processCoords), "MPI_Cart_coords scatter");
            const Block processBlock = blockFor(globalRows, globalCols, processCoords, dims);
            const std::size_t count = processBlock.rows * processBlock.cols * stateWidth;
            counts[static_cast<std::size_t>(process)] = checkedMpiCount(count);
            displacements[static_cast<std::size_t>(process)] = totalPacked;
            totalPacked += static_cast<int>(count);
        }
        rootPacked.resize(static_cast<std::size_t>(totalPacked));
        for (int process = 0; process < worldSize; ++process) {
            int processCoords[2] = {0, 0};
            checkMpi(MPI_Cart_coords(cart, process, 2, processCoords), "MPI_Cart_coords pack");
            const Block processBlock = blockFor(globalRows, globalCols, processCoords, dims);
            std::size_t cursor = static_cast<std::size_t>(displacements[static_cast<std::size_t>(process)]);
            for (std::size_t row = 0; row < processBlock.rows; ++row) {
                for (std::size_t col = 0; col < processBlock.cols; ++col) {
                    const auto packed = packCell(grid.at(processBlock.rowStart + row,
                                                          processBlock.colStart + col));
                    std::copy(packed.begin(), packed.end(), rootPacked.begin() +
                              static_cast<std::ptrdiff_t>(cursor));
                    cursor += stateWidth;
                }
            }
        }
    }

    const std::size_t localCellCount = block.rows * block.cols;
    std::vector<double> localPacked(localCellCount * stateWidth);
    checkMpi(MPI_Scatterv(rank == 0 ? rootPacked.data() : nullptr,
                         rank == 0 ? counts.data() : nullptr,
                         rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE,
                         localPacked.data(), checkedMpiCount(localPacked.size()), MPI_DOUBLE,
                         0, cart), "MPI_Scatterv initial state");

    const std::size_t stride = block.cols + 2;
    std::vector<Cell> cells((block.rows + 2) * stride);
    std::size_t packedCursor = 0;
    for (std::size_t row = 0; row < block.rows; ++row) {
        for (std::size_t col = 0; col < block.cols; ++col) {
            cells[cellIndex(row + 1, col + 1, stride)] = unpackCell(localPacked.data() + packedCursor);
            packedCursor += stateWidth;
        }
    }
    if (rank != 0) grid = Grid(1, 1, dx, dy);

    const double area = dx * dy;
    CompensatedSum initialLocalSum;
    int localStateValid = 1;
    for (std::size_t row = 1; row <= block.rows; ++row) {
        for (std::size_t col = 1; col <= block.cols; ++col) {
            const Cell& cell = cells[cellIndex(row, col, stride)];
            if (cell.h < 0.0 || !std::isfinite(cell.h) || !std::isfinite(cell.bed) ||
                !std::isfinite(cell.hu) || !std::isfinite(cell.hv)) localStateValid = 0;
            initialLocalSum.add(cell.h * area);
        }
    }
    int allStateValid = 0;
    checkMpi(MPI_Allreduce(&localStateValid, &allStateValid, 1, MPI_INT, MPI_MIN, cart),
             "MPI_Allreduce initial state validity");
    if (!allStateValid) {
        MPI_Comm_free(&cart);
        throw std::invalid_argument("MPI input contains invalid cell state");
    }
    double initialLocal = initialLocalSum.value, initialGlobal = 0.0;
    checkMpi(MPI_Allreduce(&initialLocal, &initialGlobal, 1, MPI_DOUBLE, MPI_SUM, cart),
             "MPI_Allreduce initial volume");
    const double setupSeconds = MPI_Wtime() - setupStart;

    double communicationSeconds = 0.0, synchronizationSeconds = 0.0, computationSeconds = 0.0;
    CompensatedSum rainfallLocalSum, infiltrationLocalSum, outflowLocalSum;
    double time = 0.0;
    std::size_t steps = 0;
    double syncStart = 0.0;
    while (time < config.endTime) {
        double localMaxRate = 0.0;
        const double computationStart = MPI_Wtime();
        for (std::size_t row = 1; row <= block.rows; ++row) {
            for (std::size_t col = 1; col <= block.cols; ++col) {
                const Cell& cell = cells[cellIndex(row, col, stride)];
                if (cell.h <= config.dryDepth) continue;
                const double u = cell.hu / cell.h, v = cell.hv / cell.h;
                const double wave = std::sqrt(config.gravity * cell.h);
                localMaxRate = std::max(localMaxRate, (std::abs(u) + wave) / dx +
                                                        (std::abs(v) + wave) / dy);
            }
        }
        computationSeconds += MPI_Wtime() - computationStart;
        double globalMaxRate = 0.0;
        syncStart = MPI_Wtime();
        checkMpi(MPI_Allreduce(&localMaxRate, &globalMaxRate, 1, MPI_DOUBLE, MPI_MAX, cart),
                 "MPI_Allreduce CFL rate");
        synchronizationSeconds += MPI_Wtime() - syncStart;
        double dt = globalMaxRate > 0.0 ? config.cfl / globalMaxRate : config.maxTimestep;
        dt = std::min({dt, config.maxTimestep, config.endTime - time});
        const double nextRainChange = rainfall.nextChangeAfter(time);
        if (nextRainChange > time) dt = std::min(dt, nextRainChange - time);
        if (!(dt > 0.0) || !std::isfinite(dt)) {
            MPI_Comm_free(&cart);
            throw std::runtime_error("MPI global CFL timestep became invalid");
        }

        fillPhysicalGhosts(cells, block, stride, coords, dims, config.boundary);
        exchangeStateHalos(cells, block, stride, north, south, west, east,
                           cart, communicationSeconds);

        const double faceStart = MPI_Wtime();
        const std::size_t xFaceCount = block.rows * (block.cols + 1);
        const std::size_t yFaceCount = (block.rows + 1) * block.cols;
        std::vector<FaceFlux> xFaces(xFaceCount), yFaces(yFaceCount);
        auto localIndexFor = [stride](std::size_t row, std::size_t col) {
            return cellIndex(row, col, stride);
        };
        for (std::size_t row = 0; row < block.rows; ++row) {
            for (std::size_t faceCol = 0; faceCol <= block.cols; ++faceCol) {
                const std::size_t stateRow = row + 1;
                const std::size_t leftIndex = localIndexFor(stateRow, faceCol);
                const std::size_t rightIndex = localIndexFor(stateRow, faceCol + 1);
                auto& face = xFaces[row * (block.cols + 1) + faceCol];
                face = rusanovFace(cells[leftIndex], cells[rightIndex], leftIndex, rightIndex,
                                   true, config.gravity, config.dryDepth);
                const bool lowPhysical = faceCol == 0 && west == MPI_PROC_NULL;
                const bool highPhysical = faceCol == block.cols && east == MPI_PROC_NULL;
                recordFace(face, lowPhysical, highPhysical);
            }
        }
        for (std::size_t faceRow = 0; faceRow <= block.rows; ++faceRow) {
            for (std::size_t col = 0; col < block.cols; ++col) {
                const std::size_t stateCol = col + 1;
                const std::size_t leftIndex = localIndexFor(faceRow, stateCol);
                const std::size_t rightIndex = localIndexFor(faceRow + 1, stateCol);
                auto& face = yFaces[faceRow * block.cols + col];
                face = rusanovFace(cells[leftIndex], cells[rightIndex], leftIndex, rightIndex,
                                   false, config.gravity, config.dryDepth);
                const bool lowPhysical = faceRow == 0 && north == MPI_PROC_NULL;
                const bool highPhysical = faceRow == block.rows && south == MPI_PROC_NULL;
                recordFace(face, lowPhysical, highPhysical);
            }
        }

        const std::size_t stateCount = cells.size();
        std::vector<double> outgoing(stateCount, 0.0), limiter(stateCount, 1.0);
        for (std::size_t row = 0; row < block.rows; ++row) {
            for (std::size_t col = 0; col < block.cols; ++col) {
                const std::size_t localCell = localIndexFor(row + 1, col + 1);
                const std::array<const FaceFlux*, 4> faces = {
                    &xFaces[row * (block.cols + 1) + col],
                    &xFaces[row * (block.cols + 1) + col + 1],
                    &yFaces[row * block.cols + col],
                    &yFaces[(row + 1) * block.cols + col]};
                for (const FaceFlux* face : faces)
                    if (face->donor == localCell) outgoing[localCell] += std::abs(face->massRate);
                const double available = cells[localCell].h * area;
                if (outgoing[localCell] * dt > available && outgoing[localCell] > 0.0)
                    limiter[localCell] = available / (outgoing[localCell] * dt);
            }
        }
            computationSeconds += MPI_Wtime() - faceStart;
        exchangeScalarHalos(limiter, block, stride, north, south, west, east,
                            cart, communicationSeconds);

            const double updateStart = MPI_Wtime();
        std::vector<State> delta(stateCount, State{0.0, 0.0, 0.0});
        const auto addContribution = [&](const FaceFlux& face, bool cellIsRight,
                                         double faceLength, State& cellDelta) {
            const double scale = face.donor == std::numeric_limits<std::size_t>::max()
                ? 1.0 : limiter[face.donor];
            const State& flux = cellIsRight ? face.rightFlux : face.leftFlux;
            const double sign = cellIsRight ? 1.0 : -1.0;
            for (std::size_t component = 0; component < 3; ++component)
                cellDelta[component] += sign * scale * flux[component] * faceLength / area;
        };
        for (std::size_t row = 0; row < block.rows; ++row) {
            for (std::size_t col = 0; col < block.cols; ++col) {
                const std::size_t localCell = localIndexFor(row + 1, col + 1);
                State cellDelta{0.0, 0.0, 0.0};
                addContribution(xFaces[row * (block.cols + 1) + col], true, dy, cellDelta);
                addContribution(xFaces[row * (block.cols + 1) + col + 1], false, dy, cellDelta);
                addContribution(yFaces[row * block.cols + col], true, dx, cellDelta);
                addContribution(yFaces[(row + 1) * block.cols + col], false, dx, cellDelta);
                delta[localCell] = cellDelta;
            }
        }
        for (const auto& face : xFaces) {
            if (!face.boundaryOutflow) continue;
            outflowLocalSum.add(limiter[face.donor] * std::abs(face.massRate) * dy * dt);
        }
        for (const auto& face : yFaces) {
            if (!face.boundaryOutflow) continue;
            outflowLocalSum.add(limiter[face.donor] * std::abs(face.massRate) * dx * dt);
        }

        const double rainRate = rainfall.metersPerSecond(time);
        rainfallLocalSum.add(rainRate * area * static_cast<double>(localCellCount) * dt);
        int localInvalid = 0;
        for (std::size_t row = 0; row < block.rows; ++row) {
            for (std::size_t col = 0; col < block.cols; ++col) {
                const std::size_t localCell = localIndexFor(row + 1, col + 1);
                Cell& cell = cells[localCell];
                cell.h += dt * (delta[localCell][0] + rainRate);
                cell.hu += dt * delta[localCell][1];
                cell.hv += dt * delta[localCell][2];
                if (cell.h < 0.0 && cell.h > -1e-12) cell.h = 0.0;
                if (cell.h < 0.0 || !std::isfinite(cell.h) || !std::isfinite(cell.hu) ||
                    !std::isfinite(cell.hv)) {
                    localInvalid = 1;
                    continue;
                }
                const double infiltrated = std::min(cell.h, config.infiltrationMetersPerSecond * dt);
                cell.h -= infiltrated;
                infiltrationLocalSum.add(infiltrated * area);
                if (cell.h <= config.dryDepth) {
                    cell.hu = 0.0;
                    cell.hv = 0.0;
                } else if (config.manningN > 0.0) {
                    const double u = cell.hu / cell.h, v = cell.hv / cell.h;
                    const double speed = std::hypot(u, v);
                    const double coefficient = config.gravity * config.manningN * config.manningN *
                                               speed / std::pow(cell.h, 4.0 / 3.0);
                    const double frictionScale = 1.0 / (1.0 + dt * coefficient);
                    cell.hu *= frictionScale;
                    cell.hv *= frictionScale;
                }
            }
        }
        computationSeconds += MPI_Wtime() - updateStart;
        int anyInvalid = 0;
        syncStart = MPI_Wtime();
        checkMpi(MPI_Allreduce(&localInvalid, &anyInvalid, 1, MPI_INT, MPI_MAX, cart),
                 "MPI_Allreduce state validity");
        synchronizationSeconds += MPI_Wtime() - syncStart;
        if (anyInvalid) {
            MPI_Comm_free(&cart);
            throw std::runtime_error("MPI timestep produced an invalid state");
        }
        time += dt;
        ++steps;
    }

    CompensatedSum storedLocalSum;
    double localMaxDepth = 0.0, localMinDepth = std::numeric_limits<double>::infinity();
    double localMaxVelocity = 0.0;
    int localWetCells = 0;
    std::vector<double> localOutput(localCellCount * stateWidth);
    packedCursor = 0;
    for (std::size_t row = 0; row < block.rows; ++row) {
        for (std::size_t col = 0; col < block.cols; ++col) {
            const Cell& cell = cells[cellIndex(row + 1, col + 1, stride)];
            storedLocalSum.add(cell.h * area);
            localMaxDepth = std::max(localMaxDepth, cell.h);
            localMinDepth = std::min(localMinDepth, cell.h);
            if (cell.h > config.dryDepth) {
                ++localWetCells;
                localMaxVelocity = std::max(localMaxVelocity,
                    std::hypot(cell.hu, cell.hv) / cell.h);
            }
            writeCell(localOutput.data() + packedCursor, cell);
            packedCursor += stateWidth;
        }
    }
    std::vector<double> gathered;
    if (rank == 0) gathered.resize(static_cast<std::size_t>(totalPacked));
    syncStart = MPI_Wtime();
    checkMpi(MPI_Gatherv(localOutput.data(), checkedMpiCount(localOutput.size()), MPI_DOUBLE,
                        rank == 0 ? gathered.data() : nullptr,
                        rank == 0 ? counts.data() : nullptr,
                        rank == 0 ? displacements.data() : nullptr,
                        MPI_DOUBLE, 0, cart), "MPI_Gatherv final state");
    if (rank == 0) {
        for (int process = 0; process < worldSize; ++process) {
            int processCoords[2] = {0, 0};
            checkMpi(MPI_Cart_coords(cart, process, 2, processCoords), "MPI_Cart_coords gather");
            const Block processBlock = blockFor(globalRows, globalCols, processCoords, dims);
            std::size_t cursor = static_cast<std::size_t>(displacements[static_cast<std::size_t>(process)]);
            for (std::size_t row = 0; row < processBlock.rows; ++row) {
                for (std::size_t col = 0; col < processBlock.cols; ++col) {
                    grid.at(processBlock.rowStart + row, processBlock.colStart + col) =
                        unpackCell(gathered.data() + cursor);
                    cursor += stateWidth;
                }
            }
        }
    }
    synchronizationSeconds += MPI_Wtime() - syncStart;

    const std::array<double, 4> localSums = {storedLocalSum.value, rainfallLocalSum.value,
        infiltrationLocalSum.value, outflowLocalSum.value};
    std::array<double, 4> globalSums{};
    double globalMaxDepth = 0.0, globalMinDepth = 0.0, globalMaxVelocity = 0.0;
    int globalWetCells = 0;
    syncStart = MPI_Wtime();
    checkMpi(MPI_Allreduce(localSums.data(), globalSums.data(), 4, MPI_DOUBLE, MPI_SUM, cart),
             "MPI_Allreduce volumes");
    checkMpi(MPI_Allreduce(&localMaxDepth, &globalMaxDepth, 1, MPI_DOUBLE, MPI_MAX, cart),
             "MPI_Allreduce maximum depth");
    checkMpi(MPI_Allreduce(&localMinDepth, &globalMinDepth, 1, MPI_DOUBLE, MPI_MIN, cart),
             "MPI_Allreduce minimum depth");
    checkMpi(MPI_Allreduce(&localMaxVelocity, &globalMaxVelocity, 1, MPI_DOUBLE, MPI_MAX, cart),
             "MPI_Allreduce maximum velocity");
    checkMpi(MPI_Allreduce(&localWetCells, &globalWetCells, 1, MPI_INT, MPI_SUM, cart),
             "MPI_Allreduce wet cells");
    synchronizationSeconds += MPI_Wtime() - syncStart;

    struct TimedRank { double seconds; int rank; } localCritical{MPI_Wtime() - totalStart, rank},
        critical{};
    checkMpi(MPI_Allreduce(&localCritical, &critical, 1, MPI_DOUBLE_INT, MPI_MAXLOC, cart),
             "MPI_Allreduce critical rank timing");
    const double localOther = localCritical.seconds - setupSeconds - computationSeconds -
                              communicationSeconds - synchronizationSeconds;
    double criticalPhases[5] = {setupSeconds, computationSeconds, communicationSeconds,
                                synchronizationSeconds, localOther};
    checkMpi(MPI_Bcast(criticalPhases, 5, MPI_DOUBLE, critical.rank, cart),
             "MPI_Bcast critical rank timings");

    Diagnostics diagnostics;
    diagnostics.time = time;
    diagnostics.steps = steps;
    diagnostics.storedVolume = globalSums[0];
    diagnostics.rainfallVolume = globalSums[1];
    diagnostics.infiltrationVolume = globalSums[2];
    diagnostics.outflowVolume = globalSums[3];
    diagnostics.massResidual = initialGlobal + globalSums[1] - globalSums[3] -
                               globalSums[2] - globalSums[0];
    diagnostics.maxDepth = globalMaxDepth;
    diagnostics.minDepth = globalMinDepth;
    diagnostics.maxVelocity = globalMaxVelocity;
    diagnostics.wetCells = static_cast<std::size_t>(globalWetCells);
    diagnostics.floodedArea = static_cast<double>(globalWetCells) * area;
    diagnostics.solverSeconds = critical.seconds;
    diagnostics.setupSeconds = criticalPhases[0];
    diagnostics.computationSeconds = criticalPhases[1];
    diagnostics.communicationSeconds = criticalPhases[2];
    diagnostics.synchronizationSeconds = criticalPhases[3];
    diagnostics.otherSeconds = criticalPhases[4];
    diagnostics.backend = "mpi";
    diagnostics.processes = worldSize;
    diagnostics.threads = 1;
    diagnostics.isRoot = rank == 0;

    checkMpi(MPI_Comm_free(&cart), "MPI_Comm_free");
    return diagnostics;
}

} // namespace flood