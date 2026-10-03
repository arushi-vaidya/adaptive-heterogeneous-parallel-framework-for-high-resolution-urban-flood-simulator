#include "grid/Grid.hpp"

#include <cmath>
#include <stdexcept>

namespace flood {
namespace {

std::size_t checkedCellCount(std::size_t rows, std::size_t cols, double dx, double dy) {
    if (rows == 0 || cols == 0 || !std::isfinite(dx) || !std::isfinite(dy) ||
        dx <= 0.0 || dy <= 0.0 || rows > static_cast<std::size_t>(-1) / cols) {
        throw std::invalid_argument("Grid dimensions and cell sizes must be positive and finite");
    }
    return rows * cols;
}

} // namespace

Grid::Grid(std::size_t rows, std::size_t cols, double dx, double dy)
    : rows_(rows), cols_(cols), dx_(dx), dy_(dy), cells_(checkedCellCount(rows, cols, dx, dy)) {}

Cell& Grid::at(std::size_t row, std::size_t col) {
    return cells_.at(row * cols_ + col);
}

const Cell& Grid::at(std::size_t row, std::size_t col) const {
    return cells_.at(row * cols_ + col);
}

} // namespace flood