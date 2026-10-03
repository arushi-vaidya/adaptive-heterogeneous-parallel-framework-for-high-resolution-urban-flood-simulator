#pragma once

#include <cstddef>
#include <vector>

namespace flood {

struct Cell {
    double bed = 0.0;
    double h = 0.0;
    double hu = 0.0;
    double hv = 0.0;
};

class Grid {
public:
    Grid(std::size_t rows, std::size_t cols, double dx, double dy);

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    double dx() const { return dx_; }
    double dy() const { return dy_; }
    std::size_t size() const { return cells_.size(); }
    Cell& at(std::size_t row, std::size_t col);
    const Cell& at(std::size_t row, std::size_t col) const;
    std::vector<Cell>& cells() { return cells_; }
    const std::vector<Cell>& cells() const { return cells_; }

private:
    std::size_t rows_;
    std::size_t cols_;
    double dx_;
    double dy_;
    std::vector<Cell> cells_;
};

} // namespace flood