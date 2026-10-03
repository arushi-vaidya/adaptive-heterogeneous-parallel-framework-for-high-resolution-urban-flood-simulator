#include "io/Output.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace flood {
namespace {

std::vector<double> parseRow(const std::string& line) {
    std::vector<double> values;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
        const auto first = field.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) throw std::runtime_error("Empty numeric CSV field");
        const auto last = field.find_last_not_of(" \t\r\n");
        field = field.substr(first, last - first + 1);
        std::size_t consumed = 0;
        const double value = std::stod(field, &consumed);
        if (consumed != field.size()) throw std::runtime_error("Invalid numeric CSV field");
        values.push_back(value);
    }
    return values;
}

} // namespace

Grid readTerrainCsv(const std::filesystem::path& path, double dx, double dy) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open terrain CSV: " + path.string());
    std::vector<std::vector<double>> values;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        values.push_back(parseRow(line));
    }
    if (values.empty() || values.front().empty()) throw std::runtime_error("Terrain CSV is empty");
    const std::size_t cols = values.front().size();
    Grid grid(values.size(), cols, dx, dy);
    for (std::size_t row = 0; row < values.size(); ++row) {
        if (values[row].size() != cols) throw std::runtime_error("Terrain CSV rows have different widths");
        for (std::size_t col = 0; col < cols; ++col) grid.at(row, col).bed = values[row][col];
    }
    return grid;
}

Rainfall readRainfallCsv(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open rainfall CSV: " + path.string());
    std::vector<RainPoint> points;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        try {
            const auto row = parseRow(line);
            if (row.size() != 2) throw std::runtime_error("Rainfall rows must have 2 columns");
            points.push_back({row[0], row[1]});
        } catch (const std::exception&) {
            if (!points.empty()) throw;
        }
    }
    if (points.empty()) throw std::runtime_error("Rainfall CSV has no numeric data rows");
    return Rainfall(std::move(points));
}

} // namespace flood