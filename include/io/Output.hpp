#pragma once

#include "grid/Grid.hpp"
#include "solver/SerialSolver.hpp"
#include "physics/Rainfall.hpp"

#include <filesystem>

namespace flood {

void writeOutputs(const Grid& grid, const Diagnostics& diagnostics,
                  const std::filesystem::path& directory);
Grid readTerrainCsv(const std::filesystem::path& path, double dx, double dy);
Rainfall readRainfallCsv(const std::filesystem::path& path);

} // namespace flood