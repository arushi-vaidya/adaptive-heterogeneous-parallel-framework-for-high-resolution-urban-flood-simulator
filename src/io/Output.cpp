#include "io/Output.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace flood {

void writeOutputs(const Grid& grid, const Diagnostics& d,
                  const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    std::ofstream terrain(directory / "terrain.csv");
    std::ofstream depth(directory / "depth.csv");
    std::ofstream xMomentum(directory / "hu.csv");
    std::ofstream yMomentum(directory / "hv.csv");
    terrain << std::setprecision(12);
    depth << std::setprecision(12);
    xMomentum << std::setprecision(12);
    yMomentum << std::setprecision(12);
    for (std::size_t r = 0; r < grid.rows(); ++r) {
        for (std::size_t c = 0; c < grid.cols(); ++c) {
            if (c) { terrain << ','; depth << ','; xMomentum << ','; yMomentum << ','; }
            terrain << grid.at(r, c).bed;
            depth << grid.at(r, c).h;
            xMomentum << grid.at(r, c).hu;
            yMomentum << grid.at(r, c).hv;
        }
        terrain << '\n';
        depth << '\n';
        xMomentum << '\n';
        yMomentum << '\n';
    }
    std::ofstream stats(directory / "stats.csv");
    stats << "time_s,steps,stored_m3,rainfall_m3,outflow_m3,infiltration_m3,mass_residual_m3,"
             "max_depth_m,max_velocity_m_s,wet_cells,flooded_area_m2,solver_seconds,backend,threads,"
             "processes,setup_seconds,computation_seconds,communication_seconds,synchronization_seconds,other_seconds\n";
    stats << std::setprecision(12) << d.time << ',' << d.steps << ',' << d.storedVolume << ','
          << d.rainfallVolume << ',' << d.outflowVolume << ',' << d.infiltrationVolume << ','
          << d.massResidual << ',' << d.maxDepth << ',' << d.maxVelocity << ',' << d.wetCells
          << ',' << d.floodedArea << ',' << d.solverSeconds << ',' << d.backend
          << ',' << d.threads << ',' << d.processes << ',' << d.setupSeconds << ',' << d.computationSeconds
          << ',' << d.communicationSeconds << ',' << d.synchronizationSeconds
          << ',' << d.otherSeconds << '\n';
    if (!terrain || !depth || !xMomentum || !yMomentum || !stats)
        throw std::runtime_error("Failed writing simulation output");
}

} // namespace flood