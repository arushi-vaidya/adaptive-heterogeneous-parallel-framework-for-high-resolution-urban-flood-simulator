#!/usr/bin/env python3
"""Plot terrain, final water depth, and flooded extent from solver CSV output."""
import argparse

import matplotlib.pyplot as plt
import numpy as np


def load_grid(path):
    grid = np.atleast_2d(np.loadtxt(path, delimiter=","))
    if grid.ndim != 2:
        raise ValueError(f"Expected a 2D CSV grid in {path}")
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain", default="output/run/terrain.csv")
    parser.add_argument("--depth", default="output/run/depth.csv")
    parser.add_argument("--output", default="output/run/flood_map.png")
    parser.add_argument("--cell-size", type=float, default=5.0)
    parser.add_argument("--dry-depth", type=float, default=1e-6)
    args = parser.parse_args()
    terrain, depth = load_grid(args.terrain), load_grid(args.depth)
    if terrain.shape != depth.shape:
        parser.error("terrain and depth grids must have the same dimensions")
    extent = (0, depth.shape[1] * args.cell_size, depth.shape[0] * args.cell_size, 0)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    terrain_plot = axes[0].imshow(terrain, cmap="terrain", extent=extent)
    figure.colorbar(terrain_plot, ax=axes[0], label="Elevation (m)")
    depth_plot = axes[1].imshow(depth, cmap="Blues", vmin=0, extent=extent)
    figure.colorbar(depth_plot, ax=axes[1], label="Water depth (m)")
    axes[2].imshow(depth > args.dry_depth, cmap="Blues", vmin=0, vmax=1, extent=extent)
    for axis, title in zip(axes, ("Terrain", "Water depth", "Flood extent")):
        axis.set_title(title)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    figure.savefig(args.output, dpi=160)
    print(f"Saved {args.output}; max depth={depth.max():.6g} m; wet cells={(depth > args.dry_depth).sum()}")


if __name__ == "__main__":
    main()