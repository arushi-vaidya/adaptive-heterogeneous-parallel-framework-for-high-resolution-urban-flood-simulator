#!/usr/bin/env python3
"""Compare two water-depth CSV grids and estimate their stored-volume difference."""
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--cell-size", type=float, default=5.0)
    parser.add_argument("--dry-depth", type=float, default=1e-6)
    parser.add_argument("--tolerance", type=float, default=1e-8,
                        help="maximum absolute depth error tolerance in metres")
    parser.add_argument("--reference-hu", type=argparse.FileType("r"))
    parser.add_argument("--candidate-hu", type=argparse.FileType("r"))
    parser.add_argument("--reference-hv", type=argparse.FileType("r"))
    parser.add_argument("--candidate-hv", type=argparse.FileType("r"))
    parser.add_argument("--momentum-tolerance", type=float, default=1e-8,
                        help="maximum absolute momentum error tolerance in m^2/s")
    args = parser.parse_args()
    if args.cell_size <= 0 or args.dry_depth < 0 or args.tolerance < 0 or args.momentum_tolerance < 0:
        parser.error("cell size must be positive; thresholds must be nonnegative")
    momentum_paths = (args.reference_hu, args.candidate_hu, args.reference_hv, args.candidate_hv)
    if any(momentum_paths) and not all(momentum_paths):
        parser.error("provide all four momentum CSV paths or none")
    reference = np.atleast_2d(np.loadtxt(args.reference, delimiter=","))
    candidate = np.atleast_2d(np.loadtxt(args.candidate, delimiter=","))
    if reference.shape != candidate.shape:
        parser.error("input grids must have identical dimensions")
    difference = candidate - reference
    absolute = np.abs(difference)
    scale = float(np.mean(np.abs(reference)))
    relative = float(np.mean(absolute) / scale) if scale > 0 else float("nan")
    volume_difference = float(np.sum(difference) * args.cell_size**2)
    flooded_area_difference = float(
        (np.count_nonzero(candidate > args.dry_depth) -
         np.count_nonzero(reference > args.dry_depth)) * args.cell_size**2
    )
    print(f"max_absolute_error_m={absolute.max():.12g}")
    print(f"mean_absolute_error_m={absolute.mean():.12g}")
    print(f"rmse_m={np.sqrt(np.mean(difference**2)):.12g}")
    print(f"mean_relative_error={relative:.12g}")
    print(f"water_volume_difference_m3={volume_difference:.12g}")
    print(f"flooded_area_difference_m2={flooded_area_difference:.12g}")
    print(f"max_error_tolerance_m={args.tolerance:.12g}")
    passed = float(absolute.max()) <= args.tolerance
    if all(momentum_paths):
        reference_hu = np.atleast_2d(np.loadtxt(args.reference_hu.name, delimiter=","))
        candidate_hu = np.atleast_2d(np.loadtxt(args.candidate_hu.name, delimiter=","))
        reference_hv = np.atleast_2d(np.loadtxt(args.reference_hv.name, delimiter=","))
        candidate_hv = np.atleast_2d(np.loadtxt(args.candidate_hv.name, delimiter=","))
        if any(array.shape != reference.shape for array in
               (reference_hu, candidate_hu, reference_hv, candidate_hv)):
            parser.error("momentum and depth grids must have identical dimensions")
        hu_difference = candidate_hu - reference_hu
        hv_difference = candidate_hv - reference_hv
        max_hu_error = float(np.abs(hu_difference).max())
        max_hv_error = float(np.abs(hv_difference).max())
        print(f"max_hu_error_m2_s={max_hu_error:.12g}")
        print(f"max_hv_error_m2_s={max_hv_error:.12g}")
        print(f"rmse_hu_m2_s={np.sqrt(np.mean(hu_difference**2)):.12g}")
        print(f"rmse_hv_m2_s={np.sqrt(np.mean(hv_difference**2)):.12g}")
        passed = passed and max_hu_error <= args.momentum_tolerance and max_hv_error <= args.momentum_tolerance
        print(f"momentum_tolerance_m2_s={args.momentum_tolerance:.12g}")
    print(f"within_tolerance={str(passed).lower()}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()