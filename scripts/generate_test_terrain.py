#!/usr/bin/env python3
"""Write deterministic flat, slope, or bowl elevation grids as CSV (metres)."""
import argparse
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("flat", "slope", "bowl"), default="bowl")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--output", default="data/terrain/test_terrain.csv")
    args = parser.parse_args()
    if args.rows < 1 or args.cols < 1:
        parser.error("rows and cols must be positive")
    with open(args.output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for row in range(args.rows):
            values = []
            for col in range(args.cols):
                if args.kind == "flat":
                    elevation = 0.0
                elif args.kind == "slope":
                    elevation = 0.02 * (args.cols - col)
                else:
                    x = (col - (args.cols - 1) / 2) / max(args.cols, 1)
                    y = (row - (args.rows - 1) / 2) / max(args.rows, 1)
                    elevation = 4.0 * (x * x + y * y)
                values.append(f"{elevation:.8f}")
            writer.writerow(values)


if __name__ == "__main__":
    main()