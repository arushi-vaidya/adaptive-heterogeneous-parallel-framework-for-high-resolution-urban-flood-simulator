#!/usr/bin/env python3
"""Write a deterministic rainfall time series; intensity is in mm/hour."""
import argparse
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/rainfall/storm.csv")
    args = parser.parse_args()
    series = [(0, 0), (600, 30), (1200, 60), (1800, 20), (2400, 0)]
    with open(args.output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_seconds", "intensity_mm_per_hour"))
        writer.writerows(series)


if __name__ == "__main__":
    main()