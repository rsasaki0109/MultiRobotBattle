#!/usr/bin/env python3
"""Convert a TUM-format trajectory file to the offline-ATE CSV schema.

Reads ``timestamp tx ty tz [qx qy qz qw]`` (the TUM RGB-D / EuRoC export
convention) and writes ``stamp_sec,x,y,z``, which ``mrn_eval_offline_ate``
consumes directly (``--estimated`` / ``--truth``). Orientation is dropped.

This is the dataset-side counterpart to ``mrn_eval_bag_to_csv`` (rosbag)
and ``mrn_eval_rtk_to_csv`` (RTK logger): it makes the offline ATE/RPE
helper usable against public benchmark datasets with no ROS dependency.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mrn_eval.bag_to_csv import write_csv
from mrn_eval.tum_to_csv import load_tum_trajectory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mrn_eval_tum_to_csv",
        description=(
            "Convert a TUM-format trajectory (timestamp tx ty tz [qx qy qz qw]) "
            "to the offline-ATE CSV (stamp_sec,x,y,z)."
        ),
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        samples = load_tum_trajectory(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not samples:
        print(
            f"error: {args.input} contains no trajectory rows "
            "(only comments/blank lines?)",
            file=sys.stderr,
        )
        return 3

    written = write_csv(args.output, samples)
    print(f"wrote {written} rows to {args.output} (from TUM {args.input})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
