#!/usr/bin/env python3
"""Convert an RTK logger CSV to the offline-ATE truth CSV via local ENU.

Input schema (RTK logger):

    stamp_sec,lat_deg,lon_deg,alt_m,fix_quality

Output schema (matches ``mrn_eval_offline_ate --truth``):

    stamp_sec,x,y,z

A sidecar ``<output>.origin.yaml`` records the geodetic + ECEF origin used
for the linearization so the comparison can be reproduced later or against
a different bag.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from mrn_gnss import EnuOrigin, FixQuality, GeodeticPoint

from .bag_to_csv import write_csv
from .rtk_to_csv import (
    filter_by_quality,
    load_rtk_csv,
    pick_origin_from_first,
    summarize_by_quality,
    to_enu_trajectory,
    write_origin_yaml,
)


def _resolve_origin(args, samples) -> EnuOrigin:
    if args.origin_from_first:
        geodetic = pick_origin_from_first(samples)
    else:
        geodetic = GeodeticPoint(
            lat_rad=math.radians(args.origin_lat_deg),
            lon_rad=math.radians(args.origin_lon_deg),
            alt_m=args.origin_alt_m,
        )
    return EnuOrigin.from_geodetic(geodetic)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mrn_eval_rtk_to_csv",
        description=(
            "Convert an RTK logger CSV (stamp_sec,lat_deg,lon_deg,alt_m,fix_quality) "
            "to the offline-ATE truth CSV (stamp_sec,x,y,z) in a local ENU frame."
        ),
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)

    origin_group = parser.add_mutually_exclusive_group(required=True)
    origin_group.add_argument(
        "--origin-from-first",
        action="store_true",
        help="Use the first finite-quality sample as the ENU origin.",
    )
    origin_group.add_argument(
        "--origin-lat-deg",
        type=float,
        help="Explicit ENU origin latitude in degrees.",
    )
    parser.add_argument(
        "--origin-lon-deg",
        type=float,
        help="Explicit ENU origin longitude in degrees (required with --origin-lat-deg).",
    )
    parser.add_argument(
        "--origin-alt-m",
        type=float,
        default=0.0,
        help="Explicit ENU origin ellipsoidal altitude in meters (default 0.0).",
    )
    parser.add_argument(
        "--min-fix-quality",
        type=int,
        default=int(FixQuality.RTK_FLOAT),
        help=(
            "Drop samples with worse-than-this fix quality (NMEA GGA value). "
            "Default 5 (RTK_FLOAT)."
        ),
    )

    args = parser.parse_args(argv)

    if args.origin_lat_deg is not None and args.origin_lon_deg is None:
        print(
            "error: --origin-lat-deg requires --origin-lon-deg",
            file=sys.stderr,
        )
        return 2

    try:
        samples = load_rtk_csv(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not samples:
        print(f"error: {args.input} contains no samples", file=sys.stderr)
        return 2

    try:
        min_quality = FixQuality(args.min_fix_quality)
    except ValueError as exc:
        print(f"error: invalid --min-fix-quality {args.min_fix_quality}: {exc}", file=sys.stderr)
        return 2

    try:
        kept = filter_by_quality(samples, min_quality)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not kept:
        print(
            f"error: no samples passed min-fix-quality={min_quality.name} "
            f"(input={len(samples)})",
            file=sys.stderr,
        )
        return 3

    try:
        origin = _resolve_origin(args, kept)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    trajectory = to_enu_trajectory(kept, origin)
    written = write_csv(args.output, trajectory)
    write_origin_yaml(args.output.with_suffix(args.output.suffix + ".origin.yaml"), origin)

    counts = summarize_by_quality(kept)
    summary = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    print(
        f"wrote {written} rows to {args.output} (kept {len(kept)}/{len(samples)}; {summary})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
