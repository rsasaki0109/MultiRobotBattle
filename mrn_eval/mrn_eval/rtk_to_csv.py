"""Pure-function helpers for converting an RTK logger CSV to offline-ATE truth.

The input CSV carries geodetic samples with a NMEA GGA-style quality
indicator:

    stamp_sec,lat_deg,lon_deg,alt_m,fix_quality

The output is the schema consumed by ``mrn_eval_offline_ate`` (and produced
by ``mrn_eval_bag_to_csv``):

    stamp_sec,x,y,z

The conversion uses :mod:`mrn_gnss` for the geodetic → ENU step and writes
a small ``*.origin.yaml`` sidecar so the origin used for the linearization
is recoverable for reproducing the comparison later.

Everything in this module is pure-function — no ROS, rosbag2, or numpy
dependency — so CI exercises it directly with handwritten fixtures.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from mrn_gnss import (
    EnuOrigin,
    FixQuality,
    GeodeticPoint,
    geodetic_to_enu,
)

from .offline_ate import TrajectorySample


REQUIRED_INPUT_COLUMNS = (
    "stamp_sec",
    "lat_deg",
    "lon_deg",
    "alt_m",
    "fix_quality",
)


@dataclass(frozen=True)
class RtkSample:
    """One row of the input RTK CSV (geodetic + GGA quality)."""

    stamp_sec: float
    lat_deg: float
    lon_deg: float
    alt_m: float
    fix_quality: FixQuality


def load_rtk_csv(path: Path) -> list[RtkSample]:
    """Load an RTK logger CSV. Sorts by ``stamp_sec``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RTK CSV not found: {path}")
    samples: list[RtkSample] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        missing = [
            name for name in REQUIRED_INPUT_COLUMNS if name not in reader.fieldnames
        ]
        if missing:
            raise ValueError(f"{path}: missing required CSV columns: {missing}")
        for row_index, row in enumerate(reader, start=2):
            try:
                stamp = float(row["stamp_sec"])
                lat_deg = float(row["lat_deg"])
                lon_deg = float(row["lon_deg"])
                alt_m = float(row["alt_m"])
                quality_int = int(row["fix_quality"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{row_index}: invalid number ({exc})") from exc
            try:
                quality = FixQuality(quality_int)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{row_index}: unknown fix_quality {quality_int}"
                ) from exc
            samples.append(
                RtkSample(
                    stamp_sec=stamp,
                    lat_deg=lat_deg,
                    lon_deg=lon_deg,
                    alt_m=alt_m,
                    fix_quality=quality,
                )
            )
    samples.sort(key=lambda sample: sample.stamp_sec)
    return samples


def filter_by_quality(
    samples: Iterable[RtkSample], min_quality: FixQuality
) -> list[RtkSample]:
    """Drop samples whose ``fix_quality`` ranks worse than ``min_quality``.

    Ranking follows the heuristic ENU sigma in :mod:`mrn_gnss.fix_quality`
    — smaller sigma = better fix. Anything with infinite sigma (INVALID,
    MANUAL, SIMULATION) is always dropped.
    """
    from mrn_gnss.fix_quality import HORIZONTAL_SIGMA_M

    threshold = HORIZONTAL_SIGMA_M[min_quality]
    if not math.isfinite(threshold):
        raise ValueError(
            f"min_quality {min_quality.name} has infinite sigma; pick a finite-sigma quality"
        )
    return [
        sample
        for sample in samples
        if math.isfinite(HORIZONTAL_SIGMA_M[sample.fix_quality])
        and HORIZONTAL_SIGMA_M[sample.fix_quality] <= threshold
    ]


def pick_origin_from_first(samples: Sequence[RtkSample]) -> GeodeticPoint:
    """Pick the first finite-quality sample's geodetic position as origin."""
    from mrn_gnss.fix_quality import HORIZONTAL_SIGMA_M

    for sample in samples:
        if math.isfinite(HORIZONTAL_SIGMA_M[sample.fix_quality]):
            return GeodeticPoint(
                lat_rad=math.radians(sample.lat_deg),
                lon_rad=math.radians(sample.lon_deg),
                alt_m=sample.alt_m,
            )
    raise ValueError("no usable sample to seed ENU origin (all INVALID/MANUAL/SIM)")


def to_enu_trajectory(
    samples: Iterable[RtkSample], origin: EnuOrigin
) -> list[TrajectorySample]:
    """Convert geodetic samples to ENU TrajectorySample list."""
    output: list[TrajectorySample] = []
    for sample in samples:
        enu = geodetic_to_enu(
            GeodeticPoint(
                lat_rad=math.radians(sample.lat_deg),
                lon_rad=math.radians(sample.lon_deg),
                alt_m=sample.alt_m,
            ),
            origin,
        )
        output.append(
            TrajectorySample(
                stamp_sec=sample.stamp_sec,
                x=enu.east,
                y=enu.north,
                z=enu.up,
            )
        )
    return output


def summarize_by_quality(samples: Iterable[RtkSample]) -> dict[str, int]:
    """Return a count-by-quality dict keyed by enum name."""
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.fix_quality.name] = counts.get(sample.fix_quality.name, 0) + 1
    return counts


def write_rtk_csv(path: Path, samples: Iterable[RtkSample]) -> int:
    """Write geodetic samples in the RTK input schema.

    Produces ``stamp_sec,lat_deg,lon_deg,alt_m,fix_quality`` — the schema
    that :func:`load_rtk_csv` reads back. This is what
    ``mrn_eval_bag_to_csv`` emits when the requested topic is a geodetic
    type (e.g. ``sensor_msgs/msg/NavSatFix``); the resulting CSV is then
    fed to ``mrn_eval_rtk_to_csv`` to linearize it into the ENU truth CSV.

    Returns the number of rows written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(REQUIRED_INPUT_COLUMNS)
        for sample in samples:
            writer.writerow(
                (
                    f"{sample.stamp_sec:.9f}",
                    f"{sample.lat_deg:.9f}",
                    f"{sample.lon_deg:.9f}",
                    f"{sample.alt_m:.6f}",
                    int(sample.fix_quality),
                )
            )
            count += 1
    return count


def write_origin_yaml(path: Path, origin: EnuOrigin) -> None:
    """Write a small YAML sidecar describing the ENU origin."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    geodetic = origin.geodetic
    lines = [
        "# ENU origin used to linearize the trajectory; pin this for reproducibility.",
        f"lat_deg: {math.degrees(geodetic.lat_rad):.9f}",
        f"lon_deg: {math.degrees(geodetic.lon_rad):.9f}",
        f"alt_m: {geodetic.alt_m:.6f}",
        f"ecef_x: {origin.ecef.x:.6f}",
        f"ecef_y: {origin.ecef.y:.6f}",
        f"ecef_z: {origin.ecef.z:.6f}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
