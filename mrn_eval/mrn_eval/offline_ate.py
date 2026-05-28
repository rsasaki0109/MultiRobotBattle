"""Offline ATE/RPE computation against a reference trajectory.

Pure-function helpers — no ROS, rosbag, or numpy dependency. The CLI in
``offline_ate_cli`` is what consumes these; tests exercise them directly.

The contract is intentionally narrow: trajectories are sequences of
``TrajectorySample`` (stamp_sec, x, y, z). Inputs come from CSV today; once a
real two-robot bag exists, a small bag-to-CSV exporter will feed the same path.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import isinf, isnan, nan, sqrt
from pathlib import Path
from statistics import mean, pstdev
from typing import Sequence


@dataclass(frozen=True)
class TrajectorySample:
    stamp_sec: float
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class AlignedPair:
    stamp_sec: float
    estimated: TrajectorySample
    truth: TrajectorySample
    time_offset_sec: float  # truth.stamp_sec - estimated.stamp_sec


@dataclass(frozen=True)
class AlignmentResult:
    pairs: tuple[AlignedPair, ...]
    estimated_count: int
    truth_count: int
    matched_count: int
    dropped_count: int
    max_time_offset_sec: float
    mean_time_offset_sec: float


@dataclass(frozen=True)
class ErrorStats:
    rmse: float
    mean: float
    stddev: float
    max: float
    count: int


REQUIRED_COLUMNS = ("stamp_sec", "x", "y")


def load_trajectory_csv(path: Path) -> list[TrajectorySample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"trajectory CSV not found: {path}")
    samples: list[TrajectorySample] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(
                f"{path}: missing CSV header (expected stamp_sec,x,y[,z])"
            )
        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: missing required CSV columns: {missing}")
        for row_index, row in enumerate(reader, start=2):
            try:
                stamp = float(row["stamp_sec"])
                x = float(row["x"])
                y = float(row["y"])
                z_value = row.get("z")
                z = float(z_value) if z_value not in (None, "") else 0.0
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{row_index}: invalid number ({exc})") from exc
            samples.append(TrajectorySample(stamp_sec=stamp, x=x, y=y, z=z))
    samples.sort(key=lambda sample: sample.stamp_sec)
    return samples


def time_align(
    estimated: Sequence[TrajectorySample],
    truth: Sequence[TrajectorySample],
    max_offset_sec: float = 0.05,
) -> AlignmentResult:
    if max_offset_sec < 0.0:
        raise ValueError("max_offset_sec must be non-negative")
    est_sorted = sorted(estimated, key=lambda sample: sample.stamp_sec)
    truth_sorted = sorted(truth, key=lambda sample: sample.stamp_sec)
    if not truth_sorted:
        return AlignmentResult(
            pairs=(),
            estimated_count=len(est_sorted),
            truth_count=0,
            matched_count=0,
            dropped_count=len(est_sorted),
            max_time_offset_sec=0.0,
            mean_time_offset_sec=0.0,
        )
    pairs: list[AlignedPair] = []
    dropped = 0
    truth_idx = 0
    for est in est_sorted:
        while (
            truth_idx + 1 < len(truth_sorted)
            and abs(truth_sorted[truth_idx + 1].stamp_sec - est.stamp_sec)
            <= abs(truth_sorted[truth_idx].stamp_sec - est.stamp_sec)
        ):
            truth_idx += 1
        candidate = truth_sorted[truth_idx]
        offset = candidate.stamp_sec - est.stamp_sec
        if abs(offset) <= max_offset_sec:
            pairs.append(AlignedPair(est.stamp_sec, est, candidate, offset))
        else:
            dropped += 1
    abs_offsets = [abs(pair.time_offset_sec) for pair in pairs]
    return AlignmentResult(
        pairs=tuple(pairs),
        estimated_count=len(est_sorted),
        truth_count=len(truth_sorted),
        matched_count=len(pairs),
        dropped_count=dropped,
        max_time_offset_sec=max(abs_offsets) if abs_offsets else 0.0,
        mean_time_offset_sec=mean(abs_offsets) if abs_offsets else 0.0,
    )


def _error_stats(errors: Sequence[float]) -> ErrorStats:
    if not errors:
        return ErrorStats(rmse=nan, mean=nan, stddev=nan, max=nan, count=0)
    return ErrorStats(
        rmse=sqrt(sum(error * error for error in errors) / len(errors)),
        mean=mean(errors),
        stddev=pstdev(errors) if len(errors) > 1 else 0.0,
        max=max(errors),
        count=len(errors),
    )


def compute_ate(pairs: Sequence[AlignedPair]) -> ErrorStats:
    errors = [
        sqrt(
            (pair.estimated.x - pair.truth.x) ** 2
            + (pair.estimated.y - pair.truth.y) ** 2
            + (pair.estimated.z - pair.truth.z) ** 2
        )
        for pair in pairs
    ]
    return _error_stats(errors)


def compute_rpe(
    pairs: Sequence[AlignedPair],
    delta_sec: float,
    delta_tolerance_sec: float | None = None,
) -> ErrorStats:
    if delta_sec <= 0:
        raise ValueError("delta_sec must be positive")
    tolerance = delta_tolerance_sec if delta_tolerance_sec is not None else delta_sec / 2.0
    if tolerance < 0:
        raise ValueError("delta_tolerance_sec must be non-negative")
    sorted_pairs = sorted(pairs, key=lambda pair: pair.stamp_sec)
    errors: list[float] = []
    for i, pair_i in enumerate(sorted_pairs):
        target = pair_i.stamp_sec + delta_sec
        best_j = None
        best_diff = float("inf")
        for j in range(i + 1, len(sorted_pairs)):
            diff = abs(sorted_pairs[j].stamp_sec - target)
            if diff <= best_diff:
                best_diff = diff
                best_j = j
            else:
                break
        if best_j is None or best_diff > tolerance:
            continue
        pair_j = sorted_pairs[best_j]
        diff_x = (pair_j.estimated.x - pair_i.estimated.x) - (
            pair_j.truth.x - pair_i.truth.x
        )
        diff_y = (pair_j.estimated.y - pair_i.estimated.y) - (
            pair_j.truth.y - pair_i.truth.y
        )
        diff_z = (pair_j.estimated.z - pair_i.estimated.z) - (
            pair_j.truth.z - pair_i.truth.z
        )
        errors.append(sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z))
    return _error_stats(errors)


def stats_to_dict(stats: ErrorStats) -> dict:
    def _jsonable(value: float) -> float | None:
        return None if isnan(value) or isinf(value) else value

    return {
        "rmse": _jsonable(stats.rmse),
        "mean": _jsonable(stats.mean),
        "stddev": _jsonable(stats.stddev),
        "max": _jsonable(stats.max),
        "count": stats.count,
    }
