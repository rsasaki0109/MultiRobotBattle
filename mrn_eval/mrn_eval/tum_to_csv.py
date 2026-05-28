"""Pure-function helper: TUM trajectory format -> offline-ATE CSV.

The TUM RGB-D trajectory format (also used by EuRoC ground truth exports
and many SLAM benchmarks) is whitespace-separated, one pose per line:

    timestamp tx ty tz qx qy qz qw

Lines beginning with ``#`` and blank lines are comments. A position-only
variant with just ``timestamp tx ty tz`` is also accepted. Orientation is
ignored — the offline ATE/RPE helper compares positions only, matching the
``stamp_sec,x,y,z`` schema produced by the bag and RTK exporters.

This makes ``mrn_eval_offline_ate`` usable against public benchmark
datasets without ROS or rosbag2: convert the dataset's ground-truth
trajectory (and an estimator's output, if it is also TUM-formatted) to CSV,
then run the offline comparison.

Pure-function module — no ROS, rosbag2, or numpy dependency.
"""

from __future__ import annotations

from pathlib import Path

from .offline_ate import TrajectorySample

_POSITION_ONLY_FIELDS = 4  # timestamp tx ty tz
_FULL_POSE_FIELDS = 8  # timestamp tx ty tz qx qy qz qw


def parse_tum_line(line: str, *, source: str = "<string>", lineno: int = 0):
    """Parse one TUM line into a TrajectorySample, or None for a comment/blank.

    Raises ``ValueError`` (with a ``source:lineno`` prefix) on a malformed
    line so the loader and CLI can surface a stable diagnostic.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    parts = stripped.split()
    if len(parts) not in (_POSITION_ONLY_FIELDS, _FULL_POSE_FIELDS):
        raise ValueError(
            f"{source}:{lineno}: expected {_POSITION_ONLY_FIELDS} or "
            f"{_FULL_POSE_FIELDS} fields (timestamp tx ty tz [qx qy qz qw]), "
            f"got {len(parts)}"
        )
    try:
        stamp = float(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
    except ValueError as exc:
        raise ValueError(f"{source}:{lineno}: invalid number ({exc})") from exc
    return TrajectorySample(stamp_sec=stamp, x=x, y=y, z=z)


def load_tum_trajectory(path: Path) -> list[TrajectorySample]:
    """Load a TUM-format trajectory file. Sorts by ``stamp_sec``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TUM trajectory not found: {path}")
    samples: list[TrajectorySample] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            sample = parse_tum_line(line, source=str(path), lineno=lineno)
            if sample is not None:
                samples.append(sample)
    samples.sort(key=lambda sample: sample.stamp_sec)
    return samples
