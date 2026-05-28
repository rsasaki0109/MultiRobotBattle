"""Small deterministic localization metrics."""

from __future__ import annotations

from collections import deque
from math import sqrt
from typing import Iterable, Sequence

Point2 = tuple[float, float]
TimedError = tuple[float, float]


def rmse(errors: Iterable[float]) -> float:
    values = list(errors)
    if not values:
        raise ValueError("rmse requires at least one error")
    return sqrt(sum(error * error for error in values) / len(values))


def ate_2d(estimate: Iterable[Point2], truth: Iterable[Point2]) -> float:
    est_values = list(estimate)
    truth_values = list(truth)
    if len(est_values) != len(truth_values):
        raise ValueError("estimate and truth lengths differ")
    if not est_values:
        raise ValueError("ate_2d requires at least one pose")

    errors = [
        sqrt((ex - tx) * (ex - tx) + (ey - ty) * (ey - ty))
        for (ex, ey), (tx, ty) in zip(est_values, truth_values)
    ]
    return rmse(errors)


def distance_2d(estimate: Point2, truth: Point2) -> float:
    return sqrt(
        (estimate[0] - truth[0]) * (estimate[0] - truth[0])
        + (estimate[1] - truth[1]) * (estimate[1] - truth[1])
    )


class StreamingAte:
    """Fixed-size online ATE RMSE accumulator."""

    def __init__(self, max_samples: int = 2000) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self._squared_errors: deque[float] = deque(maxlen=max_samples)

    def push(self, estimate: Point2, truth: Point2) -> None:
        error = distance_2d(estimate, truth)
        self._squared_errors.append(error * error)

    @property
    def count(self) -> int:
        return len(self._squared_errors)

    def rmse(self) -> float:
        if not self._squared_errors:
            raise ValueError("rmse requires at least one sample")
        return sqrt(sum(self._squared_errors) / len(self._squared_errors))


def rpe_translation_2d(
    estimate: Sequence[Point2],
    truth: Sequence[Point2],
    delta: int = 1,
) -> float:
    """RMSE of translation difference between estimated and truth deltas.

    For each index ``i``, the relative motion ``est[i+delta] - est[i]`` is
    compared with ``truth[i+delta] - truth[i]``. ``delta`` is an integer
    sample step; callers are responsible for ensuring uniform sample spacing.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    if len(estimate) != len(truth):
        raise ValueError("estimate and truth lengths differ")
    if len(estimate) <= delta:
        raise ValueError("need more samples than delta")

    errors: list[float] = []
    for i in range(len(estimate) - delta):
        est_dx = estimate[i + delta][0] - estimate[i][0]
        est_dy = estimate[i + delta][1] - estimate[i][1]
        truth_dx = truth[i + delta][0] - truth[i][0]
        truth_dy = truth[i + delta][1] - truth[i][1]
        diff_x = est_dx - truth_dx
        diff_y = est_dy - truth_dy
        errors.append(sqrt(diff_x * diff_x + diff_y * diff_y))
    return rmse(errors)


def recovery_time(
    samples: Sequence[TimedError],
    degraded_threshold: float,
    recovered_threshold: float,
    hold_seconds: float = 0.0,
) -> float | None:
    """Return seconds from first degraded sample to a sustained recovery.

    The trajectory is considered ``degraded`` when error crosses
    ``degraded_threshold``. Recovery begins when error returns at or below
    ``recovered_threshold`` and is confirmed once it stays at/below the
    threshold for at least ``hold_seconds``.

    Returns:
        ``0.0`` if the trajectory never degrades.
        ``None`` if it degrades but never confirms recovery in the window.
        Otherwise the duration in seconds between the first degraded sample
        and the start of the confirmed-recovery window.

    Args:
        samples: ``(time_sec, error)`` pairs, sorted by time.
        degraded_threshold: error above this counts as degraded.
        recovered_threshold: error at/below this counts as recovered.
        hold_seconds: how long recovery must persist before it is confirmed.
    """
    if degraded_threshold <= 0:
        raise ValueError("degraded_threshold must be positive")
    if recovered_threshold <= 0:
        raise ValueError("recovered_threshold must be positive")
    if recovered_threshold > degraded_threshold:
        raise ValueError(
            "recovered_threshold must be <= degraded_threshold"
        )
    if hold_seconds < 0:
        raise ValueError("hold_seconds must be non-negative")

    degraded_start: float | None = None
    recovery_window_start: float | None = None
    last_time: float | None = None
    for time_sec, error in samples:
        if last_time is not None and time_sec < last_time:
            raise ValueError("samples must be sorted by time")
        last_time = time_sec

        if degraded_start is None:
            if error > degraded_threshold:
                degraded_start = time_sec
            continue

        if error <= recovered_threshold:
            if recovery_window_start is None:
                recovery_window_start = time_sec
            if time_sec - recovery_window_start >= hold_seconds:
                return recovery_window_start - degraded_start
        else:
            recovery_window_start = None

    if degraded_start is None:
        return 0.0
    return None
