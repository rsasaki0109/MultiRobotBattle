"""Time-scheduled GNSS fix quality, for outage / reacquisition scenarios.

A :class:`FixQualitySchedule` is a step function over time: each interval
declares the fix quality that holds from its ``start_sec`` until the next
interval begins. Before the first interval the fix is :data:`FixQuality.INVALID`
(no acquisition yet).

This is what a synthetic world node uses to model an outage followed by a
staged reacquisition (e.g. ``INVALID`` → ``SINGLE`` → ``SBAS`` →
``RTK_FLOAT`` → ``RTK_FIX``): the schedule yields the active quality at a
given sim time, and :meth:`FixQualitySchedule.covariance_at` derives the
ENU position covariance via :func:`mrn_gnss.fix_quality.position_covariance`.

Pure-function module — no ROS or numpy dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .fix_quality import FixQuality, position_covariance


@dataclass(frozen=True)
class QualityInterval:
    """A fix quality that becomes active at ``start_sec``."""

    start_sec: float
    fix_quality: FixQuality


def _coerce_quality(value) -> FixQuality:
    if isinstance(value, FixQuality):
        return value
    if isinstance(value, str):
        try:
            return FixQuality[value.strip().upper()]
        except KeyError as exc:
            raise ValueError(f"unknown fix_quality name {value!r}") from exc
    return FixQuality(int(value))


@dataclass(frozen=True)
class FixQualitySchedule:
    """Step function from sim time to :class:`FixQuality`."""

    intervals: tuple[QualityInterval, ...]

    @classmethod
    def from_steps(cls, steps: Iterable[tuple[float, object]]) -> "FixQualitySchedule":
        """Build from ``(start_sec, fix_quality)`` pairs.

        ``fix_quality`` may be a :class:`FixQuality`, an NMEA GGA int, or a
        quality name string (case-insensitive). Intervals are sorted by
        ``start_sec``; duplicate start times are rejected.
        """
        items = [
            QualityInterval(float(start), _coerce_quality(quality))
            for start, quality in steps
        ]
        if not items:
            raise ValueError("FixQualitySchedule needs at least one interval")
        items.sort(key=lambda interval: interval.start_sec)
        starts = [interval.start_sec for interval in items]
        if len(set(starts)) != len(starts):
            raise ValueError(f"duplicate start_sec in schedule: {starts}")
        return cls(intervals=tuple(items))

    @classmethod
    def from_config(cls, entries: Sequence[dict]) -> "FixQualitySchedule":
        """Build from scenario YAML entries ``[{start_sec, fix_quality}, ...]``."""
        steps: list[tuple[float, object]] = []
        for index, entry in enumerate(entries):
            if "start_sec" not in entry or "fix_quality" not in entry:
                raise ValueError(
                    f"gnss_quality_schedule[{index}] needs start_sec and fix_quality"
                )
            steps.append((entry["start_sec"], entry["fix_quality"]))
        return cls.from_steps(steps)

    def quality_at(self, sim_time: float) -> FixQuality:
        """Return the active fix quality at ``sim_time``.

        Before the first interval's ``start_sec`` the fix is ``INVALID``.
        """
        active = FixQuality.INVALID
        for interval in self.intervals:
            if interval.start_sec <= sim_time:
                active = interval.fix_quality
            else:
                break
        return active

    def covariance_at(self, sim_time: float):
        """Return the 3×3 ENU position covariance active at ``sim_time``."""
        return position_covariance(self.quality_at(sim_time))
