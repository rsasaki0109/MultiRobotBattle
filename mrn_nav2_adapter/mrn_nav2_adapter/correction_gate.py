"""Pure-Python safety gates for cooperative-pose-driven Nav2 corrections.

This module intentionally has no ROS imports so the rules are testable in
isolation. The broadcaster node converts ROS messages into the dataclasses
below before consulting these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional


class CorrectionGateStatus(Enum):
    ACCEPT = "accept"
    STALE_COOPERATIVE_POSE = "stale_cooperative_pose"
    DEGRADED_COOPERATIVE_POSE = "degraded_cooperative_pose"
    INVALID_COOPERATIVE_POSE = "invalid_cooperative_pose"
    TRANSLATION_JUMP_TOO_LARGE = "translation_jump_too_large"
    ROTATION_JUMP_TOO_LARGE = "rotation_jump_too_large"
    NONFINITE_POSE = "nonfinite_pose"
    UNKNOWN_STATUS_FLAG = "unknown_status_flag"


# These mirror mrn_msgs/msg/CooperativePose status enum values.
STATUS_OK = 0
STATUS_DEGRADED = 1
STATUS_STALE = 2
STATUS_INVALID = 3
_KNOWN_STATUS_VALUES = {STATUS_OK, STATUS_DEGRADED, STATUS_STALE, STATUS_INVALID}


@dataclass(frozen=True)
class Pose2D:
    """Lightweight 2D pose used by the gate (corrections are evaluated in SE(2))."""

    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class CorrectionGateConfig:
    """Static configuration for the correction safety gates."""

    max_pose_age_sec: float = 1.0
    max_translation_jump_m: float = 1.5
    max_rotation_jump_rad: float = math.radians(20.0)
    accept_degraded: bool = False


@dataclass(frozen=True)
class CorrectionGateInput:
    """One cooperative-pose candidate being considered as a correction source."""

    stamp_sec: float
    now_sec: float
    status: int
    pose: Pose2D
    previous_pose: Optional[Pose2D] = None


@dataclass(frozen=True)
class CorrectionGateResult:
    status: CorrectionGateStatus
    pose_age_sec: float
    translation_jump_m: Optional[float]
    rotation_jump_rad: Optional[float]

    @property
    def accepted(self) -> bool:
        return self.status is CorrectionGateStatus.ACCEPT


def _normalize_angle(yaw: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    wrapped = math.fmod(yaw + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def evaluate(
    candidate: CorrectionGateInput, config: CorrectionGateConfig
) -> CorrectionGateResult:
    """Decide whether ``candidate`` should be applied as a Nav2 correction.

    The function is deterministic and side-effect free. ``previous_pose`` is
    optional: if absent, the jump gates are skipped (this is the first
    correction so there is nothing to compare against).
    """
    if not _is_finite_pose(candidate.pose):
        return CorrectionGateResult(
            status=CorrectionGateStatus.NONFINITE_POSE,
            pose_age_sec=float("nan"),
            translation_jump_m=None,
            rotation_jump_rad=None,
        )

    pose_age = candidate.now_sec - candidate.stamp_sec
    if pose_age > config.max_pose_age_sec:
        return CorrectionGateResult(
            status=CorrectionGateStatus.STALE_COOPERATIVE_POSE,
            pose_age_sec=pose_age,
            translation_jump_m=None,
            rotation_jump_rad=None,
        )

    if candidate.status not in _KNOWN_STATUS_VALUES:
        return CorrectionGateResult(
            status=CorrectionGateStatus.UNKNOWN_STATUS_FLAG,
            pose_age_sec=pose_age,
            translation_jump_m=None,
            rotation_jump_rad=None,
        )

    if candidate.status == STATUS_INVALID:
        return CorrectionGateResult(
            status=CorrectionGateStatus.INVALID_COOPERATIVE_POSE,
            pose_age_sec=pose_age,
            translation_jump_m=None,
            rotation_jump_rad=None,
        )

    if candidate.status == STATUS_STALE:
        return CorrectionGateResult(
            status=CorrectionGateStatus.STALE_COOPERATIVE_POSE,
            pose_age_sec=pose_age,
            translation_jump_m=None,
            rotation_jump_rad=None,
        )

    if candidate.status == STATUS_DEGRADED and not config.accept_degraded:
        return CorrectionGateResult(
            status=CorrectionGateStatus.DEGRADED_COOPERATIVE_POSE,
            pose_age_sec=pose_age,
            translation_jump_m=None,
            rotation_jump_rad=None,
        )

    translation_jump: Optional[float] = None
    rotation_jump: Optional[float] = None
    if candidate.previous_pose is not None:
        translation_jump = math.hypot(
            candidate.pose.x - candidate.previous_pose.x,
            candidate.pose.y - candidate.previous_pose.y,
        )
        rotation_jump = abs(
            _normalize_angle(candidate.pose.yaw - candidate.previous_pose.yaw)
        )

        if translation_jump > config.max_translation_jump_m:
            return CorrectionGateResult(
                status=CorrectionGateStatus.TRANSLATION_JUMP_TOO_LARGE,
                pose_age_sec=pose_age,
                translation_jump_m=translation_jump,
                rotation_jump_rad=rotation_jump,
            )

        if rotation_jump > config.max_rotation_jump_rad:
            return CorrectionGateResult(
                status=CorrectionGateStatus.ROTATION_JUMP_TOO_LARGE,
                pose_age_sec=pose_age,
                translation_jump_m=translation_jump,
                rotation_jump_rad=rotation_jump,
            )

    return CorrectionGateResult(
        status=CorrectionGateStatus.ACCEPT,
        pose_age_sec=pose_age,
        translation_jump_m=translation_jump,
        rotation_jump_rad=rotation_jump,
    )


def _is_finite_pose(pose: Pose2D) -> bool:
    return all(math.isfinite(v) for v in (pose.x, pose.y, pose.yaw))
