#!/usr/bin/env python3
"""Solver-independent factor-graph math for cooperative localization.

This module is the numeric core a real graph backend (GTSAM, Ceres, or a
hand-rolled fixed-lag optimizer) calls regardless of which solver is wired
in. It is pure Python — no ROS, no numpy, no GTSAM — so it stays
CI-testable and de-risks the v0.4 backend dependency question: the factor
residuals, covariance-aware weighting, and robust loss can be verified
here before any solver packaging is decided.

Poses are SE(2): ``(x, y, yaw)`` with yaw in radians. Covariances are
square matrices (``list[list[float]]``) — 3×3 for SE(2) factors, 2×2 for a
position-only GNSS prior.

The factor families in the v0.4 plan map onto these helpers:

- odometry between-factor and relative-pose factor → :func:`between_residual`
- GNSS prior factor → :func:`gnss_prior_residual`
- covariance-aware weighting → :func:`information_from_covariance`,
  :func:`mahalanobis_norm`
- robust loss → :func:`huber_weight`
- rejected factors and reasons → :func:`evaluate_factor`,
  :class:`FactorReason`
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

Pose2 = tuple[float, float, float]


# --- SE(2) pose operations -------------------------------------------------

def normalize_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    wrapped = math.fmod(angle + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def pose_compose(a: Pose2, b: Pose2) -> Pose2:
    """Return ``a ∘ b`` — b expressed in a's frame, composed onto a."""
    ax, ay, ayaw = a
    bx, by, byaw = b
    cos_a = math.cos(ayaw)
    sin_a = math.sin(ayaw)
    return (
        ax + cos_a * bx - sin_a * by,
        ay + sin_a * bx + cos_a * by,
        normalize_angle(ayaw + byaw),
    )


def pose_inverse(p: Pose2) -> Pose2:
    """Return the SE(2) inverse of ``p``."""
    x, y, yaw = p
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    return (
        -(cos_y * x + sin_y * y),
        -(-sin_y * x + cos_y * y),
        normalize_angle(-yaw),
    )


def pose_between(a: Pose2, b: Pose2) -> Pose2:
    """Return the relative pose ``a^-1 ∘ b`` (``T_a_b``, b seen from a)."""
    return pose_compose(pose_inverse(a), b)


# --- Factor residuals ------------------------------------------------------

def between_residual(
    pose_from: Pose2, pose_to: Pose2, measured_from_to: Pose2
) -> Pose2:
    """Residual of a between-factor (odometry or relative-pose).

    Compares the predicted relative pose ``pose_from^-1 ∘ pose_to`` against
    the measured ``T_from_to``. Returns the SE(2) error
    ``measured^-1 ∘ predicted`` as ``(dx, dy, dyaw)`` in the measurement's
    local frame; zero when the prediction matches the measurement.
    """
    predicted = pose_between(pose_from, pose_to)
    return pose_between(measured_from_to, predicted)


def gnss_prior_residual(
    pose: Pose2, measured_xy: tuple[float, float]
) -> tuple[float, float]:
    """Residual of a position-only GNSS prior: estimated minus measured."""
    return (pose[0] - measured_xy[0], pose[1] - measured_xy[1])


# --- Linear algebra (small dense matrices) ---------------------------------

def invert_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """Invert a small square matrix via Gauss-Jordan with partial pivoting.

    Raises ``ValueError`` if the matrix is not square, contains a
    non-finite entry, or is singular.
    """
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square")
    aug = []
    for i in range(n):
        row = [float(value) for value in matrix[i]]
        if any(not math.isfinite(value) for value in row):
            raise ValueError("matrix has non-finite entries")
        identity = [1.0 if j == i else 0.0 for j in range(n)]
        aug.append(row + identity)

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("matrix is singular")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        aug[col] = [value / pivot_value for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor != 0.0:
                aug[row] = [
                    cell - factor * aug[col][k] for k, cell in enumerate(aug[row])
                ]
    return [row[n:] for row in aug]


def covariance_is_valid(covariance: Sequence[Sequence[float]]) -> bool:
    """True if ``covariance`` is finite, square, and positive on the diagonal."""
    n = len(covariance)
    if n == 0 or any(len(row) != n for row in covariance):
        return False
    for i in range(n):
        for j in range(n):
            if not math.isfinite(covariance[i][j]):
                return False
        if covariance[i][i] <= 0.0:
            return False
    return True


def information_from_covariance(
    covariance: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Return the information matrix (inverse covariance).

    Raises ``ValueError`` if the covariance is invalid or singular.
    """
    if not covariance_is_valid(covariance):
        raise ValueError("covariance is not finite / positive-definite on diagonal")
    return invert_matrix(covariance)


def mahalanobis_norm(
    residual: Sequence[float], information: Sequence[Sequence[float]]
) -> float:
    """Return ``sqrt(rᵀ Λ r)`` — the covariance-weighted residual norm."""
    n = len(residual)
    if any(len(row) != n for row in information):
        raise ValueError("information matrix shape does not match residual")
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += residual[i] * information[i][j] * residual[j]
    if total < 0.0:
        # Numerical guard: a valid information matrix keeps this non-negative.
        total = 0.0
    return math.sqrt(total)


# --- Robust loss -----------------------------------------------------------

def huber_weight(norm: float, delta: float) -> float:
    """IRLS weight for the Huber loss at whitened-residual norm ``norm``.

    Returns 1.0 inside the quadratic region (``norm <= delta``) and
    ``delta / norm`` outside it, down-weighting outliers linearly.
    """
    if delta <= 0.0:
        raise ValueError("huber delta must be positive")
    if norm <= delta:
        return 1.0
    return delta / norm


# --- Factor evaluation -----------------------------------------------------

class FactorReason(Enum):
    ACCEPT = "accept"
    NONFINITE_MEASUREMENT = "nonfinite_measurement"
    INVALID_COVARIANCE = "invalid_covariance"
    STALE = "stale"


@dataclass(frozen=True)
class FactorEvaluation:
    reason: FactorReason
    mahalanobis_norm: float | None
    robust_weight: float | None

    @property
    def accepted(self) -> bool:
        return self.reason is FactorReason.ACCEPT


def evaluate_factor(
    residual: Sequence[float],
    covariance: Sequence[Sequence[float]],
    *,
    age_sec: float | None = None,
    max_age_sec: float | None = None,
    huber_delta: float | None = None,
) -> FactorEvaluation:
    """Decide whether a factor should enter the graph, and how to weight it.

    Deterministic and side-effect free. Returns the rejection reason, or
    ``ACCEPT`` with the covariance-weighted residual norm and a robust
    weight (1.0 when ``huber_delta`` is None).
    """
    if any(not math.isfinite(value) for value in residual):
        return FactorEvaluation(FactorReason.NONFINITE_MEASUREMENT, None, None)

    if max_age_sec is not None and age_sec is not None and age_sec > max_age_sec:
        return FactorEvaluation(FactorReason.STALE, None, None)

    try:
        information = information_from_covariance(covariance)
    except ValueError:
        return FactorEvaluation(FactorReason.INVALID_COVARIANCE, None, None)

    norm = mahalanobis_norm(residual, information)
    weight = huber_weight(norm, huber_delta) if huber_delta is not None else 1.0
    return FactorEvaluation(FactorReason.ACCEPT, norm, weight)
