#!/usr/bin/env python3
"""Pure-Python Gauss-Newton pose-graph solver (SE(2)).

A solver-dependency-free reference backend for the v0.4 graph work. It
optimizes a small 2D pose graph using the residuals, covariance weighting,
and robust loss in :mod:`factor_graph`, so the full backend behavior —
odometry factors, GNSS priors, relative-pose factors, robust outlier
handling, factor rejection — is exercised in CI without GTSAM or numpy.

GTSAM is the intended high-performance backend (verified importable from
``ros-jazzy-gtsam`` locally), but it is not installed in CI; wiring it in is
a follow-up that also adds the dependency to the CI image. This pure solver
keeps the backend contract green in the meantime and doubles as the
reference the GTSAM backend must match.

Design notes:

- Variables are named SE(2) poses ``(x, y, yaw)``.
- Jacobians are computed numerically (finite differences) with the same
  body-frame retraction used to apply the update, which keeps the
  linearization self-consistent and avoids hand-derived SE(2) Jacobian
  bugs in a reference implementation.
- At least one prior (pose or position) is required to fix the gauge;
  without it the normal equations are singular and the solve reports
  non-convergence instead of returning a meaningless result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from factor_graph import (
    FactorReason,
    between_residual,
    covariance_is_valid,
    gnss_prior_residual,
    huber_weight,
    information_from_covariance,
    invert_matrix,
    mahalanobis_norm,
    normalize_angle,
    pose_compose,
)

Pose2 = tuple[float, float, float]
_EPS = 1e-6
_DOF = 3  # SE(2)


class FactorKind(Enum):
    PRIOR_POSE = "prior_pose"
    PRIOR_POSITION = "prior_position"  # GNSS-style 2D position prior
    BETWEEN = "between"  # odometry or relative-pose


@dataclass(frozen=True)
class PriorFactor:
    """Anchors a single variable to a measured pose or 2D position."""

    variable: str
    measured: Sequence[float]  # (x, y, yaw) for pose, (x, y) for position
    covariance: Sequence[Sequence[float]]
    kind: FactorKind = FactorKind.PRIOR_POSE

    def __post_init__(self):
        if self.kind is FactorKind.PRIOR_POSE and len(self.measured) != 3:
            raise ValueError("pose prior needs a 3-vector measurement")
        if self.kind is FactorKind.PRIOR_POSITION and len(self.measured) != 2:
            raise ValueError("position prior needs a 2-vector measurement")


@dataclass(frozen=True)
class BetweenFactor:
    """Constrains the relative pose between two variables (odom or relative)."""

    variable_from: str
    variable_to: str
    measured: Pose2  # T_from_to
    covariance: Sequence[Sequence[float]]


@dataclass(frozen=True)
class FactorReport:
    label: str
    reason: FactorReason
    residual_norm: float | None


@dataclass(frozen=True)
class SolveResult:
    variables: dict[str, Pose2]
    iterations: int
    converged: bool
    final_cost: float
    factor_reports: tuple[FactorReport, ...] = field(default_factory=tuple)


def _retract(pose: Pose2, delta: Sequence[float]) -> Pose2:
    """Body-frame SE(2) retraction: pose ∘ exp(delta)."""
    return pose_compose(pose, (delta[0], delta[1], normalize_angle(delta[2])))


def _prior_residual(factor: PriorFactor, pose: Pose2):
    if factor.kind is FactorKind.PRIOR_POSITION:
        return gnss_prior_residual(pose, (factor.measured[0], factor.measured[1]))
    # pose prior: error in the body frame of the measurement
    from factor_graph import pose_between

    return pose_between(tuple(factor.measured), pose)


def _between_resid(factor: BetweenFactor, pose_from: Pose2, pose_to: Pose2):
    return between_residual(pose_from, pose_to, factor.measured)


def _numerical_jacobian(residual_fn, poses: list[Pose2], dim: int):
    """Jacobian of a residual w.r.t. variable ``dim`` of ``poses``.

    ``residual_fn(poses) -> residual vector``. Perturbs the ``dim``-th pose
    with the body-frame retraction used by the update step.
    """
    base = residual_fn(poses)
    rows = len(base)
    jac = [[0.0] * _DOF for _ in range(rows)]
    for axis in range(_DOF):
        delta = [0.0, 0.0, 0.0]
        delta[axis] = _EPS
        perturbed = list(poses)
        perturbed[dim] = _retract(poses[dim], delta)
        forward = residual_fn(perturbed)
        for r in range(rows):
            diff = forward[r] - base[r]
            if r == 2 and rows == _DOF:
                diff = normalize_angle(diff)
            jac[r][axis] = diff / _EPS
    return jac


def gauss_newton(
    initial: dict[str, Pose2],
    priors: Sequence[PriorFactor],
    betweens: Sequence[BetweenFactor],
    *,
    max_iterations: int = 25,
    tolerance: float = 1e-9,
    huber_delta: float | None = None,
) -> SolveResult:
    """Optimize a 2D pose graph by Gauss-Newton.

    Returns the optimized poses, iteration count, convergence flag, final
    total cost (sum of weighted squared residual norms over accepted
    factors), and a per-factor report including rejected factors.
    """
    variables = list(initial.keys())
    index = {name: i for i, name in enumerate(variables)}
    poses: dict[str, Pose2] = {name: tuple(p) for name, p in initial.items()}

    # Validate / classify factors once; rejected factors never enter the
    # normal equations but are reported.
    def _factor_reason(covariance, residual) -> FactorReason:
        if any(not math.isfinite(v) for v in residual):
            return FactorReason.NONFINITE_MEASUREMENT
        if not covariance_is_valid(covariance):
            return FactorReason.INVALID_COVARIANCE
        return FactorReason.ACCEPT

    n = len(variables) * _DOF
    converged = False
    iterations = 0
    last_cost = float("inf")

    for iterations in range(1, max_iterations + 1):
        hessian = [[0.0] * n for _ in range(n)]
        gradient = [0.0] * n
        cost = 0.0

        # --- prior factors ---
        for factor in priors:
            pose = poses[factor.variable]
            residual = list(_prior_residual(factor, pose))
            if _factor_reason(factor.covariance, residual) is not FactorReason.ACCEPT:
                continue
            information = information_from_covariance(factor.covariance)
            weight = (
                huber_weight(mahalanobis_norm(residual, information), huber_delta)
                if huber_delta is not None
                else 1.0
            )
            jac = _numerical_jacobian(
                lambda ps: list(_prior_residual(factor, ps[0])), [pose], 0
            )
            _accumulate(
                hessian, gradient, [index[factor.variable]], jac, residual,
                information, weight,
            )
            cost += weight * mahalanobis_norm(residual, information) ** 2

        # --- between factors ---
        for factor in betweens:
            pose_from = poses[factor.variable_from]
            pose_to = poses[factor.variable_to]
            residual = list(_between_resid(factor, pose_from, pose_to))
            if _factor_reason(factor.covariance, residual) is not FactorReason.ACCEPT:
                continue
            information = information_from_covariance(factor.covariance)
            weight = (
                huber_weight(mahalanobis_norm(residual, information), huber_delta)
                if huber_delta is not None
                else 1.0
            )

            def resid_fn(ps):
                return list(_between_resid(factor, ps[0], ps[1]))

            jac_from = _numerical_jacobian(resid_fn, [pose_from, pose_to], 0)
            jac_to = _numerical_jacobian(resid_fn, [pose_from, pose_to], 1)
            _accumulate(
                hessian, gradient,
                [index[factor.variable_from], index[factor.variable_to]],
                _hstack(jac_from, jac_to), residual, information, weight,
            )
            cost += weight * mahalanobis_norm(residual, information) ** 2

        # Solve H dx = -g
        try:
            hessian_inv = invert_matrix(hessian)
        except ValueError:
            # Singular: typically no prior to fix the gauge.
            return SolveResult(
                variables=poses, iterations=iterations, converged=False,
                final_cost=cost, factor_reports=_reports(priors, betweens, poses, huber_delta),
            )
        step = [
            -sum(hessian_inv[i][j] * gradient[j] for j in range(n)) for i in range(n)
        ]

        for name in variables:
            base = index[name] * _DOF
            poses[name] = _retract(poses[name], step[base : base + _DOF])

        if abs(last_cost - cost) <= tolerance:
            converged = True
            last_cost = cost
            break
        last_cost = cost

    return SolveResult(
        variables=poses,
        iterations=iterations,
        converged=converged,
        final_cost=last_cost,
        factor_reports=_reports(priors, betweens, poses, huber_delta),
    )


def _accumulate(hessian, gradient, var_indices, jac, residual, information, weight):
    """Add weight * Jᵀ Λ J to H and weight * Jᵀ Λ r to g (g = gradient)."""
    rows = len(residual)
    cols = len(var_indices) * _DOF
    # weighted information-times-jacobian: WJ = Λ J  (rows x cols)
    wj = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            wj[r][c] = sum(information[r][k] * jac[k][c] for k in range(rows))
    # Jᵀ (Λ J) and Jᵀ Λ r, scaled by robust weight
    for a in range(cols):
        ga = 0.0
        for r in range(rows):
            ga += jac[r][a] * sum(information[r][k] * residual[k] for k in range(rows))
        global_a = var_indices[a // _DOF] * _DOF + (a % _DOF)
        gradient[global_a] += weight * ga
        for b in range(cols):
            hab = 0.0
            for r in range(rows):
                hab += jac[r][a] * wj[r][b]
            global_b = var_indices[b // _DOF] * _DOF + (b % _DOF)
            hessian[global_a][global_b] += weight * hab


def _hstack(jac_a, jac_b):
    """Horizontally stack two Jacobian blocks (same row count)."""
    return [row_a + row_b for row_a, row_b in zip(jac_a, jac_b)]


def _reports(priors, betweens, poses, huber_delta) -> tuple[FactorReport, ...]:
    reports: list[FactorReport] = []
    for i, factor in enumerate(priors):
        pose = poses[factor.variable]
        residual = list(_prior_residual(factor, pose))
        reports.append(_one_report(f"prior[{i}]:{factor.variable}", factor.covariance, residual))
    for i, factor in enumerate(betweens):
        residual = list(
            _between_resid(factor, poses[factor.variable_from], poses[factor.variable_to])
        )
        reports.append(
            _one_report(
                f"between[{i}]:{factor.variable_from}->{factor.variable_to}",
                factor.covariance, residual,
            )
        )
    return tuple(reports)


def _one_report(label, covariance, residual) -> FactorReport:
    if any(not math.isfinite(v) for v in residual):
        return FactorReport(label, FactorReason.NONFINITE_MEASUREMENT, None)
    if not covariance_is_valid(covariance):
        return FactorReport(label, FactorReason.INVALID_COVARIANCE, None)
    information = information_from_covariance(covariance)
    return FactorReport(label, FactorReason.ACCEPT, mahalanobis_norm(residual, information))
