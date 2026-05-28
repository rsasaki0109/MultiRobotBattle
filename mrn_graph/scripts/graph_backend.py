#!/usr/bin/env python3
"""Pure backend layer wrapping the pose-graph solver.

``FixedLagBackend`` turns a window of plain agent / constraint / GNSS inputs
into optimized cooperative poses plus the diagnostics contract from
``docs/graph_backend_plugin.md`` (accepted / rejected / stale counts and a
stable rejection-reason vocabulary). It is the backend "brain": no rclpy and
no ROS message types, so it is unit-tested in CI directly. A thin rclpy node
converts ``AgentState`` / ``RelativePoseConstraint`` / GNSS messages into
these dataclasses, calls :meth:`FixedLagBackend.step`, and publishes the
result — keeping the ROS layer trivial and the optimization logic testable.

It builds on :mod:`pose_graph_solver`: non-degraded agents anchor the graph
with a tight pose prior (their local estimate is trusted), degraded agents
get a loose prior (weak regularization toward their dead-reckoned pose so the
gauge never collapses), relative constraints become between-factors, and GNSS
fixes become position priors. The solver then optimizes the whole window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from factor_graph import FactorReason
from pose_graph_solver import (
    BetweenFactor,
    FactorKind,
    PriorFactor,
    gauss_newton,
)

Pose2 = tuple[float, float, float]

# Stable rejection-reason strings (subset of the plugin-doc vocabulary).
REASON_STALE = "message_too_old"
REASON_INVALID_COVARIANCE = "nonfinite_covariance"
REASON_SELF_CONSTRAINT = "self_constraint"
REASON_UNKNOWN_AGENT = "unknown_to_agent"


@dataclass(frozen=True)
class AgentInput:
    agent_id: str
    pose: Pose2  # (x, y, yaw) in map_frame
    covariance: Sequence[Sequence[float]]  # 3x3 pose covariance
    degraded: bool = False


@dataclass(frozen=True)
class RelativeInput:
    from_id: str
    to_id: str
    measured: Pose2  # T_from_to
    covariance: Sequence[Sequence[float]]  # 3x3
    age_sec: float = 0.0


@dataclass(frozen=True)
class GnssInput:
    agent_id: str
    xy: tuple[float, float]
    covariance: Sequence[Sequence[float]]  # 2x2


@dataclass(frozen=True)
class CooperativeEstimate:
    agent_id: str
    pose: Pose2
    accepted_constraints: int
    rejected_constraints: int
    quality: float
    degraded: bool


@dataclass(frozen=True)
class GraphDiagnostics:
    accepted: int
    rejected: int
    stale: int
    converged: bool
    iterations: int
    rejection_reasons: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class FixedLagBackendConfig:
    max_constraint_age_sec: float = 2.0
    anchor_position_var: float = 1e-4   # tight: trust a non-degraded local pose
    anchor_yaw_var: float = 1e-4
    degraded_position_var: float = 1.0e4  # loose: weak gauge regularization
    degraded_yaw_var: float = 1.0e4
    huber_delta: float | None = 1.0
    base_quality: float = 0.9


def _diag3(pos_var: float, yaw_var: float):
    return [
        [pos_var, 0.0, 0.0],
        [0.0, pos_var, 0.0],
        [0.0, 0.0, yaw_var],
    ]


class FixedLagBackend:
    """Optimize one window of cooperative-localization inputs."""

    def __init__(self, config: FixedLagBackendConfig | None = None) -> None:
        self.config = config or FixedLagBackendConfig()
        self.name = "fixed_lag_python"

    def step(
        self,
        agents: Sequence[AgentInput],
        relatives: Sequence[RelativeInput] = (),
        gnss: Sequence[GnssInput] = (),
        *,
        now_sec: float | None = None,
    ) -> tuple[list[CooperativeEstimate], GraphDiagnostics]:
        cfg = self.config
        known = {a.agent_id for a in agents}
        initial: dict[str, Pose2] = {a.agent_id: tuple(a.pose) for a in agents}

        priors: list[PriorFactor] = []
        for agent in agents:
            priors.append(
                PriorFactor(
                    agent.agent_id,
                    tuple(agent.pose),
                    prior_covariance_for_agent(agent, cfg),
                    FactorKind.PRIOR_POSE,
                )
            )

        for fix in gnss:
            if fix.agent_id not in known:
                continue
            priors.append(
                PriorFactor(
                    fix.agent_id,
                    (fix.xy[0], fix.xy[1]),
                    fix.covariance,
                    FactorKind.PRIOR_POSITION,
                )
            )

        classification = classify_relatives(agents, relatives, cfg.max_constraint_age_sec)
        betweens = [
            BetweenFactor(rel.from_id, rel.to_id, tuple(rel.measured), rel.covariance)
            for rel in classification.accepted
        ]

        result = gauss_newton(
            initial, priors, betweens, huber_delta=cfg.huber_delta
        )

        estimates = build_estimates(
            agents,
            {name: result.variables[name] for name in result.variables},
            classification,
            cfg,
        )
        diagnostics = GraphDiagnostics(
            accepted=classification.accepted_count,
            rejected=classification.rejected,
            stale=classification.stale,
            converged=result.converged,
            iterations=result.iterations,
            rejection_reasons=classification.reasons,
        )
        return estimates, diagnostics


def prior_covariance_for_agent(
    agent: AgentInput, config: FixedLagBackendConfig
) -> list[list[float]]:
    """Anchor (tight) covariance for OK agents, loose for degraded ones."""
    if agent.degraded:
        return _diag3(config.degraded_position_var, config.degraded_yaw_var)
    return _diag3(config.anchor_position_var, config.anchor_yaw_var)


@dataclass(frozen=True)
class RelativeClassification:
    accepted: tuple[RelativeInput, ...]
    accepted_count: int
    rejected: int
    stale: int
    reasons: dict[str, int]
    per_agent_accepted: dict[str, int]
    per_agent_rejected: dict[str, int]


def classify_relatives(
    agents: Sequence[AgentInput],
    relatives: Sequence[RelativeInput],
    max_constraint_age_sec: float,
) -> RelativeClassification:
    """Gate relative constraints, shared by every backend implementation.

    Keeping this in one place guarantees the Python and GTSAM backends apply
    identical self / unknown-agent / age / covariance gating and produce the
    same accepted set, counts, and rejection-reason vocabulary.
    """
    known = {a.agent_id for a in agents}
    accepted: list[RelativeInput] = []
    rejected = 0
    stale = 0
    reasons: dict[str, int] = {}
    per_agent_accepted: dict[str, int] = {a.agent_id: 0 for a in agents}
    per_agent_rejected: dict[str, int] = {a.agent_id: 0 for a in agents}

    def _reject(reason: str, agent_id: str | None) -> None:
        nonlocal rejected
        rejected += 1
        reasons[reason] = reasons.get(reason, 0) + 1
        if agent_id in per_agent_rejected:
            per_agent_rejected[agent_id] += 1

    for rel in relatives:
        if rel.from_id == rel.to_id:
            _reject(REASON_SELF_CONSTRAINT, rel.to_id)
            continue
        if rel.from_id not in known or rel.to_id not in known:
            _reject(REASON_UNKNOWN_AGENT, rel.to_id)
            continue
        if rel.age_sec > max_constraint_age_sec:
            stale += 1
            reasons[REASON_STALE] = reasons.get(REASON_STALE, 0) + 1
            continue
        if any(not _all_finite(row) for row in rel.covariance) or not _spd_diag(
            rel.covariance
        ):
            _reject(REASON_INVALID_COVARIANCE, rel.to_id)
            continue
        accepted.append(rel)
        per_agent_accepted[rel.to_id] += 1

    return RelativeClassification(
        accepted=tuple(accepted),
        accepted_count=len(accepted),
        rejected=rejected,
        stale=stale,
        reasons=reasons,
        per_agent_accepted=per_agent_accepted,
        per_agent_rejected=per_agent_rejected,
    )


def build_estimates(
    agents: Sequence[AgentInput],
    optimized: dict[str, Pose2],
    classification: RelativeClassification,
    config: FixedLagBackendConfig,
) -> list[CooperativeEstimate]:
    """Assemble per-agent cooperative estimates from an optimized pose set."""
    estimates: list[CooperativeEstimate] = []
    for agent in agents:
        accepted = classification.per_agent_accepted.get(agent.agent_id, 0)
        quality = 0.0 if agent.degraded and accepted == 0 else config.base_quality
        estimates.append(
            CooperativeEstimate(
                agent_id=agent.agent_id,
                pose=optimized[agent.agent_id],
                accepted_constraints=accepted,
                rejected_constraints=classification.per_agent_rejected.get(agent.agent_id, 0),
                quality=quality,
                degraded=agent.degraded,
            )
        )
    return estimates


def _all_finite(row) -> bool:
    import math

    return all(math.isfinite(v) for v in row)


def _spd_diag(covariance) -> bool:
    n = len(covariance)
    if n == 0 or any(len(r) != n for r in covariance):
        return False
    return all(covariance[i][i] > 0.0 for i in range(n))
