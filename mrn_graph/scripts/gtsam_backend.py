#!/usr/bin/env python3
"""GTSAM-backed cooperative localization backend.

A drop-in alternative to :class:`graph_backend.FixedLagBackend` with the same
``step(agents, relatives, gnss)`` interface and the same diagnostics
contract, but using GTSAM's factor graph and Levenberg-Marquardt optimizer
instead of the pure-Python Gauss-Newton solver.

The two backends share their gating, per-agent accounting, prior covariance
selection, and estimate assembly through the helpers in
:mod:`graph_backend` (``classify_relatives``, ``prior_covariance_for_agent``,
``build_estimates``), so they apply identical constraint gating and only
differ in the optimizer. ``test_gtsam_backend.py`` asserts the two agree on
the synthetic scenarios; it is skipped wherever GTSAM is not installed.

``gtsam`` and ``numpy`` are imported at module top, so this module is only
importable where GTSAM is present. The default node backend stays the
pure-Python one; this is selected explicitly
(``fixed_lag_graph_node.py -p backend:=gtsam``).
"""

from __future__ import annotations

from typing import Sequence

import gtsam
import numpy as np

from graph_backend import (
    AgentInput,
    CooperativeEstimate,
    FixedLagBackendConfig,
    GnssInput,
    GraphDiagnostics,
    RelativeInput,
    build_estimates,
    classify_relatives,
    prior_covariance_for_agent,
)

Pose2 = tuple[float, float, float]
_GNSS_YAW_VAR = 1.0e6  # leave yaw unconstrained for a position-only prior


def _pose2(pose) -> "gtsam.Pose2":
    return gtsam.Pose2(float(pose[0]), float(pose[1]), float(pose[2]))


def _gaussian(covariance) -> "gtsam.noiseModel.Gaussian":
    return gtsam.noiseModel.Gaussian.Covariance(np.array(covariance, dtype=float))


class GtsamBackend:
    """Cooperative graph backend built on GTSAM (opt-in, requires gtsam)."""

    def __init__(self, config: FixedLagBackendConfig | None = None) -> None:
        self.config = config or FixedLagBackendConfig()
        self.name = "fixed_lag_gtsam"

    def step(
        self,
        agents: Sequence[AgentInput],
        relatives: Sequence[RelativeInput] = (),
        gnss: Sequence[GnssInput] = (),
        *,
        now_sec: float | None = None,
    ) -> tuple[list[CooperativeEstimate], GraphDiagnostics]:
        cfg = self.config
        symbol = gtsam.symbol_shorthand.X
        key_of = {agent.agent_id: index + 1 for index, agent in enumerate(agents)}
        known = set(key_of)

        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        for agent in agents:
            key = symbol(key_of[agent.agent_id])
            values.insert(key, _pose2(agent.pose))
            graph.add(
                gtsam.PriorFactorPose2(
                    key,
                    _pose2(agent.pose),
                    _gaussian(prior_covariance_for_agent(agent, cfg)),
                )
            )

        for fix in gnss:
            if fix.agent_id not in known:
                continue
            cov = np.array(fix.covariance, dtype=float)
            cov3 = np.diag([cov[0][0], cov[1][1], _GNSS_YAW_VAR])
            graph.add(
                gtsam.PriorFactorPose2(
                    symbol(key_of[fix.agent_id]),
                    gtsam.Pose2(float(fix.xy[0]), float(fix.xy[1]), 0.0),
                    _gaussian(cov3),
                )
            )

        classification = classify_relatives(agents, relatives, cfg.max_constraint_age_sec)
        for rel in classification.accepted:
            base = _gaussian(rel.covariance)
            if cfg.huber_delta is not None:
                noise = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.Huber.Create(float(cfg.huber_delta)),
                    base,
                )
            else:
                noise = base
            graph.add(
                gtsam.BetweenFactorPose2(
                    symbol(key_of[rel.from_id]),
                    symbol(key_of[rel.to_id]),
                    _pose2(rel.measured),
                    noise,
                )
            )

        converged = True
        iterations = 0
        if graph.size() > 0:
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values)
            result = optimizer.optimize()
            iterations = optimizer.iterations()
        else:
            result = values

        optimized: dict[str, Pose2] = {}
        for agent in agents:
            pose = result.atPose2(symbol(key_of[agent.agent_id]))
            optimized[agent.agent_id] = (pose.x(), pose.y(), pose.theta())

        estimates = build_estimates(agents, optimized, classification, cfg)
        diagnostics = GraphDiagnostics(
            accepted=classification.accepted_count,
            rejected=classification.rejected,
            stale=classification.stale,
            converged=converged,
            iterations=iterations,
            rejection_reasons=classification.reasons,
        )
        return estimates, diagnostics
