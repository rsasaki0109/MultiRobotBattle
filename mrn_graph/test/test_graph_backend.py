import math

from graph_backend import (
    REASON_INVALID_COVARIANCE,
    REASON_SELF_CONSTRAINT,
    REASON_STALE,
    REASON_UNKNOWN_AGENT,
    AgentInput,
    FixedLagBackend,
    FixedLagBackendConfig,
    GnssInput,
    RelativeInput,
)


def _d3(pos, yaw):
    return [[pos, 0.0, 0.0], [0.0, pos, 0.0], [0.0, 0.0, yaw]]


def _d2(var):
    return [[var, 0.0], [0.0, var]]


_COV = _d3(0.04, 0.04)


def _estimate(estimates, agent_id):
    return next(e for e in estimates if e.agent_id == agent_id)


class TestCorrection:
    def test_degraded_agent_corrected_toward_anchor(self):
        backend = FixedLagBackend()
        agents = [
            AgentInput("robot_1", (0.0, 0.0, 0.0), _COV, degraded=False),
            AgentInput("robot_2", (3.0, 3.0, 0.0), _COV, degraded=True),
        ]
        relatives = [
            RelativeInput("robot_1", "robot_2", (2.0, 0.0, 0.0), _COV, age_sec=0.1)
        ]
        estimates, diag = backend.step(agents, relatives)
        r2 = _estimate(estimates, "robot_2")
        assert math.isclose(r2.pose[0], 2.0, abs_tol=1e-2)
        assert math.isclose(r2.pose[1], 0.0, abs_tol=1e-2)
        assert r2.accepted_constraints == 1
        assert diag.accepted == 1
        assert diag.converged

    def test_anchor_stays_put(self):
        backend = FixedLagBackend()
        agents = [
            AgentInput("robot_1", (0.0, 0.0, 0.0), _COV, degraded=False),
            AgentInput("robot_2", (3.0, 3.0, 0.0), _COV, degraded=True),
        ]
        relatives = [
            RelativeInput("robot_1", "robot_2", (2.0, 0.0, 0.0), _COV, age_sec=0.1)
        ]
        estimates, _ = backend.step(agents, relatives)
        r1 = _estimate(estimates, "robot_1")
        assert math.isclose(r1.pose[0], 0.0, abs_tol=1e-3)
        assert math.isclose(r1.pose[1], 0.0, abs_tol=1e-3)


class TestGating:
    def _agents(self):
        return [
            AgentInput("robot_1", (0.0, 0.0, 0.0), _COV, degraded=False),
            AgentInput("robot_2", (3.0, 3.0, 0.0), _COV, degraded=True),
        ]

    def test_stale_constraint_counted_and_skipped(self):
        backend = FixedLagBackend(FixedLagBackendConfig(max_constraint_age_sec=2.0))
        relatives = [
            RelativeInput("robot_1", "robot_2", (2.0, 0.0, 0.0), _COV, age_sec=5.0)
        ]
        _, diag = backend.step(self._agents(), relatives)
        assert diag.stale == 1
        assert diag.accepted == 0
        assert diag.rejection_reasons.get(REASON_STALE) == 1

    def test_invalid_covariance_rejected(self):
        backend = FixedLagBackend()
        relatives = [
            RelativeInput("robot_1", "robot_2", (2.0, 0.0, 0.0), _d3(0.0, 0.04), age_sec=0.1)
        ]
        _, diag = backend.step(self._agents(), relatives)
        assert diag.rejected == 1
        assert diag.rejection_reasons.get(REASON_INVALID_COVARIANCE) == 1

    def test_self_constraint_rejected(self):
        backend = FixedLagBackend()
        relatives = [
            RelativeInput("robot_1", "robot_1", (0.0, 0.0, 0.0), _COV, age_sec=0.1)
        ]
        _, diag = backend.step(self._agents(), relatives)
        assert diag.rejection_reasons.get(REASON_SELF_CONSTRAINT) == 1

    def test_unknown_agent_rejected(self):
        backend = FixedLagBackend()
        relatives = [
            RelativeInput("robot_1", "robot_9", (2.0, 0.0, 0.0), _COV, age_sec=0.1)
        ]
        _, diag = backend.step(self._agents(), relatives)
        assert diag.rejection_reasons.get(REASON_UNKNOWN_AGENT) == 1


class TestGnss:
    def test_gnss_prior_pulls_estimate(self):
        backend = FixedLagBackend()
        # single degraded agent; only a GNSS fix anchors it.
        agents = [AgentInput("robot_1", (0.0, 0.0, 0.0), _COV, degraded=True)]
        gnss = [GnssInput("robot_1", (5.0, 0.0), _d2(0.04))]
        estimates, diag = backend.step(agents, (), gnss)
        r1 = _estimate(estimates, "robot_1")
        # loose degraded prior at origin vs tight-ish GNSS at 5.0 → close to GNSS.
        assert r1.pose[0] > 4.5
        assert diag.converged


class TestRobustness:
    def test_all_degraded_still_converges(self):
        # No anchor; loose degraded priors keep the gauge from collapsing.
        backend = FixedLagBackend()
        agents = [
            AgentInput("robot_1", (0.0, 0.0, 0.0), _COV, degraded=True),
            AgentInput("robot_2", (2.1, 0.0, 0.0), _COV, degraded=True),
        ]
        relatives = [
            RelativeInput("robot_1", "robot_2", (2.0, 0.0, 0.0), _COV, age_sec=0.1)
        ]
        estimates, diag = backend.step(agents, relatives)
        assert diag.converged
        # relative pose between the two estimates should match the measurement.
        r1 = _estimate(estimates, "robot_1")
        r2 = _estimate(estimates, "robot_2")
        assert math.isclose(r2.pose[0] - r1.pose[0], 2.0, abs_tol=0.2)

    def test_degraded_without_constraint_has_zero_quality(self):
        backend = FixedLagBackend()
        agents = [
            AgentInput("robot_1", (0.0, 0.0, 0.0), _COV, degraded=False),
            AgentInput("robot_2", (3.0, 3.0, 0.0), _COV, degraded=True),
        ]
        estimates, _ = backend.step(agents, ())
        r2 = _estimate(estimates, "robot_2")
        assert r2.quality == 0.0


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
