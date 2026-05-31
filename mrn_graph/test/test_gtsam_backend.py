"""Equivalence tests: the GTSAM backend must match the pure-Python reference.

Skipped wherever GTSAM is not installed (e.g. the default CI image), so the
always-green path never depends on GTSAM. Where ``gtsam`` is present (dev
machines, a dedicated CI job), this asserts the GTSAM-backed optimizer agrees
with ``FixedLagBackend`` on the same scenarios — which is what lets the pure
solver act as the reference the GTSAM backend is held to.
"""

import math

import pytest

# Skip (don't error) where GTSAM is absent — but keep the tests *collected* so
# pytest exits 0 instead of 5 ("no tests collected"), which ament_cmake_pytest
# would treat as a failure. A module-level importorskip would abort collection
# entirely; a skipif marker leaves the tests collected-but-skipped.
try:
    import gtsam  # noqa: F401
    _HAVE_GTSAM = True
except ImportError:
    _HAVE_GTSAM = False

from graph_backend import (  # noqa: E402
    AgentInput,
    FixedLagBackend,
    GnssInput,
    RelativeInput,
)

if _HAVE_GTSAM:
    from gtsam_backend import GtsamBackend  # noqa: E402

pytestmark = pytest.mark.skipif(not _HAVE_GTSAM, reason="gtsam not available")


def _d3(pos, yaw):
    return [[pos, 0.0, 0.0], [0.0, pos, 0.0], [0.0, 0.0, yaw]]


_COV = _d3(0.04, 0.04)


def _by_id(estimates, agent_id):
    return next(e for e in estimates if e.agent_id == agent_id)


def _assert_poses_close(a, b, tol=1e-2):
    assert math.isclose(a[0], b[0], abs_tol=tol)
    assert math.isclose(a[1], b[1], abs_tol=tol)
    assert math.isclose(a[2], b[2], abs_tol=tol)


class TestEquivalence:
    def test_backend_name(self):
        assert GtsamBackend().name == "fixed_lag_gtsam"

    def test_degraded_correction_matches_reference(self):
        agents = [
            AgentInput("robot_1", (0.0, 0.0, 0.0), _COV, degraded=False),
            AgentInput("robot_2", (3.0, 3.0, 0.0), _COV, degraded=True),
        ]
        relatives = [
            RelativeInput("robot_1", "robot_2", (2.0, 0.0, 0.0), _COV, age_sec=0.1)
        ]
        py_est, py_diag = FixedLagBackend().step(agents, relatives)
        gt_est, gt_diag = GtsamBackend().step(agents, relatives)
        _assert_poses_close(
            _by_id(gt_est, "robot_2").pose, _by_id(py_est, "robot_2").pose
        )
        _assert_poses_close(
            _by_id(gt_est, "robot_1").pose, _by_id(py_est, "robot_1").pose
        )
        # gating accounting is shared, so diagnostics must match exactly
        assert gt_diag.accepted == py_diag.accepted == 1
        assert gt_diag.rejected == py_diag.rejected == 0

    def test_gating_counts_match_reference(self):
        agents = [
            AgentInput("robot_1", (0.0, 0.0, 0.0), _COV, degraded=False),
            AgentInput("robot_2", (3.0, 3.0, 0.0), _COV, degraded=True),
        ]
        relatives = [
            RelativeInput("robot_1", "robot_2", (2.0, 0.0, 0.0), _COV, age_sec=5.0),  # stale
            RelativeInput("robot_1", "robot_2", (2.0, 0.0, 0.0), _d3(0.0, 0.04), age_sec=0.1),  # invalid
            RelativeInput("robot_1", "robot_1", (0.0, 0.0, 0.0), _COV, age_sec=0.1),  # self
        ]
        _, py_diag = FixedLagBackend().step(agents, relatives)
        _, gt_diag = GtsamBackend().step(agents, relatives)
        assert gt_diag.rejected == py_diag.rejected
        assert gt_diag.stale == py_diag.stale
        assert gt_diag.rejection_reasons == py_diag.rejection_reasons

    def test_gnss_pull_matches_reference(self):
        agents = [AgentInput("robot_1", (0.0, 0.0, 0.0), _COV, degraded=True)]
        gnss = [GnssInput("robot_1", (5.0, 0.0), [[0.04, 0.0], [0.0, 0.04]])]
        py_est, _ = FixedLagBackend().step(agents, (), gnss)
        gt_est, _ = GtsamBackend().step(agents, (), gnss)
        assert math.isclose(
            _by_id(gt_est, "robot_1").pose[0],
            _by_id(py_est, "robot_1").pose[0],
            abs_tol=1e-2,
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
