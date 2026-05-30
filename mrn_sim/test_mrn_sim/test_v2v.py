"""Tests for the sim V2V relative-pose observation and constraint emission.

The pure observation (geometry + covariance) is tested directly. The
constraint-builder is additionally checked against ``constraint_gate`` from
mrn_graph when it (and mrn_msgs) are importable, honoring the project rule that
a constraint source is correct iff its output passes the gate.
"""

import importlib.util
import math
import os
import sys
import unittest

from mrn_sim import relative_pose_observation
from mrn_sim.sensors import _XX, _YY, _YAW


class TestRelativePoseObservation(unittest.TestCase):
    def test_noiseless_geometry_matches_body_frame(self):
        x, y, yaw, cov = relative_pose_observation(
            (0.0, 0.0, 0.0), (3.0, 0.0, 0.0), xy_sigma=0.1, yaw_sigma=0.05
        )
        self.assertAlmostEqual(x, 3.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(yaw, 0.0)
        self.assertEqual(len(cov), 36)

    def test_observer_heading_rotates_measurement(self):
        # observer faces +y; target 3m north in world -> 3m forward in body
        x, y, _, _ = relative_pose_observation(
            (0.0, 0.0, math.pi / 2), (0.0, 3.0, math.pi / 2)
        )
        self.assertAlmostEqual(x, 3.0, places=9)
        self.assertAlmostEqual(y, 0.0, places=9)

    def test_covariance_diagonal_from_sigmas(self):
        _, _, _, cov = relative_pose_observation(
            (0.0, 0.0, 0.0), (1.0, 1.0, 0.0), xy_sigma=0.2, yaw_sigma=0.1
        )
        self.assertAlmostEqual(cov[_XX], 0.04)
        self.assertAlmostEqual(cov[_YY], 0.04)
        self.assertAlmostEqual(cov[_YAW], 0.01)
        # off-SE(2) axes finite, all positive
        self.assertTrue(all(cov[i] > 0.0 for i in (0, 7, 14, 21, 28, 35)))

    def test_noise_is_reproducible(self):
        import random
        a = relative_pose_observation((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), rng=random.Random(3))
        b = relative_pose_observation((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), rng=random.Random(3))
        self.assertEqual(a, b)
        self.assertNotAlmostEqual(a[0], 2.0)   # noise moved it off the true value


def _gate_available():
    if importlib.util.find_spec("mrn_msgs") is None:
        return False
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for sub in ("mrn_graph/scripts", "mrn_sync"):
        p = os.path.join(repo, sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    return importlib.util.find_spec("constraint_gate") is not None


@unittest.skipUnless(_gate_available(), "mrn_msgs / constraint_gate not available")
class TestConstraintPassesGate(unittest.TestCase):
    def test_built_constraint_is_accepted(self):
        from constraint_gate import validate_relative_pose_constraint

        from mrn_sim.v2v import build_relative_constraint

        x, y, yaw, cov = relative_pose_observation(
            (0.0, 0.0, 0.0), (3.0, 1.0, 0.2), xy_sigma=0.1, yaw_sigma=0.05
        )
        msg = build_relative_constraint(
            from_agent_id="robot_1", to_agent_id="robot_2",
            from_frame="robot_1/base_link", to_frame="robot_2/base_link",
            x=x, y=y, yaw=yaw, covariance=cov, stamp_sec=5.0,
            sequence_id=1, confidence=0.9,
        )
        result = validate_relative_pose_constraint(
            msg, known_agent_ids=["robot_1", "robot_2"]
        )
        self.assertTrue(result.accepted, getattr(result, "reason", ""))

    def test_low_confidence_rejected(self):
        from constraint_gate import validate_relative_pose_constraint

        from mrn_sim.v2v import build_relative_constraint

        x, y, yaw, cov = relative_pose_observation((0.0, 0.0, 0.0), (3.0, 0.0, 0.0))
        msg = build_relative_constraint(
            from_agent_id="robot_1", to_agent_id="robot_2",
            from_frame="robot_1/base_link", to_frame="robot_2/base_link",
            x=x, y=y, yaw=yaw, covariance=cov, stamp_sec=5.0, confidence=0.0,
        )
        result = validate_relative_pose_constraint(
            msg, known_agent_ids=["robot_1", "robot_2"]
        )
        self.assertFalse(result.accepted)


if __name__ == "__main__":
    unittest.main()
