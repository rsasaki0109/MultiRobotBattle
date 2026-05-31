"""Tests for the sim V2V relative-pose observation and constraint emission.

The pure observation (geometry + covariance) is tested directly. The
constraint-builder is checked to populate a well-formed
``mrn_msgs/RelativePoseConstraint`` (the message the localization repo,
multirobot-localization, consumes — see that repo for gate/estimator checks).
"""

import importlib.util
import math
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


@unittest.skipUnless(
    importlib.util.find_spec("mrn_msgs") is not None, "mrn_msgs not available"
)
class TestBuildConstraint(unittest.TestCase):
    def test_built_constraint_fields(self):
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
        self.assertEqual(msg.from_agent_id, "robot_1")
        self.assertEqual(msg.to_agent_id, "robot_2")
        self.assertAlmostEqual(msg.relative_pose.pose.position.x, x)
        self.assertAlmostEqual(msg.relative_pose.pose.position.y, y)
        self.assertEqual(len(msg.relative_pose.covariance), 36)
        self.assertAlmostEqual(msg.confidence, 0.9, places=6)
        self.assertEqual(msg.packet.ttl.sec, 0)   # finite TTL set


if __name__ == "__main__":
    unittest.main()
