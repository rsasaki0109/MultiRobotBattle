"""Unit tests for the cooperative-pose correction gate."""

import math
import unittest

from mrn_nav2_adapter.correction_gate import (
    STATUS_DEGRADED,
    STATUS_INVALID,
    STATUS_OK,
    STATUS_STALE,
    CorrectionGateConfig,
    CorrectionGateInput,
    CorrectionGateStatus,
    Pose2D,
    evaluate,
)


def _candidate(**overrides):
    base = {
        "stamp_sec": 100.0,
        "now_sec": 100.05,
        "status": STATUS_OK,
        "pose": Pose2D(x=1.0, y=2.0, yaw=0.0),
        "previous_pose": Pose2D(x=1.0, y=2.0, yaw=0.0),
    }
    base.update(overrides)
    return CorrectionGateInput(**base)


class TestCorrectionGate(unittest.TestCase):
    def test_accepts_fresh_status_ok_pose_without_jump(self):
        result = evaluate(_candidate(), CorrectionGateConfig())
        self.assertTrue(result.accepted)
        self.assertIs(result.status, CorrectionGateStatus.ACCEPT)
        self.assertAlmostEqual(result.translation_jump_m, 0.0)
        self.assertAlmostEqual(result.rotation_jump_rad, 0.0)

    def test_first_correction_skips_jump_gate(self):
        result = evaluate(
            _candidate(previous_pose=None, pose=Pose2D(x=100.0, y=200.0, yaw=3.0)),
            CorrectionGateConfig(
                max_translation_jump_m=0.1, max_rotation_jump_rad=0.1
            ),
        )
        self.assertTrue(result.accepted)
        self.assertIsNone(result.translation_jump_m)
        self.assertIsNone(result.rotation_jump_rad)

    def test_rejects_stale_by_age(self):
        config = CorrectionGateConfig(max_pose_age_sec=0.5)
        result = evaluate(_candidate(stamp_sec=100.0, now_sec=101.0), config)
        self.assertFalse(result.accepted)
        self.assertIs(result.status, CorrectionGateStatus.STALE_COOPERATIVE_POSE)
        self.assertAlmostEqual(result.pose_age_sec, 1.0)

    def test_rejects_status_stale_flag(self):
        result = evaluate(_candidate(status=STATUS_STALE), CorrectionGateConfig())
        self.assertIs(result.status, CorrectionGateStatus.STALE_COOPERATIVE_POSE)

    def test_rejects_status_invalid_flag(self):
        result = evaluate(_candidate(status=STATUS_INVALID), CorrectionGateConfig())
        self.assertIs(result.status, CorrectionGateStatus.INVALID_COOPERATIVE_POSE)

    def test_rejects_status_degraded_by_default(self):
        result = evaluate(_candidate(status=STATUS_DEGRADED), CorrectionGateConfig())
        self.assertIs(result.status, CorrectionGateStatus.DEGRADED_COOPERATIVE_POSE)

    def test_accepts_status_degraded_when_opted_in(self):
        result = evaluate(
            _candidate(status=STATUS_DEGRADED),
            CorrectionGateConfig(accept_degraded=True),
        )
        self.assertTrue(result.accepted)

    def test_rejects_translation_jump(self):
        config = CorrectionGateConfig(max_translation_jump_m=1.0)
        candidate = _candidate(
            pose=Pose2D(x=2.5, y=2.0, yaw=0.0),
            previous_pose=Pose2D(x=1.0, y=2.0, yaw=0.0),
        )
        result = evaluate(candidate, config)
        self.assertIs(result.status, CorrectionGateStatus.TRANSLATION_JUMP_TOO_LARGE)
        self.assertAlmostEqual(result.translation_jump_m, 1.5)

    def test_rejects_rotation_jump(self):
        config = CorrectionGateConfig(
            max_translation_jump_m=10.0, max_rotation_jump_rad=math.radians(10.0)
        )
        candidate = _candidate(
            pose=Pose2D(x=1.0, y=2.0, yaw=math.radians(30.0)),
            previous_pose=Pose2D(x=1.0, y=2.0, yaw=0.0),
        )
        result = evaluate(candidate, config)
        self.assertIs(result.status, CorrectionGateStatus.ROTATION_JUMP_TOO_LARGE)
        self.assertAlmostEqual(result.rotation_jump_rad, math.radians(30.0))

    def test_rotation_jump_uses_wrap_around(self):
        config = CorrectionGateConfig(max_rotation_jump_rad=math.radians(20.0))
        candidate = _candidate(
            pose=Pose2D(x=1.0, y=2.0, yaw=math.radians(179.0)),
            previous_pose=Pose2D(x=1.0, y=2.0, yaw=math.radians(-179.0)),
        )
        result = evaluate(candidate, config)
        self.assertTrue(result.accepted, "shortest-arc delta should be 2 deg")
        self.assertAlmostEqual(
            result.rotation_jump_rad, math.radians(2.0), places=9
        )

    def test_unknown_status_rejected(self):
        result = evaluate(_candidate(status=99), CorrectionGateConfig())
        self.assertIs(result.status, CorrectionGateStatus.UNKNOWN_STATUS_FLAG)

    def test_nonfinite_pose_rejected(self):
        result = evaluate(
            _candidate(pose=Pose2D(x=float("nan"), y=2.0, yaw=0.0)),
            CorrectionGateConfig(),
        )
        self.assertIs(result.status, CorrectionGateStatus.NONFINITE_POSE)

    def test_status_enum_has_stable_reason_strings(self):
        """The reason strings double as report keys so they must not silently move."""
        expected = {
            "accept",
            "stale_cooperative_pose",
            "degraded_cooperative_pose",
            "invalid_cooperative_pose",
            "translation_jump_too_large",
            "rotation_jump_too_large",
            "nonfinite_pose",
            "unknown_status_flag",
        }
        self.assertEqual({member.value for member in CorrectionGateStatus}, expected)


if __name__ == "__main__":
    unittest.main()
