"""Smoke tests verifying the cooperative-pose correction gate wiring.

The exhaustive gate semantics are covered by the corresponding test in
``mrn_nav2_adapter``; these tests just confirm that the same rules apply
in the autoware adapter's namespace (so a future divergence between the
two copies of ``correction_gate.py`` shows up immediately).
"""

import math
import unittest

from mrn_autoware_adapter.correction_gate import (
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


class TestCorrectionGateWiring(unittest.TestCase):
    def test_accepts_fresh_status_ok(self):
        result = evaluate(_candidate(), CorrectionGateConfig())
        self.assertTrue(result.accepted)
        self.assertIs(result.status, CorrectionGateStatus.ACCEPT)

    def test_rejects_stale_pose_by_age(self):
        result = evaluate(
            _candidate(stamp_sec=100.0, now_sec=102.0),
            CorrectionGateConfig(max_pose_age_sec=1.0),
        )
        self.assertFalse(result.accepted)
        self.assertIs(result.status, CorrectionGateStatus.STALE_COOPERATIVE_POSE)
        self.assertAlmostEqual(result.pose_age_sec, 2.0)

    def test_rejects_translation_jump(self):
        result = evaluate(
            _candidate(pose=Pose2D(x=10.0, y=2.0, yaw=0.0)),
            CorrectionGateConfig(max_translation_jump_m=1.5),
        )
        self.assertIs(result.status, CorrectionGateStatus.TRANSLATION_JUMP_TOO_LARGE)

    def test_rejects_rotation_jump(self):
        result = evaluate(
            _candidate(pose=Pose2D(x=1.0, y=2.0, yaw=math.radians(60.0))),
            CorrectionGateConfig(max_rotation_jump_rad=math.radians(20.0)),
        )
        self.assertIs(result.status, CorrectionGateStatus.ROTATION_JUMP_TOO_LARGE)

    def test_rejects_invalid_status(self):
        result = evaluate(_candidate(status=STATUS_INVALID), CorrectionGateConfig())
        self.assertIs(result.status, CorrectionGateStatus.INVALID_COOPERATIVE_POSE)

    def test_rejects_stale_status(self):
        result = evaluate(_candidate(status=STATUS_STALE), CorrectionGateConfig())
        self.assertIs(result.status, CorrectionGateStatus.STALE_COOPERATIVE_POSE)

    def test_rejects_degraded_when_not_accepting(self):
        result = evaluate(
            _candidate(status=STATUS_DEGRADED),
            CorrectionGateConfig(accept_degraded=False),
        )
        self.assertIs(result.status, CorrectionGateStatus.DEGRADED_COOPERATIVE_POSE)

    def test_accepts_degraded_when_configured(self):
        result = evaluate(
            _candidate(status=STATUS_DEGRADED),
            CorrectionGateConfig(accept_degraded=True),
        )
        self.assertIs(result.status, CorrectionGateStatus.ACCEPT)

    def test_rejects_unknown_status_flag(self):
        result = evaluate(_candidate(status=99), CorrectionGateConfig())
        self.assertIs(result.status, CorrectionGateStatus.UNKNOWN_STATUS_FLAG)

    def test_first_correction_skips_jump_gate(self):
        result = evaluate(
            _candidate(pose=Pose2D(x=100.0, y=200.0, yaw=0.0), previous_pose=None),
            CorrectionGateConfig(max_translation_jump_m=1.5),
        )
        self.assertIs(result.status, CorrectionGateStatus.ACCEPT)
        self.assertIsNone(result.translation_jump_m)


if __name__ == "__main__":
    unittest.main()
