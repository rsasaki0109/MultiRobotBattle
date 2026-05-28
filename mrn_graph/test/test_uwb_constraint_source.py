import math

from constraint_gate import validate_pose_covariance, validate_relative_pose_constraint
from uwb_constraint_source import build_uwb_constraint, range_bearing_to_relative


class TestRangeBearingMath:
    def test_bearing_zero_is_forward(self):
        x, y, yaw, _ = range_bearing_to_relative(5.0, 0.0, 0.1, 0.05)
        assert math.isclose(x, 5.0, abs_tol=1e-9)
        assert math.isclose(y, 0.0, abs_tol=1e-9)
        assert yaw == 0.0

    def test_bearing_ninety_is_left(self):
        x, y, _, _ = range_bearing_to_relative(5.0, math.pi / 2, 0.1, 0.05)
        assert math.isclose(x, 0.0, abs_tol=1e-9)
        assert math.isclose(y, 5.0, abs_tol=1e-9)

    def test_tangential_variance_grows_with_range(self):
        # At bearing 0, yy = r^2 * bearing_sigma^2 grows with range.
        _, _, _, near = range_bearing_to_relative(2.0, 0.0, 0.1, 0.05)
        _, _, _, far = range_bearing_to_relative(8.0, 0.0, 0.1, 0.05)
        assert far[7] > near[7]
        # range variance maps to the radial (x) axis, unchanged by range
        assert math.isclose(near[0], far[0], abs_tol=1e-9)

    def test_covariance_is_symmetric(self):
        _, _, _, cov = range_bearing_to_relative(5.0, 0.7, 0.1, 0.05)
        assert cov[1] == cov[6]

    def test_covariance_passes_gate(self):
        for bearing in (0.0, math.pi / 4, math.pi / 2, -0.6):
            _, _, _, cov = range_bearing_to_relative(5.0, bearing, 0.1, 0.05)
            assert validate_pose_covariance(cov).accepted

    def test_rejects_bad_inputs(self):
        for bad in (
            lambda: range_bearing_to_relative(-1.0, 0.0, 0.1, 0.05),
            lambda: range_bearing_to_relative(5.0, 0.0, 0.0, 0.05),
            lambda: range_bearing_to_relative(5.0, 0.0, 0.1, 0.0),
            lambda: range_bearing_to_relative(5.0, 0.0, 0.1, 0.05, yaw_var=0.0),
        ):
            try:
                bad()
            except ValueError:
                pass
            else:
                raise AssertionError("expected ValueError")


class TestBuildUwbConstraint:
    def _build(self, **overrides):
        kwargs = dict(
            from_agent_id="robot_1",
            to_agent_id="robot_2",
            from_frame="robot_1/base_link",
            to_frame="robot_2/base_link",
            range_m=5.0,
            bearing_rad=0.3,
            range_sigma_m=0.1,
            bearing_sigma_rad=0.05,
            stamp_sec=12.5,
            sequence_id=7,
            ttl_sec=2.0,
            confidence=0.9,
        )
        kwargs.update(overrides)
        return build_uwb_constraint(**kwargs)

    def test_built_constraint_passes_gate(self):
        msg = self._build()
        result = validate_relative_pose_constraint(
            msg, known_agent_ids=["robot_1", "robot_2"]
        )
        assert result.accepted, result.reason

    def test_fields_are_populated(self):
        from mrn_msgs.msg import RelativePoseConstraint

        msg = self._build()
        assert msg.from_agent_id == "robot_1"
        assert msg.to_agent_id == "robot_2"
        assert msg.from_frame == "robot_1/base_link"
        assert msg.source_type == RelativePoseConstraint.SOURCE_UWB
        assert msg.packet.sequence_id == 7
        assert msg.packet.ttl.sec == 2
        # T_from_to position matches the pure sensor math
        x, y, _, _ = range_bearing_to_relative(5.0, 0.3, 0.1, 0.05)
        assert math.isclose(msg.relative_pose.pose.position.x, x, abs_tol=1e-9)
        assert math.isclose(msg.relative_pose.pose.position.y, y, abs_tol=1e-9)

    def test_low_confidence_is_rejected_by_gate(self):
        msg = self._build(confidence=0.0)
        result = validate_relative_pose_constraint(
            msg, known_agent_ids=["robot_1", "robot_2"]
        )
        assert not result.accepted
        assert result.reason == "low_confidence"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
