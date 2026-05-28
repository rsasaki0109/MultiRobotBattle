import math

from factor_graph import (
    FactorReason,
    between_residual,
    covariance_is_valid,
    evaluate_factor,
    gnss_prior_residual,
    huber_weight,
    information_from_covariance,
    invert_matrix,
    mahalanobis_norm,
    normalize_angle,
    pose_between,
    pose_compose,
    pose_inverse,
)


def _diag(*values):
    n = len(values)
    return [[values[i] if i == j else 0.0 for j in range(n)] for i in range(n)]


class TestSE2:
    def test_normalize_angle_wraps(self):
        assert math.isclose(normalize_angle(3.0 * math.pi), math.pi, abs_tol=1e-9)
        assert math.isclose(normalize_angle(-3.0 * math.pi), math.pi, abs_tol=1e-9)

    def test_compose_inverse_is_identity(self):
        p = (1.0, 2.0, math.radians(30.0))
        x, y, yaw = pose_compose(p, pose_inverse(p))
        assert math.isclose(x, 0.0, abs_tol=1e-9)
        assert math.isclose(y, 0.0, abs_tol=1e-9)
        assert math.isclose(yaw, 0.0, abs_tol=1e-9)

    def test_between_recovers_relative(self):
        a = (1.0, 0.0, 0.0)
        b = (1.0, 1.0, math.radians(90.0))
        rel = pose_between(a, b)
        # b is 1m ahead in a's +y, rotated +90deg.
        assert math.isclose(rel[0], 0.0, abs_tol=1e-9)
        assert math.isclose(rel[1], 1.0, abs_tol=1e-9)
        assert math.isclose(rel[2], math.radians(90.0), abs_tol=1e-9)


class TestResiduals:
    def test_perfect_odometry_residual_is_zero(self):
        pose_i = (0.0, 0.0, 0.0)
        pose_j = (2.0, 0.0, math.radians(10.0))
        measured = pose_between(pose_i, pose_j)
        rx, ry, ryaw = between_residual(pose_i, pose_j, measured)
        assert math.isclose(rx, 0.0, abs_tol=1e-9)
        assert math.isclose(ry, 0.0, abs_tol=1e-9)
        assert math.isclose(ryaw, 0.0, abs_tol=1e-9)

    def test_odometry_residual_nonzero_on_mismatch(self):
        pose_i = (0.0, 0.0, 0.0)
        pose_j = (2.0, 0.0, 0.0)
        measured = (1.0, 0.0, 0.0)  # claims only 1m of travel
        rx, _, _ = between_residual(pose_i, pose_j, measured)
        assert math.isclose(rx, 1.0, abs_tol=1e-9)

    def test_gnss_prior_residual(self):
        assert gnss_prior_residual((3.0, 4.0, 0.0), (1.0, 1.0)) == (2.0, 3.0)


class TestLinearAlgebra:
    def test_invert_identity(self):
        identity = _diag(1.0, 1.0, 1.0)
        inverse = invert_matrix(identity)
        for i in range(3):
            for j in range(3):
                assert math.isclose(inverse[i][j], identity[i][j], abs_tol=1e-9)

    def test_invert_diagonal(self):
        inverse = invert_matrix(_diag(4.0, 0.25, 2.0))
        assert math.isclose(inverse[0][0], 0.25, abs_tol=1e-9)
        assert math.isclose(inverse[1][1], 4.0, abs_tol=1e-9)
        assert math.isclose(inverse[2][2], 0.5, abs_tol=1e-9)

    def test_invert_general_2x2(self):
        inverse = invert_matrix([[4.0, 3.0], [6.0, 3.0]])
        # det = 4*3 - 3*6 = -6; inv = 1/-6 * [[3,-3],[-6,4]]
        assert math.isclose(inverse[0][0], -0.5, abs_tol=1e-9)
        assert math.isclose(inverse[0][1], 0.5, abs_tol=1e-9)
        assert math.isclose(inverse[1][0], 1.0, abs_tol=1e-9)
        assert math.isclose(inverse[1][1], -2.0 / 3.0, abs_tol=1e-9)

    def test_singular_raises(self):
        try:
            invert_matrix([[1.0, 2.0], [2.0, 4.0]])
        except ValueError as exc:
            assert "singular" in str(exc)
        else:
            raise AssertionError("expected ValueError")

    def test_nonfinite_raises(self):
        try:
            invert_matrix([[float("inf"), 0.0], [0.0, 1.0]])
        except ValueError as exc:
            assert "non-finite" in str(exc)
        else:
            raise AssertionError("expected ValueError")


class TestCovarianceWeighting:
    def test_covariance_is_valid(self):
        assert covariance_is_valid(_diag(0.04, 0.04, 0.01))
        assert not covariance_is_valid(_diag(0.04, -0.04, 0.01))  # negative diag
        assert not covariance_is_valid([[float("nan"), 0.0], [0.0, 1.0]])
        assert not covariance_is_valid([[1.0, 0.0]])  # not square

    def test_information_is_inverse(self):
        info = information_from_covariance(_diag(0.25, 0.25, 4.0))
        assert math.isclose(info[0][0], 4.0, abs_tol=1e-9)
        assert math.isclose(info[2][2], 0.25, abs_tol=1e-9)

    def test_information_invalid_raises(self):
        try:
            information_from_covariance(_diag(0.0, 1.0, 1.0))
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    def test_mahalanobis_norm_scales_with_information(self):
        # residual (1,0,0) under unit covariance → norm 1
        info = information_from_covariance(_diag(1.0, 1.0, 1.0))
        assert math.isclose(mahalanobis_norm((1.0, 0.0, 0.0), info), 1.0, abs_tol=1e-9)
        # tighter covariance (0.25) → information 4 → norm 2
        tight = information_from_covariance(_diag(0.25, 1.0, 1.0))
        assert math.isclose(mahalanobis_norm((1.0, 0.0, 0.0), tight), 2.0, abs_tol=1e-9)


class TestHuberWeight:
    def test_inside_linear_region(self):
        assert huber_weight(0.5, 1.0) == 1.0
        assert huber_weight(1.0, 1.0) == 1.0

    def test_outside_region_downweights(self):
        assert math.isclose(huber_weight(2.0, 1.0), 0.5, abs_tol=1e-9)
        assert math.isclose(huber_weight(4.0, 2.0), 0.5, abs_tol=1e-9)

    def test_nonpositive_delta_raises(self):
        try:
            huber_weight(1.0, 0.0)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


class TestEvaluateFactor:
    def test_accepts_and_weights(self):
        result = evaluate_factor((1.0, 0.0, 0.0), _diag(1.0, 1.0, 1.0))
        assert result.accepted
        assert result.reason is FactorReason.ACCEPT
        assert math.isclose(result.mahalanobis_norm, 1.0, abs_tol=1e-9)
        assert result.robust_weight == 1.0

    def test_huber_downweights_outlier(self):
        result = evaluate_factor(
            (4.0, 0.0, 0.0), _diag(1.0, 1.0, 1.0), huber_delta=1.0
        )
        assert result.accepted
        assert math.isclose(result.robust_weight, 0.25, abs_tol=1e-9)

    def test_nonfinite_measurement_rejected(self):
        result = evaluate_factor((float("nan"), 0.0, 0.0), _diag(1.0, 1.0, 1.0))
        assert result.reason is FactorReason.NONFINITE_MEASUREMENT
        assert not result.accepted

    def test_invalid_covariance_rejected(self):
        result = evaluate_factor((1.0, 0.0, 0.0), _diag(0.0, 1.0, 1.0))
        assert result.reason is FactorReason.INVALID_COVARIANCE

    def test_stale_rejected(self):
        result = evaluate_factor(
            (0.1, 0.0, 0.0), _diag(1.0, 1.0, 1.0), age_sec=3.0, max_age_sec=2.0
        )
        assert result.reason is FactorReason.STALE

    def test_fresh_within_age_accepted(self):
        result = evaluate_factor(
            (0.1, 0.0, 0.0), _diag(1.0, 1.0, 1.0), age_sec=1.0, max_age_sec=2.0
        )
        assert result.accepted
