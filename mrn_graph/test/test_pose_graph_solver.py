import math

from factor_graph import FactorReason
from pose_graph_solver import (
    BetweenFactor,
    FactorKind,
    PriorFactor,
    gauss_newton,
)


def _diag(*values):
    n = len(values)
    return [[values[i] if i == j else 0.0 for j in range(n)] for i in range(n)]


_TIGHT = _diag(1e-6, 1e-6, 1e-6)
_LOOSE = _diag(0.04, 0.04, 0.04)


class TestTwoPose:
    def test_between_with_anchor_converges_to_exact(self):
        priors = [PriorFactor("a", (0.0, 0.0, 0.0), _TIGHT, FactorKind.PRIOR_POSE)]
        betweens = [BetweenFactor("a", "b", (2.0, 0.0, 0.0), _LOOSE)]
        result = gauss_newton(
            {"a": (0.0, 0.0, 0.0), "b": (5.0, 3.0, 1.0)}, priors, betweens
        )
        assert result.converged
        bx, by, byaw = result.variables["b"]
        assert math.isclose(bx, 2.0, abs_tol=1e-4)
        assert math.isclose(by, 0.0, abs_tol=1e-4)
        assert math.isclose(byaw, 0.0, abs_tol=1e-4)
        assert result.final_cost < 1e-6


class TestTriangleLoop:
    def test_consistent_loop_closes(self):
        # a anchored at origin; b at (2,0); c at (2,2); loop a->b->c->a consistent.
        priors = [PriorFactor("a", (0.0, 0.0, 0.0), _TIGHT, FactorKind.PRIOR_POSE)]
        betweens = [
            BetweenFactor("a", "b", (2.0, 0.0, 0.0), _LOOSE),
            BetweenFactor("b", "c", (0.0, 2.0, 0.0), _LOOSE),
            BetweenFactor("a", "c", (2.0, 2.0, 0.0), _LOOSE),
        ]
        result = gauss_newton(
            {"a": (0.0, 0.0, 0.0), "b": (1.0, 1.0, 0.5), "c": (0.0, 0.0, 0.0)},
            priors,
            betweens,
        )
        assert result.converged
        assert math.isclose(result.variables["b"][0], 2.0, abs_tol=1e-3)
        assert math.isclose(result.variables["c"][0], 2.0, abs_tol=1e-3)
        assert math.isclose(result.variables["c"][1], 2.0, abs_tol=1e-3)


class TestGnssPrior:
    def test_position_prior_pulls_estimate(self):
        # between says b is at x=2.0; GNSS prior says x=2.5; equal weight -> 2.25.
        priors = [
            PriorFactor("a", (0.0, 0.0, 0.0), _TIGHT, FactorKind.PRIOR_POSE),
            PriorFactor("b", (2.5, 0.0), _diag(0.04, 0.04), FactorKind.PRIOR_POSITION),
        ]
        betweens = [BetweenFactor("a", "b", (2.0, 0.0, 0.0), _diag(0.04, 0.04, 0.04))]
        result = gauss_newton(
            {"a": (0.0, 0.0, 0.0), "b": (0.0, 0.0, 0.0)}, priors, betweens
        )
        assert result.converged
        assert math.isclose(result.variables["b"][0], 2.25, abs_tol=1e-3)


class TestRobustLoss:
    def _scene(self):
        priors = [PriorFactor("a", (0.0, 0.0, 0.0), _TIGHT, FactorKind.PRIOR_POSE)]
        betweens = [
            BetweenFactor("a", "b", (2.0, 0.0, 0.0), _LOOSE),
            BetweenFactor("a", "b", (2.0, 0.0, 0.0), _LOOSE),
            BetweenFactor("a", "b", (10.0, 0.0, 0.0), _LOOSE),  # outlier
        ]
        return priors, betweens

    def test_plain_least_squares_is_pulled_by_outlier(self):
        priors, betweens = self._scene()
        result = gauss_newton({"a": (0.0, 0.0, 0.0), "b": (0.0, 0.0, 0.0)}, priors, betweens)
        # mean of 2, 2, 10 = 4.667
        assert math.isclose(result.variables["b"][0], 4.667, abs_tol=1e-2)

    def test_huber_downweights_outlier(self):
        priors, betweens = self._scene()
        result = gauss_newton(
            {"a": (0.0, 0.0, 0.0), "b": (2.5, 0.0, 0.0)},
            priors,
            betweens,
            huber_delta=1.0,
        )
        # Robust optimum sits near the two inliers, well below the LS 4.667.
        assert result.variables["b"][0] < 3.0
        assert result.variables["b"][0] > 1.5


class TestFactorReports:
    def test_invalid_covariance_factor_reported_and_skipped(self):
        priors = [PriorFactor("a", (0.0, 0.0, 0.0), _TIGHT, FactorKind.PRIOR_POSE)]
        betweens = [
            BetweenFactor("a", "b", (2.0, 0.0, 0.0), _LOOSE),
            BetweenFactor("a", "b", (99.0, 0.0, 0.0), _diag(0.0, 0.04, 0.04)),  # invalid
        ]
        result = gauss_newton(
            {"a": (0.0, 0.0, 0.0), "b": (0.0, 0.0, 0.0)}, priors, betweens
        )
        # The invalid factor is skipped, so b follows the valid 2.0 constraint.
        assert math.isclose(result.variables["b"][0], 2.0, abs_tol=1e-3)
        reasons = {r.label: r.reason for r in result.factor_reports}
        assert reasons["between[1]:a->b"] is FactorReason.INVALID_COVARIANCE
        assert reasons["between[0]:a->b"] is FactorReason.ACCEPT

    def test_no_prior_is_singular_and_not_converged(self):
        betweens = [BetweenFactor("a", "b", (2.0, 0.0, 0.0), _LOOSE)]
        result = gauss_newton({"a": (0.0, 0.0, 0.0), "b": (0.0, 0.0, 0.0)}, [], betweens)
        assert not result.converged


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
