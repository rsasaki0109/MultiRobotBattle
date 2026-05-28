from math import isclose
import unittest

from mrn_eval.metrics import (
    StreamingAte,
    ate_2d,
    distance_2d,
    recovery_time,
    rmse,
    rpe_translation_2d,
)


class TestMetrics(unittest.TestCase):
    def test_rmse(self):
        self.assertTrue(isclose(rmse([3.0, 4.0]), 3.5355339059327378))

    def test_ate_2d(self):
        self.assertTrue(
            isclose(
                ate_2d([(0.0, 0.0), (1.1, 0.0)], [(0.0, 0.0), (1.0, 0.0)]),
                0.07071067811865482,
            )
        )

    def test_distance_2d(self):
        self.assertTrue(isclose(distance_2d((3.0, 4.0), (0.0, 0.0)), 5.0))

    def test_streaming_ate_window(self):
        metric = StreamingAte(max_samples=2)
        metric.push((1.0, 0.0), (0.0, 0.0))
        metric.push((2.0, 0.0), (0.0, 0.0))
        metric.push((2.0, 0.0), (0.0, 0.0))
        self.assertEqual(metric.count, 2)
        self.assertTrue(isclose(metric.rmse(), 2.0))


class TestRpe(unittest.TestCase):
    def test_zero_when_estimate_matches_truth(self):
        truth = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        self.assertTrue(isclose(rpe_translation_2d(truth, truth), 0.0))

    def test_constant_offset_is_zero_rpe(self):
        truth = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        estimate = [(10.0, 5.0), (11.0, 5.0), (12.0, 5.0)]
        self.assertTrue(isclose(rpe_translation_2d(estimate, truth), 0.0))

    def test_uniform_drift(self):
        truth = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        estimate = [(0.0, 0.0), (1.1, 0.0), (2.2, 0.0), (3.3, 0.0)]
        self.assertTrue(
            isclose(rpe_translation_2d(estimate, truth), 0.1, rel_tol=1e-9)
        )

    def test_delta_greater_than_one(self):
        truth = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        estimate = [(0.0, 0.0), (1.0, 0.0), (2.5, 0.0), (3.5, 0.0)]
        # Compare pairs at delta=2: (est[2]-est[0]) vs (truth[2]-truth[0])=2.5-2.0=0.5
        # And (est[3]-est[1]) vs (truth[3]-truth[1])=2.5-2.0=0.5
        self.assertTrue(
            isclose(rpe_translation_2d(estimate, truth, delta=2), 0.5)
        )

    def test_invalid_delta(self):
        truth = [(0.0, 0.0), (1.0, 0.0)]
        with self.assertRaises(ValueError):
            rpe_translation_2d(truth, truth, delta=0)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            rpe_translation_2d([(0.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)])

    def test_not_enough_samples(self):
        truth = [(0.0, 0.0)]
        with self.assertRaises(ValueError):
            rpe_translation_2d(truth, truth, delta=1)


class TestRecoveryTime(unittest.TestCase):
    def test_never_degrades_returns_zero(self):
        samples = [(0.0, 0.1), (1.0, 0.2), (2.0, 0.1)]
        self.assertEqual(
            recovery_time(samples, degraded_threshold=1.0, recovered_threshold=0.5),
            0.0,
        )

    def test_returns_none_if_never_recovers(self):
        samples = [(0.0, 0.1), (1.0, 2.0), (2.0, 3.0), (3.0, 2.5)]
        self.assertIsNone(
            recovery_time(samples, degraded_threshold=1.0, recovered_threshold=0.3)
        )

    def test_simple_recovery(self):
        samples = [
            (0.0, 0.1),
            (1.0, 2.0),  # degraded
            (2.0, 1.5),  # still degraded
            (3.0, 0.2),  # recovered, hold=0 -> confirmed immediately
        ]
        result = recovery_time(
            samples, degraded_threshold=1.0, recovered_threshold=0.3
        )
        self.assertIsNotNone(result)
        self.assertTrue(isclose(result, 2.0))

    def test_hold_seconds_requires_sustained_recovery(self):
        samples = [
            (0.0, 0.1),
            (1.0, 2.0),  # degraded
            (2.0, 0.2),  # recovered window starts
            (2.5, 1.5),  # spikes back above recovered_threshold -> reset
            (3.0, 0.2),  # recovered window restarts
            (4.0, 0.2),  # held for 1.0s
        ]
        result = recovery_time(
            samples,
            degraded_threshold=1.0,
            recovered_threshold=0.3,
            hold_seconds=1.0,
        )
        self.assertIsNotNone(result)
        self.assertTrue(isclose(result, 2.0))

    def test_rejects_unsorted_samples(self):
        samples = [(1.0, 0.1), (0.5, 0.2)]
        with self.assertRaises(ValueError):
            recovery_time(samples, 1.0, 0.5)

    def test_rejects_invalid_thresholds(self):
        with self.assertRaises(ValueError):
            recovery_time([(0.0, 0.1)], degraded_threshold=0.0, recovered_threshold=0.1)
        with self.assertRaises(ValueError):
            recovery_time([(0.0, 0.1)], degraded_threshold=1.0, recovered_threshold=0.0)
        with self.assertRaises(ValueError):
            recovery_time(
                [(0.0, 0.1)], degraded_threshold=0.3, recovered_threshold=0.5
            )


if __name__ == "__main__":
    unittest.main()
