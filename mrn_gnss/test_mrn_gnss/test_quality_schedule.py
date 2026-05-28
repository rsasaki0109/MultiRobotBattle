import math
import unittest

from mrn_gnss.fix_quality import FixQuality, position_covariance
from mrn_gnss.quality_schedule import FixQualitySchedule, QualityInterval


class TestFromSteps(unittest.TestCase):
    def test_sorts_by_start(self):
        schedule = FixQualitySchedule.from_steps(
            [(10.0, FixQuality.RTK_FIX), (0.0, FixQuality.SINGLE)]
        )
        self.assertEqual(
            [interval.start_sec for interval in schedule.intervals], [0.0, 10.0]
        )

    def test_empty_raises(self):
        with self.assertRaisesRegex(ValueError, "at least one interval"):
            FixQualitySchedule.from_steps([])

    def test_duplicate_start_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate start_sec"):
            FixQualitySchedule.from_steps(
                [(0.0, FixQuality.SINGLE), (0.0, FixQuality.RTK_FIX)]
            )

    def test_accepts_int_and_name(self):
        schedule = FixQualitySchedule.from_steps([(0.0, 4), (1.0, "rtk_float")])
        self.assertEqual(schedule.quality_at(0.0), FixQuality.RTK_FIX)
        self.assertEqual(schedule.quality_at(1.0), FixQuality.RTK_FLOAT)


class TestQualityAt(unittest.TestCase):
    def test_before_first_interval_is_invalid(self):
        schedule = FixQualitySchedule.from_steps([(5.0, FixQuality.RTK_FIX)])
        self.assertEqual(schedule.quality_at(0.0), FixQuality.INVALID)
        self.assertEqual(schedule.quality_at(4.999), FixQuality.INVALID)

    def test_step_transitions_at_boundaries(self):
        schedule = FixQualitySchedule.from_steps(
            [(0.0, FixQuality.SINGLE), (10.0, FixQuality.RTK_FIX)]
        )
        self.assertEqual(schedule.quality_at(0.0), FixQuality.SINGLE)
        self.assertEqual(schedule.quality_at(9.999), FixQuality.SINGLE)
        self.assertEqual(schedule.quality_at(10.0), FixQuality.RTK_FIX)
        self.assertEqual(schedule.quality_at(100.0), FixQuality.RTK_FIX)

    def test_reacquisition_sequence(self):
        schedule = FixQualitySchedule.from_steps(
            [
                (0.0, FixQuality.RTK_FIX),
                (15.0, FixQuality.INVALID),
                (30.0, FixQuality.SINGLE),
                (35.0, FixQuality.SBAS),
                (40.0, FixQuality.RTK_FLOAT),
                (45.0, FixQuality.RTK_FIX),
            ]
        )
        self.assertEqual(schedule.quality_at(5.0), FixQuality.RTK_FIX)
        self.assertEqual(schedule.quality_at(20.0), FixQuality.INVALID)
        self.assertEqual(schedule.quality_at(31.0), FixQuality.SINGLE)
        self.assertEqual(schedule.quality_at(36.0), FixQuality.SBAS)
        self.assertEqual(schedule.quality_at(41.0), FixQuality.RTK_FLOAT)
        self.assertEqual(schedule.quality_at(50.0), FixQuality.RTK_FIX)


class TestFromConfig(unittest.TestCase):
    def test_parses_entries(self):
        schedule = FixQualitySchedule.from_config(
            [
                {"start_sec": 0.0, "fix_quality": "RTK_FIX"},
                {"start_sec": 15.0, "fix_quality": 0},
            ]
        )
        self.assertEqual(schedule.quality_at(0.0), FixQuality.RTK_FIX)
        self.assertEqual(schedule.quality_at(15.0), FixQuality.INVALID)

    def test_missing_keys_raise(self):
        with self.assertRaisesRegex(ValueError, "needs start_sec and fix_quality"):
            FixQualitySchedule.from_config([{"start_sec": 0.0}])


class TestCovarianceAt(unittest.TestCase):
    def test_matches_position_covariance(self):
        schedule = FixQualitySchedule.from_steps(
            [(0.0, FixQuality.INVALID), (10.0, FixQuality.RTK_FIX)]
        )
        self.assertEqual(
            schedule.covariance_at(20.0), position_covariance(FixQuality.RTK_FIX)
        )

    def test_invalid_covariance_is_infinite(self):
        schedule = FixQualitySchedule.from_steps([(5.0, FixQuality.RTK_FIX)])
        cov = schedule.covariance_at(0.0)  # before first interval → INVALID
        self.assertTrue(math.isinf(cov[0][0]))


if __name__ == "__main__":
    unittest.main()
