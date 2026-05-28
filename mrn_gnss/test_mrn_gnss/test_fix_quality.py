import math
import unittest

from mrn_gnss.fix_quality import (
    HORIZONTAL_SIGMA_M,
    VERTICAL_SIGMA_M,
    FixQuality,
    position_covariance,
)


class TestFixQualityEnum(unittest.TestCase):
    def test_enum_values_match_nmea_gga(self):
        self.assertEqual(int(FixQuality.INVALID), 0)
        self.assertEqual(int(FixQuality.SINGLE), 1)
        self.assertEqual(int(FixQuality.DGPS), 2)
        self.assertEqual(int(FixQuality.RTK_FIX), 4)
        self.assertEqual(int(FixQuality.RTK_FLOAT), 5)
        self.assertEqual(int(FixQuality.SBAS), 9)


class TestFromNavSatStatus(unittest.TestCase):
    def test_no_fix(self):
        self.assertEqual(FixQuality.from_navsatstatus(-1), FixQuality.INVALID)

    def test_status_fix_is_single(self):
        self.assertEqual(FixQuality.from_navsatstatus(0), FixQuality.SINGLE)

    def test_sbas(self):
        self.assertEqual(FixQuality.from_navsatstatus(1), FixQuality.SBAS)

    def test_gbas(self):
        self.assertEqual(FixQuality.from_navsatstatus(2), FixQuality.DGPS)

    def test_unknown_status_falls_back_to_invalid(self):
        self.assertEqual(FixQuality.from_navsatstatus(99), FixQuality.INVALID)


class TestSigmaTables(unittest.TestCase):
    def test_invalid_is_infinite(self):
        self.assertTrue(math.isinf(HORIZONTAL_SIGMA_M[FixQuality.INVALID]))
        self.assertTrue(math.isinf(VERTICAL_SIGMA_M[FixQuality.INVALID]))

    def test_rtk_fix_is_centimeter_level(self):
        self.assertLess(HORIZONTAL_SIGMA_M[FixQuality.RTK_FIX], 0.05)

    def test_rtk_float_worse_than_rtk_fix(self):
        self.assertGreater(
            HORIZONTAL_SIGMA_M[FixQuality.RTK_FLOAT],
            HORIZONTAL_SIGMA_M[FixQuality.RTK_FIX],
        )

    def test_vertical_is_twice_horizontal_for_finite_values(self):
        for quality, h_sigma in HORIZONTAL_SIGMA_M.items():
            if not math.isfinite(h_sigma):
                continue
            self.assertAlmostEqual(
                VERTICAL_SIGMA_M[quality], 2.0 * h_sigma, places=12
            )


class TestPositionCovariance(unittest.TestCase):
    def test_diagonal_matches_sigma_squared(self):
        cov = position_covariance(FixQuality.RTK_FIX)
        h_var = HORIZONTAL_SIGMA_M[FixQuality.RTK_FIX] ** 2
        v_var = VERTICAL_SIGMA_M[FixQuality.RTK_FIX] ** 2
        self.assertAlmostEqual(cov[0][0], h_var, places=12)
        self.assertAlmostEqual(cov[1][1], h_var, places=12)
        self.assertAlmostEqual(cov[2][2], v_var, places=12)

    def test_off_diagonal_zero(self):
        cov = position_covariance(FixQuality.SBAS)
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                self.assertEqual(cov[i][j], 0.0)

    def test_invalid_returns_inf_diagonal(self):
        cov = position_covariance(FixQuality.INVALID)
        for i in range(3):
            self.assertTrue(math.isinf(cov[i][i]))

    def test_unknown_quality_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown FixQuality"):
            position_covariance(999)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
