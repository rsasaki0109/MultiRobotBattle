import math
import unittest

from mrn_gnss.wgs84 import (
    WGS84_A,
    WGS84_B,
    EcefPoint,
    GeodeticPoint,
    ecef_to_geodetic,
    geodetic_to_ecef,
)


class TestGeodeticToEcef(unittest.TestCase):
    def test_equator_prime_meridian_lies_on_x_axis(self):
        ecef = geodetic_to_ecef(GeodeticPoint(0.0, 0.0, 0.0))
        self.assertAlmostEqual(ecef.x, WGS84_A, places=6)
        self.assertAlmostEqual(ecef.y, 0.0, places=6)
        self.assertAlmostEqual(ecef.z, 0.0, places=6)

    def test_north_pole_lies_on_z_axis(self):
        ecef = geodetic_to_ecef(GeodeticPoint(math.pi / 2.0, 0.0, 0.0))
        self.assertAlmostEqual(ecef.x, 0.0, places=4)
        self.assertAlmostEqual(ecef.y, 0.0, places=4)
        self.assertAlmostEqual(ecef.z, WGS84_B, places=4)

    def test_altitude_increases_radius(self):
        ground = geodetic_to_ecef(GeodeticPoint(0.0, 0.0, 0.0))
        up_100 = geodetic_to_ecef(GeodeticPoint(0.0, 0.0, 100.0))
        self.assertAlmostEqual(up_100.x - ground.x, 100.0, places=6)


class TestEcefToGeodetic(unittest.TestCase):
    def test_round_trip_at_ground_level(self):
        # Tokyo-ish reference (lat 35.6N, lon 139.7E, h 40m)
        original = GeodeticPoint(math.radians(35.6), math.radians(139.7), 40.0)
        ecef = geodetic_to_ecef(original)
        recovered = ecef_to_geodetic(ecef)
        self.assertAlmostEqual(recovered.lat_rad, original.lat_rad, places=10)
        self.assertAlmostEqual(recovered.lon_rad, original.lon_rad, places=10)
        self.assertAlmostEqual(recovered.alt_m, original.alt_m, places=6)

    def test_round_trip_at_altitude(self):
        # Mid-latitude reference at airliner altitude.
        original = GeodeticPoint(math.radians(45.0), math.radians(-122.0), 11_000.0)
        ecef = geodetic_to_ecef(original)
        recovered = ecef_to_geodetic(ecef)
        self.assertAlmostEqual(recovered.lat_rad, original.lat_rad, places=10)
        self.assertAlmostEqual(recovered.lon_rad, original.lon_rad, places=10)
        self.assertAlmostEqual(recovered.alt_m, original.alt_m, places=4)

    def test_polar_axis_handled(self):
        recovered = ecef_to_geodetic(EcefPoint(0.0, 0.0, WGS84_B + 1.0))
        self.assertAlmostEqual(recovered.lat_rad, math.pi / 2.0, places=10)
        self.assertAlmostEqual(recovered.alt_m, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
