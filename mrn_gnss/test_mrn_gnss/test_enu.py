import math
import unittest

from mrn_gnss.enu import (
    EnuOrigin,
    EnuPoint,
    ecef_to_enu,
    enu_to_ecef,
    enu_to_geodetic,
    geodetic_to_enu,
)
from mrn_gnss.wgs84 import GeodeticPoint, geodetic_to_ecef


def _origin(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> EnuOrigin:
    return EnuOrigin.from_geodetic(
        GeodeticPoint(math.radians(lat_deg), math.radians(lon_deg), alt_m)
    )


class TestOriginIdentity(unittest.TestCase):
    def test_origin_maps_to_zero(self):
        origin = _origin(35.6, 139.7, 40.0)
        enu = geodetic_to_enu(origin.geodetic, origin)
        self.assertAlmostEqual(enu.east, 0.0, places=6)
        self.assertAlmostEqual(enu.north, 0.0, places=6)
        self.assertAlmostEqual(enu.up, 0.0, places=6)


class TestAxisDirections(unittest.TestCase):
    def test_one_degree_north_is_north_axis(self):
        origin = _origin(35.0, 139.0, 0.0)
        north_1_deg = GeodeticPoint(math.radians(36.0), math.radians(139.0), 0.0)
        enu = geodetic_to_enu(north_1_deg, origin)
        # Meridional arc of 1 deg is roughly 111 km. Small east bleed is OK due
        # to the ellipsoid, but the displacement is overwhelmingly along north.
        self.assertGreater(enu.north, 110_000.0)
        self.assertLess(enu.north, 112_000.0)
        self.assertLess(abs(enu.east), 1.0)

    def test_one_degree_east_is_east_axis_at_equator(self):
        origin = _origin(0.0, 0.0, 0.0)
        east_1_deg = GeodeticPoint(math.radians(0.0), math.radians(1.0), 0.0)
        enu = geodetic_to_enu(east_1_deg, origin)
        # Equator: 1 deg of longitude ≈ 111.319 km of east displacement.
        self.assertGreater(enu.east, 111_000.0)
        self.assertLess(enu.east, 112_000.0)
        self.assertLess(abs(enu.north), 1.0)

    def test_up_displacement(self):
        origin = _origin(45.0, 100.0, 0.0)
        above = GeodeticPoint(origin.geodetic.lat_rad, origin.geodetic.lon_rad, 50.0)
        enu = geodetic_to_enu(above, origin)
        self.assertAlmostEqual(enu.up, 50.0, places=3)
        self.assertLess(abs(enu.east), 1.0e-3)
        self.assertLess(abs(enu.north), 1.0e-3)


class TestRoundTrip(unittest.TestCase):
    def test_ecef_enu_ecef_round_trip(self):
        origin = _origin(35.6, 139.7, 40.0)
        target = geodetic_to_ecef(
            GeodeticPoint(math.radians(35.61), math.radians(139.71), 45.0)
        )
        enu = ecef_to_enu(target, origin)
        recovered = enu_to_ecef(enu, origin)
        self.assertAlmostEqual(recovered.x, target.x, places=4)
        self.assertAlmostEqual(recovered.y, target.y, places=4)
        self.assertAlmostEqual(recovered.z, target.z, places=4)

    def test_geodetic_enu_geodetic_round_trip(self):
        origin = _origin(-33.86, 151.21, 10.0)
        target = GeodeticPoint(math.radians(-33.85), math.radians(151.22), 12.0)
        enu = geodetic_to_enu(target, origin)
        recovered = enu_to_geodetic(enu, origin)
        self.assertAlmostEqual(recovered.lat_rad, target.lat_rad, places=10)
        self.assertAlmostEqual(recovered.lon_rad, target.lon_rad, places=10)
        self.assertAlmostEqual(recovered.alt_m, target.alt_m, places=4)


class TestEnuOriginCaching(unittest.TestCase):
    def test_origin_caches_trig(self):
        origin = _origin(45.0, 90.0, 0.0)
        self.assertAlmostEqual(origin.sin_lat, math.sin(math.radians(45.0)), places=12)
        self.assertAlmostEqual(origin.cos_lat, math.cos(math.radians(45.0)), places=12)
        self.assertAlmostEqual(origin.sin_lon, math.sin(math.radians(90.0)), places=12)
        self.assertAlmostEqual(origin.cos_lon, math.cos(math.radians(90.0)), places=12)

    def test_zero_enu_input_returns_origin_ecef(self):
        origin = _origin(35.6, 139.7, 40.0)
        recovered = enu_to_ecef(EnuPoint(0.0, 0.0, 0.0), origin)
        self.assertAlmostEqual(recovered.x, origin.ecef.x, places=6)
        self.assertAlmostEqual(recovered.y, origin.ecef.y, places=6)
        self.assertAlmostEqual(recovered.z, origin.ecef.z, places=6)


if __name__ == "__main__":
    unittest.main()
