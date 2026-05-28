"""WGS84 geodetic ↔ ECEF conversion.

Pure-function module — no ROS, numpy, or external library dependency.
Constants follow `EPSG:4326 / WGS84` (NGA TR 8350.2).

Conventions:

- Latitude / longitude in radians (use ``math.radians`` for degrees).
- Altitude in meters (height above the WGS84 ellipsoid, *not* MSL).
- ECEF axes: x toward (lat=0, lon=0); z toward the north pole.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin, sqrt

# WGS84 ellipsoid constants (NGA TR 8350.2).
WGS84_A = 6378137.0  # semi-major axis [m]
WGS84_F = 1.0 / 298.257223563  # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)  # semi-minor axis [m]
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)  # first eccentricity squared
WGS84_EP2 = WGS84_E2 / (1.0 - WGS84_E2)  # second eccentricity squared


@dataclass(frozen=True)
class GeodeticPoint:
    """Geodetic position on the WGS84 ellipsoid.

    Attributes:
        lat_rad: latitude in radians, range [-pi/2, +pi/2].
        lon_rad: longitude in radians, range [-pi, +pi].
        alt_m: ellipsoidal height in meters (above WGS84, not MSL).
    """

    lat_rad: float
    lon_rad: float
    alt_m: float = 0.0


@dataclass(frozen=True)
class EcefPoint:
    """Earth-Centered Earth-Fixed Cartesian position in meters."""

    x: float
    y: float
    z: float


def geodetic_to_ecef(point: GeodeticPoint) -> EcefPoint:
    """Convert WGS84 geodetic (lat, lon, h) to ECEF (x, y, z) in meters."""
    sin_lat = sin(point.lat_rad)
    cos_lat = cos(point.lat_rad)
    sin_lon = sin(point.lon_rad)
    cos_lon = cos(point.lon_rad)
    n = WGS84_A / sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + point.alt_m) * cos_lat * cos_lon
    y = (n + point.alt_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + point.alt_m) * sin_lat
    return EcefPoint(x=x, y=y, z=z)


def ecef_to_geodetic(point: EcefPoint) -> GeodeticPoint:
    """Convert ECEF (x, y, z) to WGS84 geodetic via Bowring's closed form.

    Stable for typical Earth-surface inputs (|h| << a). Returns lat in
    [-pi/2, +pi/2] and lon in (-pi, +pi].
    """
    x, y, z = point.x, point.y, point.z
    lon = atan2(y, x)
    p = sqrt(x * x + y * y)
    if p < 1.0e-12:
        # On the polar axis. lat = ±pi/2, lon arbitrary (return 0).
        lat = (1.0 if z >= 0.0 else -1.0) * (3.141592653589793 / 2.0)
        alt = abs(z) - WGS84_B
        return GeodeticPoint(lat_rad=lat, lon_rad=0.0, alt_m=alt)
    theta = atan2(z * WGS84_A, p * WGS84_B)
    sin_theta = sin(theta)
    cos_theta = cos(theta)
    lat = atan2(
        z + WGS84_EP2 * WGS84_B * sin_theta * sin_theta * sin_theta,
        p - WGS84_E2 * WGS84_A * cos_theta * cos_theta * cos_theta,
    )
    sin_lat = sin(lat)
    n = WGS84_A / sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / cos(lat) - n
    return GeodeticPoint(lat_rad=lat, lon_rad=lon, alt_m=alt)
