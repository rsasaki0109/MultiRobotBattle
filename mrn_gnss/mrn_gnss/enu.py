"""Local-tangent ENU frame conversion at a fixed origin.

The ENU (East-North-Up) frame is a right-handed Cartesian frame whose
origin is a fixed geodetic point ``(lat0, lon0, h0)`` and whose axes are:

- east  (x): tangent to the ellipsoid, pointing east at the origin.
- north (y): tangent to the ellipsoid, pointing north at the origin.
- up    (z): along the ellipsoid normal at the origin.

For small operating areas (<< 100 km from the origin) this is a good
approximation of a flat metric map suitable for SLAM / cooperative
localization. The error from treating the Earth as flat grows with the
distance from the origin; this module makes no attempt to hide that —
callers pick the origin and accept the linearization.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

from .wgs84 import (
    EcefPoint,
    GeodeticPoint,
    ecef_to_geodetic,
    geodetic_to_ecef,
)


@dataclass(frozen=True)
class EnuOrigin:
    """Geodetic origin of a local ENU frame, with cached ECEF and rotation.

    Construct via :meth:`from_geodetic` so the trig values used for every
    conversion are computed once.
    """

    geodetic: GeodeticPoint
    ecef: EcefPoint
    sin_lat: float
    cos_lat: float
    sin_lon: float
    cos_lon: float

    @classmethod
    def from_geodetic(cls, origin: GeodeticPoint) -> "EnuOrigin":
        return cls(
            geodetic=origin,
            ecef=geodetic_to_ecef(origin),
            sin_lat=sin(origin.lat_rad),
            cos_lat=cos(origin.lat_rad),
            sin_lon=sin(origin.lon_rad),
            cos_lon=cos(origin.lon_rad),
        )


@dataclass(frozen=True)
class EnuPoint:
    """Position in the local ENU frame in meters."""

    east: float
    north: float
    up: float


def ecef_to_enu(point: EcefPoint, origin: EnuOrigin) -> EnuPoint:
    """Convert an ECEF position to the local ENU frame at ``origin``."""
    dx = point.x - origin.ecef.x
    dy = point.y - origin.ecef.y
    dz = point.z - origin.ecef.z
    east = -origin.sin_lon * dx + origin.cos_lon * dy
    north = (
        -origin.sin_lat * origin.cos_lon * dx
        - origin.sin_lat * origin.sin_lon * dy
        + origin.cos_lat * dz
    )
    up = (
        origin.cos_lat * origin.cos_lon * dx
        + origin.cos_lat * origin.sin_lon * dy
        + origin.sin_lat * dz
    )
    return EnuPoint(east=east, north=north, up=up)


def enu_to_ecef(point: EnuPoint, origin: EnuOrigin) -> EcefPoint:
    """Convert a local ENU position back to ECEF."""
    e, n, u = point.east, point.north, point.up
    dx = (
        -origin.sin_lon * e
        - origin.sin_lat * origin.cos_lon * n
        + origin.cos_lat * origin.cos_lon * u
    )
    dy = (
        origin.cos_lon * e
        - origin.sin_lat * origin.sin_lon * n
        + origin.cos_lat * origin.sin_lon * u
    )
    dz = origin.cos_lat * n + origin.sin_lat * u
    return EcefPoint(
        x=origin.ecef.x + dx,
        y=origin.ecef.y + dy,
        z=origin.ecef.z + dz,
    )


def geodetic_to_enu(point: GeodeticPoint, origin: EnuOrigin) -> EnuPoint:
    """Convert a geodetic position to the local ENU frame at ``origin``."""
    return ecef_to_enu(geodetic_to_ecef(point), origin)


def enu_to_geodetic(point: EnuPoint, origin: EnuOrigin) -> GeodeticPoint:
    """Convert a local ENU position back to geodetic."""
    return ecef_to_geodetic(enu_to_ecef(point, origin))
