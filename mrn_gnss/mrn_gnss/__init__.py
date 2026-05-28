"""GNSS / WGS84 / local ENU utilities for multirobot-navigation.

Pure-function modules with no ROS or numpy dependency:

- :mod:`mrn_gnss.wgs84` — geodetic ↔ ECEF (WGS84 ellipsoid).
- :mod:`mrn_gnss.enu` — local-tangent ENU frame conversion at a fixed origin.
- :mod:`mrn_gnss.fix_quality` — NMEA GGA fix quality enum and a baseline
  position-covariance heuristic for ENU.

Consumers (RTK→CSV exporter, dataset adapters, Autoware adapter) compose
these modules without taking on numpy or rclpy as a build dependency.
"""

from .wgs84 import EcefPoint, GeodeticPoint, ecef_to_geodetic, geodetic_to_ecef
from .enu import (
    EnuOrigin,
    EnuPoint,
    ecef_to_enu,
    enu_to_ecef,
    enu_to_geodetic,
    geodetic_to_enu,
)
from .fix_quality import (
    FixQuality,
    HORIZONTAL_SIGMA_M,
    VERTICAL_SIGMA_M,
    position_covariance,
)
from .quality_schedule import FixQualitySchedule, QualityInterval

__all__ = [
    "EcefPoint",
    "EnuOrigin",
    "EnuPoint",
    "FixQuality",
    "FixQualitySchedule",
    "GeodeticPoint",
    "HORIZONTAL_SIGMA_M",
    "QualityInterval",
    "VERTICAL_SIGMA_M",
    "ecef_to_enu",
    "ecef_to_geodetic",
    "enu_to_ecef",
    "enu_to_geodetic",
    "geodetic_to_ecef",
    "geodetic_to_enu",
    "position_covariance",
]
