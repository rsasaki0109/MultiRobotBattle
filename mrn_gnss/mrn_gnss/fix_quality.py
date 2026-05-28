"""NMEA GGA fix-quality enum + baseline position-covariance heuristic.

The values follow the NMEA 0183 GGA "quality indicator" field, which is
what most RTK receivers expose. They are *not* the same as
``sensor_msgs/NavSatStatus.status`` (-1/0/1/2 SBAS/GBAS); the
:meth:`FixQuality.from_navsatstatus` helper converts when an upstream
producer publishes ``NavSatStatus`` instead of raw GGA.

The sigma tables here are intentionally heuristic, not receiver-derived:

- They give a *baseline* covariance when the receiver does not publish a
  per-message ``position_covariance``.
- A real bag pipeline should prefer the receiver's reported covariance
  when ``position_covariance_type`` is ``COVARIANCE_TYPE_KNOWN`` and only
  fall back to these defaults otherwise.
- The values are conservative on purpose so cooperative localization
  weighs RTK_FLOAT and SBAS more cautiously than RTK_FIX.

Map ENU covariance from fix quality with :func:`position_covariance`.
Callers that need their own multiplier strategy can read
``HORIZONTAL_SIGMA_M`` / ``VERTICAL_SIGMA_M`` directly.
"""

from __future__ import annotations

import math
from enum import IntEnum


class FixQuality(IntEnum):
    """NMEA GGA quality indicator values."""

    INVALID = 0
    SINGLE = 1
    DGPS = 2
    PPS = 3
    RTK_FIX = 4
    RTK_FLOAT = 5
    DEAD_RECKONING = 6
    MANUAL = 7
    SIMULATION = 8
    SBAS = 9

    @classmethod
    def from_navsatstatus(cls, status: int) -> "FixQuality":
        """Map ``sensor_msgs/NavSatStatus.status`` to a baseline FixQuality.

        sensor_msgs/NavSatStatus does not distinguish RTK fix from float;
        callers that need that detail should use NMEA GGA directly.

        - STATUS_NO_FIX  (-1) → INVALID
        - STATUS_FIX     ( 0) → SINGLE
        - STATUS_SBAS_FIX ( 1) → SBAS
        - STATUS_GBAS_FIX ( 2) → DGPS  (closest GGA bucket for ground-based aug.)
        """
        if status < 0:
            return cls.INVALID
        if status == 0:
            return cls.SINGLE
        if status == 1:
            return cls.SBAS
        if status == 2:
            return cls.DGPS
        return cls.INVALID


# Heuristic 1-sigma horizontal accuracy in meters. Values are deliberately
# conservative — receivers that publish a tighter `position_covariance`
# should be trusted over this table.
HORIZONTAL_SIGMA_M: dict[FixQuality, float] = {
    FixQuality.INVALID: math.inf,
    FixQuality.SINGLE: 3.0,
    FixQuality.DGPS: 1.0,
    FixQuality.PPS: 3.0,
    FixQuality.RTK_FIX: 0.02,
    FixQuality.RTK_FLOAT: 0.20,
    FixQuality.DEAD_RECKONING: 10.0,
    FixQuality.MANUAL: math.inf,
    FixQuality.SIMULATION: math.inf,
    FixQuality.SBAS: 1.5,
}

# Vertical 1-sigma typically runs ~2x horizontal for GNSS-only solutions.
VERTICAL_SIGMA_M: dict[FixQuality, float] = {
    quality: sigma * 2.0 if math.isfinite(sigma) else sigma
    for quality, sigma in HORIZONTAL_SIGMA_M.items()
}


def position_covariance(
    quality: FixQuality,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Return a 3×3 ENU position covariance matrix for ``quality``.

    Diagonal only — east/north share the horizontal variance, up uses the
    vertical variance. The matrix is returned as nested tuples so it is
    JSON-friendly and avoids a numpy dependency.
    """
    if quality not in HORIZONTAL_SIGMA_M:
        raise ValueError(f"unknown FixQuality: {quality!r}")
    sigma_h = HORIZONTAL_SIGMA_M[quality]
    sigma_v = VERTICAL_SIGMA_M[quality]
    var_h = sigma_h * sigma_h if math.isfinite(sigma_h) else math.inf
    var_v = sigma_v * sigma_v if math.isfinite(sigma_v) else math.inf
    return (
        (var_h, 0.0, 0.0),
        (0.0, var_h, 0.0),
        (0.0, 0.0, var_v),
    )
