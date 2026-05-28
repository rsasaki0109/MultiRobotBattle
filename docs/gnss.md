# GNSS / WGS84 / Local ENU

Status: experimental (added during v0.3.0 scaffolding).

`mrn_gnss` is the small pure-function library that converts between WGS84
geodetic, ECEF, and a local ENU frame, plus a baseline NMEA GGA fix-quality
→ ENU position covariance heuristic. There is no ROS, numpy, or pyproj
dependency — the modules can be imported by Python in any environment that
has `math` available.

The library is intentionally narrow: it answers "given an RTK fix and a
fixed origin, what is my ENU position and what covariance should I
believe?" and stops there. Trajectory smoothing, INS coupling, and
GNSS/INS fusion are out of scope.

## Coordinate Frames

| Frame | Axes | Origin | Use |
| --- | --- | --- | --- |
| Geodetic (WGS84) | latitude, longitude, ellipsoidal height | Earth center | what receivers publish |
| ECEF | x toward (lat=0, lon=0), z to north pole | Earth center | intermediate |
| Local ENU | east, north, up | fixed `EnuOrigin` on the ellipsoid | what cooperative localization consumes |

The ENU frame is right-handed. East and north are tangent to the
WGS84 ellipsoid at the origin; up is along the ellipsoid normal at the
origin.

## Origin Choice

An ENU frame is a linearization around a single fixed origin. The
linearization error grows quadratically with distance from the origin —
typically a few centimeters at 10 km and meters past 100 km. Pick the
origin:

- once per dataset, recorded in the bag manifest;
- close to the working area (e.g. the experiment start location);
- with enough precision that re-recording does not change it (8+
  decimal places of lat/lon).

`EnuOrigin.from_geodetic(origin)` caches the trig values used for every
conversion, so constructing it once and re-using it for every fix is
cheap.

## Units

- Latitude / longitude are **radians** in the API (use `math.radians`
  when reading degree-valued YAML or CSV).
- Altitude is **meters above the WGS84 ellipsoid**, not mean sea level.
  Subtract the geoid height (e.g. EGM96) if your receiver publishes MSL
  altitude.
- ENU coordinates and distances are **meters**.

## Fix Quality and Covariance

`FixQuality` follows the NMEA 0183 GGA quality indicator field, which is
what most RTK receivers expose. Use `FixQuality.from_navsatstatus` to map
from `sensor_msgs/NavSatStatus` when the upstream node publishes that
message instead of raw GGA.

| FixQuality | NMEA GGA | Horizontal σ [m] | Vertical σ [m] |
| --- | ---: | ---: | ---: |
| INVALID | 0 | ∞ | ∞ |
| SINGLE | 1 | 3.0 | 6.0 |
| DGPS | 2 | 1.0 | 2.0 |
| PPS | 3 | 3.0 | 6.0 |
| RTK_FIX | 4 | 0.02 | 0.04 |
| RTK_FLOAT | 5 | 0.20 | 0.40 |
| DEAD_RECKONING | 6 | 10.0 | 20.0 |
| MANUAL | 7 | ∞ | ∞ |
| SIMULATION | 8 | ∞ | ∞ |
| SBAS | 9 | 1.5 | 3.0 |

`position_covariance(quality)` returns a 3×3 ENU diagonal covariance
matrix (nested tuples, no numpy) suitable for the
`geometry_msgs/PoseWithCovariance.covariance[0..8]` sub-block.

These sigmas are **heuristic**:

- They give a baseline when the receiver does not publish a per-message
  `position_covariance`.
- When the receiver publishes covariance with
  `position_covariance_type == COVARIANCE_TYPE_KNOWN`, trust that instead.
- They are conservative on purpose so cooperative localization weights
  RTK_FLOAT and SBAS less aggressively than RTK_FIX.

## Example

```python
import math

from mrn_gnss import (
    EnuOrigin,
    FixQuality,
    GeodeticPoint,
    geodetic_to_enu,
    position_covariance,
)

origin = EnuOrigin.from_geodetic(
    GeodeticPoint(math.radians(35.6), math.radians(139.7), 40.0)
)

fix = GeodeticPoint(math.radians(35.6005), math.radians(139.7008), 41.2)
enu = geodetic_to_enu(fix, origin)
cov = position_covariance(FixQuality.RTK_FIX)

print(enu, cov)
```

## Fix Quality Schedule

`FixQualitySchedule` is a step function from time to `FixQuality`, used to
model an outage and a staged reacquisition without real RTK data. Each
interval's quality holds until the next interval begins; before the first
interval the fix is `INVALID`.

```python
from mrn_gnss import FixQuality, FixQualitySchedule

schedule = FixQualitySchedule.from_steps([
    (0.0, FixQuality.RTK_FIX),
    (15.0, FixQuality.INVALID),   # outage
    (30.0, FixQuality.SINGLE),    # coarse reacquisition
    (44.0, FixQuality.RTK_FLOAT),
    (50.0, FixQuality.RTK_FIX),   # full reacquisition
])

schedule.quality_at(20.0)      # FixQuality.INVALID
schedule.covariance_at(5.0)    # position_covariance(RTK_FIX)
```

`FixQualitySchedule.from_config([...])` parses scenario YAML entries of the
form `{start_sec, fix_quality}`, where `fix_quality` is a quality name or
the NMEA GGA integer. The synthetic world node consumes this via the
`faults.gnss_quality_schedule` block — see
[`docs/experiments.md`](experiments.md) → "GNSS Quality Transition".

## Related Documents

- [`docs/offline_ate.md`](offline_ate.md) → "Feeding from RTK" — RTK truth
  feeds the offline ATE/RPE comparison; the `mrn_eval_rtk_to_csv` CLI
  uses this library to linearize geodetic samples around an ENU origin.
- [`docs/experiments.md`](experiments.md) → "GNSS Quality Transition" —
  the synthetic outage/reacquisition scenario driven by
  `FixQualitySchedule`.
- [`docs/interfaces.md`](interfaces.md) — covariance encoding inside
  `AgentState`/`CooperativePose`.
- [`PLAN.md`](../PLAN.md) §12 — v0.3.0 acceptance (GNSS/ENU utilities,
  RTK fix quality, Autoware adapter skeleton).
