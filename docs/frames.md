# Frames

Frame semantics are part of the public API.

## Initial Tree

```text
earth
  map
    robot_1/odom
      robot_1/base_link
    robot_2/odom
      robot_2/base_link
    robot_3/odom
      robot_3/base_link
```

## Rules

- `earth` is reserved for future WGS84/ECEF integration.
- MVP demos use `map` as a local ENU mission frame.
- every robot must use a namespace prefix.
- `map -> robot_i/odom` is the cooperative correction transform.
- `robot_i/odom -> robot_i/base_link` remains owned by the local estimator.
- V2V messages must not rely on raw `/tf` from another robot.
- relative pose direction follows `T_from_to`.

## `earth`, `map`, and Local ENU

This tree follows [REP-105](https://www.ros.org/reps/rep-0105.html): `earth`
is global, `map` is a world-fixed local frame, `odom` is continuous, and
`base_link` is the robot body. The MRN-specific assumptions that make this
concrete:

- **`earth` is ECEF (WGS84).** When GNSS is in play, `earth` is the
  Earth-Centered Earth-Fixed frame defined by the WGS84 ellipsoid — exactly
  the `EcefPoint` frame in [`mrn_gnss`](gnss.md). It has no meaning in a
  GNSS-free MVP run and stays unpublished there.
- **`map` is a local-tangent ENU plane anchored at a pinned geodetic
  origin.** The `earth -> map` transform *is* the ENU origin: a single
  fixed `(lat0, lon0, h0)` realized by `mrn_gnss.EnuOrigin`. East is `map`
  `+x`, north is `+y`, up is `+z`. For a working area within a few km of
  the origin the flat-`map` approximation holds to centimeters; the error
  grows with distance (see [`docs/gnss.md`](gnss.md) → "Origin Choice").
- **The ENU origin is part of the dataset, not the code.** Pin it in the
  bag `manifest.md` and in any RTK truth export
  (`mrn_eval_rtk_to_csv` writes a `*.origin.yaml` sidecar). Re-deriving an
  origin per run silently shifts `map`, which makes trajectories from two
  runs incomparable.
- **All robots in one cooperative session share one `map` / one ENU
  origin.** Cooperative constraints and ATE comparison are only meaningful
  in a common frame; a per-robot origin breaks both. A future
  multi-session or global deployment is where `earth -> map` per region
  starts to matter.
- **Altitude is ellipsoidal, not MSL.** The `up` axis and any `h0` are
  heights above the WGS84 ellipsoid; subtract a geoid model if a receiver
  reports MSL (see [`docs/gnss.md`](gnss.md) → "Units").

## Failure Modes

- missing namespace prefix
- duplicate `base_link` names across robots
- stale TF lookup
- relative pose direction reversed
- covariance not transformed with the pose
- ENU origin re-derived per run (silently shifts `map`)
- per-robot ENU origins in one cooperative session
- ellipsoidal vs MSL altitude mismatch at the origin

## Related Documents

- [`docs/gnss.md`](gnss.md) — WGS84 ↔ ECEF ↔ ENU conversion and the
  `EnuOrigin` that realizes `earth -> map`.
- [`docs/offline_ate.md`](offline_ate.md) — trajectory comparison assumes
  a shared `map` / ENU origin for estimate and truth.
- [`PLAN.md`](../PLAN.md) §17 — frame contract and planned validator work.
