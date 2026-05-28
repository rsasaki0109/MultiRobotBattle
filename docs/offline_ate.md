# Offline ATE / RPE Helper

Status: experimental (added during v0.2.0 work).

`mrn_eval_offline_ate` computes Absolute Trajectory Error (ATE) and Relative
Pose Error (RPE) between an estimated trajectory and a reference trajectory.
It is the post-hoc counterpart to `mrn_online_ate`: when a real bag carries no
`/<agent_id>/ground_truth/pose` topic (e.g. an outdoor recording with RTK truth
on a separate logger), this CLI is what produces the comparison.

The CLI consumes CSV today so it is fully unit-testable in CI. The
companion `mrn_eval_bag_to_csv` CLI (see [Feeding from a Bag](#feeding-from-a-bag))
exports one topic from a recorded rosbag2 directory into the same CSV format.
Pure-function helpers live in `mrn_eval/mrn_eval/offline_ate.py` and
`mrn_eval/mrn_eval/bag_to_csv.py`, exercised directly by
`test_mrn_eval/test_offline_ate.py` and `test_mrn_eval/test_bag_to_csv.py`.

## CSV Format

Both `--estimated` and `--truth` accept the same schema:

```csv
stamp_sec,x,y,z
0.000,0.000,0.000,0.000
0.100,0.100,0.000,0.000
0.200,0.200,0.000,0.000
```

- `stamp_sec` — seconds since some shared epoch. Both files must use the
  **same** clock; `--max-offset-sec` is the alignment tolerance, not an epoch
  correction.
- `x`, `y` — required, in meters.
- `z` — optional. When omitted, every sample is treated as `z=0`.
- Rows do not need to be sorted; the loader sorts by `stamp_sec`.

## Invocation

```bash
ros2 run mrn_eval mrn_eval_offline_ate \
  --estimated  out/robot_1_cooperative.csv \
  --truth      out/robot_1_rtk_truth.csv \
  --estimated-label cooperative \
  --truth-label rtk \
  --max-offset-sec 0.05 \
  --rpe-delta-sec 1.0 \
  --rpe-delta-sec 5.0 \
  --output-dir out/offline_ate/robot_1
```

Outputs:

- `out/offline_ate/robot_1/metrics.json` — machine-readable summary
  (alignment counts, ATE rmse/mean/stddev/max, per-delta RPE).
- `out/offline_ate/robot_1/report.md` — markdown report for PR diff review.

Without `--output-dir` the markdown report is written to stdout, which is
useful for a quick eyeball check during a bag replay debug loop.

## Exit Codes

- `0` — success.
- `2` — input error (missing CSV, malformed columns, non-numeric value).
- `3` — alignment failed (no estimated sample fell within
  `--max-offset-sec` of any truth sample). Loosen the tolerance or confirm the
  two files actually share a clock.

## How It Fits

```text
recorded bag                          RTK truth (separate logger)
  + estimated cooperative pose          + geodetic + GGA quality
       |                                       |
       v                                       v
   mrn_eval_bag_to_csv                   mrn_eval_rtk_to_csv
       |                                       |
       v                                       v
       estimated.csv                       truth.csv + .origin.yaml
            \                              /
             v                            v
              mrn_eval_offline_ate --estimated ... --truth ...
                          |
                          v
                metrics.json + report.md
```

Both exporters emit the same `stamp_sec,x,y,z` schema in meters; the RTK
exporter linearizes around an ENU origin recorded in a sidecar
`<output>.origin.yaml` so the comparison is reproducible. See
[Feeding from RTK](#feeding-from-rtk) below for the input CSV schema and
filter behavior.

## Feeding from a Bag

`mrn_eval_bag_to_csv` exports one topic from a rosbag2 directory. The
output schema depends on the recorded message type (use `--list-types` to
print the full set).

**Pose types → `stamp_sec,x,y,z` directly** (feed straight to
`mrn_eval_offline_ate`):

- `mrn_msgs/msg/CooperativePose` — fused cooperative estimate
- `mrn_msgs/msg/AgentState` — local pose with V2V packet header
- `nav_msgs/msg/Odometry` — local-only baseline
- `geometry_msgs/msg/PoseStamped`
- `geometry_msgs/msg/PoseWithCovarianceStamped`

**Geodetic types → RTK input CSV** (`stamp_sec,lat_deg,lon_deg,alt_m,fix_quality`;
feed to `mrn_eval_rtk_to_csv`, see [Feeding from RTK](#feeding-from-rtk)):

- `sensor_msgs/msg/NavSatFix`

```bash
ros2 run mrn_eval mrn_eval_bag_to_csv \
  path/to/two_robot_demo_2026-06-01 \
  --topic /robot_1/mrn/cooperative_pose \
  --output out/robot_1_cooperative.csv

ros2 run mrn_eval mrn_eval_bag_to_csv \
  path/to/two_robot_demo_2026-06-01 \
  --topic /robot_1/odom \
  --output out/robot_1_local_only.csv

# NavSatFix → geodetic CSV, then linearize to ENU truth:
ros2 run mrn_eval mrn_eval_bag_to_csv \
  path/to/two_robot_demo_2026-06-01 \
  --topic /robot_1/sensor/gnss/fix \
  --output out/robot_1_gnss.csv
ros2 run mrn_eval mrn_eval_rtk_to_csv \
  --input out/robot_1_gnss.csv \
  --output out/robot_1_gnss_truth.csv \
  --origin-from-first \
  --min-fix-quality 2
```

Exit codes match the offline CLI: `0` success, `2` input error (missing
bag, unsupported message type, `rosbag2_py` not installed), `3` no
messages on the requested topic. For pose types, orientation is ignored —
only the position is exported, matching the `stamp_sec,x,y,z` schema.

> **NavSatFix quality ceiling.** `sensor_msgs/NavSatFix` carries a
> `NavSatStatus` whose best value is `STATUS_GBAS_FIX`, which maps to
> `DGPS` (≈1 m). NavSatStatus cannot express an RTK fix, so a
> NavSatFix-derived truth never reaches RTK quality. The default
> `mrn_eval_rtk_to_csv --min-fix-quality` of `RTK_FLOAT` would drop every
> NavSatFix sample — pass `--min-fix-quality 2` (DGPS) or lower, or record
> an NMEA-GGA-derived CSV directly when you have RTK truth.

## Feeding from RTK

`mrn_eval_rtk_to_csv` converts an RTK logger CSV (geodetic samples with a
NMEA GGA quality indicator) into the offline-ATE truth CSV via the local
ENU frame in `mrn_gnss` (see [`docs/gnss.md`](gnss.md) for the frame and
fix-quality conventions).

Input schema (RTK side):

```csv
stamp_sec,lat_deg,lon_deg,alt_m,fix_quality
0.000,35.6000000,139.7000000,40.0,4
0.100,35.6000010,139.7000010,40.0,4
```

`fix_quality` matches the NMEA GGA quality indicator (0=INVALID,
1=SINGLE, 2=DGPS, 4=RTK_FIX, 5=RTK_FLOAT, 9=SBAS, ...). Samples with
`fix_quality` worse than `--min-fix-quality` are dropped before
linearization. The default is 5 (RTK_FLOAT) so SINGLE and SBAS fixes do
not silently degrade the truth trajectory.

```bash
# Use the first kept sample as the ENU origin (typical for one-bag runs):
ros2 run mrn_eval mrn_eval_rtk_to_csv \
  --input  rtk_logger.csv \
  --output out/robot_1_rtk_truth.csv \
  --origin-from-first \
  --min-fix-quality 4

# Pin an explicit origin (use this across multiple bags for a stable frame):
ros2 run mrn_eval mrn_eval_rtk_to_csv \
  --input  rtk_logger.csv \
  --output out/robot_1_rtk_truth.csv \
  --origin-lat-deg 35.6 \
  --origin-lon-deg 139.7 \
  --origin-alt-m   40.0
```

Outputs:

- `out/robot_1_rtk_truth.csv` — offline-ATE truth in ENU meters.
- `out/robot_1_rtk_truth.csv.origin.yaml` — `lat_deg`, `lon_deg`,
  `alt_m`, and `ecef_*` of the origin used for the linearization. Pin
  this in the bag's `manifest.md` so re-running the comparison later
  gives identical numbers.

Exit codes mirror the bag exporter: `0` success, `2` input error
(missing CSV, unsupported quality, missing `--origin-lon-deg`), `3` no
samples passed the quality filter.

## Feeding from a Public Dataset

`mrn_eval_tum_to_csv` converts a TUM-format trajectory into the offline-ATE
CSV. The TUM RGB-D format — also emitted by EuRoC ground-truth exports and
many SLAM benchmarks — is whitespace-separated, one pose per line:

```text
# timestamp tx ty tz qx qy qz qw
1305031910.765 1.342 0.620 1.628 0.659 0.610 -0.292 -0.328
1305031910.797 1.343 0.621 1.628 0.659 0.610 -0.292 -0.328
```

A position-only variant (`timestamp tx ty tz`) is also accepted. Lines
beginning with `#` and blank lines are ignored. Orientation is dropped —
only the position is used, matching the `stamp_sec,x,y,z` schema.

```bash
ros2 run mrn_eval mrn_eval_tum_to_csv \
  --input  dataset/groundtruth.tum \
  --output out/groundtruth.csv

ros2 run mrn_eval mrn_eval_tum_to_csv \
  --input  dataset/orbslam_trajectory.tum \
  --output out/estimate.csv

ros2 run mrn_eval mrn_eval_offline_ate \
  --estimated out/estimate.csv \
  --truth     out/groundtruth.csv \
  --output-dir out/offline_ate/dataset
```

Exit codes: `0` success, `2` input error (missing file, malformed line),
`3` the file had no trajectory rows (only comments/blank lines).

> **Shared clock.** As with the bag and RTK paths, the estimate and truth
> must share a clock. TUM timestamps are seconds; if the dataset's
> estimate and ground truth use different epochs, align them before
> conversion — `--max-offset-sec` is an alignment tolerance, not an epoch
> correction.

## Related Documents

- [`docs/experiments.md`](experiments.md) — `mrn_experiment run` (online ATE
  path; consumes `/mrn/eval/summary` from `mrn_online_ate`).
- [`docs/bag_capture.md`](bag_capture.md) — bag capture procedure that will
  eventually feed this CLI.
- [`docs/gnss.md`](gnss.md) — frames, units, and the fix-quality →
  covariance heuristic used by `mrn_eval_rtk_to_csv`.
- [`PLAN.md`](../PLAN.md) §11 — v0.2.0 acceptance.
