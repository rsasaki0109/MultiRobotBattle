import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from mrn_eval.offline_ate import (
    TrajectorySample,
    compute_ate,
    compute_drift_rate,
    compute_rpe,
    load_trajectory_csv,
    time_align,
)
from mrn_eval.offline_ate_cli import main as offline_main


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


class TestLoadTrajectoryCsv(unittest.TestCase):
    def test_loads_2d_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traj.csv"
            _write_csv(path, "stamp_sec,x,y", ["0.0,0.0,0.0", "1.0,1.0,2.0"])
            samples = load_trajectory_csv(path)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0], TrajectorySample(0.0, 0.0, 0.0, 0.0))
        self.assertEqual(samples[1], TrajectorySample(1.0, 1.0, 2.0, 0.0))

    def test_loads_3d_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traj.csv"
            _write_csv(path, "stamp_sec,x,y,z", ["0.0,1.0,2.0,3.0"])
            samples = load_trajectory_csv(path)
        self.assertEqual(samples[0].z, 3.0)

    def test_sorts_by_stamp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traj.csv"
            _write_csv(path, "stamp_sec,x,y", ["2.0,2.0,0.0", "0.0,0.0,0.0"])
            samples = load_trajectory_csv(path)
        self.assertEqual([sample.stamp_sec for sample in samples], [0.0, 2.0])

    def test_rejects_missing_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traj.csv"
            _write_csv(path, "stamp_sec,x", ["0.0,0.0"])
            with self.assertRaisesRegex(ValueError, "missing required CSV columns"):
                load_trajectory_csv(path)

    def test_rejects_non_numeric(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traj.csv"
            _write_csv(path, "stamp_sec,x,y", ["0.0,oops,0.0"])
            with self.assertRaisesRegex(ValueError, "invalid number"):
                load_trajectory_csv(path)

    def test_raises_when_missing(self):
        with self.assertRaises(FileNotFoundError):
            load_trajectory_csv(Path("/nonexistent/path.csv"))


class TestTimeAlign(unittest.TestCase):
    def test_matches_nearest_truth_within_tolerance(self):
        estimated = [TrajectorySample(0.0, 0.0, 0.0), TrajectorySample(1.0, 1.0, 1.0)]
        truth = [TrajectorySample(0.01, 0.0, 0.0), TrajectorySample(0.99, 1.0, 1.0)]
        result = time_align(estimated, truth, max_offset_sec=0.05)
        self.assertEqual(result.matched_count, 2)
        self.assertEqual(result.dropped_count, 0)
        self.assertAlmostEqual(result.pairs[0].time_offset_sec, 0.01)
        self.assertAlmostEqual(result.pairs[1].time_offset_sec, -0.01)

    def test_drops_samples_outside_tolerance(self):
        estimated = [TrajectorySample(0.0, 0.0, 0.0), TrajectorySample(2.0, 2.0, 0.0)]
        truth = [TrajectorySample(0.01, 0.0, 0.0)]
        result = time_align(estimated, truth, max_offset_sec=0.05)
        self.assertEqual(result.matched_count, 1)
        self.assertEqual(result.dropped_count, 1)
        self.assertAlmostEqual(result.max_time_offset_sec, 0.01)

    def test_empty_truth_drops_all(self):
        estimated = [TrajectorySample(0.0, 0.0, 0.0)]
        result = time_align(estimated, [], max_offset_sec=0.05)
        self.assertEqual(result.matched_count, 0)
        self.assertEqual(result.dropped_count, 1)

    def test_negative_tolerance_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            time_align([], [], max_offset_sec=-1.0)


class TestComputeATE(unittest.TestCase):
    def test_perfect_trajectory_gives_zero(self):
        truth = [TrajectorySample(float(t), float(t), 0.0) for t in range(5)]
        estimated = list(truth)
        result = time_align(estimated, truth, max_offset_sec=0.001)
        stats = compute_ate(result.pairs)
        self.assertEqual(stats.count, 5)
        self.assertAlmostEqual(stats.rmse, 0.0)
        self.assertAlmostEqual(stats.max, 0.0)

    def test_constant_offset_matches_exact_magnitude(self):
        truth = [TrajectorySample(float(t), 0.0, 0.0) for t in range(4)]
        estimated = [TrajectorySample(float(t), 3.0, 4.0) for t in range(4)]
        result = time_align(estimated, truth, max_offset_sec=0.001)
        stats = compute_ate(result.pairs)
        self.assertAlmostEqual(stats.rmse, 5.0)
        self.assertAlmostEqual(stats.mean, 5.0)
        self.assertAlmostEqual(stats.max, 5.0)
        self.assertEqual(stats.count, 4)

    def test_empty_pairs_gives_nan_stats(self):
        stats = compute_ate(())
        self.assertEqual(stats.count, 0)
        self.assertTrue(math.isnan(stats.rmse))


class TestComputeRPE(unittest.TestCase):
    def test_perfect_constant_velocity_gives_zero(self):
        # 0.1s spacing, identical motion → RPE 0
        truth = [TrajectorySample(0.1 * i, 0.1 * i, 0.0) for i in range(20)]
        estimated = list(truth)
        align = time_align(estimated, truth, max_offset_sec=0.001)
        stats = compute_rpe(align.pairs, delta_sec=1.0, delta_tolerance_sec=0.05)
        self.assertGreater(stats.count, 0)
        self.assertAlmostEqual(stats.rmse, 0.0, places=6)

    def test_extra_drift_in_estimate_shows_up_as_rpe(self):
        # Truth moves at 1m/s. Estimated drifts +0.5m extra over each 1s window.
        truth = [TrajectorySample(0.1 * i, 0.1 * i, 0.0) for i in range(20)]
        estimated = [
            TrajectorySample(0.1 * i, 0.1 * i + 0.5 * (0.1 * i), 0.0)
            for i in range(20)
        ]
        align = time_align(estimated, truth, max_offset_sec=0.001)
        stats = compute_rpe(align.pairs, delta_sec=1.0, delta_tolerance_sec=0.05)
        self.assertGreater(stats.count, 0)
        self.assertAlmostEqual(stats.rmse, 0.5, places=2)

    def test_rejects_zero_delta(self):
        with self.assertRaisesRegex(ValueError, "delta_sec must be positive"):
            compute_rpe((), delta_sec=0.0)


class TestComputeDriftRate(unittest.TestCase):
    def test_perfect_trajectory_gives_zero_drift(self):
        # straight line, 1 m spacing along x; estimate == truth -> 0 drift.
        truth = [TrajectorySample(float(i), float(i), 0.0) for i in range(11)]
        estimated = list(truth)
        align = time_align(estimated, truth, max_offset_sec=0.001)
        stats = compute_drift_rate(align.pairs, segment_m=2.0, segment_tolerance_m=0.5)
        self.assertGreater(stats.count, 0)
        self.assertAlmostEqual(stats.rmse, 0.0, places=9)

    def test_constant_drift_fraction(self):
        # Truth moves 1 m/step along x (path length = distance). Estimate adds
        # 10% extra in y per step -> 0.1 m drift per 1 m travelled = 0.1.
        truth = [TrajectorySample(float(i), float(i), 0.0) for i in range(11)]
        estimated = [TrajectorySample(float(i), float(i), 0.1 * i) for i in range(11)]
        align = time_align(estimated, truth, max_offset_sec=0.001)
        stats = compute_drift_rate(align.pairs, segment_m=2.0, segment_tolerance_m=0.5)
        self.assertGreater(stats.count, 0)
        self.assertAlmostEqual(stats.mean, 0.1, places=6)

    def test_rejects_zero_segment(self):
        with self.assertRaisesRegex(ValueError, "segment_m must be positive"):
            compute_drift_rate((), segment_m=0.0)

    def test_empty_pairs_gives_nan(self):
        stats = compute_drift_rate((), segment_m=1.0)
        self.assertEqual(stats.count, 0)
        self.assertTrue(math.isnan(stats.rmse))


class TestCli(unittest.TestCase):
    def _write_traj(self, path: Path, samples: list[tuple[float, float, float]]) -> None:
        path.write_text(
            "stamp_sec,x,y\n"
            + "\n".join(f"{t},{x},{y}" for t, x, y in samples)
            + "\n",
            encoding="utf-8",
        )

    def test_writes_metrics_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            est_path = base / "est.csv"
            truth_path = base / "truth.csv"
            self._write_traj(est_path, [(0.0, 0.0, 0.0), (1.0, 1.5, 0.0), (2.0, 3.0, 0.0)])
            self._write_traj(truth_path, [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 2.0, 0.0)])
            output_dir = base / "out"
            rc = offline_main(
                [
                    "--estimated",
                    str(est_path),
                    "--truth",
                    str(truth_path),
                    "--output-dir",
                    str(output_dir),
                    "--rpe-delta-sec",
                    "1.0",
                ]
            )
            self.assertEqual(rc, 0)
            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            report = (output_dir / "report.md").read_text(encoding="utf-8")
        self.assertEqual(metrics["ate"]["count"], 3)
        self.assertAlmostEqual(metrics["ate"]["max"], 1.0, places=6)
        self.assertIn("1s", metrics["rpe"])
        self.assertGreater(metrics["rpe"]["1s"]["count"], 0)
        self.assertIn("Offline ATE Report", report)
        self.assertIn("Absolute Trajectory Error", report)
        self.assertIn("Relative Pose Error", report)
        # drift section is opt-in; absent unless --drift-segment-m is given.
        self.assertNotIn("drift", metrics)
        self.assertNotIn("Drift Rate", report)

    def test_drift_segment_adds_drift_section(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            est_path = base / "est.csv"
            truth_path = base / "truth.csv"
            # straight x-line truth; estimate drifts in y.
            self._write_traj(
                est_path,
                [(float(i), float(i), 0.1 * i) for i in range(11)],
            )
            self._write_traj(
                truth_path,
                [(float(i), float(i), 0.0) for i in range(11)],
            )
            output_dir = base / "out"
            rc = offline_main(
                [
                    "--estimated",
                    str(est_path),
                    "--truth",
                    str(truth_path),
                    "--output-dir",
                    str(output_dir),
                    "--drift-segment-m",
                    "2.0",
                ]
            )
            self.assertEqual(rc, 0)
            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            report = (output_dir / "report.md").read_text(encoding="utf-8")
        self.assertIn("2m", metrics["drift"])
        self.assertGreater(metrics["drift"]["2m"]["count"], 0)
        self.assertAlmostEqual(metrics["drift"]["2m"]["mean"], 0.1, places=6)
        self.assertIn("Drift Rate", report)

    def test_stdout_only_when_no_output_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            est_path = base / "est.csv"
            truth_path = base / "truth.csv"
            self._write_traj(est_path, [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)])
            self._write_traj(truth_path, [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)])
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                rc = offline_main(
                    ["--estimated", str(est_path), "--truth", str(truth_path)]
                )
        self.assertEqual(rc, 0)
        self.assertIn("Offline ATE Report", stream.getvalue())

    def test_missing_estimated_returns_2(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = offline_main(
                ["--estimated", "/nonexistent/x.csv", "--truth", "/nonexistent/y.csv"]
            )
        self.assertEqual(rc, 2)
        self.assertIn("trajectory CSV not found", err.getvalue())

    def test_no_matched_samples_returns_3(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            est_path = base / "est.csv"
            truth_path = base / "truth.csv"
            # 10-second gap, far outside default 0.05s tolerance.
            self._write_traj(est_path, [(0.0, 0.0, 0.0)])
            self._write_traj(truth_path, [(10.0, 0.0, 0.0)])
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = offline_main(
                    ["--estimated", str(est_path), "--truth", str(truth_path)]
                )
        self.assertEqual(rc, 3)
        self.assertIn("no samples matched", err.getvalue())


if __name__ == "__main__":
    unittest.main()
