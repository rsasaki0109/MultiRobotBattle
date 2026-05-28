import io
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from mrn_gnss import EnuOrigin, FixQuality, GeodeticPoint

from mrn_eval.offline_ate import load_trajectory_csv
from mrn_eval.rtk_to_csv import (
    REQUIRED_INPUT_COLUMNS,
    RtkSample,
    filter_by_quality,
    load_rtk_csv,
    pick_origin_from_first,
    summarize_by_quality,
    to_enu_trajectory,
    write_origin_yaml,
)
from mrn_eval.rtk_to_csv_cli import main as rtk_main


def _write_input(path: Path, rows: list[tuple]) -> None:
    header = ",".join(REQUIRED_INPUT_COLUMNS)
    body = "\n".join(",".join(str(value) for value in row) for row in rows)
    path.write_text(header + "\n" + body + "\n", encoding="utf-8")


class TestLoadRtkCsv(unittest.TestCase):
    def test_loads_and_sorts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rtk.csv"
            _write_input(
                path,
                [
                    (1.0, 35.6, 139.7, 40.0, 4),
                    (0.0, 35.6, 139.7, 40.0, 4),
                ],
            )
            samples = load_rtk_csv(path)
        self.assertEqual([sample.stamp_sec for sample in samples], [0.0, 1.0])
        self.assertEqual(samples[0].fix_quality, FixQuality.RTK_FIX)

    def test_missing_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rtk.csv"
            path.write_text("stamp_sec,lat_deg,lon_deg,alt_m\n0,0,0,0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required CSV columns"):
                load_rtk_csv(path)

    def test_bad_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rtk.csv"
            _write_input(path, [(0.0, "oops", 139.7, 40.0, 4)])
            with self.assertRaisesRegex(ValueError, "invalid number"):
                load_rtk_csv(path)

    def test_unknown_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rtk.csv"
            _write_input(path, [(0.0, 35.6, 139.7, 40.0, 42)])
            with self.assertRaisesRegex(ValueError, "unknown fix_quality"):
                load_rtk_csv(path)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_rtk_csv(Path("/nonexistent/x.csv"))


class TestFilterByQuality(unittest.TestCase):
    def _sample(self, q: FixQuality) -> RtkSample:
        return RtkSample(0.0, 35.6, 139.7, 40.0, q)

    def test_keeps_rtk_fix_drops_single_when_threshold_is_rtk_float(self):
        samples = [
            self._sample(FixQuality.SINGLE),
            self._sample(FixQuality.RTK_FLOAT),
            self._sample(FixQuality.RTK_FIX),
            self._sample(FixQuality.INVALID),
        ]
        kept = filter_by_quality(samples, FixQuality.RTK_FLOAT)
        kept_qualities = {sample.fix_quality for sample in kept}
        self.assertEqual(kept_qualities, {FixQuality.RTK_FLOAT, FixQuality.RTK_FIX})

    def test_invalid_threshold_rejected(self):
        with self.assertRaisesRegex(ValueError, "infinite sigma"):
            filter_by_quality([], FixQuality.INVALID)

    def test_invalid_samples_always_dropped(self):
        samples = [self._sample(FixQuality.INVALID), self._sample(FixQuality.RTK_FIX)]
        kept = filter_by_quality(samples, FixQuality.SBAS)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].fix_quality, FixQuality.RTK_FIX)


class TestPickOriginFromFirst(unittest.TestCase):
    def test_skips_invalid(self):
        samples = [
            RtkSample(0.0, 0.0, 0.0, 0.0, FixQuality.INVALID),
            RtkSample(1.0, 35.6, 139.7, 40.0, FixQuality.RTK_FIX),
        ]
        origin = pick_origin_from_first(samples)
        self.assertAlmostEqual(math.degrees(origin.lat_rad), 35.6)
        self.assertAlmostEqual(origin.alt_m, 40.0)

    def test_all_invalid_raises(self):
        samples = [RtkSample(0.0, 0.0, 0.0, 0.0, FixQuality.INVALID)]
        with self.assertRaisesRegex(ValueError, "no usable sample"):
            pick_origin_from_first(samples)


class TestToEnuTrajectory(unittest.TestCase):
    def test_first_sample_is_origin_when_origin_from_first(self):
        samples = [
            RtkSample(0.0, 35.6, 139.7, 40.0, FixQuality.RTK_FIX),
            RtkSample(1.0, 35.60001, 139.70001, 40.0, FixQuality.RTK_FIX),
        ]
        origin = EnuOrigin.from_geodetic(pick_origin_from_first(samples))
        traj = to_enu_trajectory(samples, origin)
        self.assertEqual(len(traj), 2)
        self.assertAlmostEqual(traj[0].x, 0.0, places=6)
        self.assertAlmostEqual(traj[0].y, 0.0, places=6)
        self.assertAlmostEqual(traj[0].z, 0.0, places=6)
        # A ~1e-5° step is roughly ~1 m at these latitudes.
        self.assertGreater(math.hypot(traj[1].x, traj[1].y), 0.5)
        self.assertLess(math.hypot(traj[1].x, traj[1].y), 5.0)

    def test_north_one_degree_is_about_111km(self):
        origin = EnuOrigin.from_geodetic(GeodeticPoint(math.radians(35.0), math.radians(139.0), 0.0))
        samples = [RtkSample(0.0, 36.0, 139.0, 0.0, FixQuality.RTK_FIX)]
        traj = to_enu_trajectory(samples, origin)
        self.assertGreater(traj[0].y, 110_000.0)
        self.assertLess(traj[0].y, 112_000.0)
        self.assertLess(abs(traj[0].x), 1.0)


class TestSummarizeByQuality(unittest.TestCase):
    def test_counts(self):
        samples = [
            RtkSample(0.0, 0.0, 0.0, 0.0, FixQuality.RTK_FIX),
            RtkSample(1.0, 0.0, 0.0, 0.0, FixQuality.RTK_FIX),
            RtkSample(2.0, 0.0, 0.0, 0.0, FixQuality.RTK_FLOAT),
        ]
        counts = summarize_by_quality(samples)
        self.assertEqual(counts, {"RTK_FIX": 2, "RTK_FLOAT": 1})


class TestWriteOriginYaml(unittest.TestCase):
    def test_writes_fields(self):
        origin = EnuOrigin.from_geodetic(GeodeticPoint(math.radians(35.6), math.radians(139.7), 40.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "out.origin.yaml"
            write_origin_yaml(path, origin)
            text = path.read_text(encoding="utf-8")
        self.assertIn("lat_deg: 35.600000000", text)
        self.assertIn("lon_deg: 139.700000000", text)
        self.assertIn("alt_m: 40.000000", text)
        self.assertIn("ecef_x:", text)


class TestCli(unittest.TestCase):
    def _write_input_file(self, base: Path, rows) -> Path:
        path = base / "rtk.csv"
        _write_input(path, rows)
        return path

    def test_origin_from_first_writes_csv_and_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            input_path = self._write_input_file(
                base,
                [
                    (0.0, 35.6, 139.7, 40.0, 4),
                    (1.0, 35.60001, 139.70001, 40.0, 4),
                ],
            )
            output = base / "out" / "truth.csv"
            stream = io.StringIO()
            with redirect_stdout(stream):
                rc = rtk_main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output),
                        "--origin-from-first",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertIn("RTK_FIX=2", stream.getvalue())
            samples = load_trajectory_csv(output)
            sidecar = output.with_suffix(output.suffix + ".origin.yaml").read_text(
                encoding="utf-8"
            )
        self.assertEqual(len(samples), 2)
        self.assertAlmostEqual(samples[0].x, 0.0, places=6)
        self.assertAlmostEqual(samples[0].y, 0.0, places=6)
        self.assertIn("lat_deg: 35.600000000", sidecar)

    def test_filter_drops_below_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            input_path = self._write_input_file(
                base,
                [
                    (0.0, 35.6, 139.7, 40.0, 1),  # SINGLE → dropped
                    (1.0, 35.6, 139.7, 40.0, 4),  # RTK_FIX → kept
                ],
            )
            output = base / "truth.csv"
            stream = io.StringIO()
            with redirect_stdout(stream):
                rc = rtk_main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output),
                        "--origin-from-first",
                        "--min-fix-quality",
                        str(int(FixQuality.RTK_FLOAT)),
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertIn("kept 1/2", stream.getvalue())

    def test_no_samples_pass_filter_returns_3(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            input_path = self._write_input_file(
                base, [(0.0, 35.6, 139.7, 40.0, 0)]  # INVALID
            )
            output = base / "truth.csv"
            err = io.StringIO()
            with redirect_stderr(err):
                rc = rtk_main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output),
                        "--origin-from-first",
                    ]
                )
        self.assertEqual(rc, 3)
        self.assertIn("no samples passed", err.getvalue())

    def test_missing_input_returns_2(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "truth.csv"
            err = io.StringIO()
            with redirect_stderr(err):
                rc = rtk_main(
                    [
                        "--input",
                        "/nonexistent/in.csv",
                        "--output",
                        str(output),
                        "--origin-from-first",
                    ]
                )
        self.assertEqual(rc, 2)
        self.assertIn("RTK CSV not found", err.getvalue())

    def test_explicit_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            input_path = self._write_input_file(
                base,
                [
                    (0.0, 35.0, 139.0, 0.0, 4),
                    (1.0, 36.0, 139.0, 0.0, 4),
                ],
            )
            output = base / "truth.csv"
            rc = rtk_main(
                [
                    "--input",
                    str(input_path),
                    "--output",
                    str(output),
                    "--origin-lat-deg",
                    "35.0",
                    "--origin-lon-deg",
                    "139.0",
                    "--origin-alt-m",
                    "0.0",
                ]
            )
            self.assertEqual(rc, 0)
            samples = load_trajectory_csv(output)
        self.assertAlmostEqual(samples[0].x, 0.0, places=6)
        self.assertAlmostEqual(samples[0].y, 0.0, places=6)
        # ~1° north → ~111 km
        self.assertGreater(samples[1].y, 110_000.0)


if __name__ == "__main__":
    unittest.main()
