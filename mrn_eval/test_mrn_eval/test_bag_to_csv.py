import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from mrn_gnss import FixQuality

from mrn_eval.bag_to_csv import (
    CSV_HEADER,
    EXTRACTORS,
    extract_geodetic_sample,
    extract_sample,
    is_geodetic_message_type,
    supported_geodetic_message_types,
    supported_message_types,
    write_csv,
)
from mrn_eval.bag_to_csv_cli import main as bag_to_csv_main
from mrn_eval.offline_ate import TrajectorySample, load_trajectory_csv
from mrn_eval.rtk_to_csv import RtkSample, load_rtk_csv


def _ns(**fields) -> types.SimpleNamespace:
    return types.SimpleNamespace(**fields)


def _stamp(sec: int, nanosec: int) -> types.SimpleNamespace:
    return _ns(sec=sec, nanosec=nanosec)


def _header(sec: int, nanosec: int) -> types.SimpleNamespace:
    return _ns(stamp=_stamp(sec, nanosec), frame_id="map")


def _point(x: float, y: float, z: float = 0.0) -> types.SimpleNamespace:
    return _ns(x=x, y=y, z=z)


def _agent_state(sec, nanosec, x, y, z=0.0):
    return _ns(
        packet=_ns(header=_header(sec, nanosec)),
        pose=_ns(pose=_ns(position=_point(x, y, z))),
    )


def _cooperative_pose(sec, nanosec, x, y, z=0.0):
    return _ns(
        header=_header(sec, nanosec),
        pose=_ns(pose=_ns(position=_point(x, y, z))),
    )


def _odometry(sec, nanosec, x, y, z=0.0):
    return _ns(
        header=_header(sec, nanosec),
        pose=_ns(pose=_ns(position=_point(x, y, z))),
    )


def _pose_stamped(sec, nanosec, x, y, z=0.0):
    return _ns(
        header=_header(sec, nanosec),
        pose=_ns(position=_point(x, y, z)),
    )


def _pose_with_cov_stamped(sec, nanosec, x, y, z=0.0):
    return _ns(
        header=_header(sec, nanosec),
        pose=_ns(pose=_ns(position=_point(x, y, z))),
    )


def _navsatfix(sec, nanosec, lat, lon, alt, status):
    return _ns(
        header=_header(sec, nanosec),
        status=_ns(status=status, service=1),
        latitude=lat,
        longitude=lon,
        altitude=alt,
    )


class TestExtractSample(unittest.TestCase):
    def test_supported_message_types_registered(self):
        self.assertEqual(set(supported_message_types()), set(EXTRACTORS))
        self.assertIn("mrn_msgs/msg/AgentState", supported_message_types())
        self.assertIn("mrn_msgs/msg/CooperativePose", supported_message_types())
        self.assertIn("nav_msgs/msg/Odometry", supported_message_types())
        self.assertIn("geometry_msgs/msg/PoseStamped", supported_message_types())
        self.assertIn(
            "geometry_msgs/msg/PoseWithCovarianceStamped", supported_message_types()
        )

    def test_agent_state(self):
        sample = extract_sample(
            "mrn_msgs/msg/AgentState", _agent_state(1, 500_000_000, 1.0, 2.0, 3.0)
        )
        self.assertEqual(sample, TrajectorySample(1.5, 1.0, 2.0, 3.0))

    def test_cooperative_pose(self):
        sample = extract_sample(
            "mrn_msgs/msg/CooperativePose",
            _cooperative_pose(2, 0, -1.0, 0.5, 0.0),
        )
        self.assertEqual(sample, TrajectorySample(2.0, -1.0, 0.5, 0.0))

    def test_odometry(self):
        sample = extract_sample(
            "nav_msgs/msg/Odometry", _odometry(3, 250_000_000, 4.0, 5.0, 6.0)
        )
        self.assertAlmostEqual(sample.stamp_sec, 3.25)
        self.assertEqual((sample.x, sample.y, sample.z), (4.0, 5.0, 6.0))

    def test_pose_stamped(self):
        sample = extract_sample(
            "geometry_msgs/msg/PoseStamped", _pose_stamped(0, 0, 7.0, 8.0, 9.0)
        )
        self.assertEqual(sample, TrajectorySample(0.0, 7.0, 8.0, 9.0))

    def test_pose_with_covariance_stamped(self):
        sample = extract_sample(
            "geometry_msgs/msg/PoseWithCovarianceStamped",
            _pose_with_cov_stamped(5, 100_000_000, 1.5, 2.5, 3.5),
        )
        self.assertAlmostEqual(sample.stamp_sec, 5.1)
        self.assertEqual((sample.x, sample.y, sample.z), (1.5, 2.5, 3.5))

    def test_unsupported_type_raises(self):
        with self.assertRaisesRegex(ValueError, "unsupported message type"):
            extract_sample("std_msgs/msg/String", _ns(data="hello"))


class TestExtractGeodeticSample(unittest.TestCase):
    def test_navsatfix_registered(self):
        self.assertIn(
            "sensor_msgs/msg/NavSatFix", supported_geodetic_message_types()
        )
        self.assertTrue(is_geodetic_message_type("sensor_msgs/msg/NavSatFix"))
        # A pose type is not geodetic.
        self.assertFalse(is_geodetic_message_type("nav_msgs/msg/Odometry"))

    def test_navsatfix_status_fix_maps_to_single(self):
        # NavSatStatus.STATUS_FIX == 0 → FixQuality.SINGLE
        sample = extract_geodetic_sample(
            "sensor_msgs/msg/NavSatFix",
            _navsatfix(2, 500_000_000, 35.6, 139.7, 40.0, 0),
        )
        self.assertIsInstance(sample, RtkSample)
        self.assertAlmostEqual(sample.stamp_sec, 2.5)
        self.assertAlmostEqual(sample.lat_deg, 35.6)
        self.assertAlmostEqual(sample.lon_deg, 139.7)
        self.assertAlmostEqual(sample.alt_m, 40.0)
        self.assertEqual(sample.fix_quality, FixQuality.SINGLE)

    def test_navsatfix_no_fix_maps_to_invalid(self):
        sample = extract_geodetic_sample(
            "sensor_msgs/msg/NavSatFix",
            _navsatfix(0, 0, 0.0, 0.0, 0.0, -1),
        )
        self.assertEqual(sample.fix_quality, FixQuality.INVALID)

    def test_navsatfix_sbas_maps_to_sbas(self):
        sample = extract_geodetic_sample(
            "sensor_msgs/msg/NavSatFix",
            _navsatfix(0, 0, 1.0, 2.0, 3.0, 1),
        )
        self.assertEqual(sample.fix_quality, FixQuality.SBAS)

    def test_unsupported_geodetic_type_raises(self):
        with self.assertRaisesRegex(ValueError, "unsupported geodetic message type"):
            extract_geodetic_sample("std_msgs/msg/String", _ns(data="x"))


class TestNavSatFixRoundTrip(unittest.TestCase):
    def test_extract_then_write_rtk_csv_round_trips(self):
        from mrn_eval.rtk_to_csv import write_rtk_csv

        msgs = [
            _navsatfix(0, 0, 35.6, 139.7, 40.0, 0),
            _navsatfix(1, 0, 35.6001, 139.7001, 40.5, 1),
        ]
        samples = [
            extract_geodetic_sample("sensor_msgs/msg/NavSatFix", msg) for msg in msgs
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geodetic.csv"
            written = write_rtk_csv(path, samples)
            self.assertEqual(written, 2)
            reloaded = load_rtk_csv(path)
        self.assertEqual(len(reloaded), 2)
        self.assertAlmostEqual(reloaded[0].lat_deg, 35.6)
        self.assertEqual(reloaded[0].fix_quality, FixQuality.SINGLE)
        self.assertEqual(reloaded[1].fix_quality, FixQuality.SBAS)


class TestWriteCsv(unittest.TestCase):
    def test_writes_header_and_rows_round_trips(self):
        samples = [
            TrajectorySample(0.0, 0.0, 0.0, 0.0),
            TrajectorySample(0.1, 1.0, 2.0, 0.0),
            TrajectorySample(0.2, 2.0, 4.0, 0.0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "out.csv"
            written = write_csv(path, samples)
            self.assertEqual(written, 3)
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.splitlines()[0], ",".join(CSV_HEADER))
            reloaded = load_trajectory_csv(path)
        self.assertEqual(len(reloaded), 3)
        self.assertAlmostEqual(reloaded[1].x, 1.0)
        self.assertAlmostEqual(reloaded[2].y, 4.0)

    def test_empty_input_writes_header_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.csv"
            written = write_csv(path, [])
            self.assertEqual(written, 0)
            text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        self.assertEqual(lines, [",".join(CSV_HEADER)])


class TestCli(unittest.TestCase):
    def test_list_types_prints_registry(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            rc = bag_to_csv_main(
                ["--list-types", "--topic", "x", "--output", "/dev/null", "bag"]
            )
        self.assertEqual(rc, 0)
        for message_type in supported_message_types():
            self.assertIn(message_type, stream.getvalue())
        for message_type in supported_geodetic_message_types():
            self.assertIn(message_type, stream.getvalue())
        self.assertIn("sensor_msgs/msg/NavSatFix", stream.getvalue())

    def test_missing_bag_dir_returns_2(self):
        err = io.StringIO()
        with redirect_stderr(err):
            rc = bag_to_csv_main(
                [
                    "/nonexistent/bag",
                    "--topic",
                    "/x",
                    "--output",
                    "/tmp/out.csv",
                ]
            )
        self.assertEqual(rc, 2)
        self.assertIn("bag directory not found", err.getvalue())

    def test_rosbag2_py_unavailable_returns_2(self):
        # When rosbag2_py is not installed, the CLI must fail with a stable
        # message instead of crashing with an unhelpful ImportError.
        with tempfile.TemporaryDirectory() as directory:
            bag_dir = Path(directory) / "bag"
            bag_dir.mkdir()
            output = Path(directory) / "out.csv"
            saved = sys.modules.pop("rosbag2_py", None)
            sys.modules["rosbag2_py"] = None  # forces ImportError on import
            try:
                err = io.StringIO()
                with redirect_stderr(err):
                    rc = bag_to_csv_main(
                        [
                            str(bag_dir),
                            "--topic",
                            "/robot_1/mrn/cooperative_pose",
                            "--output",
                            str(output),
                        ]
                    )
            finally:
                if saved is None:
                    sys.modules.pop("rosbag2_py", None)
                else:
                    sys.modules["rosbag2_py"] = saved
        self.assertEqual(rc, 2)
        self.assertIn("rosbag2_py not available", err.getvalue())


if __name__ == "__main__":
    unittest.main()
