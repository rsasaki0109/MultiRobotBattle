import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from mrn_eval.offline_ate import TrajectorySample, load_trajectory_csv
from mrn_eval.tum_to_csv import load_tum_trajectory, parse_tum_line
from mrn_eval.tum_to_csv_cli import main as tum_main


class TestParseTumLine(unittest.TestCase):
    def test_full_pose_line_drops_orientation(self):
        sample = parse_tum_line("1.5 1.0 2.0 3.0 0.0 0.0 0.0 1.0")
        self.assertEqual(sample, TrajectorySample(1.5, 1.0, 2.0, 3.0))

    def test_position_only_line(self):
        sample = parse_tum_line("2.0 4.0 5.0 6.0")
        self.assertEqual(sample, TrajectorySample(2.0, 4.0, 5.0, 6.0))

    def test_comment_returns_none(self):
        self.assertIsNone(parse_tum_line("# this is a comment"))

    def test_blank_returns_none(self):
        self.assertIsNone(parse_tum_line("   "))

    def test_tab_separated(self):
        sample = parse_tum_line("1.0\t1.0\t2.0\t3.0")
        self.assertEqual(sample, TrajectorySample(1.0, 1.0, 2.0, 3.0))

    def test_wrong_field_count_raises(self):
        with self.assertRaisesRegex(ValueError, "expected 4 or 8 fields"):
            parse_tum_line("1.0 2.0 3.0", source="f", lineno=7)

    def test_non_numeric_raises(self):
        with self.assertRaisesRegex(ValueError, "invalid number"):
            parse_tum_line("1.0 oops 2.0 3.0", source="f", lineno=3)


class TestLoadTumTrajectory(unittest.TestCase):
    def _write(self, directory: str, text: str) -> Path:
        path = Path(directory) / "traj.tum"
        path.write_text(text, encoding="utf-8")
        return path

    def test_loads_and_sorts_skipping_comments(self):
        text = (
            "# TUM trajectory\n"
            "\n"
            "2.0 2.0 0.0 0.0 0 0 0 1\n"
            "0.0 0.0 0.0 0.0 0 0 0 1\n"
            "1.0 1.0 0.0 0.0 0 0 0 1\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, text)
            samples = load_tum_trajectory(path)
        self.assertEqual([s.stamp_sec for s in samples], [0.0, 1.0, 2.0])

    def test_mixed_field_counts(self):
        text = "0.0 0.0 0.0 0.0\n1.0 1.0 2.0 3.0 0 0 0 1\n"
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, text)
            samples = load_tum_trajectory(path)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[1], TrajectorySample(1.0, 1.0, 2.0, 3.0))

    def test_malformed_line_reports_lineno(self):
        text = "0.0 0.0 0.0 0.0\nbad line here only five fields x\n"
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, text)
            with self.assertRaisesRegex(ValueError, ":2:"):
                load_tum_trajectory(path)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_tum_trajectory(Path("/nonexistent/traj.tum"))


class TestCli(unittest.TestCase):
    def test_writes_offline_ate_csv(self):
        text = "0.0 0.0 0.0 0.0 0 0 0 1\n1.0 1.0 2.0 0.0 0 0 0 1\n"
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            input_path = base / "gt.tum"
            input_path.write_text(text, encoding="utf-8")
            output = base / "out" / "gt.csv"
            stream = io.StringIO()
            with redirect_stdout(stream):
                rc = tum_main(["--input", str(input_path), "--output", str(output)])
            self.assertEqual(rc, 0)
            self.assertIn("wrote 2 rows", stream.getvalue())
            samples = load_trajectory_csv(output)
        self.assertEqual(len(samples), 2)
        self.assertAlmostEqual(samples[1].x, 1.0)
        self.assertAlmostEqual(samples[1].y, 2.0)

    def test_missing_input_returns_2(self):
        err = io.StringIO()
        with redirect_stderr(err):
            rc = tum_main(["--input", "/nonexistent/x.tum", "--output", "/tmp/o.csv"])
        self.assertEqual(rc, 2)
        self.assertIn("TUM trajectory not found", err.getvalue())

    def test_only_comments_returns_3(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            input_path = base / "empty.tum"
            input_path.write_text("# header only\n\n", encoding="utf-8")
            output = base / "o.csv"
            err = io.StringIO()
            with redirect_stderr(err):
                rc = tum_main(["--input", str(input_path), "--output", str(output)])
        self.assertEqual(rc, 3)
        self.assertIn("no trajectory rows", err.getvalue())


if __name__ == "__main__":
    unittest.main()
