"""Tests for the decentralized formation controller."""

import math
import unittest

from mrn_coord.formation import (
    FormationSpec,
    formation_control_from_relative,
    formation_error,
    line_formation,
    polygon_formation,
    relative_measurement,
    relative_measurements,
    simulate,
)


def _centroid(positions):
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


class TestSpec(unittest.TestCase):
    def test_desired_relative(self):
        spec = FormationSpec({"a": (0.0, 0.0), "b": (2.0, 0.0)})
        self.assertEqual(spec.desired_relative("a", "b"), (2.0, 0.0))
        self.assertEqual(spec.desired_relative("b", "a"), (-2.0, 0.0))

    def test_line_formation_spacing(self):
        spec = line_formation(["a", "b", "c"], spacing=1.5)
        self.assertEqual(spec.offsets["a"], (0.0, 0.0))
        self.assertAlmostEqual(spec.offsets["b"][0], 1.5)
        self.assertAlmostEqual(spec.offsets["c"][0], 3.0)

    def test_line_formation_rejects_zero_axis(self):
        with self.assertRaises(ValueError):
            line_formation(["a", "b"], axis=(0.0, 0.0))

    def test_polygon_is_equilateral_triangle(self):
        spec = polygon_formation(["a", "b", "c"], radius=1.0)
        # all three at unit radius
        for p in spec.offsets.values():
            self.assertAlmostEqual(math.hypot(*p), 1.0, places=9)
        # pairwise distances equal (equilateral)
        pts = list(spec.offsets.values())
        d01 = math.dist(pts[0], pts[1])
        d12 = math.dist(pts[1], pts[2])
        self.assertAlmostEqual(d01, d12, places=9)


class TestMeasurements(unittest.TestCase):
    def test_relative_measurement_is_antisymmetric(self):
        positions = {"a": (1.0, 1.0), "b": (4.0, 5.0)}
        rab = relative_measurement(positions, "a", "b")
        rba = relative_measurement(positions, "b", "a")
        self.assertEqual(rab, (3.0, 4.0))
        self.assertEqual(rba, (-3.0, -4.0))

    def test_relative_measurements_builds_both_directions(self):
        positions = {"a": (0.0, 0.0), "b": (1.0, 0.0)}
        meas = relative_measurements(positions, [("a", "b")])
        self.assertIn(("a", "b"), meas)
        self.assertIn(("b", "a"), meas)


class TestControlLaw(unittest.TestCase):
    def test_zero_command_when_already_in_formation(self):
        spec = polygon_formation(["a", "b", "c"], radius=2.0)
        # positions equal to the spec offsets -> error 0 -> zero command
        positions = dict(spec.offsets)
        edges = [("a", "b"), ("b", "c"), ("a", "c")]
        self.assertAlmostEqual(formation_error(positions, spec, edges), 0.0, places=9)
        meas = relative_measurements(positions, edges)
        commands = formation_control_from_relative(meas, spec, gain=1.0)
        for ux, uy in commands.values():
            self.assertAlmostEqual(ux, 0.0, places=9)
            self.assertAlmostEqual(uy, 0.0, places=9)

    def test_error_invariant_to_global_translation(self):
        spec = polygon_formation(["a", "b", "c"], radius=2.0)
        edges = [("a", "b"), ("b", "c"), ("a", "c")]
        shifted = {a: (p[0] + 10.0, p[1] - 3.0) for a, p in spec.offsets.items()}
        self.assertAlmostEqual(formation_error(shifted, spec, edges), 0.0, places=9)

    def test_fixed_agent_gets_zero_command(self):
        spec = line_formation(["a", "b"], spacing=1.0)
        positions = {"a": (0.0, 0.0), "b": (5.0, 0.0)}  # not in formation
        meas = relative_measurements(positions, [("a", "b")])
        commands = formation_control_from_relative(meas, spec, gain=1.0, fixed=["a"])
        self.assertEqual(commands["a"], (0.0, 0.0))
        self.assertNotEqual(commands["b"], (0.0, 0.0))


class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.agents = ["a", "b", "c"]
        self.spec = polygon_formation(self.agents, radius=2.0)
        self.edges = [("a", "b"), ("b", "c"), ("a", "c")]
        self.start = {"a": (0.0, 0.0), "b": (5.0, 1.0), "c": (1.0, 4.0)}

    def test_converges_to_formation(self):
        _, errors = simulate(
            self.start, self.spec, self.edges, gain=1.0, dt=0.1, steps=300
        )
        self.assertGreater(errors[0], 1.0)        # starts out of formation
        self.assertLess(errors[-1], 1e-3)         # converges
        # monotone-ish: final is much smaller than a quarter of the way in
        self.assertLess(errors[-1], errors[len(errors) // 4])

    def test_centroid_is_invariant_without_leader(self):
        traj, _ = simulate(
            self.start, self.spec, self.edges, gain=1.0, dt=0.1, steps=200
        )
        c0 = _centroid(traj[0])
        cN = _centroid(traj[-1])
        self.assertAlmostEqual(c0[0], cN[0], places=6)
        self.assertAlmostEqual(c0[1], cN[1], places=6)

    def test_leader_stays_put_and_shape_anchors(self):
        traj, errors = simulate(
            self.start, self.spec, self.edges,
            gain=1.0, dt=0.1, steps=300, leader="a",
        )
        # leader 'a' never moves (zero leader velocity)
        self.assertEqual(traj[-1]["a"], self.start["a"])
        self.assertLess(errors[-1], 1e-3)

    def test_leader_translation_is_tracked(self):
        traj, errors = simulate(
            self.start, self.spec, self.edges,
            gain=2.0, dt=0.05, steps=600, leader="a", leader_velocity=(0.5, 0.0),
        )
        # the leader advances steadily in +x
        self.assertGreater(traj[-1]["a"][0], self.start["a"][0] + 10.0)
        # a proportional controller tracking a constant-velocity leader keeps a
        # *bounded steady-state lag* (not zero error): the error settles to a
        # constant rather than growing, and stays bounded.
        tail = errors[-50:]
        self.assertLess(max(tail) - min(tail), 1e-2)   # settled to steady state
        self.assertLess(errors[-1], 1.0)               # bounded lag


if __name__ == "__main__":
    unittest.main()
