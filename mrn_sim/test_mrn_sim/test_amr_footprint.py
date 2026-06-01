"""Tests for the bodied-AMR footprint executor (mrn_sim.amr_footprint).

Covers the convex-polygon clearance primitive and the two headline gaps the
executor exposes: turning cost (a differential-drive body pays for reorientation
the point plan never counts) and footprint overlap (a rectangular body sweeps
into shelves / other robots once the aisle tightens toward its size).
"""

import math
import unittest

from mrn_sim.amr_footprint import (AmrExecResult, Footprint, execute_amr,
                                    poly_clearance)


def _sq(cx, cy, half):
    return [(cx - half, cy - half), (cx + half, cy - half),
            (cx + half, cy + half), (cx - half, cy + half)]


class TestPolyClearance(unittest.TestCase):
    def test_disjoint_distance_is_exact(self):
        a = _sq(0.0, 0.0, 0.5)              # [-0.5,0.5]^2
        b = _sq(2.0, 0.0, 0.5)              # gap along x = 1.0
        self.assertAlmostEqual(poly_clearance(a, b), 1.0, places=6)

    def test_corner_to_corner_distance(self):
        a = _sq(0.0, 0.0, 0.5)
        b = _sq(1.5, 1.5, 0.5)             # nearest corners (0.5,0.5)-(1.0,1.0)
        self.assertAlmostEqual(poly_clearance(a, b),
                               math.hypot(0.5, 0.5), places=6)

    def test_overlap_is_negative(self):
        a = _sq(0.0, 0.0, 0.5)
        b = _sq(0.5, 0.0, 0.5)            # overlap of 0.5 along x
        self.assertLess(poly_clearance(a, b), 0.0)

    def test_touching_is_zero(self):
        a = _sq(0.0, 0.0, 0.5)
        b = _sq(1.0, 0.0, 0.5)            # share an edge
        self.assertAlmostEqual(poly_clearance(a, b), 0.0, places=6)


class TestFootprint(unittest.TestCase):
    def test_corners_axis_aligned(self):
        fp = Footprint(2.0, 1.0)
        corners = fp.corners((0.0, 0.0, 0.0))
        xs = sorted({round(c[0], 6) for c in corners})
        ys = sorted({round(c[1], 6) for c in corners})
        self.assertEqual(xs, [-1.0, 1.0])
        self.assertEqual(ys, [-0.5, 0.5])

    def test_corners_rotated_90deg(self):
        fp = Footprint(2.0, 1.0)
        corners = fp.corners((0.0, 0.0, math.pi / 2))
        # length now spans y, width spans x
        xs = sorted({round(c[0], 6) for c in corners})
        ys = sorted({round(c[1], 6) for c in corners})
        self.assertEqual(xs, [-0.5, 0.5])
        self.assertEqual(ys, [-1.0, 1.0])


class TestExecuteAmr(unittest.TestCase):
    def test_turning_stretches_makespan(self):
        # An L-path forces a 90 deg turn the discrete plan treats as free.
        paths = {"a": [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]}
        res = execute_amr(paths, set(), cell_size=1.0, max_v=1.0, max_omega=2.5)
        ideal = res.discrete_makespan * 1.0 / 1.0       # no-turn drive time
        self.assertGreater(res.turn_time_frac, 0.0)
        self.assertGreater(res.makespan_sec, ideal)     # turning adds wall-clock
        self.assertTrue(res.success)

    # Two robots approach head-on and end in adjacent cells along their travel
    # axis (a legal MAPF configuration — no shared cell, no swap). Their bodies
    # then sit centre-to-centre one cell apart, so the gap is cell_size minus the
    # 0.7 m body length: roomy when the cell is large, overlapping when it shrinks.
    _HEAD_ON = {"a": [(0, 0), (1, 0)], "b": [(3, 0), (2, 0)]}

    def test_wide_aisle_is_collision_free(self):
        res = execute_amr(self._HEAD_ON, set(), cell_size=2.0,
                          footprint=Footprint(0.7, 0.45))
        self.assertEqual(res.footprint_collisions, 0)
        self.assertGreater(res.min_robot_clearance, 0.0)

    def test_tight_aisle_breaks_the_point_guarantee(self):
        # The *same* discrete plan, squeezed toward the body size, makes the
        # rectangles overlap — the cell-level guarantee no longer holds.
        res = execute_amr(self._HEAD_ON, set(), cell_size=0.5,
                          footprint=Footprint(0.7, 0.45))
        self.assertLess(res.min_robot_clearance, 0.0)
        self.assertGreater(res.footprint_collisions, 0)

    def test_clearance_shrinks_monotonically_with_cell(self):
        wide = execute_amr(self._HEAD_ON, set(), cell_size=2.0,
                           footprint=Footprint(0.7, 0.45))
        tight = execute_amr(self._HEAD_ON, set(), cell_size=1.0,
                            footprint=Footprint(0.7, 0.45))
        self.assertGreater(wide.min_robot_clearance, tight.min_robot_clearance)

    def test_deterministic(self):
        paths = {"a": [(0, 0), (1, 0), (1, 1)], "b": [(2, 0), (1, 0), (1, 1)]}
        r1 = execute_amr(paths, {(0, 1)}, cell_size=1.0)
        r2 = execute_amr(paths, {(0, 1)}, cell_size=1.0)
        self.assertEqual(r1.as_dict(), r2.as_dict())

    def test_result_is_serializable(self):
        res = execute_amr({"a": [(0, 0), (1, 0)]}, set())
        self.assertIsInstance(res, AmrExecResult)
        self.assertEqual(set(res.as_dict()),
                         {"success", "discrete_makespan", "continuous_steps",
                          "makespan_sec", "turn_time_frac", "min_shelf_clearance",
                          "min_robot_clearance", "footprint_collisions"})


if __name__ == "__main__":
    unittest.main()
