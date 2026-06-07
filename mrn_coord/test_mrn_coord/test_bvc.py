"""Tests for Buffered Voronoi Cells (BVC) — Zhou, Wang, Bandyopadhyay & Schwager (2017)."""

import math
import random
import unittest

from mrn_coord.mapf.bvc import (
    buffered_voronoi_cell,
    min_separation,
    project_to_cell,
    simulate,
    step_bvc,
)


R = 0.2


class TestCellGeometry(unittest.TestCase):
    def test_own_position_in_cell_when_apart(self):
        # two robots 4 apart (>= 2r) -> each robot's own position satisfies its BVC
        pos = [(0.0, 0.0), (4.0, 0.0)]
        planes = buffered_voronoi_cell(pos, 0, R)
        for (a, b) in planes:
            self.assertLessEqual(a[0] * pos[0][0] + a[1] * pos[0][1], b + 1e-9)

    def test_buffer_retracts_boundary(self):
        # the buffered boundary sits short of the midpoint by exactly r
        pos = [(0.0, 0.0), (2.0, 0.0)]
        (a, b), = buffered_voronoi_cell(pos, 0, R)
        # boundary x: a.x = b -> x = b / a[0]
        x_boundary = b / a[0]
        self.assertAlmostEqual(x_boundary, 1.0 - R)   # midpoint 1.0 minus r


class TestProjection(unittest.TestCase):
    def test_target_inside_returns_target(self):
        planes = [((1.0, 0.0), 5.0)]      # x <= 5
        self.assertEqual(project_to_cell((2.0, 3.0), planes), (2.0, 3.0))

    def test_project_single_halfplane(self):
        planes = [((1.0, 0.0), 1.0)]      # x <= 1
        p = project_to_cell((3.0, 2.0), planes)
        self.assertAlmostEqual(p[0], 1.0)
        self.assertAlmostEqual(p[1], 2.0)

    def test_project_corner_two_planes(self):
        planes = [((1.0, 0.0), 1.0), ((0.0, 1.0), 1.0)]   # x<=1, y<=1
        p = project_to_cell((3.0, 3.0), planes)
        self.assertAlmostEqual(p[0], 1.0, places=4)
        self.assertAlmostEqual(p[1], 1.0, places=4)


class TestSimulate(unittest.TestCase):
    def test_random_arrives_collision_free(self):
        rng = random.Random(100)
        pts = []
        while len(pts) < 8:
            p = (rng.uniform(0, 4), rng.uniform(0, 4))
            if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= 3 * R for q in pts):
                pts.append(p)
        res = simulate(pts[:4], pts[4:], R, step_size=0.1, goal_radius=0.12,
                       max_steps=600)
        self.assertTrue(res.arrived)
        self.assertGreaterEqual(min_separation(res.paths, R), -1e-6)

    def test_collision_free_guarantee_in_deadlock(self):
        # head-on swap deadlocks but never collides
        res = simulate([(0, 0), (4, 0)], [(4, 0), (0, 0)], R, step_size=0.1,
                       goal_radius=0.1, max_steps=400)
        self.assertGreaterEqual(min_separation(res.paths, R), -1e-6)

    def test_symmetric_circle_deadlocks_but_safe(self):
        n = 6
        rad = 2.5
        s = [(rad * math.cos(2 * math.pi * k / n),
              rad * math.sin(2 * math.pi * k / n)) for k in range(n)]
        g = [(rad * math.cos(2 * math.pi * k / n + math.pi),
              rad * math.sin(2 * math.pi * k / n + math.pi)) for k in range(n)]
        res = simulate(s, g, R, step_size=0.08, goal_radius=0.15, max_steps=600)
        self.assertGreaterEqual(min_separation(res.paths, R), -1e-6)
        self.assertLess(res.num_arrived, n)        # honest deadlock

    def test_deterministic(self):
        a = simulate([(0, 0), (4, 0)], [(4, 0), (0, 0)], R, step_size=0.1,
                     max_steps=400)
        b = simulate([(0, 0), (4, 0)], [(4, 0), (0, 0)], R, step_size=0.1,
                     max_steps=400)
        self.assertEqual(a.paths, b.paths)
        self.assertEqual(a.steps, b.steps)


if __name__ == "__main__":
    unittest.main()
