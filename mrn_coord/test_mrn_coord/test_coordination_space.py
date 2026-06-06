"""Tests for path-velocity decomposition / coordination-space scheduling
(Kant & Zucker 1986; O'Donnell & Lozano-Perez 1989)."""

import math
import unittest

from mrn_coord.mapf.coordination_space import (
    CoordinationProblem,
    build_collision_table,
    discretize_path,
    min_clearance,
    schedule,
    schedule_to_trajectories,
)


M = 0.15


def _baseline_collides(p, margin=M):
    table = build_collision_table(p, margin)
    for k in range(p.m):
        st = tuple(min(p.m - 1, k) for _ in range(p.n))
        for (a, b), mask in table.items():
            if mask[st[a]][st[b]]:
                return True
    return False


def _has_wait(s, n):
    return any(s.states[t][r] == s.states[t + 1][r]
               for t in range(len(s.states) - 1) for r in range(n))


class TestDiscretize(unittest.TestCase):
    def test_endpoints_and_uniform(self):
        pts = discretize_path([(0, 0), (10, 0)], 11)
        self.assertEqual(len(pts), 11)
        self.assertAlmostEqual(pts[0][0], 0.0)
        self.assertAlmostEqual(pts[-1][0], 10.0)
        self.assertAlmostEqual(pts[5][0], 5.0)

    def test_polyline_arc_length(self):
        pts = discretize_path([(0, 0), (0, 2), (2, 2)], 5)
        # total length 4, midpoint at arc-length 2 == the corner (0, 2)
        self.assertAlmostEqual(pts[2][0], 0.0)
        self.assertAlmostEqual(pts[2][1], 2.0)


class TestSchedule(unittest.TestCase):
    def test_perpendicular_crossing_resolved(self):
        p = CoordinationProblem([[(0, 4), (8, 4)], [(4, 0), (4, 8)]],
                                [0.5, 0.5], m=20)
        self.assertTrue(_baseline_collides(p))
        s = schedule(p, safety_margin=M)
        self.assertIsNotNone(s)
        cl = min_clearance(schedule_to_trajectories(p, s), p.radii)
        self.assertGreaterEqual(cl, -1e-6)
        self.assertTrue(_has_wait(s, p.n))
        self.assertGreater(s.makespan, p.m - 1)   # timing delay incurred

    def test_shared_bridge_resolved(self):
        p = CoordinationProblem([[(0, 0), (4, 4), (8, 0)],
                                 [(0, 8), (4, 4), (8, 8)]], [0.5, 0.5], m=20)
        s = schedule(p, safety_margin=M)
        self.assertIsNotNone(s)
        self.assertGreaterEqual(
            min_clearance(schedule_to_trajectories(p, s), p.radii), -1e-6)

    def test_head_on_shared_corridor_unsolvable(self):
        # same corridor, opposite directions -> velocity tuning cannot reroute
        p = CoordinationProblem([[(0, 4), (8, 4)], [(8, 4), (0, 4)]],
                                [0.5, 0.5], m=20)
        self.assertIsNone(schedule(p, safety_margin=M))

    def test_monotone_forward_only(self):
        p = CoordinationProblem([[(0, 4), (8, 4)], [(4, 0), (4, 8)]],
                                [0.5, 0.5], m=16)
        s = schedule(p, safety_margin=M)
        for t in range(len(s.states) - 1):
            for r in range(p.n):
                self.assertGreaterEqual(s.states[t + 1][r], s.states[t][r])

    def test_makespan_optimal_vs_bfs(self):
        from collections import deque
        p = CoordinationProblem([[(0, 4), (8, 4)], [(4, 0), (4, 8)]],
                                [0.5, 0.5], m=12)
        table = build_collision_table(p, M)
        start, goal = (0, 0), (p.m - 1, p.m - 1)
        dist = {start: 0}
        q = deque([start])
        moves = [(1, 0), (0, 1), (1, 1)]
        best = None
        while q:
            u = q.popleft()
            if u == goal:
                best = dist[u]
                break
            for mv in moves:
                v = (min(p.m - 1, u[0] + mv[0]), min(p.m - 1, u[1] + mv[1]))
                if v == u or v in dist or table[(0, 1)][v[0]][v[1]]:
                    continue
                dist[v] = dist[u] + 1
                q.append(v)
        s = schedule(p, safety_margin=M)
        self.assertEqual(s.makespan, best)

    def test_three_robots(self):
        p = CoordinationProblem([[(0, 4), (8, 4)], [(4, 0), (4, 8)],
                                 [(0, 0), (8, 8)]], [0.4, 0.4, 0.4], m=16)
        s = schedule(p, safety_margin=M)
        self.assertIsNotNone(s)
        self.assertGreaterEqual(
            min_clearance(schedule_to_trajectories(p, s), p.radii), -1e-6)

    def test_deterministic(self):
        p = CoordinationProblem([[(0, 4), (8, 4)], [(4, 0), (4, 8)]],
                                [0.5, 0.5], m=14)
        a = schedule(p, safety_margin=M)
        b = schedule(p, safety_margin=M)
        self.assertEqual(a.states, b.states)


if __name__ == "__main__":
    unittest.main()
