"""Tests for discrete RRT (dRRT) — Solovey, Salzman & Halperin (WAFR 2014)."""

import math
import random
import unittest

from mrn_coord.mapf.drrt import (
    Obstacle,
    build_roadmap,
    composite_optimum,
    direction_oracle,
    drrt,
    drrt_star,
    moving_min_distance,
    segment_point_distance,
    solution_clearance,
    tensor_product_size,
)


class TestGeometry(unittest.TestCase):
    def test_segment_point_distance_endpoints_and_interior(self):
        # foot of perpendicular falls inside the segment
        self.assertAlmostEqual(segment_point_distance((0, 0), (2, 0), (1, 1)), 1.0)
        # closest point is an endpoint when the foot is outside
        self.assertAlmostEqual(segment_point_distance((0, 0), (2, 0), (3, 0)), 1.0)

    def test_moving_min_distance_crossing(self):
        # two points crossing the origin in opposite directions meet exactly
        d = moving_min_distance((-1, 0), (1, 0), (0, -1), (0, 1))
        self.assertAlmostEqual(d, 0.0, places=6)

    def test_moving_min_distance_parallel(self):
        # parallel motion keeps a constant gap
        d = moving_min_distance((0, 0), (1, 0), (0, 1), (1, 1))
        self.assertAlmostEqual(d, 1.0)


class TestRoadmap(unittest.TestCase):
    def test_start_goal_indices_and_connectivity(self):
        rng = random.Random(0)
        rm = build_roadmap((0.1, 0.1), (0.9, 0.9), [], 0.05, 1.0, 1.0,
                           n_samples=30, k=10, rng=rng)
        self.assertEqual(rm.points[rm.start], (0.1, 0.1))
        self.assertEqual(rm.points[rm.goal], (0.9, 0.9))
        # adjacency is symmetric
        for i, nbrs in enumerate(rm.adj):
            for j in nbrs:
                self.assertIn(i, rm.adj[j])

    def test_samples_avoid_obstacle(self):
        rng = random.Random(1)
        obs = [Obstacle(0.5, 0.5, 0.2)]
        rm = build_roadmap((0.05, 0.05), (0.95, 0.95), obs, 0.05, 1.0, 1.0,
                           n_samples=40, k=8, rng=rng)
        for p in rm.points:
            self.assertGreaterEqual(math.hypot(p[0] - 0.5, p[1] - 0.5),
                                    0.2 + 0.05 - 1e-9)

    def test_tensor_product_size(self):
        rng = random.Random(2)
        rms = [build_roadmap((0.1, 0.1), (0.9, 0.9), [], 0.05, 1.0, 1.0,
                             n_samples=10, k=6, rng=rng) for _ in range(3)]
        prod = 1
        for rm in rms:
            prod *= len(rm)
        self.assertEqual(tensor_product_size(rms), prod)


class TestOracle(unittest.TestCase):
    def test_oracle_moves_toward_target(self):
        rng = random.Random(3)
        rm = build_roadmap((0.0, 0.0), (1.0, 1.0), [], 0.05, 1.0, 1.0,
                           n_samples=40, k=12, rng=rng)
        node = (rm.start,)
        # heading straight at the goal should pick a neighbour closer to it
        new = direction_oracle([rm], node, [rm.points[rm.goal]])
        before = math.hypot(*[a - b for a, b in
                              zip(rm.points[rm.start], rm.points[rm.goal])])
        after = math.hypot(*[a - b for a, b in
                             zip(rm.points[new[0]], rm.points[rm.goal])])
        self.assertLess(after, before)

    def test_oracle_stays_when_at_target(self):
        rng = random.Random(4)
        rm = build_roadmap((0.0, 0.0), (1.0, 1.0), [], 0.05, 1.0, 1.0,
                           n_samples=20, k=8, rng=rng)
        node = (rm.goal,)
        new = direction_oracle([rm], node, [rm.points[rm.goal]])
        self.assertEqual(new, node)


class TestSearch(unittest.TestCase):
    def _roadmaps(self, starts, goals, obstacles, seed):
        rng = random.Random(seed)
        return [build_roadmap(s, g, obstacles, 0.05, 1.0, 1.0,
                              n_samples=30, k=12, rng=rng)
                for s, g in zip(starts, goals)]

    def test_solution_collision_free_and_endpoints(self):
        starts = [(0.1, 0.2), (0.9, 0.8), (0.1, 0.8)]
        goals = [(0.9, 0.8), (0.1, 0.2), (0.9, 0.2)]
        rms = self._roadmaps(starts, goals, [], 5)
        sol = drrt(rms, [], 0.05, max_iters=3000, rng=random.Random(5))
        self.assertIsNotNone(sol)
        for i in range(3):
            self.assertEqual(sol.paths[i][0], starts[i])
            self.assertEqual(sol.paths[i][-1], goals[i])
        min_pair, _ = solution_clearance(sol.paths, [], 0.05)
        self.assertGreaterEqual(min_pair, 2 * 0.05 - 1e-6)

    def test_swap_around_obstacle(self):
        obs = [Obstacle(0.5, 0.5, 0.12)]
        starts, goals = [(0.1, 0.5), (0.9, 0.5)], [(0.9, 0.5), (0.1, 0.5)]
        rms = self._roadmaps(starts, goals, obs, 3)
        sol = drrt(rms, obs, 0.05, max_iters=3000, rng=random.Random(99))
        self.assertIsNotNone(sol)
        min_pair, min_obs = solution_clearance(sol.paths, obs, 0.05)
        self.assertGreaterEqual(min_pair, 2 * 0.05 - 1e-6)
        self.assertGreaterEqual(min_obs, -1e-6)

    def test_implicit_exploration_tiny_fraction(self):
        starts = [(0.1, 0.1), (0.9, 0.9), (0.1, 0.9), (0.9, 0.1)]
        goals = [(0.9, 0.9), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9)]
        rng = random.Random(7)
        rms = [build_roadmap(s, g, [], 0.05, 1.0, 1.0,
                             n_samples=40, k=12, rng=rng)
               for s, g in zip(starts, goals)]
        prod = tensor_product_size(rms)
        sol = drrt(rms, [], 0.05, max_iters=4000, rng=random.Random(7))
        self.assertIsNotNone(sol)
        self.assertLess(sol.tree_size / prod, 1e-3)

    def test_oracle_beats_random_neighbour(self):
        starts = [(0.1, 0.2), (0.9, 0.8), (0.1, 0.8)]
        goals = [(0.9, 0.8), (0.1, 0.2), (0.9, 0.2)]
        rms = self._roadmaps(starts, goals, [], 5)
        s_dir = drrt(rms, [], 0.05, max_iters=2000, rng=random.Random(5),
                     oracle="direction")
        rms = self._roadmaps(starts, goals, [], 5)
        s_rnd = drrt(rms, [], 0.05, max_iters=2000, rng=random.Random(5),
                     oracle="random")
        self.assertIsNotNone(s_dir)
        # the direction oracle solves with a far smaller tree (when random
        # solves at all); here it must at least not be worse.
        if s_rnd is not None:
            self.assertLessEqual(s_dir.tree_size, s_rnd.tree_size)

    def test_deterministic(self):
        obs = [Obstacle(0.5, 0.5, 0.12)]
        starts, goals = [(0.1, 0.5), (0.9, 0.5)], [(0.9, 0.5), (0.1, 0.5)]
        a = drrt(self._roadmaps(starts, goals, obs, 3), obs, 0.05,
                 max_iters=3000, rng=random.Random(99))
        b = drrt(self._roadmaps(starts, goals, obs, 3), obs, 0.05,
                 max_iters=3000, rng=random.Random(99))
        self.assertEqual(a.tree_size, b.tree_size)
        self.assertAlmostEqual(a.makespan, b.makespan, places=9)


class TestStar(unittest.TestCase):
    def _roadmaps(self, starts, goals, obstacles, seed, n_samples=12):
        rng = random.Random(seed)
        return [build_roadmap(s, g, obstacles, 0.05, 1.0, 1.0,
                              n_samples=n_samples, k=8, rng=rng)
                for s, g in zip(starts, goals)]

    def test_converges_to_brute_optimum(self):
        # dRRT* must reach (or get within 2% of) the shortest path over the FULL
        # implicit composite roadmap, computed independently by brute Dijkstra.
        starts, goals = [(0.15, 0.2), (0.85, 0.8)], [(0.85, 0.8), (0.15, 0.2)]
        rms = self._roadmaps(starts, goals, [], 5)
        opt, _ = composite_optimum(rms, [], 0.05)
        star = drrt_star(rms, [], 0.05, max_iters=1200, rng=random.Random(5))
        self.assertIsNotNone(star)
        self.assertLessEqual(star.cost, opt * 1.02 + 1e-9)

    def test_beats_plain_drrt_first_solution(self):
        starts, goals = [(0.15, 0.2), (0.85, 0.8)], [(0.85, 0.8), (0.15, 0.2)]
        rms = self._roadmaps(starts, goals, [], 6)
        star = drrt_star(rms, [], 0.05, max_iters=1000, rng=random.Random(6))
        rms = self._roadmaps(starts, goals, [], 6)
        plain = drrt(rms, [], 0.05, max_iters=1000, rng=random.Random(6))
        self.assertIsNotNone(star)
        self.assertIsNotNone(plain)
        self.assertLessEqual(star.cost, plain.total_length + 1e-9)

    def test_cost_history_monotone_non_increasing(self):
        starts, goals = [(0.15, 0.2), (0.85, 0.8)], [(0.85, 0.8), (0.15, 0.2)]
        rms = self._roadmaps(starts, goals, [], 5)
        star = drrt_star(rms, [], 0.05, max_iters=800, rng=random.Random(5))
        hist = [c for c in star.cost_history if c != float("inf")]
        for i in range(len(hist) - 1):
            self.assertGreaterEqual(hist[i], hist[i + 1] - 1e-9)

    def test_informed_focuses_search(self):
        starts = [(0.05, 0.5), (0.95, 0.5)]
        goals = [(0.95, 0.5), (0.05, 0.5)]
        s_inf = drrt_star(self._roadmaps(starts, goals, [], 11, n_samples=14),
                          [], 0.05, max_iters=600, rng=random.Random(11),
                          informed=True)
        s_uninf = drrt_star(self._roadmaps(starts, goals, [], 11, n_samples=14),
                            [], 0.05, max_iters=600, rng=random.Random(11),
                            informed=False)
        self.assertLess(s_inf.graph_size, s_uninf.graph_size)

    def test_swap_around_obstacle_optimal_and_clear(self):
        obs = [Obstacle(0.5, 0.5, 0.12)]
        starts, goals = [(0.1, 0.5), (0.9, 0.5)], [(0.9, 0.5), (0.1, 0.5)]
        rms = self._roadmaps(starts, goals, obs, 3)
        opt, _ = composite_optimum(rms, obs, 0.05)
        star = drrt_star(rms, obs, 0.05, max_iters=1200, rng=random.Random(7))
        self.assertIsNotNone(star)
        self.assertAlmostEqual(star.cost, opt, places=6)
        mp, mo = solution_clearance(star.paths, obs, 0.05)
        self.assertGreaterEqual(mp, 2 * 0.05 - 1e-6)
        self.assertGreaterEqual(mo, -1e-6)

    def test_deterministic(self):
        obs = [Obstacle(0.5, 0.5, 0.12)]
        starts, goals = [(0.1, 0.5), (0.9, 0.5)], [(0.9, 0.5), (0.1, 0.5)]
        a = drrt_star(self._roadmaps(starts, goals, obs, 3), obs, 0.05,
                      max_iters=800, rng=random.Random(99))
        b = drrt_star(self._roadmaps(starts, goals, obs, 3), obs, 0.05,
                      max_iters=800, rng=random.Random(99))
        self.assertEqual(a.graph_size, b.graph_size)
        self.assertAlmostEqual(a.cost, b.cost, places=9)


if __name__ == "__main__":
    unittest.main()
