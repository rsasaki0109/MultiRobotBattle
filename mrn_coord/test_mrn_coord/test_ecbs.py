"""Tests for Enhanced CBS (ECBS), the bounded-suboptimal MAPF solver.

The contract is a *provable* one: ECBS returns collision-free paths whose
sum-of-costs is at most ``w`` times the optimum CBS would find. So the tests
pin (a) ``w = 1`` reproduces the optimal cost, (b) for ``w > 1`` the solution
is collision-free and within the ``w * optimal`` bound on many random
instances, and (c) ECBS never fails on an instance CBS solves. A focused case
checks ECBS expands no more high-level nodes than CBS on a conflict-heavy
instance — the reason it exists.
"""

import random
import unittest

from mrn_coord.mapf import GridWorld, cbs, ecbs
from mrn_coord.mapf.conflicts import detect_first_conflict


def _random_instance(rng):
    w = rng.randint(4, 6)
    h = rng.randint(4, 6)
    blocked = frozenset(
        (rng.randrange(w), rng.randrange(h)) for _ in range(rng.randint(0, 4)))
    grid = GridWorld(w, h, blocked)
    free = [(x, y) for x in range(w) for y in range(h) if grid.is_free((x, y))]
    n = rng.randint(2, 4)
    if len(free) < 2 * n:
        return None
    pts = rng.sample(free, 2 * n)
    agents = {str(i): (pts[2 * i], pts[2 * i + 1]) for i in range(n)}
    return grid, agents


class TestEcbsBasics(unittest.TestCase):
    def test_solves_a_simple_swap(self):
        grid = GridWorld(5, 3, blocked={(2, 0), (2, 2)})
        agents = {"a": ((0, 1), (4, 1)), "b": ((4, 1), (0, 1))}
        sol = ecbs(grid, agents, w=1.5)
        self.assertIsNotNone(sol)
        self.assertIsNone(detect_first_conflict(sol.paths))

    def test_infeasible_returns_none(self):
        # A wall splits the corridor, so the agent can never reach its goal.
        grid = GridWorld(3, 1, blocked={(1, 0)})
        agents = {"a": ((0, 0), (2, 0))}
        self.assertIsNone(ecbs(grid, agents, w=1.5))

    def test_reports_expansions(self):
        grid = GridWorld(5, 3, blocked={(2, 0), (2, 2)})
        agents = {"a": ((0, 1), (4, 1)), "b": ((4, 1), (0, 1))}
        stats = {}
        ecbs(grid, agents, w=1.5, stats=stats)
        self.assertIn("expansions", stats)
        self.assertGreaterEqual(stats["expansions"], 1)


class TestEcbsBoundedSuboptimal(unittest.TestCase):
    def test_w1_matches_optimal_cost(self):
        rng = random.Random(11)
        checked = 0
        for _ in range(120):
            inst = _random_instance(rng)
            if inst is None:
                continue
            grid, agents = inst
            opt = cbs(grid, agents, max_expansions=20_000)
            if opt is None:
                continue
            sol = ecbs(grid, agents, w=1.0, max_expansions=20_000)
            self.assertIsNotNone(sol)
            self.assertEqual(sol.cost, opt.cost)
            checked += 1
        self.assertGreater(checked, 40)

    def test_within_bound_and_collision_free(self):
        rng = random.Random(99)
        checked = 0
        for _ in range(200):
            inst = _random_instance(rng)
            if inst is None:
                continue
            grid, agents = inst
            opt = cbs(grid, agents, max_expansions=20_000)
            if opt is None:
                continue
            for w in (1.2, 1.5, 2.0):
                sol = ecbs(grid, agents, w=w, max_expansions=20_000)
                self.assertIsNotNone(sol)  # solvable -> ECBS must find one
                self.assertIsNone(detect_first_conflict(sol.paths))
                self.assertLessEqual(sol.cost, w * opt.cost + 1e-9)
            checked += 1
        self.assertGreater(checked, 60)


class TestEcbsExpandsNoMoreThanCbs(unittest.TestCase):
    def test_fewer_or_equal_high_level_nodes(self):
        # A doorway both agents must cross drives a few CBS branches; ECBS with
        # slack resolves it without expanding more nodes.
        grid = GridWorld(7, 3, blocked={(3, 0), (3, 2)})
        agents = {"a": ((0, 1), (6, 1)), "b": ((6, 1), (0, 1)),
                  "c": ((0, 0), (6, 2))}
        cstats, estats = {}, {}
        cbs(grid, agents, max_expansions=20_000, stats=cstats)
        ecbs(grid, agents, w=1.5, max_expansions=20_000, stats=estats)
        self.assertLessEqual(estats["expansions"], cstats["expansions"])


if __name__ == "__main__":
    unittest.main()
