"""Tests for MAPF-LNS: validity, monotone anytime improvement, determinism.

LNS is anytime, not optimal, so the contract is: it returns a collision-free
solution, never worse than the one it started from, deterministic for a fixed
seed, and — given enough rounds on instances CBS can solve — it closes most of
the gap to the optimum.
"""

import random
import unittest

from mrn_coord.mapf import GridWorld, cbs
from mrn_coord.mapf.conflicts import detect_first_conflict
from mrn_coord.mapf.lns import mapf_lns
from mrn_coord.mapf.solution import pad_paths, sum_of_costs


def _valid(grid, agents, sol) -> bool:
    if sol is None:
        return False
    if detect_first_conflict(pad_paths(sol.paths)) is not None:
        return False
    for a, (start, goal) in agents.items():
        path = sol.paths[a]
        if path[0] != start or path[-1] != goal:
            return False
    return True


class TestLnsBasics(unittest.TestCase):
    def test_solves_and_is_collision_free(self):
        grid = GridWorld(5, 3, blocked={(2, 0), (2, 2)})
        agents = {"a": ((0, 1), (4, 1)), "b": ((4, 1), (0, 1))}
        sol = mapf_lns(grid, agents, iterations=30, seed=0)
        self.assertTrue(_valid(grid, agents, sol))

    def test_never_worse_than_initial(self):
        rng = random.Random(2)
        checked = 0
        for _ in range(60):
            w = h = rng.randint(4, 6)
            blocked = frozenset(
                (rng.randrange(w), rng.randrange(h))
                for _ in range(rng.randint(0, 4)))
            grid = GridWorld(w, h, blocked)
            free = [(x, y) for x in range(w) for y in range(h)
                    if grid.is_free((x, y))]
            n = rng.randint(2, 5)
            if len(free) < 2 * n:
                continue
            pts = rng.sample(free, 2 * n)
            agents = {str(i): (pts[2 * i], pts[2 * i + 1]) for i in range(n)}
            stats = {}
            sol = mapf_lns(grid, agents, iterations=40, seed=1, stats=stats)
            if sol is None:
                continue
            self.assertTrue(_valid(grid, agents, sol))
            self.assertLessEqual(stats["final_cost"], stats["initial_cost"])
            checked += 1
        self.assertGreater(checked, 20)

    def test_closes_gap_to_optimal_on_average(self):
        # Over solvable instances, LNS should reach the CBS optimum most of the
        # time and never beat it (optimal is a lower bound).
        rng = random.Random(5)
        optimal_hits = 0
        total = 0
        for _ in range(60):
            grid = GridWorld(5, 5)
            free = [(x, y) for x in range(5) for y in range(5)]
            n = 4
            pts = rng.sample(free, 2 * n)
            agents = {str(i): (pts[2 * i], pts[2 * i + 1]) for i in range(n)}
            opt = cbs(grid, agents, max_expansions=20_000)
            if opt is None:
                continue
            sol = mapf_lns(grid, agents, iterations=80, seed=3)
            self.assertGreaterEqual(sol.cost, opt.cost)      # optimal is a bound
            optimal_hits += sol.cost == opt.cost
            total += 1
        self.assertGreater(optimal_hits / total, 0.6)

    def test_deterministic(self):
        grid = GridWorld(6, 6, blocked={(3, 2), (2, 3)})
        agents = {"a": ((0, 0), (5, 5)), "b": ((5, 0), (0, 5)),
                  "c": ((0, 5), (5, 0))}
        a = mapf_lns(grid, agents, iterations=50, seed=7)
        b = mapf_lns(grid, agents, iterations=50, seed=7)
        self.assertEqual(a.paths, b.paths)

    def test_improves_a_poor_initial(self):
        # Hand a deliberately padded initial solution; LNS must shorten it.
        grid = GridWorld(7, 1)
        agents = {"a": ((0, 0), (6, 0))}
        from mrn_coord.mapf.solution import Solution
        padded = Solution(paths={"a": [(0, 0)] * 5 + [(i, 0) for i in range(7)]},
                          cost=0)
        padded.cost = sum_of_costs(padded.paths)
        stats = {}
        sol = mapf_lns(grid, agents, iterations=10, seed=0, init=padded,
                       stats=stats)
        self.assertEqual(sol.cost, 6)                         # straight line
        self.assertLess(stats["final_cost"], stats["initial_cost"])


if __name__ == "__main__":
    unittest.main()
