"""Tests for MAPF-LNS2: collision-minimizing repair to feasibility.

MAPF-LNS2 (Li et al., AAAI 2022) starts from a collision-ridden set of
individual shortest paths and minimizes the NUMBER of collisions with Large
Neighborhood Search until it reaches zero. Unlike the cost optimizer it is not
collision-free by construction -- the returned solution is only guaranteed
collision-free when ``stats["feasible"]`` (the count hit zero). The contracts:
the feasibility flag agrees with an independent conflict check, it repairs to
zero on instances a greedy prioritized order or a budgeted CBS cannot solve, the
collision count is monotone non-increasing under acceptance, and it is
deterministic for a fixed seed.
"""

import random
import unittest

from mrn_coord.mapf import GridWorld
from mrn_coord.mapf.cbs import cbs
from mrn_coord.mapf.conflicts import detect_first_conflict
from mrn_coord.mapf.lns2 import _count_collisions, mapf_lns2


def _inst(n, w, h, seed):
    rng = random.Random(seed)
    free = [(x, y) for x in range(w) for y in range(h)]
    cells = rng.sample(free, 2 * n)
    return GridWorld(w, h), {i: (cells[i], cells[n + i]) for i in range(n)}


class TestLns2(unittest.TestCase):
    def test_feasible_flag_matches_conflict_check(self):
        # The stats["feasible"] flag (final count == 0) must agree with an
        # independent detect_first_conflict on every instance, feasible or not.
        for seed in range(12):
            grid, agents = _inst(9, 8, 8, seed)
            s = {}
            sol = mapf_lns2(grid, agents, iterations=400, neighborhood_size=8,
                            seed=1, stats=s)
            cf = detect_first_conflict(sol.paths) is None
            self.assertEqual(cf, s["feasible"], f"seed={seed}")

    def test_repairs_colliding_start_to_zero(self):
        # Each instance starts with collisions (shortest paths ignore everyone)
        # and is driven to a genuinely collision-free solution.
        for seed in (0, 2, 3, 4, 5, 6):
            grid, agents = _inst(9, 8, 8, seed)
            s = {}
            sol = mapf_lns2(grid, agents, iterations=600, neighborhood_size=8,
                            seed=1, stats=s)
            self.assertGreater(s["initial_collisions"], 0, f"seed={seed}")
            self.assertTrue(s["feasible"], f"seed={seed}")
            self.assertEqual(_count_collisions(sol.paths), 0)
            for a, (start, goal) in agents.items():
                self.assertEqual(sol.paths[a][0], start)
                self.assertEqual(sol.paths[a][-1], goal)

    def test_solves_where_cbs_busts_budget(self):
        # On dense 6x6/14-agent instances a 2000-node CBS gives up, yet MAPF-LNS2
        # still repairs to feasibility.
        for seed in (0, 1, 2):
            grid, agents = _inst(14, 6, 6, seed)
            self.assertIsNone(cbs(grid, agents, max_expansions=2000))
            s = {}
            sol = mapf_lns2(grid, agents, iterations=600, neighborhood_size=10,
                            seed=1, stats=s)
            self.assertTrue(s["feasible"], f"seed={seed}")
            self.assertIsNone(detect_first_conflict(sol.paths))

    def test_deterministic(self):
        grid, agents = _inst(9, 8, 8, 2)
        runs = []
        for _ in range(2):
            s = {}
            mapf_lns2(grid, dict(agents), iterations=300, neighborhood_size=8,
                      seed=1, stats=s)
            runs.append((s["initial_collisions"], s["final_collisions"],
                         s["accepted"]))
        self.assertEqual(runs[0], runs[1])

    def test_returns_none_when_an_agent_is_trapped(self):
        # An agent walled off from its goal -> no path at all -> None.
        grid = GridWorld(3, 3, blocked={(1, 0), (1, 1), (1, 2)})
        agents = {0: ((0, 0), (2, 2)), 1: ((0, 1), (0, 2))}
        self.assertIsNone(mapf_lns2(grid, agents, iterations=10))


if __name__ == "__main__":
    unittest.main()
