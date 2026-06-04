"""Tests for CBS-TA — CBS with optimal Target Assignment (Hönig et al., 2018).

CBS-TA leaves the goal assignment open: each agent may serve any target from a
pool, and the solver finds the jointly optimal assignment *and* paths by laying a
forest of assignment roots (unfolded by Murty's K-best matching) over CBS's
constraint tree. The contracts: Murty yields assignment costs in sorted order;
one distinct goal per agent degenerates to plain CBS; with a shared pool the
combined cost equals a brute force over every assignment; and the cheapest
matching is not always the jointly optimal one.
"""

import itertools
import random
import unittest

from mrn_coord.mapf import GridWorld
from mrn_coord.mapf.cbs import cbs
from mrn_coord.mapf.cbs_ta import _murty, cbs_ta
from mrn_coord.mapf.conflicts import detect_first_conflict

INF = float("inf")


class TestCBSTA(unittest.TestCase):
    def test_murty_kbest_sorted_order(self):
        def brute(cost):
            R, C = len(cost), len(cost[0])
            out = []
            for p in itertools.permutations(range(C), R):
                t = sum(cost[i][p[i]] for i in range(R))
                if t < INF:
                    out.append(t)
            return sorted(out)

        for seed in range(60):
            rng = random.Random(seed)
            R = rng.randint(2, 4)
            C = rng.randint(R, R + 2)
            cost = [[float(rng.randint(1, 9)) if rng.random() > 0.15 else INF
                     for _ in range(C)] for _ in range(R)]
            bt = brute(cost)
            if not bt:
                continue
            got = [t for (_a, t), _ in zip(_murty(cost), range(min(len(bt), 6)))]
            self.assertEqual(got, bt[:len(got)], f"seed={seed}")

    def test_degenerate_equals_cbs(self):
        for seed in range(60):
            rng = random.Random(seed)
            cells = rng.sample([(x, y) for x in range(5) for y in range(5)], 6)
            grid = GridWorld(5, 5)
            agents = {i: (cells[i], cells[i + 3]) for i in range(3)}
            c = cbs(grid, agents)
            ct = cbs_ta(grid, {i: (cells[i], [cells[i + 3]]) for i in range(3)})
            self.assertEqual(c is None, ct is None, f"seed={seed}")
            if c is not None:
                self.assertEqual(c.cost, ct.cost, f"seed={seed}")
                self.assertIsNone(detect_first_conflict(ct.paths))

    def test_jointly_optimal_vs_brute(self):
        for seed in range(40):
            rng = random.Random(seed)
            cells = rng.sample([(x, y) for x in range(5) for y in range(5)], 6)
            grid = GridWorld(5, 5)
            starts, targets = cells[:3], cells[3:6]
            best = None
            for combo in itertools.permutations(targets, 3):
                sol = cbs(grid, {i: (starts[i], combo[i]) for i in range(3)})
                if sol is not None and (best is None or sol.cost < best):
                    best = sol.cost
            ct = cbs_ta(grid, {i: (starts[i], targets) for i in range(3)})
            if best is None or ct is None:
                continue
            self.assertEqual(ct.cost, best, f"seed={seed}")
            self.assertIsNone(detect_first_conflict(ct.paths))

    def test_assignment_matters_showcase(self):
        # seed 8: the distance-cheapest assignment forces a conflict (cost 8);
        # CBS-TA swaps two agents' targets for a cheaper joint plan (cost 7).
        rng = random.Random(8)
        cells = rng.sample([(x, y) for x in range(5) for y in range(5)], 6)
        grid = GridWorld(5, 5)
        starts, targets = cells[:3], cells[3:6]
        st: dict = {}
        ct = cbs_ta(grid, {i: (starts[i], targets) for i in range(3)}, stats=st)
        self.assertEqual(ct.cost, 7)
        self.assertEqual(st["roots"], 3)
        self.assertIsNone(detect_first_conflict(ct.paths))
        # every agent ends on one of the pool targets, all distinct
        ends = {ct.paths[i][-1] for i in range(3)}
        self.assertEqual(ends, set(targets))

    def test_infeasible_fewer_targets(self):
        grid = GridWorld(3, 3, blocked={(1, 0), (1, 1), (1, 2)})
        self.assertIsNone(
            cbs_ta(grid, {0: ((0, 0), [(2, 2)])}, max_expansions=200))


if __name__ == "__main__":
    unittest.main()
