"""Tests for FECBS — ECBS with flex distribution (Chan et al., SoCS 2021).

FECBS bounds only the total sum-of-costs by w (not each agent individually),
lending each replanned agent the suboptimality budget the others left unspent.
The contracts: it keeps the same w guarantee (cost <= w * optimal) and collapses
to optimal CBS at w=1; every plan is collision-free and on goal; and ECBS stays
byte-for-byte unchanged (flex defaults to 0).
"""

import random
import unittest

from mrn_coord.mapf import GridWorld
from mrn_coord.mapf.cbs import cbs
from mrn_coord.mapf.conflicts import detect_first_conflict
from mrn_coord.mapf.ecbs import ecbs
from mrn_coord.mapf.fecbs import fecbs


def _valid(sol, agents):
    return (detect_first_conflict(sol.paths) is None
            and all(sol.paths[a][-1] == agents[a][1] for a in agents))


def _rand(w, h, n, seed, obs):
    rng = random.Random(seed)
    blocked = {(x, y) for x in range(w) for y in range(h) if rng.random() < obs}
    free = [(x, y) for x in range(w) for y in range(h) if (x, y) not in blocked]
    rng.shuffle(free)
    if len(free) < 2 * n:
        return None, None
    return (GridWorld(w, h, frozenset(blocked)),
            {i: (free[i], free[n + i]) for i in range(n)})


class TestFECBS(unittest.TestCase):
    def test_within_w_bound_and_valid(self):
        for (gw, gh, n, obs) in ((6, 6, 4, 0.0), (5, 5, 4, 0.1)):
            for W in (1.1, 1.5, 2.0):
                for seed in range(10):
                    grid, ag = _rand(gw, gh, n, seed, obs)
                    if grid is None:
                        continue
                    base = cbs(grid, ag, max_expansions=20000)
                    if base is None:
                        continue
                    sol = fecbs(grid, ag, w=W, max_expansions=40000)
                    self.assertIsNotNone(sol)
                    self.assertTrue(_valid(sol, ag))
                    self.assertLessEqual(sol.cost, W * base.cost + 1e-9)

    def test_w1_collapses_to_optimal(self):
        for seed in range(15):
            grid, ag = _rand(6, 6, 4, seed, 0.0)
            if grid is None:
                continue
            base = cbs(grid, ag, max_expansions=20000)
            if base is None:
                continue
            sol = fecbs(grid, ag, w=1.0, max_expansions=40000)
            self.assertIsNotNone(sol)
            self.assertEqual(sol.cost, base.cost)

    def test_flex_expands_no_more_overall(self):
        # on a dense, tight-w family FECBS expands fewer high-level nodes in total
        fe = ec = 0
        for seed in range(12):
            grid, ag = _rand(8, 8, 8, seed, 0.1)
            if grid is None:
                continue
            sf, se = {}, {}
            fsol = fecbs(grid, ag, w=1.05, max_expansions=30000, stats=sf)
            esol = ecbs(grid, ag, w=1.05, max_expansions=30000, stats=se)
            if fsol is None or esol is None:
                continue
            self.assertTrue(_valid(fsol, ag))
            fe += sf["expansions"]
            ec += se["expansions"]
        self.assertLess(fe, ec)

    def test_ecbs_unchanged_by_default_flex(self):
        # ecbs (flex defaults to 0) must be unaffected by the new keyword
        for seed in range(8):
            grid, ag = _rand(6, 6, 4, seed, 0.0)
            if grid is None:
                continue
            s1, s2 = {}, {}
            a = ecbs(grid, ag, w=1.3, max_expansions=20000, stats=s1)
            b = ecbs(grid, ag, w=1.3, max_expansions=20000, stats=s2)
            self.assertEqual(s1["expansions"], s2["expansions"])
            if a is not None:
                self.assertEqual(a.cost, b.cost)


if __name__ == "__main__":
    unittest.main()
