"""Tests for BCBS — Bounded CBS (Barer et al., 2014), ECBS's sibling.

BCBS runs focal search at both levels but bounds the high-level focal against the
best cost, so its factors multiply: cost <= w_high * w_low * optimal. The
contracts: BCBS(1,1) is optimal CBS; the product bound holds; the independent
w_high/w_low knobs (which ECBS lacks) stay bounded and collision-free; and on a
known instance BCBS trades a higher cost for fewer expansions than ECBS.
"""

import random
import unittest

from mrn_coord.mapf import GridWorld
from mrn_coord.mapf.bcbs import bcbs
from mrn_coord.mapf.cbs import cbs
from mrn_coord.mapf.conflicts import detect_first_conflict
from mrn_coord.mapf.ecbs import ecbs


def _rand(seed, w, h, n, ob):
    rng = random.Random(seed)
    blocked = {(x, y) for x in range(w) for y in range(h) if rng.random() < ob}
    free = [(x, y) for x in range(w) for y in range(h) if (x, y) not in blocked]
    if len(free) < 2 * n:
        return None
    rng.shuffle(free)
    return GridWorld(w, h, frozenset(blocked)), \
        {i: (free[i], free[n + i]) for i in range(n)}


class TestBCBS(unittest.TestCase):
    def test_optimal_at_1_1_and_product_bound(self):
        W = 1.5
        for seed in range(40):
            r = _rand(seed, 5, 5, 4, 0.1)
            if r is None:
                continue
            grid, ag = r
            base = cbs(grid, ag, max_expansions=40000)
            if base is None:
                continue
            b11 = bcbs(grid, ag, w_high=1.0, w_low=1.0)
            bww = bcbs(grid, ag, w_high=W, w_low=W)
            self.assertEqual(b11.cost, base.cost, f"seed={seed}")
            self.assertLessEqual(bww.cost, W * W * base.cost + 1e-9)
            self.assertIsNone(detect_first_conflict(bww.paths))

    def test_independent_knobs_bounded(self):
        grid, ag = _rand(23, 5, 5, 7, 0.05)
        opt = cbs(grid, ag, max_expansions=80000).cost
        for wh, wl in ((1.0, 3.0), (3.0, 1.0), (2.0, 1.0)):
            r = bcbs(grid, ag, w_high=wh, w_low=wl)
            self.assertLessEqual(r.cost, wh * wl * opt + 1e-9)
            self.assertIsNone(detect_first_conflict(r.paths))

    def test_diverges_from_ecbs_showcase(self):
        # seed 23: BCBS's looser product bound stops earlier (fewer expansions)
        # at a higher cost than ECBS's tight w bound.
        grid, ag = _rand(23, 5, 5, 7, 0.05)
        sb: dict = {}
        se: dict = {}
        b = bcbs(grid, ag, w_high=1.5, w_low=1.5, stats=sb)
        e = ecbs(grid, ag, w=1.5, stats=se)
        self.assertLess(sb["expansions"], se["expansions"])
        self.assertGreater(b.cost, e.cost)
        self.assertIsNone(detect_first_conflict(b.paths))


if __name__ == "__main__":
    unittest.main()
