"""Tests for BCP rectangle cuts (bcp.py's rectangle=True) — the Lam et al.
specialized cut family on the branch-and-cut-and-price frame.

When two agents cross an open rectangle in the same direction, plain
branch-and-price enumerates the symmetric crossings; a single rectangle cut
sum_{B1} y_a1 + sum_{B2} y_a2 <= 1 collapses it. The contracts: the cut never
changes the optimum (== cbs and == plain BCP), it collapses the branch tree on a
rectangle crossing, and it is byte-for-byte off by default.
"""

import random
import unittest

from mrn_coord.mapf import GridWorld
from mrn_coord.mapf.bcp import bcp
from mrn_coord.mapf.cbs import cbs
from mrn_coord.mapf.conflicts import detect_first_conflict


class TestBCPRectangle(unittest.TestCase):
    def test_rectangle_collapses_branching_same_optimum(self):
        # same-direction anti-diagonal crossing = a rectangle symmetry
        grid = GridWorld(7, 7)
        ag = {0: ((2, 0), (6, 6)), 1: ((0, 2), (6, 4))}
        base = cbs(grid, ag)
        soff: dict = {}
        off = bcp(grid, ag, rectangle=False, stats=soff)
        son: dict = {}
        on = bcp(grid, ag, rectangle=True, stats=son)
        self.assertEqual(on.cost, base.cost)
        self.assertEqual(on.cost, off.cost)            # cut drops no solution
        self.assertIsNone(detect_first_conflict(on.paths))
        self.assertGreater(son["rcuts"], 0)            # a rectangle cut fired
        self.assertLess(son["nodes"], soff["nodes"])   # branch tree collapsed

    def test_off_matches_plain_on_random(self):
        # rectangle=True must not change the answer where no rectangle exists,
        # and rectangle=False is the plain solver -- both == cbs.
        for seed in range(20):
            rng = random.Random(seed)
            free = [(x, y) for x in range(5) for y in range(5)]
            rng.shuffle(free)
            grid = GridWorld(5, 5)
            ag = {i: (free[i], free[3 + i]) for i in range(3)}
            base = cbs(grid, ag)
            off = bcp(grid, ag, rectangle=False)
            on = bcp(grid, ag, rectangle=True)
            if base is None:
                continue
            self.assertEqual(off.cost, base.cost, f"seed={seed}")
            self.assertEqual(on.cost, base.cost, f"seed={seed}")
            self.assertIsNone(detect_first_conflict(on.paths))

    def test_rectangle_lp_bound_certifies(self):
        grid = GridWorld(6, 6)
        ag = {0: ((2, 0), (4, 5)), 1: ((1, 1), (4, 2))}
        st: dict = {}
        sol = bcp(grid, ag, rectangle=True, stats=st)
        self.assertIsNotNone(sol)
        self.assertLessEqual(st["lp_bound"], sol.cost + 1e-6)


if __name__ == "__main__":
    unittest.main()
