"""Tests for highway heuristics (Cohen, Uras & Koenig, 2015).

A highway is a set of directed edges layered on ECBS: it reorders the low-level
FOCAL list to prefer flowing with the highway, leaving OPEN -- and the w bound --
untouched. The contracts: an empty highway is byte-for-byte plain ECBS; on a
two-lane corridor a keep-to-one-side highway cuts high-level expansions while the
solution stays within w * optimal and collision-free; and a highway is advice,
so it may raise cost (within the bound) when it does not match the instance.
"""

import unittest

from mrn_coord.mapf import GridWorld
from mrn_coord.mapf.cbs import cbs
from mrn_coord.mapf.conflicts import detect_first_conflict
from mrn_coord.mapf.ecbs import ecbs
from mrn_coord.mapf.highway import (ecbs_highway, keep_side_highway,
                                    ring_highway)


def _two_lane(width):
    g = GridWorld(width, 2)
    agents = {0: ((0, 0), (width - 1, 0)), 1: ((0, 1), (width - 1, 1)),
              2: ((width - 1, 0), (0, 0)), 3: ((width - 1, 1), (0, 1))}
    return g, agents


class TestHighway(unittest.TestCase):
    def test_empty_highway_is_plain_ecbs(self):
        for width in (4, 5, 6, 7):
            g, ag = _two_lane(width)
            for w in (1.5, 2.0, 3.0):
                s1, s2 = {}, {}
                a = ecbs(g, ag, w=w, stats=s1)
                b = ecbs_highway(g, ag, w=w, highways=frozenset(), stats=s2)
                self.assertEqual(s1["expansions"], s2["expansions"])
                self.assertEqual(a.cost, b.cost)

    def test_highway_cuts_expansions_within_bound(self):
        # 2x5 corridor, w=1.5: the keep-side highway gives each direction a lane,
        # cutting expansions 12 -> 3 at the same (optimal) cost.
        g, ag = _two_lane(5)
        hwy = keep_side_highway(g, axis="x")
        so, sh = {}, {}
        off = ecbs(g, ag, w=1.5, stats=so)
        on = ecbs_highway(g, ag, w=1.5, highways=hwy, stats=sh)
        opt = cbs(g, ag, max_expansions=20000)
        self.assertLess(sh["expansions"], so["expansions"])
        self.assertLessEqual(on.cost, 1.5 * opt.cost + 1e-9)
        self.assertIsNone(detect_first_conflict(on.paths))
        self.assertEqual(on.cost, opt.cost)

    def test_highway_is_advice_can_cost_more(self):
        # 2x6 corridor, w=2.0: lane discipline raises cost (28 -> 30) but stays
        # within the bound -- a highway is advice, not a constraint.
        g, ag = _two_lane(6)
        hwy = keep_side_highway(g, axis="x")
        off = ecbs(g, ag, w=2.0)
        on = ecbs_highway(g, ag, w=2.0, highways=hwy)
        opt = cbs(g, ag, max_expansions=20000)
        self.assertGreater(on.cost, off.cost)
        self.assertLessEqual(on.cost, 2.0 * opt.cost + 1e-9)
        self.assertIsNone(detect_first_conflict(on.paths))

    def test_highway_builders(self):
        g = GridWorld(4, 2)
        ks = keep_side_highway(g, axis="x")
        # even row 0 flows +x, odd row 1 flows -x
        self.assertIn(((0, 0), (1, 0)), ks)
        self.assertIn(((3, 1), (2, 1)), ks)
        self.assertNotIn(((0, 0), (0, 1)), ks)     # no off-axis edges
        ring = ring_highway(g, [(0, 0), (1, 0), (1, 1), (0, 1)])
        self.assertIn(((0, 0), (1, 0)), ring)
        self.assertIn(((0, 1), (0, 0)), ring)      # closes the loop


if __name__ == "__main__":
    unittest.main()
