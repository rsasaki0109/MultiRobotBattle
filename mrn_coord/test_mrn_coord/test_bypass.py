"""Tests for CBS with bypassing conflicts (Boyarski et al., ICAPS 2015).

BP adopts a same-cost, fewer-conflicts child's path into the current node instead
of splitting the constraint tree. The contracts: it returns the CBS optimum
(collision-free); a cardinal conflict can never be bypassed; bypassing shrinks
the search (fewer expansions and generated nodes) and never enlarges it; and it
is deterministic.
"""

import random
import unittest

from mrn_coord.mapf import GridWorld
from mrn_coord.mapf.cbs import cbs
from mrn_coord.mapf.bypass import cbs_bypass
from mrn_coord.mapf.conflicts import detect_first_conflict


def _inst(seed, n, w, h):
    rng = random.Random(seed)
    free = [(x, y) for x in range(w) for y in range(h)]
    cells = rng.sample(free, 2 * n)
    return GridWorld(w, h), {i: (cells[i], cells[n + i]) for i in range(n)}


class TestCbsBypass(unittest.TestCase):
    def test_optimal_and_collision_free(self):
        for seed in range(40):
            for n, w, h in ((3, 5, 5), (4, 5, 5), (4, 6, 6)):
                grid, agents = _inst(seed, n, w, h)
                base = cbs(grid, agents, max_expansions=40000)
                if base is None:
                    continue
                sol = cbs_bypass(grid, agents, max_expansions=40000)
                self.assertIsNotNone(sol, f"seed={seed} n={n}")
                self.assertEqual(sol.cost, base.cost, f"seed={seed} n={n}")
                self.assertIsNone(detect_first_conflict(sol.paths))

    def test_bypass_never_enlarges_the_search(self):
        # Over a battery, BP must never expand more nodes than the no-bypass
        # ablation, and must strictly help somewhere.
        helped = 0
        for seed in range(40):
            for n, w, h in ((4, 6, 6), (5, 6, 6)):
                grid, agents = _inst(seed, n, w, h)
                base = cbs(grid, agents, max_expansions=40000)
                if base is None:
                    continue
                so: dict = {}
                sb: dict = {}
                cbs_bypass(grid, agents, bypass=False, max_expansions=40000,
                           stats=so)
                cbs_bypass(grid, agents, bypass=True, max_expansions=40000,
                           stats=sb)
                self.assertLessEqual(sb["expansions"], so["expansions"])
                if sb["expansions"] < so["expansions"]:
                    helped += 1
        self.assertGreater(helped, 0)

    def test_showcase_collapses_the_tree(self):
        # Seed 54, 5 agents on 6x6: same optimum, but the tree collapses.
        grid, agents = _inst(54, 5, 6, 6)
        base = cbs(grid, agents)
        so: dict = {}
        sb: dict = {}
        off = cbs_bypass(grid, agents, bypass=False, stats=so)
        on = cbs_bypass(grid, agents, bypass=True, stats=sb)
        self.assertEqual(off.cost, base.cost)
        self.assertEqual(on.cost, base.cost)
        self.assertLess(sb["expansions"], so["expansions"])
        self.assertLess(sb["generated"], so["generated"])
        self.assertGreater(sb["bypasses"], 0)

    def test_returns_none_when_infeasible(self):
        grid = GridWorld(3, 3, blocked={(1, 0), (1, 1), (1, 2)})
        agents = {0: ((0, 0), (2, 2)), 1: ((2, 2), (0, 0))}
        self.assertIsNone(cbs_bypass(grid, agents, max_expansions=2000))

    def test_deterministic(self):
        grid, agents = _inst(7, 4, 6, 6)
        a = cbs_bypass(grid, agents)
        b = cbs_bypass(grid, agents)
        self.assertEqual(a.cost, b.cost)
        self.assertEqual(a.paths, b.paths)


if __name__ == "__main__":
    unittest.main()
