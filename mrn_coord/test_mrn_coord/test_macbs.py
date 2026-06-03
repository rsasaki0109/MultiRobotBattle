"""Tests for Meta-Agent CBS (Sharon et al., 2012/2015).

MA-CBS interpolates between fully-decoupled CBS and a fully-coupled joint search
with the conflict bound ``B``: two meta-agents that conflict more than ``B`` times
are merged into one, solved by a coupled low level. The contracts: it returns the
CBS optimum (collision-free) for every ``B``; ``B = inf`` never merges (it *is*
standard CBS); a bottleneck that explodes the CBS tree is absorbed into one
coupled meta-agent so the high-level search shrinks; and it is deterministic.
"""

import random
import unittest

from mrn_coord.mapf import GridWorld
from mrn_coord.mapf.cbs import cbs
from mrn_coord.mapf.conflicts import detect_first_conflict
from mrn_coord.mapf.macbs import macbs

BIG = 10 ** 9


def _inst(seed, n, w, h):
    rng = random.Random(seed)
    free = [(x, y) for x in range(w) for y in range(h)]
    cells = rng.sample(free, 2 * n)
    return GridWorld(w, h), {i: (cells[i], cells[n + i]) for i in range(n)}


class TestMacbs(unittest.TestCase):
    def test_optimal_and_collision_free_for_every_B(self):
        for seed in range(40):
            for n, w, h in ((3, 5, 5), (3, 4, 4), (4, 5, 4)):
                grid, agents = _inst(seed, n, w, h)
                base = cbs(grid, agents, max_expansions=40000)
                if base is None:
                    continue
                for b in (BIG, 2, 1, 0):
                    sol = macbs(grid, agents, merge_bound=b, max_expansions=40000)
                    self.assertIsNotNone(sol, f"seed={seed} n={n} B={b}")
                    self.assertEqual(sol.cost, base.cost,
                                     f"seed={seed} n={n} B={b}")
                    self.assertIsNone(detect_first_conflict(sol.paths))

    def test_B_infinity_is_standard_cbs(self):
        # No merges, every group a singleton, same optimum.
        for seed in range(20):
            grid, agents = _inst(seed, 3, 5, 5)
            base = cbs(grid, agents, max_expansions=40000)
            if base is None:
                continue
            st = {}
            sol = macbs(grid, agents, merge_bound=BIG, max_expansions=40000,
                        stats=st)
            self.assertEqual(sol.cost, base.cost)
            self.assertEqual(st["merges"], 0)
            self.assertEqual(st["max_group_size"], 1)

    def test_merging_shrinks_the_search_on_a_bottleneck(self):
        # A 3-agent symmetry bottleneck: B=inf explodes, B=0 absorbs the
        # conflicting agents into one coupled meta-agent for the same optimum.
        grid = GridWorld(4, 4)
        agents = {0: ((2, 3), (1, 0)), 1: ((0, 3), (1, 1)), 2: ((3, 2), (0, 0))}
        base = cbs(grid, agents)
        inf, zero = {}, {}
        s_inf = macbs(grid, agents, merge_bound=BIG, stats=inf)
        s_zero = macbs(grid, agents, merge_bound=0, stats=zero)
        self.assertEqual(s_inf.cost, base.cost)
        self.assertEqual(s_zero.cost, base.cost)
        self.assertLess(zero["expansions"], inf["expansions"])
        self.assertEqual(zero["max_group_size"], 3)
        self.assertGreater(inf["merges"], -1)
        self.assertEqual(inf["merges"], 0)

    def test_corridor_swap_optimal_under_merging(self):
        cfree = set((x, 0) for x in range(5))
        cfree.add((2, 1))
        cblocked = {(x, y) for x in range(5) for y in range(2)} - cfree
        grid = GridWorld(5, 2, blocked=frozenset(cblocked))
        agents = {0: ((0, 0), (4, 0)), 1: ((4, 0), (0, 0))}
        base = cbs(grid, agents)
        for b in (BIG, 1, 0):
            sol = macbs(grid, agents, merge_bound=b)
            self.assertIsNotNone(sol)
            self.assertEqual(sol.cost, base.cost)
            self.assertIsNone(detect_first_conflict(sol.paths))

    def test_returns_none_when_infeasible(self):
        grid = GridWorld(3, 3, blocked={(1, 0), (1, 1), (1, 2)})
        agents = {0: ((0, 0), (2, 2)), 1: ((2, 2), (0, 0))}
        self.assertIsNone(macbs(grid, agents, merge_bound=0))

    def test_deterministic(self):
        grid, agents = _inst(7, 3, 5, 5)
        a = macbs(grid, agents, merge_bound=1)
        b = macbs(grid, agents, merge_bound=1)
        self.assertEqual(a.cost, b.cost)
        self.assertEqual(a.paths, b.paths)


if __name__ == "__main__":
    unittest.main()
