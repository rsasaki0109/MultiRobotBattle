"""Tests for rM* (recursive M*, Wagner & Choset 2011/2015).

rM* refines basic M* by keeping a PARTITION of the agents into independent
collision groups instead of a single flat collision set: it couples only agents
that genuinely collide, so peak coupling is the largest irreducible interacting
group rather than the union of all collisions. The contracts: it returns CBS's
exact sum-of-costs optimum (on random and constructed instances); every plan is
collision-free and on goal; on disjoint swaps it keeps each pair an independent
size-2 group while basic M* unions them all; and it is sound (infeasible -> None).
"""

import random
import unittest

from mrn_coord.mapf import GridWorld
from mrn_coord.mapf.cbs import cbs
from mrn_coord.mapf.conflicts import detect_first_conflict
from mrn_coord.mapf.mstar import mstar
from mrn_coord.mapf.rmstar import rmstar
from mrn_coord.mapf.solution import sum_of_costs


def _valid(sol, agents):
    return (detect_first_conflict(sol.paths) is None
            and all(sol.paths[a][-1] == agents[a][1] for a in agents))


def _rand(w, h, n, seed, obs):
    rng = random.Random(seed)
    blocked = {(x, y) for x in range(w) for y in range(h) if rng.random() < obs}
    free = [(x, y) for x in range(w) for y in range(h) if (x, y) not in blocked]
    rng.shuffle(free)
    return (GridWorld(w, h, frozenset(blocked)),
            {i: (free[i], free[n + i]) for i in range(n)})


def _disjoint_swaps(k):
    W, H = 2, 4 * k - 1
    blocked = {(x, 4 * b + 3) for b in range(k - 1) for x in range(W)}
    grid = GridWorld(W, H, frozenset(blocked))
    ag = {}
    for b in range(k):
        y0 = 4 * b
        ag[2 * b] = ((0, y0), (0, y0 + 2))
        ag[2 * b + 1] = ((0, y0 + 2), (0, y0))
    return grid, ag


class TestRMStar(unittest.TestCase):
    def test_optimal_matches_cbs_random(self):
        # tractable maps (n<=3) so every instance solves within budget; the
        # benchmark gate carries the broader random battery.
        for (w, h, n, obs) in ((4, 4, 2, 0.0), (4, 4, 3, 0.0), (3, 4, 3, 0.0)):
            for seed in range(10):
                grid, ag = _rand(w, h, n, seed, obs)
                base = cbs(grid, ag, max_expansions=20000)
                if base is None:
                    continue
                sol = rmstar(grid, ag, max_expansions=60000)
                self.assertIsNotNone(sol)
                self.assertTrue(_valid(sol, ag))
                self.assertEqual(sum_of_costs(sol.paths), base.cost)

    def test_partition_keeps_disjoint_swaps_independent(self):
        # basic M* unions all k pairs at the shared start (peak == 2k); rM* keeps
        # each pair an independent size-2 group and expands far fewer configs.
        # Capped at k=3: at k=4 basic M*'s 8-way coupling branches 5**8 per node
        # (intractable) -- exactly what rM* avoids; the gate carries rM* to k=4.
        for k in (2, 3):
            grid, ag = _disjoint_swaps(k)
            base = cbs(grid, ag, max_expansions=50000)
            sr, sm = {}, {}
            sol = rmstar(grid, ag, stats=sr, max_expansions=200000)
            mstar(grid, ag, stats=sm, max_expansions=200000)
            self.assertEqual(sum_of_costs(sol.paths), base.cost)
            self.assertTrue(_valid(sol, ag))
            self.assertEqual(sr["max_group"], 2)
            self.assertEqual(sm["max_collision_set"], 2 * k)
            self.assertLess(sr["expansions"], sm["expansions"])

    def test_scales_where_basic_mstar_cannot(self):
        # rM* keeps k=4 disjoint swaps a set of size-2 groups and solves cheaply.
        grid, ag = _disjoint_swaps(4)
        base = cbs(grid, ag, max_expansions=50000)
        sr: dict = {}
        sol = rmstar(grid, ag, stats=sr, max_expansions=200000)
        self.assertIsNotNone(sol)
        self.assertTrue(_valid(sol, ag))
        self.assertEqual(sum_of_costs(sol.paths), base.cost)
        self.assertEqual(sr["max_group"], 2)

    def test_sound_returns_none_when_infeasible(self):
        # head-on swap in a 1-wide corridor: no passing, so unsolvable
        corr = GridWorld(1, 4)
        self.assertIsNone(rmstar(corr, {0: ((0, 0), (0, 3)),
                                        1: ((0, 3), (0, 0))}, max_expansions=50000))

    def test_theta_swap_couples_the_pair(self):
        grid = GridWorld(2, 3)
        ag = {0: ((0, 0), (0, 2)), 1: ((0, 2), (0, 0))}
        stats: dict = {}
        sol = rmstar(grid, ag, stats=stats)
        self.assertIsNotNone(sol)
        self.assertTrue(_valid(sol, ag))
        self.assertEqual(stats["max_group"], 2)


if __name__ == "__main__":
    unittest.main()
