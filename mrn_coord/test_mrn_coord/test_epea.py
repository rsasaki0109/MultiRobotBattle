"""Tests for EPEA* — Enhanced Partial Expansion A* (Goldenberg et al., 2014).

EPEA* generates only the successors whose ``f`` matches the node's, via an
Operator Selection Function, re-inserting the node at its next child ``f``. The
contracts: it returns the CBS optimum (collision-free); it generates far fewer
nodes than the fully-expanding joint A* and never more; and it is deterministic.
"""

import random
import unittest

from mrn_coord.mapf import GridWorld
from mrn_coord.mapf.cbs import cbs
from mrn_coord.mapf.conflicts import detect_first_conflict
from mrn_coord.mapf.epea import epea_star
from mrn_coord.mapf.mstar import joint_astar


def _inst(seed, n, w, h):
    rng = random.Random(seed)
    free = [(x, y) for x in range(w) for y in range(h)]
    cells = rng.sample(free, 2 * n)
    return GridWorld(w, h), {i: (cells[i], cells[n + i]) for i in range(n)}


class TestEpea(unittest.TestCase):
    def test_optimal_and_collision_free(self):
        for seed in range(40):
            for n, w, h in ((2, 5, 5), (3, 5, 5), (3, 4, 4)):
                grid, agents = _inst(seed, n, w, h)
                base = cbs(grid, agents, max_expansions=40000)
                if base is None:
                    continue
                sol = epea_star(grid, agents)
                self.assertIsNotNone(sol, f"seed={seed} n={n}")
                self.assertEqual(sol.cost, base.cost, f"seed={seed} n={n}")
                self.assertIsNone(detect_first_conflict(sol.paths))

    def test_generates_fewer_nodes_than_joint_astar(self):
        helped = 0
        for seed in range(40):
            for n, w, h in ((3, 5, 5), (3, 4, 4)):
                grid, agents = _inst(seed, n, w, h)
                base = cbs(grid, agents, max_expansions=40000)
                if base is None:
                    continue
                sja: dict = {}
                sep: dict = {}
                ja = joint_astar(grid, agents, stats=sja)
                ep = epea_star(grid, agents, stats=sep)
                if ja is None or ep is None:
                    continue
                self.assertLessEqual(sep["generated"], sja["generated"])
                if sep["generated"] < sja["generated"]:
                    helped += 1
        self.assertGreater(helped, 0)

    def test_showcase(self):
        grid, agents = _inst(87, 3, 5, 5)
        base = cbs(grid, agents)
        sja: dict = {}
        sep: dict = {}
        joint_astar(grid, agents, stats=sja)
        sol = epea_star(grid, agents, stats=sep)
        self.assertEqual(sol.cost, base.cost)
        self.assertLess(sep["generated"], sja["generated"])

    def test_returns_none_when_infeasible(self):
        grid = GridWorld(3, 3, blocked={(1, 0), (1, 1), (1, 2)})
        agents = {0: ((0, 0), (2, 2)), 1: ((2, 2), (0, 0))}
        self.assertIsNone(epea_star(grid, agents, max_expansions=5000))

    def test_deterministic(self):
        grid, agents = _inst(87, 3, 5, 5)
        a = epea_star(grid, agents)
        b = epea_star(grid, agents)
        self.assertEqual(a.cost, b.cost)
        self.assertEqual(a.paths, b.paths)


if __name__ == "__main__":
    unittest.main()
