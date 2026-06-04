"""Tests for CBM — Conflict-Based Min-cost-flow for TAPF (Ma & Koenig, 2016).

TAPF assigns interchangeable targets within teams and routes everyone
collision-free, minimizing makespan. The contracts: a single team degenerates to
anonymous network flow; singleton teams degenerate to makespan-optimal labeled
MAPF (matched against brute force); and team plans are collision-free with every
agent on one of its own team's targets.
"""

import itertools
import random
import unittest
from collections import deque

from mrn_coord.mapf import GridWorld
from mrn_coord.mapf.cbm import cbm
from mrn_coord.mapf.conflicts import detect_first_conflict
from mrn_coord.mapf.flow import anonymous_makespan


def _brute_labeled(grid, starts, goals):
    na = len(starts)
    cells = [(x, y) for x in range(grid.width) for y in range(grid.height)
             if grid.is_free((x, y))]
    nbr = {c: grid.neighbors(c) for c in cells}
    start, goal = tuple(starts), tuple(goals)
    if start == goal:
        return 0
    seen = {start}
    q = deque([(start, 0)])
    while q:
        cfg, t = q.popleft()
        if t > 25:
            continue
        for nxt in itertools.product(*[nbr[c] for c in cfg]):
            if len(set(nxt)) != na:
                continue
            bad = any(nxt[i] == cfg[j] and nxt[j] == cfg[i]
                      for i in range(na) for j in range(i + 1, na))
            if bad:
                continue
            if nxt == goal:
                return t + 1
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, t + 1))
    return None


class TestCBM(unittest.TestCase):
    def test_one_team_is_anonymous_flow(self):
        for seed in range(40):
            rng = random.Random(seed)
            cells = rng.sample([(x, y) for x in range(5) for y in range(5)], 8)
            grid = GridWorld(5, 5)
            starts, goals = cells[:4], cells[4:]
            fmk = anonymous_makespan(grid, starts, goals)
            res = cbm(grid, [(starts, goals)])
            self.assertIsNotNone(res)
            self.assertEqual(res[1], fmk[1], f"seed={seed}")
            self.assertIsNone(detect_first_conflict(res[0]))

    def test_singleton_teams_are_labeled_optimum(self):
        for seed in range(80):
            rng = random.Random(seed)
            cells = rng.sample([(x, y) for x in range(4) for y in range(4)], 4)
            grid = GridWorld(4, 4)
            starts, goals = cells[:2], cells[2:]
            bms = _brute_labeled(grid, starts, goals)
            res = cbm(grid, [([starts[i]], [goals[i]]) for i in range(2)])
            if res is None or bms is None:
                continue
            paths, ms = res
            self.assertEqual(ms, bms, f"seed={seed}")
            self.assertIsNone(detect_first_conflict(paths))
            for i in range(2):
                self.assertEqual(paths[(i, 0)][-1], goals[i])

    def test_anonymous_lower_bounds_labeled(self):
        for seed in range(40):
            rng = random.Random(seed)
            cells = rng.sample([(x, y) for x in range(5) for y in range(5)], 6)
            grid = GridWorld(5, 5)
            starts, goals = cells[:3], cells[3:]
            one = cbm(grid, [(starts, goals)])
            many = cbm(grid, [([starts[i]], [goals[i]]) for i in range(3)])
            if one is None or many is None:
                continue
            self.assertLessEqual(one[1], many[1], f"seed={seed}")

    def test_two_team_showcase(self):
        rng = random.Random(0)
        cells = rng.sample([(x, y) for x in range(5) for y in range(5)], 8)
        grid = GridWorld(5, 5)
        tA = (cells[0:2], cells[2:4])
        tB = (cells[4:6], cells[6:8])
        st: dict = {}
        paths, ms = cbm(grid, [tA, tB], stats=st)
        self.assertIsNone(detect_first_conflict(paths))
        for ti, (s, g) in enumerate((tA, tB)):
            ends = {paths[(ti, ai)][-1] for ai in range(len(s))}
            self.assertEqual(ends, set(g))
        self.assertGreater(st["expansions"], 1)

    def test_infeasible(self):
        grid = GridWorld(3, 3, blocked={(1, 0), (1, 1), (1, 2)})
        self.assertIsNone(cbm(grid, [([(0, 0)], [(2, 2)])], max_makespan=8))


if __name__ == "__main__":
    unittest.main()
