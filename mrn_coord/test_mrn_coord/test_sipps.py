"""Tests for SIPPS — Safe Interval Path Planning with Soft constraints.

SIPPS (Li et al., 2022) is the low level of MAPF-LNS2: hard constraints define
the safe intervals, the other agents' paths are soft (passable at one collision
each, counted even while waiting), and it returns the fewest-collision, then
shortest, path. The contracts: with no soft constraints it is plain SIPP; with
soft constraints it returns the true ``(collisions, length)`` optimum; and it
searches the safe-interval state space, not the time-expanded one.
"""

import heapq
import random
import unittest

from mrn_coord.mapf import GridWorld
from mrn_coord.mapf.lns2 import _soft_reservations
from mrn_coord.mapf.sipp import plan_sipp
from mrn_coord.mapf.sipps import plan_sipps
from mrn_coord.mapf.space_time_astar import plan_path


def _sh(sv, se):
    h = 0
    for k in sv:
        h = max(h, k[1])
    for k in se:
        h = max(h, k[2])
    return h


def _count(path, sv, se):
    H = max(_sh(sv, se), len(path) - 1)
    col = 0
    for t in range(H + 1):
        cell = path[t] if t < len(path) else path[-1]
        col += sv.get((cell, t), 0)
    for t in range(1, len(path)):
        if path[t] != path[t - 1]:
            col += se.get((path[t - 1], path[t], t), 0)
    return col


def _brute(grid, s, go, sv, se):
    H = _sh(sv, se)
    maxt = H + 2 * grid.width * grid.height + 5
    sc = sv.get((s, 0), 0)
    pq = [(sc, 0, s)]
    best = {(s, 0): sc}
    ans = None
    while pq:
        col, t, cell = heapq.heappop(pq)
        if best.get((cell, t), 1e9) < col:
            continue
        if cell == go:
            tot = col + sum(sv.get((go, tt), 0) for tt in range(t + 1, H + 1))
            ans = tot if ans is None else min(ans, tot)
        if t >= maxt:
            continue
        for nc in grid.neighbors(cell):
            nt = t + 1
            add = sv.get((nc, nt), 0)
            if nc != cell:
                add += se.get((cell, nc, nt), 0)
            ncol = col + add
            if best.get((nc, nt), 1e9) > ncol:
                best[(nc, nt)] = ncol
                heapq.heappush(pq, (ncol, nt, nc))
    return ans


class TestSipps(unittest.TestCase):
    def test_matches_plain_sipp_without_soft(self):
        grid = GridWorld(8, 8)
        for seed in range(100):
            rng = random.Random(seed)
            s, go = rng.sample([(x, y) for x in range(8) for y in range(8)], 2)
            p1 = plan_sipp(grid, s, go)
            st: dict = {}
            p2 = plan_sipps(grid, s, go, stats=st)
            self.assertEqual(p1 is None, p2 is None)
            if p1 is not None:
                self.assertEqual(len(p1), len(p2))
                self.assertEqual(st["collisions"], 0)

    def test_minimizes_collisions_optimally(self):
        for seed in range(120):
            rng = random.Random(seed)
            cells = rng.sample([(x, y) for x in range(7) for y in range(7)], 10)
            grid = GridWorld(7, 7)
            ag = {i: (cells[i], cells[5 + i]) for i in range(5)}
            paths = {}
            ok = True
            for i in range(1, 5):
                p = plan_path(grid, ag[i][0], ag[i][1])
                if p is None:
                    ok = False
                    break
                paths[i] = p
            if not ok:
                continue
            horizon = max(len(p) for p in paths.values()) + 10
            sv, se = _soft_reservations({**paths}, set(), horizon)
            s, go = ag[0]
            opt = _brute(grid, s, go, sv, se)
            sol = plan_sipps(grid, s, go, soft_vertex=sv, soft_edge=se)
            if sol is None or opt is None:
                continue
            self.assertEqual(_count(sol, sv, se), opt, f"seed={seed}")

    def test_safe_interval_compresses_long_wait(self):
        # A long hard-blocked stretch: SIPPS waits in one interval state.
        grid = GridWorld(12, 1)
        hard = frozenset({((5, 0), t) for t in range(3, 40)})
        st: dict = {}
        sol = plan_sipps(grid, (0, 0), (11, 0), hard_vertex=hard, stats=st)
        self.assertIsNotNone(sol)
        self.assertEqual(st["collisions"], 0)
        self.assertLess(st["expansions"], 30)
        # honors the hard block (never on (5,0) during [3,40))
        for t, cell in enumerate(sol):
            if cell == (5, 0):
                self.assertFalse(3 <= t < 40)

    def test_vacates_goal_for_zero_collision_settle(self):
        # A soft agent crosses the goal late; the optimum waits it out.
        grid = GridWorld(6, 1)
        soft = {((5, 0), t): 1 for t in range(3, 20)}
        st: dict = {}
        sol = plan_sipps(grid, (0, 0), (5, 0), soft_vertex=soft, stats=st)
        self.assertEqual(st["collisions"], 0)
        self.assertEqual(sol[-1], (5, 0))

    def test_returns_none_when_walled(self):
        grid = GridWorld(3, 3, blocked={(1, 0), (1, 1), (1, 2)})
        self.assertIsNone(plan_sipps(grid, (0, 0), (2, 2)))

    def test_deterministic(self):
        grid = GridWorld(7, 7)
        soft = {((3, 3), 4): 1, ((4, 3), 5): 1}
        a = plan_sipps(grid, (0, 0), (6, 6), soft_vertex=soft)
        b = plan_sipps(grid, (0, 0), (6, 6), soft_vertex=soft)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
