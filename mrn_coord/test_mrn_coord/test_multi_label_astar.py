"""Tests for Multi-Label A* (Grenouilleau, van Hoeve & Hooker, ICAPS 2019).

MLA* plans a single agent through an ordered pickup->delivery in one search over
(cell, time, label) states, passing through the pickup instead of resting there.
The contracts: every returned path is valid (visits the pickup before ending at
the delivery, collision-free against the reservation table) and optimal (matches
a brute multi-label BFS); it finds a path where the two-step baseline is
over-constrained (pickup reserved indefinitely -> None); and it is at least as
short where two_step must wait.
"""

import random
import unittest
from collections import deque

from mrn_coord.mapf import GridWorld
from mrn_coord.mapf.multi_label_astar import mla_star, two_step_plan


def _valid(grid, path, start, pickup, delivery, vertex, edge):
    if path is None or path[0] != start or path[-1] != delivery:
        return False
    if pickup not in path:
        return False
    for t, c in enumerate(path):
        if (c, t) in vertex:
            return False
        if t > 0 and (path[t - 1], c, t) in edge:
            return False
    return True


def _brute(grid, start, pickup, delivery, vertex, horizon=80):
    q = deque([(start, 0, 1)])
    seen = {(start, 0, 1)}
    while q:
        cell, t, lab = q.popleft()
        if lab == 2 and cell == delivery:
            return t
        if t >= horizon:
            continue
        nbrs = []
        if lab == 1 and cell == pickup:
            nbrs.append((pickup, t, 2))
        nt = t + 1
        for nc in grid.neighbors(cell):
            if (nc, nt) not in vertex:
                nbrs.append((nc, nt, lab))
        for st in nbrs:
            if st not in seen:
                seen.add(st)
                q.append(st)
    return None


class TestMultiLabelAStar(unittest.TestCase):
    def test_valid_and_optimal_vs_brute(self):
        for seed in range(120):
            rng = random.Random(seed)
            grid = GridWorld(7, 7)
            cells = [(x, y) for x in range(7) for y in range(7)]
            s, pk, dl = rng.sample(cells, 3)
            others = rng.sample([c for c in cells if c != s], 4)
            resv = frozenset((c, t) for c in others
                             for t in range(1, rng.randint(4, 12)))
            path = mla_star(grid, s, pk, dl, resv, max_time=80)
            opt = _brute(grid, s, pk, dl, resv)
            if opt is None:
                continue
            self.assertTrue(_valid(grid, path, s, pk, dl, resv, frozenset()))
            self.assertEqual(len(path) - 1, opt)

    def test_case1_passes_through_reserved_pickup(self):
        # pickup parked-on indefinitely: two_step cannot settle -> None; MLA*
        # passes through before the reservation starts.
        grid = GridWorld(9, 1)
        s, pk, dl = (0, 0), (4, 0), (8, 0)
        resv = frozenset((pk, t) for t in range(6, 60))
        mla = mla_star(grid, s, pk, dl, resv, max_time=60)
        self.assertIsNotNone(mla)
        self.assertTrue(_valid(grid, mla, s, pk, dl, resv, frozenset()))
        self.assertIsNone(two_step_plan(grid, s, pk, dl, resv, max_time=60))

    def test_case2_shorter_than_two_step(self):
        grid = GridWorld(9, 1)
        s, pk, dl = (0, 0), (4, 0), (8, 0)
        resv = frozenset({(pk, 10)})
        mla = mla_star(grid, s, pk, dl, resv, max_time=60)
        two = two_step_plan(grid, s, pk, dl, resv, max_time=60)
        self.assertIsNotNone(mla)
        self.assertIsNotNone(two)
        self.assertLess(len(mla), len(two))

    def test_infeasible_returns_none(self):
        # pickup walled off
        grid = GridWorld(3, 3, frozenset({(1, 0), (0, 1), (1, 2), (2, 1)}))
        # (1,1) is enclosed; route through it is impossible
        self.assertIsNone(
            mla_star(grid, (0, 0), (1, 1), (2, 2)))


if __name__ == "__main__":
    unittest.main()
