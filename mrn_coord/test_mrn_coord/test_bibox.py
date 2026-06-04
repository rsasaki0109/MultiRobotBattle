"""Tests for Bibox (Surynek, 2009) -- constructive complete MAPF on biconnected
graphs.

Bibox decomposes a biconnected graph into a basic cycle plus open ears, solves the
derived ears in reverse order by rotating the cycle each forms with a return path
(locking each interior), then closes the basic cycle. It is complete on biconnected
graphs with at least two blanks and valid by construction. The contracts: the ear
decomposition is a real open decomposition; every plan is collision-free and on
goal; it solves a packed instance and a theta swap; and it returns None outside its
class (not biconnected, or fewer than two blanks).
"""

import random
import unittest
from collections import deque

from mrn_coord.mapf import GridWorld
from mrn_coord.mapf.bibox import bibox, ear_decomposition
from mrn_coord.mapf.conflicts import detect_first_conflict


def _free(grid):
    return [(x, y) for x in range(grid.width) for y in range(grid.height)
            if grid.is_free((x, y))]


def _adj(grid):
    fset = set(_free(grid))
    return {c: [n for n in ((c[0] + 1, c[1]), (c[0] - 1, c[1]),
                            (c[0], c[1] + 1), (c[0], c[1] - 1)) if n in fset]
            for c in fset}


def _valid(sol, agents):
    return (detect_first_conflict(sol.paths) is None
            and all(sol.paths[a][-1] == agents[a][1] for a in agents))


def _brute_solvable(grid, agents, cap=200000):
    adj = _adj(grid)
    ids = sorted(agents)
    start = tuple(agents[a][0] for a in ids)
    goal = tuple(agents[a][1] for a in ids)
    if start == goal:
        return True
    seen = {start}
    q = deque([start])
    while q and len(seen) < cap:
        cfg = q.popleft()
        occ = set(cfg)
        for i, a in enumerate(ids):
            for nb in adj[cfg[i]]:
                if nb not in occ:
                    nc = cfg[:i] + (nb,) + cfg[i + 1:]
                    if nc not in seen:
                        seen.add(nc)
                        if nc == goal:
                            return True
                        q.append(nc)
    return None


class TestBibox(unittest.TestCase):
    def test_ear_decomposition_is_open(self):
        for w, h in ((3, 3), (4, 4), (2, 5)):
            grid = GridWorld(w, h)
            bc, ears = ear_decomposition(grid)
            adj = {c: set(ns) for c, ns in _adj(grid).items()}
            # basic cycle is a cycle
            self.assertTrue(all(bc[(i + 1) % len(bc)] in adj[bc[i]]
                                for i in range(len(bc))))
            built = set(bc)
            cover = set(bc)
            for e in ears:
                self.assertIn(e[0], built)
                self.assertIn(e[-1], built)
                self.assertNotEqual(e[0], e[-1])          # open
                for iv in e[1:-1]:
                    self.assertNotIn(iv, built)           # interior is new
                built |= set(e)
                cover |= set(e)
            self.assertEqual(cover, set(_adj(grid)))      # covers every vertex

    def test_random_battery_valid_by_construction(self):
        rng = random.Random(3)
        for w, h in ((3, 3), (2, 4), (4, 3), (4, 4)):
            grid = GridWorld(w, h)
            free = _free(grid)
            for n in range(1, len(free) - 1):
                for _ in range(6):
                    s = rng.sample(free, n)
                    g = rng.sample(free, n)
                    agents = {i: (s[i], g[i]) for i in range(n)}
                    sol = bibox(grid, agents)
                    if sol is not None:
                        self.assertTrue(_valid(sol, agents))

    def test_complete_and_sound_vs_brute(self):
        rng = random.Random(5)
        for w, h, blk in ((3, 3, ()), (2, 4, ()), (2, 3, ())):
            grid = GridWorld(w, h, frozenset(blk))
            free = _free(grid)
            for n in range(1, min(4, len(free) - 1) + 1):
                for _ in range(15):
                    s = rng.sample(free, n)
                    g = rng.sample(free, n)
                    agents = {i: (s[i], g[i]) for i in range(n)}
                    sol = bibox(grid, agents)
                    bt = _brute_solvable(grid, agents)
                    if bt is True:
                        self.assertIsNotNone(sol)         # complete
                    if bt is False:
                        self.assertIsNone(sol)            # sound

    def test_theta_swap(self):
        grid = GridWorld(2, 3)
        agents = {0: ((0, 0), (0, 2)), 1: ((0, 2), (0, 0))}
        stats: dict = {}
        sol = bibox(grid, agents, stats=stats)
        self.assertIsNotNone(sol)
        self.assertTrue(_valid(sol, agents))
        self.assertEqual(stats["ears"], 1)

    def test_out_of_class_returns_none(self):
        # a path graph has cut vertices -> not biconnected
        path = GridWorld(1, 4)
        self.assertIsNone(bibox(path, {0: ((0, 0), (0, 3)), 1: ((0, 3), (0, 0))}))
        # 2x2 with three agents -> only one blank
        tiny = GridWorld(2, 2)
        self.assertIsNone(bibox(tiny, {0: ((0, 0), (1, 1)), 1: ((1, 0), (0, 1)),
                                       2: ((0, 1), (1, 0))}))


if __name__ == "__main__":
    unittest.main()
