"""Tests for DDM — database-driven multi-robot path planning (Han & Yu, 2020).

DDM resolves conflicts locally with a precomputed optimal sub-problem database
and spreads robots with path diversification. The contracts: the database is
makespan-optimal and collision-free within a window; it caches translation-
invariantly; it performs the canonical local maneuvers (rotation, swap) a single
cell cannot; diversification lowers the space-time footprint overlap; and the
database-driven online loop is collision-free by construction whenever it returns
a solution (it is incomplete, like the paper).
"""

import random
import unittest
from collections import deque
from itertools import product

from mrn_coord.mapf import GridWorld
from mrn_coord.mapf.conflicts import detect_first_conflict
from mrn_coord.mapf.ddm import (
    LocalDatabase, _dist_field, _diversified_paths, _shortest_paths, ddm,
)


def _brute_makespan(cells, starts, goals):
    order = sorted(starts)
    cs = set(cells)

    def nb(c):
        x, y = c
        return [c] + [n for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
                      if n in cs]

    s = tuple(starts[r] for r in order)
    g = tuple(goals[r] for r in order)
    n = len(order)
    seen = {s}
    q = deque([(s, 0)])
    while q:
        u, d = q.popleft()
        if u == g:
            return d
        for v in product(*[nb(u[i]) for i in range(n)]):
            if len(set(v)) != n:
                continue
            if any(u[i] == v[j] and u[j] == v[i] and u[i] != u[j]
                   for i in range(n) for j in range(i + 1, n)):
                continue
            if v not in seen:
                seen.add(v)
                q.append((v, d + 1))
    return None


def _valid(plan, goals):
    for cfg in plan:
        if len(set(cfg.values())) != len(cfg):
            return False
    for t in range(len(plan) - 1):
        u, v = plan[t], plan[t + 1]
        for a in u:
            for b in u:
                if a < b and u[a] == v[b] and u[b] == v[a] and u[a] != u[b]:
                    return False
    return all(plan[-1][r] == goals[r] for r in goals)


def _inst(seed, n, w, h):
    rng = random.Random(seed)
    free = [(x, y) for x in range(w) for y in range(h)]
    c = rng.sample(free, 2 * n)
    return GridWorld(w, h), {i: (c[i], c[n + i]) for i in range(n)}


class TestLocalDatabase(unittest.TestCase):
    def test_optimal_vs_brute(self):
        for rw, rh in ((3, 2), (3, 3)):
            cells = [(x, y) for x in range(rw) for y in range(rh)]
            db = LocalDatabase()
            rng = random.Random(7)
            for _ in range(150):
                k = rng.randint(2, min(4, len(cells) // 2))
                pts = rng.sample(cells, 2 * k)
                s = {i: pts[i] for i in range(k)}
                g = {i: pts[k + i] for i in range(k)}
                pl = db.solve(cells, s, g)
                bm = _brute_makespan(cells, s, g)
                if bm is None:
                    self.assertIsNone(pl)
                else:
                    self.assertIsNotNone(pl)
                    self.assertEqual(len(pl) - 1, bm)
                    self.assertTrue(_valid(pl, g))

    def test_translation_invariant_cache(self):
        db = LocalDatabase()
        cells = [(x, y) for x in range(3) for y in range(2)]
        db.solve(cells, {0: (0, 0), 1: (1, 0), 2: (2, 0)},
                 {0: (1, 0), 1: (2, 0), 2: (0, 0)})
        after = db.solves
        db.solve([(x + 9, y + 4) for x, y in cells],
                 {0: (9, 4), 1: (10, 4), 2: (11, 4)},
                 {0: (10, 4), 1: (11, 4), 2: (9, 4)})
        self.assertEqual(db.solves, after)   # reused, no fresh solve

    def test_canonical_maneuvers(self):
        cells = [(x, y) for x in range(3) for y in range(2)]
        db = LocalDatabase()
        rot = db.solve(cells, {0: (0, 0), 1: (1, 0), 2: (2, 0)},
                       {0: (1, 0), 1: (2, 0), 2: (0, 0)})
        swp = db.solve(cells, {0: (0, 0), 1: (2, 0)}, {0: (2, 0), 1: (0, 0)})
        self.assertEqual(len(rot) - 1, 3)
        self.assertEqual(len(swp) - 1, 4)


class TestDiversification(unittest.TestCase):
    def test_reduces_overlap(self):
        def overlap(paths):
            claimed: dict = {}
            o = 0
            for p in paths.values():
                for t, c in enumerate(p):
                    o += claimed.get((c, t), 0)
                    claimed[(c, t)] = claimed.get((c, t), 0) + 1
            return o

        on = off = 0
        for seed in range(60):
            grid, ag = _inst(seed, 8, 8, 8)
            fields = {r: _dist_field(grid, ag[r][1]) for r in ag}
            d = _diversified_paths(grid, ag, fields, candidates=4)
            f = {r: _shortest_paths(grid, ag[r][0], ag[r][1], fields[r], 1)[0]
                 for r in ag}
            if d is None:
                continue
            on += overlap(d)
            off += overlap(f)
        self.assertLess(on, off)


class TestDdm(unittest.TestCase):
    def test_collision_free_by_construction(self):
        # Every returned solution must be collision-free and on-goal (DDM is
        # incomplete, so it may also return None -- that is allowed).
        solved = 0
        for seed in range(120):
            grid, ag = _inst(seed, 6, 8, 8)
            sol = ddm(grid, ag)
            if sol is None:
                continue
            solved += 1
            self.assertIsNone(detect_first_conflict(sol.paths),
                              f"seed={seed}")
            for r, (start, goal) in ag.items():
                self.assertEqual(sol.paths[r][0], start)
                self.assertEqual(sol.paths[r][-1], goal)
        self.assertGreater(solved, 0)

    def test_showcase_fires_database(self):
        grid, ag = _inst(4, 5, 5, 5)
        st: dict = {}
        sol = ddm(grid, ag, stats=st)
        self.assertIsNotNone(sol)
        self.assertIsNone(detect_first_conflict(sol.paths))
        self.assertGreater(st["database_solves"], 0)

    def test_deterministic(self):
        grid, ag = _inst(13, 5, 6, 6)
        a = ddm(grid, ag)
        b = ddm(grid, ag)
        self.assertEqual(a is None, b is None)
        if a is not None:
            self.assertEqual(a.paths, b.paths)


if __name__ == "__main__":
    unittest.main()
