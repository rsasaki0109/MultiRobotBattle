"""Tests for lifelong task allocation: Hungarian optimality, auction validity,
and that cost-aware allocation lifts throughput over round-robin."""

import itertools
import random
import unittest

from mrn_coord.lifelong import TaskStream, make_warehouse, run_lifelong
from mrn_coord.lifelong.allocation import INF, auction, hungarian


def _no_collisions(history) -> bool:
    for prev, cur in zip(history, history[1:]):
        if len(set(cur.values())) != len(cur):
            return False
        for a in cur:
            for b in cur:
                if a < b and cur[a] == prev[b] and cur[b] == prev[a]:
                    return False
    return True


def _total(cost, asn):
    return sum(cost[r][c] for r, c in asn.items())


def _brute_optimal(cost):
    rows, cols = len(cost), len(cost[0])
    c = cost
    if rows > cols:                       # transpose so rows <= cols
        c = [[cost[i][j] for i in range(rows)] for j in range(cols)]
        rows, cols = cols, rows
    best = INF
    for perm in itertools.permutations(range(cols), rows):
        best = min(best, sum(c[i][perm[i]] for i in range(rows)))
    return best


class TestHungarian(unittest.TestCase):
    def test_matches_brute_force_optimum(self):
        rng = random.Random(3)
        for _ in range(800):
            r = rng.randint(1, 5)
            c = rng.randint(1, 5)
            cost = [[rng.randint(0, 20) for _ in range(c)] for _ in range(r)]
            asn = hungarian(cost)
            self.assertEqual(len(asn), min(r, c))            # full matching
            self.assertEqual(len(set(asn.values())), len(asn))  # distinct cols
            self.assertEqual(_total(cost, asn), _brute_optimal(cost))

    def test_respects_forbidden_entries(self):
        cost = [[INF, 3, 5], [2, INF, 4], [6, 1, INF]]
        asn = hungarian(cost)
        for r, c in asn.items():
            self.assertLess(cost[r][c], INF)


class TestAuction(unittest.TestCase):
    def test_valid_and_never_below_optimal(self):
        rng = random.Random(4)
        for _ in range(800):
            r = rng.randint(1, 5)
            c = rng.randint(1, 5)
            cost = [[rng.randint(0, 20) for _ in range(c)] for _ in range(r)]
            asn = auction(cost)
            self.assertEqual(len(asn), min(r, c))
            self.assertEqual(len(set(asn.values())), len(asn))
            self.assertGreaterEqual(_total(cost, asn), _brute_optimal(cost))

    def test_deterministic(self):
        cost = [[5, 2, 9], [3, 8, 1], [4, 4, 4]]
        self.assertEqual(auction(cost), auction(cost))


class TestAllocatorInLifelong(unittest.TestCase):
    def _run(self, allocator, n=6, steps=120, **kw):
        grid, endpoints = make_warehouse(rows=2, cols=3)
        starts = {f"r{i}": endpoints[i] for i in range(n)}
        return run_lifelong(grid, starts, TaskStream(list(endpoints)),
                            max_steps=steps, allocator=allocator, **kw)

    def test_cost_aware_beats_round_robin(self):
        base = self._run("stream").completed
        auc = self._run("auction").completed
        hun = self._run("hungarian").completed
        self.assertGreater(auc, base)
        self.assertGreater(hun, base)

    def test_allocators_stay_collision_free(self):
        for allocator in ("auction", "hungarian"):
            res = self._run(allocator, keep_history=True)
            self.assertTrue(_no_collisions(res.history),
                            msg=f"{allocator} produced a collision")
            self.assertEqual(sum(res.per_agent.values()), res.completed)

    def test_default_is_round_robin(self):
        # The default path must be byte-identical to the explicit stream one
        # (so the existing benchmark expectations are untouched).
        grid, endpoints = make_warehouse(rows=2, cols=3)
        starts = {f"r{i}": endpoints[i] for i in range(6)}
        a = run_lifelong(grid, dict(starts), TaskStream(list(endpoints)),
                         max_steps=120)
        b = run_lifelong(grid, dict(starts), TaskStream(list(endpoints)),
                         max_steps=120, allocator="stream")
        self.assertEqual(a.as_dict(), b.as_dict())

    def test_unknown_allocator_raises(self):
        grid, endpoints = make_warehouse(rows=2, cols=2)
        starts = {"r0": endpoints[0]}
        with self.assertRaises(ValueError):
            run_lifelong(grid, starts, TaskStream(list(endpoints)),
                         max_steps=10, allocator="nope")


if __name__ == "__main__":
    unittest.main()
