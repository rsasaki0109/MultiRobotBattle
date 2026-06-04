"""Tests for k-robust CBS (Atzmon et al., 2018).

A k-robust plan stays collision-free as long as no agent is delayed by more than
k steps. The contracts: k=0 reproduces plain CBS; for k>=1 the plan has no
k-delay conflict and survives single-agent delays up to k; and cost is monotone
non-decreasing in k.
"""

import random
import unittest

from mrn_coord.mapf import GridWorld
from mrn_coord.mapf.cbs import cbs
from mrn_coord.mapf.conflicts import detect_first_conflict
from mrn_coord.mapf.k_robust import detect_first_k_conflict, k_robust_cbs


def _inst(seed, n, w, h):
    rng = random.Random(seed)
    free = [(x, y) for x in range(w) for y in range(h)]
    cells = rng.sample(free, 2 * n)
    return GridWorld(w, h), {i: (cells[i], cells[n + i]) for i in range(n)}


def _collides_under_delay(paths, k):
    for a in paths:
        for d in range(1, k + 1):
            dp = dict(paths)
            dp[a] = [paths[a][0]] * d + list(paths[a])
            if detect_first_conflict(dp) is not None:
                return True
    return False


class TestKRobust(unittest.TestCase):
    def test_k0_is_plain_cbs(self):
        for seed in range(40):
            for n, w, h in ((2, 5, 5), (3, 5, 5), (3, 4, 4)):
                grid, ag = _inst(seed, n, w, h)
                base = cbs(grid, ag, max_expansions=40000)
                if base is None:
                    continue
                sol = k_robust_cbs(grid, ag, k=0, max_expansions=40000)
                self.assertIsNotNone(sol)
                self.assertEqual(sol.cost, base.cost, f"seed={seed} n={n}")

    def test_guarantees_k_robustness_and_survives_delay(self):
        for k in (1, 2):
            for seed in range(40):
                for n, w, h in ((3, 5, 5), (3, 6, 6)):
                    grid, ag = _inst(seed, n, w, h)
                    sol = k_robust_cbs(grid, ag, k=k, max_expansions=120000)
                    if sol is None:
                        continue
                    self.assertIsNone(detect_first_k_conflict(sol.paths, k),
                                      f"k={k} seed={seed}")
                    self.assertFalse(_collides_under_delay(sol.paths, k),
                                     f"k={k} seed={seed}")

    def test_cost_monotone_in_k(self):
        for seed in range(40):
            grid, ag = _inst(seed, 3, 6, 6)
            s0 = k_robust_cbs(grid, ag, k=0, max_expansions=40000)
            s1 = k_robust_cbs(grid, ag, k=1, max_expansions=80000)
            s2 = k_robust_cbs(grid, ag, k=2, max_expansions=120000)
            if None in (s0, s1, s2):
                continue
            self.assertLessEqual(s0.cost, s1.cost)
            self.assertLessEqual(s1.cost, s2.cost)

    def test_showcase_delay_robustness(self):
        grid, ag = _inst(1, 3, 5, 5)
        base = cbs(grid, ag)
        sol = k_robust_cbs(grid, ag, k=1)
        self.assertTrue(_collides_under_delay(base.paths, 1))
        self.assertFalse(_collides_under_delay(sol.paths, 1))
        self.assertGreater(sol.cost, base.cost)

    def test_infeasible(self):
        grid = GridWorld(3, 3, blocked={(1, 0), (1, 1), (1, 2)})
        ag = {0: ((0, 0), (2, 2)), 1: ((2, 2), (0, 0))}
        self.assertIsNone(k_robust_cbs(grid, ag, k=1, max_expansions=5000))


if __name__ == "__main__":
    unittest.main()
