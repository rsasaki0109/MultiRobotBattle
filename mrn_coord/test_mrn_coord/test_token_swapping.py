"""Tests for Token Swapping — Yamanaka et al., "Swapping Labeled Tokens on Graphs" (2014/2015)."""

import random
import unittest

from mrn_coord.mapf import token_swapping as ts


class TestMechanics(unittest.TestCase):
    def test_apply_swap_exchanges(self):
        p = {0: "a", 1: "b"}
        self.assertEqual(ts.apply_swap(p, 0, 1), {0: "b", 1: "a"})

    def test_replay_rejects_non_edge(self):
        g = ts.make_path_graph(3)  # 0-1-2, no 0-2 edge
        with self.assertRaises(ValueError):
            ts.replay(g, {0: 0, 1: 1, 2: 2}, [(0, 2)])

    def test_replay_reaches_target(self):
        g = ts.make_path_graph(3)
        init = {0: 0, 1: 1, 2: 2}
        out = ts.replay(g, init, [(0, 1), (1, 2)])
        self.assertEqual(out, {0: 1, 1: 2, 2: 0})

    def test_num_misplaced(self):
        self.assertEqual(ts.num_misplaced({0: 1, 1: 0}, {0: 0, 1: 1}), 2)
        self.assertEqual(ts.num_misplaced({0: 0, 1: 1}, {0: 0, 1: 1}), 0)


class TestPathClosedForm(unittest.TestCase):
    def test_reversal_is_n_choose_2(self):
        # reversing 0..n-1 on a path takes C(n,2) adjacent swaps
        for n in range(2, 8):
            init = {i: i for i in range(n)}
            tgt = {i: n - 1 - i for i in range(n)}
            self.assertEqual(ts.path_inversions(init, tgt), n * (n - 1) // 2)

    def test_optimum_equals_inversions(self):
        rng = random.Random(1)
        for _ in range(60):
            n = rng.randint(2, 6)
            g = ts.make_path_graph(n)
            init = {i: i for i in range(n)}
            perm = list(range(n))
            rng.shuffle(perm)
            tgt = {i: perm[i] for i in range(n)}
            opt = ts.optimal_swaps(g, init, tgt)
            self.assertIsNotNone(opt)
            self.assertEqual(opt.num_swaps, ts.path_inversions(init, tgt))

    def test_construction_valid(self):
        rng = random.Random(2)
        for _ in range(60):
            n = rng.randint(2, 7)
            g = ts.make_path_graph(n)
            init = {i: i for i in range(n)}
            perm = list(range(n))
            rng.shuffle(perm)
            tgt = {i: perm[i] for i in range(n)}
            sw = ts.path_swaps(init, tgt)
            self.assertTrue(ts.is_solved(ts.replay(g, init, sw), tgt))
            self.assertEqual(len(sw), ts.path_inversions(init, tgt))


class TestCompleteClosedForm(unittest.TestCase):
    def test_cycle_count(self):
        self.assertEqual(ts.cycle_count([0, 1, 2]), 3)  # all fixed
        self.assertEqual(ts.cycle_count([1, 0, 2]), 2)  # one 2-cycle + fixed
        self.assertEqual(ts.cycle_count([1, 2, 0]), 1)  # one 3-cycle

    def test_optimum_equals_n_minus_cycles(self):
        rng = random.Random(3)
        for _ in range(60):
            n = rng.randint(2, 6)
            g = ts.make_complete_graph(n)
            init = {i: i for i in range(n)}
            perm = list(range(n))
            rng.shuffle(perm)
            tgt = {i: perm[i] for i in range(n)}
            opt = ts.optimal_swaps(g, init, tgt)
            self.assertIsNotNone(opt)
            self.assertEqual(opt.num_swaps, ts.complete_min_swaps(init, tgt))

    def test_construction_valid(self):
        rng = random.Random(4)
        for _ in range(60):
            n = rng.randint(2, 6)
            g = ts.make_complete_graph(n)
            init = {i: i for i in range(n)}
            perm = list(range(n))
            rng.shuffle(perm)
            tgt = {i: perm[i] for i in range(n)}
            sw = ts.complete_swaps(init, tgt)
            self.assertTrue(ts.is_solved(ts.replay(g, init, sw), tgt))
            self.assertEqual(len(sw), ts.complete_min_swaps(init, tgt))


class TestLowerBound(unittest.TestCase):
    def test_optimum_meets_lower_bound(self):
        rng = random.Random(5)
        for _ in range(80):
            n = rng.randint(2, 6)
            g = ts.make_path_graph(n)
            for _ in range(rng.randint(0, n)):
                a, b = rng.sample(range(n), 2)
                g[a].add(b)
                g[b].add(a)
            init = {i: i for i in range(n)}
            perm = list(range(n))
            rng.shuffle(perm)
            tgt = {i: perm[i] for i in range(n)}
            opt = ts.optimal_swaps(g, init, tgt)
            if opt is None:
                continue
            self.assertGreaterEqual(opt.num_swaps, ts.lower_bound(g, init, tgt))


class TestDescentNegative(unittest.TestCase):
    def test_descent_stalls_on_ring_rotation(self):
        g = ts.make_cycle_graph(4)
        init = {i: i for i in range(4)}
        tgt = {i: (i - 1) % 4 for i in range(4)}
        _, solved = ts.descent_swaps(g, init, tgt)
        self.assertFalse(solved)

    def test_descent_stalls_on_path_reversal(self):
        g = ts.make_path_graph(4)
        init = {i: i for i in range(4)}
        tgt = {i: 3 - i for i in range(4)}
        _, solved = ts.descent_swaps(g, init, tgt)
        self.assertFalse(solved)


class TestScalability(unittest.TestCase):
    def test_bfs_busts_where_closed_form_solves(self):
        n = 11
        init = {i: i for i in range(n)}
        gp = ts.make_path_graph(n)
        tgt = {i: n - 1 - i for i in range(n)}
        sw = ts.path_swaps(init, tgt)
        self.assertTrue(ts.is_solved(ts.replay(gp, init, sw), tgt))
        self.assertIsNone(ts.optimal_swaps(gp, init, tgt, max_states=60_000))


if __name__ == "__main__":
    unittest.main()
