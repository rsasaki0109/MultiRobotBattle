"""Tests for Push and Swap (Luna & Bekris, 2011), push_and_rotate's ancestor.

Push and Swap uses two primitives only -- push and swap, no rotate. It is
complete and valid by construction wherever there is slack (matching
push_and_rotate on sparse maps), but it stalls on cyclic, slack-free regions (a
packed rectangle, a full ring) where only a cycle rotation can advance an agent
past another -- the exact completeness gap Push and Rotate closes. The contracts:
it solves the sparse cases push_and_rotate does (collision-free, on-goal); it
fails the single-blank packed case push_and_rotate solves; and its swap primitive
exchanges two agents around a degree-3 hub.
"""

import random
import unittest

from mrn_coord.mapf import GridWorld
from mrn_coord.mapf.conflicts import detect_first_conflict
from mrn_coord.mapf.push_and_rotate import push_and_rotate
from mrn_coord.mapf.push_and_swap import push_and_swap


def _instance(w, h, n, seed):
    rng = random.Random(seed)
    free = [(x, y) for x in range(w) for y in range(h)]
    rng.shuffle(free)
    return GridWorld(w, h), {i: (free[i], free[n + i]) for i in range(n)}


def _packed(w, h, blanks, seed):
    rng = random.Random(seed * 131 + w * 7 + h * 3 + blanks)
    grid = GridWorld(w, h)
    cells = [(x, y) for y in range(h) for x in range(w)]
    goal = cells[:len(cells) - blanks]
    n = len(goal)
    pos = {i: goal[i] for i in range(n)}
    occ = {goal[i]: i for i in range(n)}
    empt = set(cells) - set(goal)

    def nb(c):
        return [d for d in ((c[0] + 1, c[1]), (c[0] - 1, c[1]),
                            (c[0], c[1] + 1), (c[0], c[1] - 1)) if grid.is_free(d)]

    for _ in range(30 * n):
        e = rng.choice(sorted(empt))
        cand = [c for c in nb(e) if c in occ]
        if not cand:
            continue
        c = rng.choice(cand)
        a = occ.pop(c)
        occ[e] = a
        pos[a] = e
        empt.discard(e)
        empt.add(c)
    return grid, {i: (pos[i], goal[i]) for i in range(n)}


def _valid(sol, agents):
    return (detect_first_conflict(sol.paths) is None
            and all(sol.paths[k][-1] == g for k, (s, g) in agents.items()))


class TestPushAndSwap(unittest.TestCase):
    def test_empty_and_trivial(self):
        self.assertEqual(push_and_swap(GridWorld(3, 3), {}).cost, 0)
        g = GridWorld(3, 3)
        sol = push_and_swap(g, {0: ((0, 0), (0, 0))})
        self.assertIsNotNone(sol)
        self.assertEqual(sol.paths[0][-1], (0, 0))

    def test_slack_matches_push_and_rotate(self):
        # On sparse maps the swap-only core is already complete: it solves exactly
        # what push_and_rotate solves, collision-free and on-goal.
        for w, h, n in ((4, 4, 4), (5, 5, 5), (6, 6, 6)):
            for seed in range(10):
                grid, ag = _instance(w, h, n, seed)
                sps = push_and_swap(grid, ag)
                spr = push_and_rotate(grid, ag)
                self.assertEqual(sps is not None, spr is not None,
                                 f"{w}x{h} seed={seed}")
                if sps is not None:
                    self.assertTrue(_valid(sps, ag))

    def test_single_blank_packed_gap(self):
        # The tightest 15-puzzle regime: push_and_rotate's tracked BFS solves it,
        # the bare push/swap core cannot (no rotate to turn the cyclic dependency).
        any_gap = False
        for w, h in ((4, 4), (5, 5)):
            for seed in range(8):
                grid, ag = _packed(w, h, 1, seed)
                sps = push_and_swap(grid, ag)
                spr = push_and_rotate(grid, ag)
                self.assertIsNotNone(spr)        # rotate closes it
                self.assertIsNone(sps)           # swap-only stalls
                any_gap = True
                if sps is not None:              # never, but stay honest
                    self.assertTrue(_valid(sps, ag))
        self.assertTrue(any_gap)

    def test_swap_primitive_fires_on_hub(self):
        # T-junction: hub (1,1) has degree 3; two agents must exchange ends, which
        # push alone cannot do -- only the swap primitive, rotating them around it.
        grid = GridWorld(3, 2, frozenset({(0, 0), (2, 0)}))
        ag = {0: ((0, 1), (2, 1)), 1: ((2, 1), (0, 1))}
        sol = push_and_swap(grid, ag)
        self.assertIsNotNone(sol)
        self.assertTrue(_valid(sol, ag))


if __name__ == "__main__":
    unittest.main()
