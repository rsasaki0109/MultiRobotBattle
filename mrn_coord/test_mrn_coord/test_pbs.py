"""Tests for Priority-Based Search (PBS) and its windowed variant.

PBS branches on priority *orderings* rather than constraints. The load-bearing
properties: it returns collision-free plans, it resolves the order-sensitivity
that fixed-order prioritized planning cannot, and — windowed — it resolves only
the conflicts inside the lookahead horizon (what RHCR relies on).
"""

import unittest

from mrn_coord.mapf import GridWorld, cbs, pbs, prioritized_planning
from mrn_coord.mapf.conflicts import detect_first_conflict


class TestPBS(unittest.TestCase):
    def test_resolves_order_sensitivity_that_pp_cannot(self):
        # A 1-wide corridor (row 1); row 0 is blocked except the alcove (2, 0),
        # so b has no detour. Agent a's goal (2, 1) sits in b's only corridor:
        # planned first (the default insertion order), a parks on its goal and
        # blocks b forever, so fixed-order prioritized planning fails. PBS is free
        # to *reorder* — let b pass, then a steps out of the alcove — so it finds
        # the plan the bad fixed order missed.
        grid = GridWorld(5, 2, blocked=frozenset({(0, 0), (1, 0), (3, 0), (4, 0)}))
        agents = {"a": ((2, 0), (2, 1)), "b": ((0, 1), (4, 1))}

        self.assertIsNone(prioritized_planning(grid, agents))   # default order [a, b]
        sol = pbs(grid, agents)
        self.assertIsNotNone(sol)
        self.assertIsNone(detect_first_conflict(sol.paths))
        for agent, (start, goal) in agents.items():
            self.assertEqual(sol.paths[agent][0], start)
            self.assertEqual(sol.paths[agent][-1], goal)

    def test_head_on_swap_is_collision_free(self):
        # Two agents swap ends of a corridor; one must step aside a row.
        grid = GridWorld(5, 3)
        agents = {"a": ((0, 1), (4, 1)), "b": ((4, 1), (0, 1))}
        sol = pbs(grid, agents)
        self.assertIsNotNone(sol)
        self.assertIsNone(detect_first_conflict(sol.paths))

    def test_agrees_with_cbs_on_solvability(self):
        # Four-way crossing: PBS need not match CBS's optimal cost, but it must
        # find a collision-free plan wherever CBS does, and never cheaper than the
        # optimum.
        grid = GridWorld(7, 7)
        agents = {"0": ((0, 3), (6, 3)), "1": ((6, 3), (0, 3)),
                  "2": ((3, 0), (3, 6)), "3": ((3, 6), (3, 0))}
        opt = cbs(grid, agents)
        sol = pbs(grid, agents)
        self.assertIsNotNone(opt)
        self.assertIsNotNone(sol)
        self.assertIsNone(detect_first_conflict(sol.paths))
        self.assertGreaterEqual(sol.cost, opt.cost)

    def test_deterministic(self):
        grid = GridWorld(7, 7)
        agents = {"0": ((0, 3), (6, 3)), "1": ((6, 3), (0, 3)),
                  "2": ((3, 0), (3, 6)), "3": ((3, 6), (3, 0))}
        a, b = pbs(grid, agents), pbs(grid, agents)
        self.assertEqual(a.paths, b.paths)

    def test_window_resolves_within_horizon(self):
        # Windowed PBS's contract is one-directional: whatever it returns is
        # conflict-free *inside* the window (it may, being incomplete, return
        # nothing on a hard instance — which is why RHCR keeps a PIBT fallback).
        # A 2-agent head-on is always resolvable, so assert both here.
        grid = GridWorld(7, 3)
        agents = {"a": ((0, 1), (6, 1)), "b": ((6, 1), (0, 1))}
        for w in (2, 4, 6):
            sol = pbs(grid, agents, window=w)
            self.assertIsNotNone(sol, f"window={w}")
            self.assertIsNone(detect_first_conflict(sol.paths, window=w),
                              f"window={w}")


if __name__ == "__main__":
    unittest.main()
