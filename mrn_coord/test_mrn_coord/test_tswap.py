"""Tests for Offline TSWAP: constructive complete anonymous MAPF.

Offline TSWAP (Okumura & Defago, ICAPS 2022) solves the anonymous (target-
interchangeable) problem by taking an arbitrary initial assignment and repeating
one-timestep planning with target swapping until every agent is on a target. The
contracts: it is collision-free BY CONSTRUCTION, it covers the goal set, it is
complete regardless of the initial assignment, its makespan never beats the flow
optimum (it is sub-optimal), the two repair mechanisms (target swap on a settled
blocker, target rotation on a deadlock cycle) fire on the scenarios that demand
them, and it is deterministic.
"""

import random
import unittest

from mrn_coord.mapf import GridWorld
from mrn_coord.mapf.conflicts import detect_first_conflict
from mrn_coord.mapf.flow import anonymous_makespan
from mrn_coord.mapf.tswap import tswap


def _inst(w, h, n, seed):
    rng = random.Random(seed)
    free = [(x, y) for x in range(w) for y in range(h)]
    rng.shuffle(free)
    return GridWorld(w, h), free[:n], free[n:2 * n]


class TestTswap(unittest.TestCase):
    def test_collision_free_and_covers_goals(self):
        # Every random instance is solved, pairwise collision-free, starts where
        # asked, and ends on the GOAL SET (anonymous coverage).
        for seed in range(20):
            grid, starts, goals = _inst(6, 6, 4, seed)
            paths = tswap(grid, starts, goals)
            self.assertIsNotNone(paths, f"seed={seed}")
            self.assertIsNone(detect_first_conflict(paths), f"seed={seed}")
            self.assertEqual(sorted(p[-1] for p in paths.values()),
                             sorted(goals))
            for i, s in enumerate(starts):
                self.assertEqual(paths[i][0], s)

    def test_never_beats_the_flow_optimum(self):
        # Sub-optimal but sound: TSWAP's makespan is >= the anonymous optimum
        # computed by flow, and equals it on at least some instances.
        equal = 0
        for seed in range(15):
            grid, starts, goals = _inst(5, 5, 3, seed)
            st = {}
            tswap(grid, starts, goals, stats=st)
            opt = anonymous_makespan(grid, starts, goals)
            self.assertGreaterEqual(st["makespan"], opt[1], f"seed={seed}")
            equal += int(st["makespan"] == opt[1])
        self.assertGreater(equal, 0)

    def test_complete_for_arbitrary_assignment(self):
        # Completeness does not rely on the matching: a deliberately reversed
        # (bad) assignment is still repaired to a valid collision-free solution.
        for seed in range(10):
            grid, starts, goals = _inst(6, 6, 4, seed)
            paths = tswap(grid, starts, goals,
                          assignment=list(reversed(range(4))))
            self.assertIsNotNone(paths, f"seed={seed}")
            self.assertIsNone(detect_first_conflict(paths), f"seed={seed}")
            self.assertEqual(sorted(p[-1] for p in paths.values()),
                             sorted(goals))

    def test_target_swap_fires_on_settled_blocker(self):
        # An agent must pass two agents sitting on their own targets in a 1-wide
        # corridor -> exactly the target-SWAP branch fires, no rotation.
        grid = GridWorld(5, 1)
        st = {}
        paths = tswap(grid, [(0, 0), (2, 0), (3, 0)],
                      [(4, 0), (2, 0), (3, 0)], assignment=[0, 1, 2], stats=st)
        self.assertEqual(st["swaps"], 2)
        self.assertEqual(st["rotations"], 0)
        self.assertIsNone(detect_first_conflict(paths))
        self.assertEqual(sorted(p[-1] for p in paths.values()),
                         [(2, 0), (3, 0), (4, 0)])

    def test_target_rotation_fires_on_head_on_deadlock(self):
        # Two agents crossing a 1-wide corridor head-on -> the deadlock cycle is
        # detected and the targets ROTATE (no settled-blocker swap).
        grid = GridWorld(5, 1)
        st = {}
        paths = tswap(grid, [(0, 0), (4, 0)], [(4, 0), (0, 0)],
                      assignment=[0, 1], stats=st)
        self.assertGreaterEqual(st["rotations"], 1)
        self.assertEqual(st["swaps"], 0)
        self.assertIsNone(detect_first_conflict(paths))
        self.assertEqual(sorted(p[-1] for p in paths.values()),
                         [(0, 0), (4, 0)])

    def test_deterministic(self):
        grid, starts, goals = _inst(6, 6, 4, 3)
        runs = []
        for _ in range(2):
            st = {}
            tswap(grid, starts, goals, stats=st)
            runs.append((st["swaps"], st["rotations"], st["makespan"],
                         st["final_assignment"]))
        self.assertEqual(runs[0], runs[1])

    def test_unreachable_goal_returns_none(self):
        grid = GridWorld(3, 3, blocked={(1, 0), (1, 1), (1, 2)})
        # goal (2, 2) is unreachable from start (0, 0)
        self.assertIsNone(tswap(grid, [(0, 0)], [(2, 2)]))


if __name__ == "__main__":
    unittest.main()
