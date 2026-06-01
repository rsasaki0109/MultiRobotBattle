"""Tests for Safe Interval Path Planning (SIPP).

The contract: SIPP is a drop-in for time-expanded A* — same minimal-time path,
same constraint vocabulary — but explores far fewer states when an agent must
wait out a reservation. So the tests check (a) it agrees with ``plan_path`` on
arrival time over many random instances with vertex *and* edge constraints,
(b) returned paths actually respect those constraints and step legally, and
(c) the headline win: a long forced wait costs SIPP a handful of states, not one
per timestep.
"""

import random
import unittest

from mrn_coord.mapf import GridWorld, plan_path, plan_sipp, prioritized_planning
from mrn_coord.mapf.conflicts import detect_first_conflict
from mrn_coord.mapf.sipp import INF, _safe_intervals


def _length(path):
    return None if path is None else len(path) - 1


def _legal_walk(grid, path):
    """Every step is a wait or a single 4-connected move onto a free cell."""
    for t, cell in enumerate(path):
        if not grid.is_free(cell):
            return False
        if t > 0:
            p = path[t - 1]
            if cell != p and abs(cell[0] - p[0]) + abs(cell[1] - p[1]) != 1:
                return False
    return True


class TestSafeIntervals(unittest.TestCase):
    def test_empty_is_one_unbounded_interval(self):
        self.assertEqual(_safe_intervals([]), [(0, INF)])

    def test_gaps_become_intervals(self):
        self.assertEqual(_safe_intervals([3, 4, 7]), [(0, 2), (5, 6), (8, INF)])

    def test_blocked_from_zero(self):
        self.assertEqual(_safe_intervals([0]), [(1, INF)])
        self.assertEqual(_safe_intervals([0, 1, 2]), [(3, INF)])

    def test_dedup_and_sort(self):
        self.assertEqual(_safe_intervals([4, 3, 4]), [(0, 2), (5, INF)])


class TestSippMatchesAStar(unittest.TestCase):
    def test_straight_line(self):
        grid = GridWorld(5, 1)
        self.assertEqual(plan_sipp(grid, (0, 0), (4, 0)),
                         [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])

    def test_start_equals_goal(self):
        self.assertEqual(plan_sipp(GridWorld(3, 3), (1, 1), (1, 1)), [(1, 1)])

    def test_blocked_start_or_goal_is_none(self):
        grid = GridWorld(3, 3, blocked={(1, 1)})
        self.assertIsNone(plan_sipp(grid, (1, 1), (0, 0)))
        self.assertIsNone(plan_sipp(grid, (0, 0), (1, 1)))

    def test_agrees_with_astar_on_random_instances(self):
        rng = random.Random(1234)
        checked = 0
        for _ in range(500):
            w = h = rng.randint(3, 6)
            blocked = frozenset(
                (rng.randrange(w), rng.randrange(h))
                for _ in range(rng.randint(0, 5)))
            grid = GridWorld(w, h, blocked)
            free = [(x, y) for x in range(w) for y in range(h)
                    if grid.is_free((x, y))]
            if len(free) < 2:
                continue
            start, goal = rng.sample(free, 2)
            vc = frozenset((rng.choice(free), rng.randint(0, 8))
                           for _ in range(rng.randint(0, 6)))
            if (start, 0) in vc:
                continue
            ec = set()
            for _ in range(rng.randint(0, 4)):
                c = rng.choice(free)
                moves = [n for n in grid.neighbors(c) if n != c]
                if moves:
                    ec.add((c, rng.choice(moves), rng.randint(1, 8)))
            ec = frozenset(ec)

            astar = plan_path(grid, start, goal, vc, ec)
            sipp = plan_sipp(grid, start, goal, vc, ec)
            # Same solvability and same optimal arrival time.
            self.assertEqual(_length(astar), _length(sipp))
            if sipp is not None:
                self.assertTrue(_legal_walk(grid, sipp))
                for t, cell in enumerate(sipp):
                    self.assertNotIn((cell, t), vc)
                    if t > 0:
                        self.assertNotIn((sipp[t - 1], cell, t), ec)
                checked += 1
        self.assertGreater(checked, 100)


class TestSippExpansions(unittest.TestCase):
    def test_long_wait_costs_few_states(self):
        # The only path to the goal passes through a cell blocked for 200 ticks;
        # time-expanded A* re-expands a state per waited tick, SIPP does not.
        grid = GridWorld(2, 1)
        vc = frozenset(((1, 0), t) for t in range(1, 201))
        astar_stats, sipp_stats = {}, {}
        astar = plan_path(grid, (0, 0), (1, 0), vc, stats=astar_stats)
        sipp = plan_sipp(grid, (0, 0), (1, 0), vc, stats=sipp_stats)
        self.assertEqual(_length(astar), _length(sipp))   # both wait then step
        self.assertLessEqual(sipp_stats["expansions"], 5)
        self.assertGreater(astar_stats["expansions"],
                           10 * sipp_stats["expansions"])


class TestSippInPrioritized(unittest.TestCase):
    def test_drop_in_low_level_stays_collision_free(self):
        # A doorway both agents want to cross; prioritized planning with the
        # SIPP low level must still produce collision-free paths.
        grid = GridWorld(5, 3, blocked={(2, 0), (2, 2)})
        agents = {"a": ((0, 1), (4, 1)), "b": ((4, 1), (0, 1))}
        sol = prioritized_planning(grid, agents, low_level=plan_sipp)
        self.assertIsNotNone(sol)
        self.assertIsNone(detect_first_conflict(sol.paths))

    def test_matches_astar_low_level_cost_on_open_grid(self):
        grid = GridWorld(6, 6)
        agents = {"a": ((0, 0), (5, 5)), "b": ((5, 0), (0, 5))}
        a = prioritized_planning(grid, agents, low_level=plan_path)
        b = prioritized_planning(grid, agents, low_level=plan_sipp)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertEqual(a.cost, b.cost)


if __name__ == "__main__":
    unittest.main()
