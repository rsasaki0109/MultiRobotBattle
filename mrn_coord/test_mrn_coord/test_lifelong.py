"""Tests for lifelong / online MAPF: collision-freedom, throughput, determinism."""

import random
import unittest

from mrn_coord.lifelong import (
    LifelongResult,
    TaskStream,
    make_warehouse,
    run_lifelong,
)
from mrn_coord.mapf import GridWorld


def _no_collisions(history) -> bool:
    """No two agents share a cell, and no pair swaps, between consecutive steps."""
    for prev, cur in zip(history, history[1:]):
        if len(set(cur.values())) != len(cur):
            return False                      # vertex collision
        for a in cur:
            for b in cur:
                if a < b and cur[a] == prev[b] and cur[b] == prev[a]:
                    return False              # edge / swap collision
    return True


class TestTaskStream(unittest.TestCase):
    def test_round_robin_and_avoid(self):
        s = TaskStream([(0, 0), (1, 1), (2, 2)])
        self.assertEqual(s.next_goal(), (0, 0))
        self.assertEqual(s.next_goal(), (1, 1))
        # avoid skips the cell the agent already stands on
        self.assertEqual(s.next_goal(avoid=(2, 2)), (0, 0))

    def test_deterministic(self):
        a = TaskStream([(0, 0), (1, 1)])
        b = TaskStream([(0, 0), (1, 1)])
        self.assertEqual([a.next_goal() for _ in range(5)],
                         [b.next_goal() for _ in range(5)])


class TestLifelong(unittest.TestCase):
    def test_completes_tasks_and_is_collision_free(self):
        grid = GridWorld(8, 8)
        starts = {"a": (0, 0), "b": (7, 0), "c": (0, 7), "d": (7, 7)}
        stream = TaskStream([(4, 4), (1, 6), (6, 1), (3, 3), (5, 5)])
        res = run_lifelong(grid, starts, stream, horizon=16, max_steps=120,
                           keep_history=True)
        self.assertIsInstance(res, LifelongResult)
        self.assertGreater(res.completed, 0)
        self.assertTrue(_no_collisions(res.history))
        self.assertEqual(sum(res.per_agent.values()), res.completed)

    def test_collision_free_through_a_doorway(self):
        # a 1-cell gap in a wall forces agents to take turns -> stresses the
        # reservation logic; movement must still never collide.
        blocked = {(3, y) for y in range(7) if y != 3}
        grid = GridWorld(7, 7, blocked=frozenset(blocked))
        starts = {"a": (0, 1), "b": (0, 5), "c": (6, 1), "d": (6, 5)}
        stream = TaskStream([(6, 3), (0, 3), (6, 0), (0, 6)])
        res = run_lifelong(grid, starts, stream, horizon=20, max_steps=150,
                           keep_history=True)
        self.assertTrue(_no_collisions(res.history))
        self.assertGreater(res.completed, 0)

    def test_throughput_is_deterministic(self):
        grid, endpoints = make_warehouse(rows=2, cols=3)
        starts = {f"r{i}": endpoints[i] for i in range(4)}
        r1 = run_lifelong(grid, dict(starts), TaskStream(list(endpoints)),
                          max_steps=100)
        r2 = run_lifelong(grid, dict(starts), TaskStream(list(endpoints)),
                          max_steps=100)
        self.assertEqual(r1.as_dict(), r2.as_dict())
        self.assertAlmostEqual(r1.throughput, r1.completed / 100)

    def test_more_agents_complete_more(self):
        grid, endpoints = make_warehouse(rows=2, cols=3)
        stream_pool = list(endpoints)

        def run(n):
            starts = {f"r{i}": endpoints[i % len(endpoints)] for i in range(n)}
            return run_lifelong(grid, starts, TaskStream(list(stream_pool)),
                                max_steps=120).completed

        few, many = run(2), run(6)
        self.assertGreaterEqual(many, few)

    def test_fleet_allocator_beats_round_robin(self):
        # At fleet scale (40 AMRs in a 4x6 warehouse, the README hero), routing
        # idle robots to *near* tasks is the whole game: a cost-aware allocator
        # must clear far more tasks than geometry-blind round-robin. This guards
        # the contrast itself, so a change that quietly neutralizes the allocator
        # fails even if every per-case throughput baseline is also nudged.
        grid, endpoints = make_warehouse(rows=4, cols=6, aisle=1)
        starts = {f"r{i}": endpoints[i] for i in range(40)}

        def run(allocator):
            return run_lifelong(grid, dict(starts), TaskStream(list(endpoints)),
                                max_steps=60, allocator=allocator).completed

        stream = run("stream")
        for allocator in ("hungarian", "auction"):
            served = run(allocator)
            # a comfortable margin below the observed ~6.5x, not a tight latch
            self.assertGreater(served, 3 * stream,
                               f"{allocator} lost its lead over round-robin")

    def test_liveness_bounded_under_distinct_goals(self):
        # The repo claims arrival-driven goals prevent a permanent standoff. That
        # claim was never tested. Probe it adversarially: a battery of densely
        # packed warehouses, random start permutations, random *distinct*-endpoint
        # task streams (one station per cell, the realistic regime). The liveness
        # contract is that no run stalls for long — `longest_stall` (consecutive
        # steps finishing zero tasks) stays well bounded. Over ~3000 wider seeds
        # the worst seen is 8 steps; gate at 15 so a regression that introduces a
        # genuine livelock (e.g. a priority/tie-break change) fails here.
        shapes = [(2, 2, 1), (2, 3, 1), (3, 3, 1), (3, 4, 1), (2, 2, 2), (3, 4, 2)]
        worst = 0
        for rows, cols, aisle in shapes:
            grid, endpoints = make_warehouse(rows=rows, cols=cols, aisle=aisle)
            for seed in range(12):
                rng = random.Random(seed * 101 + 7)
                n = len(endpoints) if seed % 2 else max(2, len(endpoints) - 2)
                order = endpoints[:]
                rng.shuffle(order)
                starts = {f"r{i}": order[i] for i in range(n)}
                pool = endpoints[:]            # distinct endpoints, shuffled
                rng.shuffle(pool)
                for allocator in ("stream", "hungarian"):
                    res = run_lifelong(grid, dict(starts), TaskStream(list(pool)),
                                       max_steps=120, allocator=allocator)
                    worst = max(worst, res.longest_stall())
        self.assertLess(worst, 15, f"a distinct-goal run stalled {worst} steps")

    def test_duplicate_goals_break_liveness(self):
        # The precise boundary of the claim above (a characterization tripwire,
        # not an endorsement). Funnel *every* task to a single contested cell with
        # cost-aware allocation: the agent already standing there is never assigned
        # its own cell, so it idles and squats while the others pile into the
        # corner around it. PIBT's push then has nowhere to shove the squatter, and
        # the cluster deadlocks permanently — zero tasks ever complete. This is the
        # random-tie-break escape the deterministic engine deliberately forgoes
        # (see docs/coordination.md). If a future change makes this complete tasks,
        # the engine gained a livelock escape and this test should assert that win.
        grid, endpoints = make_warehouse(rows=2, cols=2, aisle=1)
        starts = {f"r{i}": endpoints[i] for i in range(4)}
        res = run_lifelong(grid, starts, TaskStream([endpoints[0]]),
                           max_steps=60, allocator="hungarian")
        self.assertEqual(res.completed, 0)
        self.assertGreater(res.longest_stall(), 50)

    def test_warehouse_layout(self):
        grid, endpoints = make_warehouse(rows=2, cols=2, aisle=1)
        self.assertTrue(len(endpoints) > 0)
        self.assertTrue(all(grid.is_free(c) for c in endpoints))
        # shelves are actually blocked
        self.assertFalse(grid.is_free((1, 1)))


if __name__ == "__main__":
    unittest.main()
