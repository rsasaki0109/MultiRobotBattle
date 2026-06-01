"""Tests for lifelong / online MAPF: collision-freedom, throughput, determinism."""

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

    def test_warehouse_layout(self):
        grid, endpoints = make_warehouse(rows=2, cols=2, aisle=1)
        self.assertTrue(len(endpoints) > 0)
        self.assertTrue(all(grid.is_free(c) for c in endpoints))
        # shelves are actually blocked
        self.assertFalse(grid.is_free((1, 1)))


if __name__ == "__main__":
    unittest.main()
