"""Tests for RHCR (Rolling-Horizon Collision Resolution) lifelong MAPF.

RHCR re-plans windowed MAPF every ``h`` steps and commits the next ``h``. The
contracts: motion is collision-free however the window is solved, runs are
deterministic, and the framework degenerates *exactly* to the one-step PIBT
engine when the window solver is a PIBT rollout committed one step at a time.
"""

import unittest

from mrn_coord.lifelong import (
    TaskStream,
    make_warehouse,
    run_lifelong,
    run_rhcr,
)
from mrn_coord.mapf import GridWorld


def _no_collisions(history) -> bool:
    for prev, cur in zip(history, history[1:]):
        if len(set(cur.values())) != len(cur):
            return False                      # vertex collision
        for a in cur:
            for b in cur:
                if a < b and cur[a] == prev[b] and cur[b] == prev[a]:
                    return False              # edge / swap collision
    return True


class TestRHCR(unittest.TestCase):
    def test_collision_free_every_solver(self):
        grid, endpoints = make_warehouse(rows=2, cols=3)
        starts = {f"r{i}": endpoints[i] for i in range(6)}
        for solver in ("pbs", "pp", "pibt"):
            res = run_rhcr(grid, dict(starts), TaskStream(list(endpoints)),
                           max_steps=120, window=8, replan_period=4,
                           solver=solver, keep_history=True)
            self.assertGreater(res.completed, 0, solver)
            self.assertTrue(_no_collisions(res.history),
                            f"{solver} produced a collision")

    def test_collision_free_through_a_doorway(self):
        # A 1-cell doorway forces agents to take turns: stresses the windowed
        # solver's conflict resolution. PBS should still never collide.
        blocked = {(3, y) for y in range(7) if y != 3}
        grid = GridWorld(7, 7, blocked=frozenset(blocked))
        starts = {"a": (0, 1), "b": (0, 5), "c": (6, 1), "d": (6, 5)}
        stream = TaskStream([(6, 3), (0, 3), (6, 0), (0, 6)])
        res = run_rhcr(grid, starts, stream, max_steps=150, window=10,
                       replan_period=5, solver="pbs", keep_history=True)
        self.assertTrue(_no_collisions(res.history))
        self.assertGreater(res.completed, 0)

    def test_deterministic(self):
        grid, endpoints = make_warehouse(rows=2, cols=3)
        starts = {f"r{i}": endpoints[i] for i in range(6)}
        kw = dict(max_steps=120, window=8, replan_period=4, solver="pbs",
                  allocator="hungarian")
        a = run_rhcr(grid, dict(starts), TaskStream(list(endpoints)), **kw)
        b = run_rhcr(grid, dict(starts), TaskStream(list(endpoints)), **kw)
        self.assertEqual(a.as_dict(), b.as_dict())

    def test_reduces_to_pibt_engine_at_unit_horizon(self):
        # The framework's sanity anchor: a PIBT-rollout window committed one step
        # at a time (h=1) is just the one-step PIBT engine with immediate
        # reassignment — it must reproduce run_lifelong metric-for-metric, for
        # every allocator.
        grid, endpoints = make_warehouse(rows=2, cols=3)
        starts = {f"r{i}": endpoints[i] for i in range(6)}
        for allocator in ("stream", "auction", "hungarian"):
            rhcr = run_rhcr(grid, dict(starts), TaskStream(list(endpoints)),
                            max_steps=120, window=12, replan_period=1,
                            solver="pibt", allocator=allocator)
            pibt = run_lifelong(grid, dict(starts), TaskStream(list(endpoints)),
                                max_steps=120, allocator=allocator)
            self.assertEqual(rhcr.as_dict(), pibt.as_dict(), allocator)

    def test_shorter_commit_serves_at_least_as_many(self):
        # Committing fewer steps before replanning (smaller h) reassigns finished
        # robots sooner, so throughput is non-increasing in h. Guards the
        # documented compute/throughput trade-off.
        grid, endpoints = make_warehouse(rows=2, cols=3)
        starts = {f"r{i}": endpoints[i] for i in range(6)}

        def served(h):
            return run_rhcr(grid, dict(starts), TaskStream(list(endpoints)),
                            max_steps=120, window=8, replan_period=h,
                            solver="pbs", allocator="hungarian").completed

        self.assertGreaterEqual(served(1), served(4))
        self.assertGreaterEqual(served(4), served(8))

    def test_rejects_bad_horizon(self):
        grid, endpoints = make_warehouse(rows=2, cols=3)
        starts = {"r0": endpoints[0]}
        with self.assertRaises(ValueError):
            run_rhcr(grid, starts, TaskStream(list(endpoints)),
                     max_steps=10, window=4, replan_period=5)   # h > w


if __name__ == "__main__":
    unittest.main()
