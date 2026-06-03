"""Tests for Token Passing (Ma et al. 2017) lifelong MAPF.

Token Passing commits full space-time paths into a shared reservation token, so
its contracts are: motion is collision-free *by construction* (the defining
property — no per-step rule, no fallback), runs are deterministic, it stays live
and matches the PIBT/RHCR baselines on a well-formed (roomy) warehouse, and it
parks idle agents on home endpoints disjoint from the task endpoints.
"""

import unittest

from mrn_coord.lifelong import (
    TaskStream,
    make_warehouse,
    run_lifelong,
    run_rhcr,
    run_token_passing,
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


def _well_formed(rows, cols, aisle, agents):
    """A warehouse with parking homes disjoint from the task endpoints."""
    grid, eps = make_warehouse(rows=rows, cols=cols, aisle=aisle)
    n = min(agents, len(eps) // 2)
    homes = {f"r{i}": eps[i] for i in range(n)}
    tasks = eps[n:]
    return grid, homes, tasks


class TestTokenPassing(unittest.TestCase):
    def test_collision_free_by_construction(self):
        # The defining property: reservations make motion collision-free on every
        # map -- including a cramped one where the planner stalls.
        for rows, cols, aisle, agents, steps, hz in (
            (3, 4, 2, 8, 120, None), (2, 3, 1, 5, 40, 12),
        ):
            grid, homes, tasks = _well_formed(rows, cols, aisle, agents)
            res = run_token_passing(grid, dict(homes), TaskStream(list(tasks)),
                                    max_steps=steps, allocator="hungarian",
                                    homes=homes, horizon=hz, keep_history=True)
            self.assertTrue(_no_collisions(res.history),
                            f"{rows}x{cols} aisle={aisle} collided")

    def test_matches_baselines_when_well_formed(self):
        # On a roomy warehouse Token Passing is live (never blocked) and serves
        # exactly as many tasks as both PIBT and RHCR.
        grid, homes, tasks = _well_formed(3, 4, 2, 8)
        kw = dict(max_steps=120, allocator="hungarian")
        pibt = run_lifelong(grid, dict(homes), TaskStream(list(tasks)), **kw)
        rhcr = run_rhcr(grid, dict(homes), TaskStream(list(tasks)),
                        window=10, replan_period=2, solver="pbs", **kw)
        tp = run_token_passing(grid, dict(homes), TaskStream(list(tasks)),
                               homes=homes, **kw)
        self.assertEqual(tp.blocked, 0)
        self.assertEqual(tp.completed, pibt.completed)
        self.assertEqual(tp.completed, rhcr.completed)

    def test_stalls_when_not_well_formed(self):
        # In cramped single-aisle corridors the well-formed property fails: the
        # reservation planner gets blocked and serves fewer tasks than greedy
        # PIBT -- yet never collides.
        grid, homes, tasks = _well_formed(2, 3, 1, 5)
        kw = dict(max_steps=40, allocator="hungarian")
        pibt = run_lifelong(grid, dict(homes), TaskStream(list(tasks)), **kw)
        tp = run_token_passing(grid, dict(homes), TaskStream(list(tasks)),
                               homes=homes, horizon=12, keep_history=True, **kw)
        self.assertGreater(tp.blocked, 0)
        self.assertLess(tp.completed, pibt.completed)
        self.assertTrue(_no_collisions(tp.history))

    def test_deterministic(self):
        grid, homes, tasks = _well_formed(3, 4, 2, 8)
        kw = dict(max_steps=120, allocator="hungarian", homes=homes)
        a = run_token_passing(grid, dict(homes), TaskStream(list(tasks)), **kw)
        b = run_token_passing(grid, dict(homes), TaskStream(list(tasks)), **kw)
        self.assertEqual(a.as_dict(), b.as_dict())
        self.assertEqual((a.blocked, a.replans), (b.blocked, b.replans))

    def test_idle_agents_park_at_home(self):
        # With more homes than tasks, some agents stay idle; they must rest on
        # their home endpoint, never colliding.
        grid, eps = make_warehouse(rows=3, cols=4, aisle=2)
        homes = {f"r{i}": eps[i] for i in range(6)}
        tasks = eps[6:]
        # only two open tasks at a time, so most of the six agents stay idle
        res = run_token_passing(grid, dict(homes), TaskStream(list(tasks)),
                                max_steps=60, allocator="hungarian", open_tasks=2,
                                homes=homes, keep_history=True)
        self.assertTrue(_no_collisions(res.history))
        final = res.history[-1]
        idle_at_home = sum(1 for a in homes if final[a] == homes[a])
        self.assertGreater(idle_at_home, 0)


if __name__ == "__main__":
    unittest.main()
