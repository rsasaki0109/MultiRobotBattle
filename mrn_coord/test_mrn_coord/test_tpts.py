"""Tests for Token Passing with Task Swaps (TPTS, Ma et al. 2017, Algorithm 2).

TPTS extends Token Passing with one rule: a freshly-free agent may *steal* a
task that is assigned to a farther-away holder which has not yet reached the
pickup. Its contracts: motion stays collision-free *by construction* (the shared
reservation token, unchanged from TP), swaps fire only when enabled and never
lose a delivery, a forced swap shortens service time, and runs are deterministic.
"""

import unittest

from mrn_coord.lifelong import PickupDelivery, make_warehouse, run_tpts
from mrn_coord.mapf import GridWorld
from mrn_coord.mapf.conflicts import detect_first_conflict


def _no_collisions(history) -> bool:
    for prev, cur in zip(history, history[1:]):
        if len(set(cur.values())) != len(cur):
            return False                      # vertex collision
        for a in cur:
            for b in cur:
                if a < b and cur[a] == prev[b] and cur[b] == prev[a]:
                    return False              # edge / swap collision
    return True


def _forced_swap_instance():
    """An open 12x3 grid where exactly one beneficial steal is forced.

    r1 collects a task under it and frees up beside T0's pickup while r0 -- the
    only other free agent -- is still walking toward the farther T0 it was handed.
    """
    grid = GridWorld(12, 3, blocked=frozenset())
    homes = {"r0": (0, 1), "r1": (10, 1)}
    tasks = [
        PickupDelivery((10, 1), (6, 1)),
        PickupDelivery((7, 1), (7, 2)),
        PickupDelivery((1, 1), (1, 2)),
    ]
    return grid, homes, tasks


def _well_formed(rows, cols, aisle, agents):
    grid, eps = make_warehouse(rows=rows, cols=cols, aisle=aisle)
    n = min(agents, len(eps) // 2)
    homes = {f"r{i}": eps[i] for i in range(n)}
    pool = eps[n:]
    pairs = [(pool[i], pool[i + 1]) for i in range(0, len(pool) - 1, 2)]
    return grid, homes, pairs


class TestTpts(unittest.TestCase):
    def test_collision_free_by_construction(self):
        # The reservation token makes motion collision-free with swaps on and off.
        grid, homes, tasks = _forced_swap_instance()
        for sw in (False, True):
            res = run_tpts(grid, dict(homes),
                           [PickupDelivery(t.pickup, t.delivery) for t in tasks],
                           swaps=sw, max_steps=40, homes=homes, keep_history=True)
            self.assertTrue(_no_collisions(res.history))

    def test_forced_swap_fires_once_and_helps(self):
        # Exactly one steal, and it shortens service and the worst wait, while
        # both runs still deliver every task.
        grid, homes, tasks = _forced_swap_instance()
        mk = lambda: [PickupDelivery(t.pickup, t.delivery) for t in tasks]
        off = run_tpts(grid, dict(homes), mk(), swaps=False, max_steps=40,
                       homes=homes, keep_history=True)
        on = run_tpts(grid, dict(homes), mk(), swaps=True, max_steps=40,
                      homes=homes, keep_history=True)
        self.assertEqual(off.swaps_fired, 0)
        self.assertEqual(on.swaps_fired, 1)
        self.assertEqual(on.completed, off.completed)
        self.assertLess(on.avg_service_time, off.avg_service_time)
        self.assertLess(on.max_wait, off.max_wait)

    def test_swaps_help_on_warehouse_without_losing_deliveries(self):
        grid, homes, pairs = _well_formed(3, 4, 2, 4)
        mk = lambda: [PickupDelivery(p, d) for p, d in pairs]
        kw = dict(max_steps=60, homes=homes)
        off = run_tpts(grid, dict(homes), mk(), swaps=False, **kw)
        on = run_tpts(grid, dict(homes), mk(), swaps=True, **kw)
        self.assertGreater(on.swaps_fired, 0)
        self.assertEqual(on.completed, off.completed)
        self.assertLessEqual(on.avg_service_time, off.avg_service_time)

    def test_deliveries_complete_two_legs(self):
        # A delivery counts only after the package is collected at the pickup AND
        # carried to the delivery -- a two-leg task, not a single goal.
        grid = GridWorld(8, 3, blocked=frozenset())
        homes = {"r0": (0, 1)}
        res = run_tpts(grid, dict(homes), [PickupDelivery((6, 1), (6, 2))],
                       swaps=True, max_steps=30, homes=homes, keep_history=True)
        self.assertEqual(res.completed, 1)
        visited = [snap["r0"] for snap in res.history]
        # it collected at the pickup and then carried the package to the delivery
        self.assertIn((6, 1), visited)
        self.assertIn((6, 2), visited)
        self.assertLess(visited.index((6, 1)), visited.index((6, 2)))

    def test_deterministic(self):
        grid, homes, tasks = _forced_swap_instance()
        mk = lambda: [PickupDelivery(t.pickup, t.delivery) for t in tasks]
        a = run_tpts(grid, dict(homes), mk(), swaps=True, max_steps=40, homes=homes)
        b = run_tpts(grid, dict(homes), mk(), swaps=True, max_steps=40, homes=homes)
        self.assertEqual(a.as_dict(), b.as_dict())
        self.assertEqual((a.swaps_fired, a.blocked), (b.swaps_fired, b.blocked))


if __name__ == "__main__":
    unittest.main()
