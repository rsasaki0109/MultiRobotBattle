"""Tests for the MAPF core: grid, space-time A*, conflicts, CBS, prioritized."""

import unittest

from mrn_coord.mapf import (
    EdgeConflict,
    GridWorld,
    Solution,
    VertexConflict,
    cbs,
    cell_at,
    detect_first_conflict,
    makespan,
    manhattan,
    pad_paths,
    plan_path,
    prioritized_planning,
    sum_of_costs,
)


class TestGrid(unittest.TestCase):
    def test_bounds_and_blocking(self):
        grid = GridWorld(3, 3, blocked={(1, 1)})
        self.assertTrue(grid.is_free((0, 0)))
        self.assertFalse(grid.is_free((1, 1)))
        self.assertFalse(grid.is_free((3, 0)))
        self.assertFalse(grid.is_free((-1, 0)))

    def test_neighbors_includes_wait_and_filters_blocked(self):
        grid = GridWorld(3, 3, blocked={(1, 0)})
        ns = set(grid.neighbors((0, 0)))
        self.assertIn((0, 0), ns)        # wait
        self.assertIn((0, 1), ns)
        self.assertNotIn((1, 0), ns)     # blocked
        self.assertNotIn((-1, 0), ns)    # out of bounds

    def test_rejects_nonpositive_dims(self):
        for bad in ((0, 3), (3, -1)):
            with self.assertRaises(ValueError):
                GridWorld(*bad)


class TestSpaceTimeAStar(unittest.TestCase):
    def test_straight_line(self):
        grid = GridWorld(5, 1)
        path = plan_path(grid, (0, 0), (4, 0))
        self.assertEqual(path, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
        self.assertEqual(len(path) - 1, 4)

    def test_start_equals_goal(self):
        grid = GridWorld(3, 3)
        self.assertEqual(plan_path(grid, (1, 1), (1, 1)), [(1, 1)])

    def test_detours_around_obstacle(self):
        # Wall at x=1 for y in {0,1} forces a detour through (1,2).
        grid = GridWorld(3, 3, blocked={(1, 0), (1, 1)})
        path = plan_path(grid, (0, 0), (2, 0))
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (2, 0))
        self.assertIn((1, 2), path)       # had to go around the top
        self.assertEqual(manhattan((0, 0), (2, 0)), 2)
        self.assertGreater(len(path) - 1, 2)

    def test_unreachable_returns_none(self):
        # Fully wall off the goal column.
        grid = GridWorld(3, 3, blocked={(1, 0), (1, 1), (1, 2)})
        self.assertIsNone(plan_path(grid, (0, 0), (2, 0)))

    def test_vertex_constraint_forces_wait(self):
        grid = GridWorld(3, 1)
        # Forbid being at (1,0) at t=1, so the agent must wait one step.
        path = plan_path(grid, (0, 0), (2, 0), vertex_constraints={((1, 0), 1)})
        self.assertIsNotNone(path)
        self.assertEqual(path[-1], (2, 0))
        self.assertEqual(len(path) - 1, 3)        # one extra step vs optimal 2
        self.assertNotEqual(path[1], (1, 0))      # not at (1,0) at t=1

    def test_edge_constraint_blocks_transition(self):
        grid = GridWorld(3, 1)
        # Forbid the move (0,0)->(1,0) arriving at t=1.
        path = plan_path(
            grid, (0, 0), (2, 0), edge_constraints={((0, 0), (1, 0), 1)}
        )
        self.assertIsNotNone(path)
        self.assertFalse(path[0] == (0, 0) and path[1] == (1, 0))

    def test_goal_constraint_delays_settling(self):
        grid = GridWorld(3, 1)
        # Goal blocked at t=2 -> must arrive later / wait.
        path = plan_path(grid, (0, 0), (2, 0), vertex_constraints={((2, 0), 2)})
        self.assertIsNotNone(path)
        self.assertEqual(path[-1], (2, 0))
        self.assertNotEqual(cell_at(path, 2), (2, 0))


class TestConflicts(unittest.TestCase):
    def test_vertex_conflict(self):
        paths = {"a": [(0, 0), (1, 0), (2, 0)], "b": [(2, 0), (1, 0), (0, 0)]}
        c = detect_first_conflict(paths)
        self.assertIsInstance(c, VertexConflict)
        self.assertEqual(c.cell, (1, 0))
        self.assertEqual(c.time, 1)

    def test_edge_swap_conflict(self):
        paths = {"a": [(0, 0), (1, 0)], "b": [(1, 0), (0, 0)]}
        c = detect_first_conflict(paths)
        self.assertIsInstance(c, EdgeConflict)
        self.assertEqual(c.time, 1)
        self.assertEqual({c.cell_a, c.cell_b}, {(0, 0), (1, 0)})

    def test_no_conflict(self):
        paths = {"a": [(0, 0), (0, 1)], "b": [(2, 0), (2, 1)]}
        self.assertIsNone(detect_first_conflict(paths))

    def test_cell_at_holds_goal(self):
        path = [(0, 0), (1, 0)]
        self.assertEqual(cell_at(path, 5), (1, 0))

    def test_stay_at_goal_collision_is_detected(self):
        # 'a' parks on (1,0); 'b' arrives there later -> vertex conflict.
        paths = {"a": [(1, 0)], "b": [(0, 0), (1, 0)]}
        c = detect_first_conflict(paths)
        self.assertIsInstance(c, VertexConflict)
        self.assertEqual(c.cell, (1, 0))
        self.assertEqual(c.time, 1)


class TestCBS(unittest.TestCase):
    def test_parallel_is_optimal(self):
        grid = GridWorld(5, 2)
        agents = {"a": ((0, 0), (4, 0)), "b": ((0, 1), (4, 1))}
        sol = cbs(grid, agents)
        self.assertIsNotNone(sol)
        self.assertIsNone(detect_first_conflict(sol.paths))
        self.assertEqual(sol.cost, 8)        # 4 + 4, no interference

    def test_crossing_resolves(self):
        grid = GridWorld(5, 5)
        agents = {"a": ((0, 2), (4, 2)), "b": ((2, 0), (2, 4))}
        sol = cbs(grid, agents)
        self.assertIsNotNone(sol)
        self.assertIsNone(detect_first_conflict(sol.paths))
        # base cost is 8; resolving the center crossing costs at least one wait
        self.assertGreaterEqual(sol.cost, 9)
        self.assertEqual(sol.paths["a"][-1], (4, 2))
        self.assertEqual(sol.paths["b"][-1], (2, 4))

    def test_swap_through_passing_row(self):
        grid = GridWorld(3, 2)
        agents = {"a": ((0, 0), (2, 0)), "b": ((2, 0), (0, 0))}
        sol = cbs(grid, agents)
        self.assertIsNotNone(sol)
        self.assertIsNone(detect_first_conflict(sol.paths))

    def test_unsolvable_corridor_returns_none(self):
        # 1-wide corridor, agents must swap ends -> impossible.
        grid = GridWorld(3, 1)
        agents = {"a": ((0, 0), (2, 0)), "b": ((2, 0), (0, 0))}
        self.assertIsNone(cbs(grid, agents, max_expansions=2000))

    def test_single_agent(self):
        grid = GridWorld(4, 1)
        sol = cbs(grid, {"a": ((0, 0), (3, 0))})
        self.assertIsNotNone(sol)
        self.assertEqual(sol.cost, 3)


class TestPrioritized(unittest.TestCase):
    def test_parallel_succeeds(self):
        grid = GridWorld(5, 2)
        agents = {"a": ((0, 0), (4, 0)), "b": ((0, 1), (4, 1))}
        sol = prioritized_planning(grid, agents)
        self.assertIsNotNone(sol)
        self.assertIsNone(detect_first_conflict(sol.paths))

    def test_crossing_is_collision_free(self):
        grid = GridWorld(5, 5)
        agents = {"a": ((0, 2), (4, 2)), "b": ((2, 0), (2, 4))}
        sol = prioritized_planning(grid, agents)
        self.assertIsNotNone(sol)
        self.assertIsNone(detect_first_conflict(sol.paths))

    def test_order_changes_nothing_about_safety(self):
        grid = GridWorld(5, 5)
        agents = {"a": ((0, 2), (4, 2)), "b": ((2, 0), (2, 4))}
        sol = prioritized_planning(grid, agents, order=["b", "a"])
        self.assertIsNotNone(sol)
        self.assertIsNone(detect_first_conflict(sol.paths))

    def test_unsolvable_corridor_returns_none(self):
        grid = GridWorld(3, 1)
        agents = {"a": ((0, 0), (2, 0)), "b": ((2, 0), (0, 0))}
        self.assertIsNone(prioritized_planning(grid, agents))


class TestSolutionHelpers(unittest.TestCase):
    def test_costs_and_makespan(self):
        paths = {"a": [(0, 0), (1, 0), (2, 0)], "b": [(0, 1), (1, 1)]}
        self.assertEqual(sum_of_costs(paths), 3)   # 2 + 1
        self.assertEqual(makespan(paths), 2)

    def test_pad_paths(self):
        paths = {"a": [(0, 0), (1, 0), (2, 0)], "b": [(0, 1)]}
        padded = pad_paths(paths)
        self.assertEqual(len(padded["a"]), 3)
        self.assertEqual(len(padded["b"]), 3)
        self.assertEqual(padded["b"], [(0, 1), (0, 1), (0, 1)])

    def test_solution_makespan_property(self):
        sol = Solution(paths={"a": [(0, 0), (1, 0)]}, cost=1)
        self.assertEqual(sol.makespan, 1)


if __name__ == "__main__":
    unittest.main()
