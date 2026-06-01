"""Tests for executing a discrete MAPF plan in the continuous world.

The contract is the plan-vs-reality story: free-running pursuit (no schedule)
can collide, while the Temporal-Plan-Graph execution preserves the discrete
coordination and is collision-free and successful — at the cost of some makespan
stretch. Needs ``mrn_coord`` (the planners); each test imports it at run time and
skips cleanly when it is not importable (so collection never touches it).
"""

import unittest


class TestMapfExecution(unittest.TestCase):
    def _run(self, controller, grid_dims=(7, 7), blocked=frozenset(),
             agents=None):
        try:
            from mrn_coord.mapf import GridWorld
            from mrn_sim.mapf_exec import execute_mapf_plan
        except ModuleNotFoundError:
            self.skipTest("mrn_coord not importable (needs colcon build)")
        grid = GridWorld(*grid_dims, blocked=blocked)
        if agents is None:
            agents = {"0": ((0, 3), (6, 3)), "1": ((6, 3), (0, 3)),
                      "2": ((3, 0), (3, 6)), "3": ((3, 6), (3, 0))}
        return execute_mapf_plan(grid, agents, solver="lacam",
                                 controller=controller)

    def test_free_pursuit_exposes_the_gap(self):
        # The discrete plan is collision-free, but running it spatially without
        # the schedule lets the discs collide at the shared centre.
        r = self._run("pursuit")
        self.assertTrue(r.solved)
        self.assertGreater(r.robot_collisions, 0)

    def test_tpg_execution_is_collision_free_and_succeeds(self):
        r = self._run("tpg")
        self.assertTrue(r.success)
        self.assertEqual(r.robot_collisions, 0)
        # honoring the schedule costs wall-clock: continuous >= discrete makespan
        self.assertGreaterEqual(r.continuous_steps, r.discrete_makespan)

    def test_dwa_execution_is_collision_free(self):
        r = self._run("dwa")
        self.assertTrue(r.success)
        self.assertEqual(r.robot_collisions, 0)

    def test_deterministic(self):
        a = self._run("tpg")
        b = self._run("tpg")
        self.assertEqual(a.as_dict(), b.as_dict())

    def test_infeasible_plan_reports_unsolved(self):
        # a wall splits the corridor: no plan exists, so nothing to execute
        r = self._run("tpg", grid_dims=(3, 1), blocked=frozenset({(1, 0)}),
                      agents={"a": ((0, 0), (2, 0))})
        self.assertFalse(r.solved)


if __name__ == "__main__":
    unittest.main()
