"""Tests for point-to-point navigation: occupancy, planning, and a drive loop."""

import importlib.util
import math
import unittest

from mrn_sim import Obstacle, Robot, World


@unittest.skipUnless(
    importlib.util.find_spec("mrn_coord") is not None, "mrn_coord not available"
)
class TestNavigate(unittest.TestCase):
    def _world(self):
        return World(12.0, 8.0, {}, [Obstacle(6.0, 4.0, 1.5)])

    def test_world_cell_round_trip(self):
        from mrn_sim.navigate import cell_to_world, world_to_cell
        self.assertEqual(world_to_cell((2.6, 2.6), 0.5), (5, 5))
        self.assertEqual(cell_to_world((5, 5), 0.5), (2.75, 2.75))

    def test_occupancy_blocks_obstacle_cells(self):
        from mrn_sim.navigate import occupancy_from_world, world_to_cell
        grid = occupancy_from_world(self._world(), cell_size=0.5, inflation=0.3)
        self.assertFalse(grid.is_free(world_to_cell((6.0, 4.0), 0.5)))   # obstacle center
        self.assertTrue(grid.is_free(world_to_cell((1.0, 1.0), 0.5)))    # open space

    def test_plan_avoids_obstacle(self):
        from mrn_sim.navigate import occupancy_from_world, plan_world_path, world_to_cell
        path = plan_world_path(self._world(), (1.0, 4.0), (11.0, 4.0), cell_size=0.5)
        self.assertIsNotNone(path)
        self.assertAlmostEqual(path[-1][0], 11.0)
        self.assertAlmostEqual(path[-1][1], 4.0)
        # every waypoint clears the obstacle (planned with inflation)
        grid = occupancy_from_world(self._world(), cell_size=0.5, inflation=0.3)
        for (wx, wy) in path[:-1]:
            self.assertTrue(grid.is_free(world_to_cell((wx, wy), 0.5)))

    def test_plan_none_when_goal_blocked(self):
        from mrn_sim.navigate import plan_world_path
        # goal inside the obstacle
        self.assertIsNone(plan_world_path(self._world(), (1.0, 1.0), (6.0, 4.0)))

    def test_drive_to_goal(self):
        # plan + pure-pursuit drive a unicycle robot to the goal around the obstacle
        from mrn_coord.mapf.path_follower import pure_pursuit
        from mrn_sim.navigate import plan_world_path
        from mrn_sim.world import step

        world = World(12.0, 8.0, {"r": Robot("r", (1.0, 4.0, 0.0), 0.25)},
                      [Obstacle(6.0, 4.0, 1.5)])
        goal = (11.0, 4.0)
        path = plan_world_path(world, (1.0, 4.0), goal, cell_size=0.5)
        self.assertIsNotNone(path)
        for _ in range(400):
            pose = world.robots["r"].pose
            v, omega, reached = pure_pursuit(pose, path, lookahead=0.9,
                                             v_nominal=1.2, goal_tolerance=0.3)
            if reached:
                break
            world = step(world, {"r": (v, omega)}, 0.1)
        x, y, _ = world.robots["r"].pose
        self.assertLess(math.hypot(x - goal[0], y - goal[1]), 0.5)   # reached
        # never entered the obstacle
        clr = math.hypot(x - 6.0, y - 4.0) - 1.5 - 0.25
        self.assertGreater(clr, -1e-6)


    def test_reciprocal_multirobot_navigation(self):
        # parallel lanes plus one robot going the other way that must weave
        # between them: they reach their goals while never colliding. (Reactive
        # avoidance is collision-free but not deadlock-free, so the scenario
        # avoids a symmetric all-cross chokepoint.)
        from mrn_sim.navigate import navigate_step, plan_world_path

        starts = {"1": (1.5, 3.0), "2": (1.5, 7.0), "3": (18.5, 5.0)}
        goals = {"1": (18.5, 3.0), "2": (18.5, 7.0), "3": (1.5, 5.0)}
        robots = {a: Robot(a, (p[0], p[1], 0.0), 0.25) for a, p in starts.items()}
        world = World(20.0, 10.0, robots, [])
        paths = {a: plan_world_path(world, starts[a], goals[a], cell_size=0.5,
                                    inflation=0.35) for a in starts}
        for a in starts:
            self.assertIsNotNone(paths[a])

        min_pair = float("inf")
        all_reached = False
        for _ in range(500):
            world, reached = navigate_step(world, paths, dt=0.1, w_mutual=1.6,
                                           mutual_radius=1.6)
            ps = [r.pose for r in world.robots.values()]
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    dd = math.hypot(ps[i][0] - ps[j][0], ps[i][1] - ps[j][1])
                    min_pair = min(min_pair, dd)
            if all(reached.values()):
                all_reached = True
                break
        self.assertTrue(all_reached)
        # never came within two robot radii of each other
        self.assertGreater(min_pair, 0.5)


if __name__ == "__main__":
    unittest.main()
