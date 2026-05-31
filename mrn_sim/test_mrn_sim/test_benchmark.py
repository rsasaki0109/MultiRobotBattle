"""Tests for the benchmark environment: scenario, runner, metrics, policy."""

import importlib.util
import math
import unittest

from mrn_sim.benchmark import BenchmarkResult, Scenario, run_scenario


class TestScenario(unittest.TestCase):
    def test_from_dict_and_world(self):
        sc = Scenario.from_dict({
            "name": "t", "width": 10.0, "height": 6.0,
            "robots": {"1": [1.0, 1.0, 0.0]},
            "obstacles": [[5.0, 3.0, 1.0]],
            "goals": {"1": [9.0, 1.0]},
        })
        w = sc.world()
        self.assertEqual(w.width, 10.0)
        self.assertIn("1", w.robots)
        self.assertEqual(len(w.obstacles), 1)
        self.assertEqual(sc.goals["1"], (9.0, 1.0))


class TestRunner(unittest.TestCase):
    def test_metrics_with_straight_policy(self):
        # one robot, no obstacles, a policy that drives straight toward the goal
        sc = Scenario(name="line", width=12.0, height=4.0,
                      robots={"r": (1.0, 2.0, 0.0)}, goals={"r": (10.0, 2.0)})

        def policy(world):
            return {"r": (1.5, 0.0)}   # drive +x at constant speed

        res = run_scenario(sc, policy, dt=0.1, max_steps=200)
        self.assertIsInstance(res, BenchmarkResult)
        self.assertTrue(res.success)
        self.assertEqual(res.goals_reached, 1)
        self.assertGreater(res.total_path_length, 8.0)   # travelled ~9 m
        self.assertEqual(res.collisions, 0)
        self.assertGreater(res.makespan_sec, 0.0)

    def test_collision_metric_counts_overlap(self):
        # two robots driven into the same point -> robot-robot overlap recorded
        sc = Scenario(name="x", width=10.0, height=10.0,
                      robots={"a": (4.5, 5.0, 0.0), "b": (5.5, 5.0, math.pi)},
                      goals={"a": (6.0, 5.0), "b": (4.0, 5.0)})

        def policy(world):
            return {"a": (1.0, 0.0), "b": (1.0, 0.0)}   # drive toward each other

        res = run_scenario(sc, policy, dt=0.1, max_steps=40)
        self.assertGreater(res.collisions, 0)
        self.assertLess(res.min_robot_distance, 0.5)

    def test_not_success_when_goal_unreached(self):
        sc = Scenario(name="stuck", width=10.0, height=4.0,
                      robots={"r": (1.0, 2.0, 0.0)}, goals={"r": (9.0, 2.0)})

        def policy(world):
            return {"r": (0.0, 0.0)}   # never moves

        res = run_scenario(sc, policy, dt=0.1, max_steps=30)
        self.assertFalse(res.success)
        self.assertEqual(res.goals_reached, 0)


@unittest.skipUnless(
    importlib.util.find_spec("mrn_coord") is not None, "mrn_coord not available"
)
class TestNavigatePolicy(unittest.TestCase):
    def test_navigate_policy_solves_a_scenario(self):
        from mrn_sim.benchmark import navigate_policy

        sc = Scenario(name="around", width=14.0, height=8.0,
                      robots={"r": (1.0, 4.0, 0.0)},
                      obstacles=[(7.0, 4.0, 1.5)],
                      goals={"r": (13.0, 4.0)})
        res = run_scenario(sc, navigate_policy(sc), dt=0.1, max_steps=400)
        self.assertTrue(res.success)
        self.assertGreaterEqual(res.min_obstacle_clearance, -0.05)   # stayed clear

    def test_orca_policy_solves_crossing_collision_free(self):
        from mrn_sim.benchmark import orca_policy

        # Three robots cross past a central obstacle; ORCA reciprocal avoidance
        # must get them all to their goals without any collision.
        sc = Scenario(name="crossing", width=20.0, height=10.0,
                      robots={"r1": (1.5, 3.0, 0.0), "r2": (1.5, 7.0, 0.0),
                              "r3": (18.5, 5.0, math.pi)},
                      obstacles=[(10.0, 5.0, 1.2)],
                      goals={"r1": (18.5, 3.0), "r2": (18.5, 7.0), "r3": (1.5, 5.0)})
        res = run_scenario(sc, orca_policy(sc), dt=0.1, max_steps=400)
        self.assertTrue(res.success)
        self.assertEqual(res.collisions, 0)
        self.assertGreater(res.min_robot_distance, 2 * sc.robot_radius)
        self.assertGreaterEqual(res.min_obstacle_clearance, -0.05)


if __name__ == "__main__":
    unittest.main()
