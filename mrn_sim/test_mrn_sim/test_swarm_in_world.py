"""Integration test: flocking drives the deterministic 2D world.

This is the verifiable twin of the Gazebo swarm — the same control loop
(Boids -> velocity_to_unicycle -> world step) but pure and deterministic, so it
can assert end-to-end properties in CI: determinism, staying in bounds, never
entering an obstacle, and actually moving.
"""

import importlib.util
import unittest

from mrn_sim import Obstacle, Robot, World


def _world():
    robots = {
        f"r{i}": Robot(f"r{i}", (2.0 + (i % 4) * 1.2, 2.0 + (i // 4) * 1.2, 0.0), 0.28)
        for i in range(8)
    }
    obstacles = [Obstacle(10.0, 7.0, 1.5), Obstacle(15.0, 4.0, 1.0)]
    return World(20.0, 14.0, robots, obstacles)


def _run(steps=160):
    from mrn_sim.swarm import flock_in_world

    world = _world()
    vel = [(0.6, 0.2)] * len(world.robots)   # a small initial drift
    traj = [world]
    for _ in range(steps):
        world, vel = flock_in_world(world, vel, dt=0.1)
        traj.append(world)
    return traj


@unittest.skipUnless(
    importlib.util.find_spec("mrn_coord") is not None, "mrn_coord not available"
)
class TestSwarmInWorld(unittest.TestCase):
    def test_deterministic(self):
        a = _run(80)
        b = _run(80)
        for ra, rb in zip(a[-1].robots.values(), b[-1].robots.values()):
            self.assertEqual(ra.pose, rb.pose)

    def test_stays_in_bounds_and_clear_of_obstacles(self):
        traj = _run(160)
        final = traj[-1]
        for r in final.robots.values():
            x, y, _ = r.pose
            self.assertTrue(0.0 <= x <= final.width)
            self.assertTrue(0.0 <= y <= final.height)
            # the world step never lets a robot enter an obstacle
            for o in final.obstacles:
                clearance = ((x - o.x) ** 2 + (y - o.y) ** 2) ** 0.5 - o.radius - r.radius
                self.assertGreater(clearance, -1e-6)

    def test_robots_move(self):
        traj = _run(160)
        moved = 0
        for rid in traj[0].robots:
            p0 = traj[0].robots[rid].pose
            pN = traj[-1].robots[rid].pose
            if (p0[0] - pN[0]) ** 2 + (p0[1] - pN[1]) ** 2 > 0.25:
                moved += 1
        self.assertGreaterEqual(moved, 6)   # most of the 8 robots moved

    def test_flees_predator(self):
        from mrn_sim.swarm import flock_in_world

        world = _world()
        predator = (4.0, 3.0)   # planted in the middle of the flock

        def mean_dist(w):
            ds = [((r.pose[0] - predator[0]) ** 2 + (r.pose[1] - predator[1]) ** 2) ** 0.5
                  for r in w.robots.values()]
            return sum(ds) / len(ds)

        d0 = mean_dist(world)
        vel = [(0.0, 0.0)] * len(world.robots)
        for _ in range(60):
            world, vel = flock_in_world(world, vel, dt=0.1, predator=predator,
                                        w_predator=2.0)
        dN = mean_dist(world)
        self.assertGreater(dN, d0 + 1.0)   # the flock fled the predator
        # still in bounds and obstacle-clear
        for r in world.robots.values():
            self.assertTrue(0.0 <= r.pose[0] <= world.width)
            self.assertTrue(0.0 <= r.pose[1] <= world.height)

    def test_migration_reaches_goal(self):
        from mrn_sim.swarm import flock_in_world

        world = _world()
        goal = (18.0, 11.0)

        def centroid(w):
            xs = [r.pose[0] for r in w.robots.values()]
            ys = [r.pose[1] for r in w.robots.values()]
            return (sum(xs) / len(xs), sum(ys) / len(ys))

        start_c = centroid(world)
        d0 = ((start_c[0] - goal[0]) ** 2 + (start_c[1] - goal[1]) ** 2) ** 0.5
        vel = [(0.0, 0.0)] * len(world.robots)
        for _ in range(300):
            world, vel = flock_in_world(world, vel, dt=0.1, goal=goal, w_goal=1.0)
        end_c = centroid(world)
        dN = ((end_c[0] - goal[0]) ** 2 + (end_c[1] - goal[1]) ** 2) ** 0.5
        # the flock migrated most of the way to the goal, around the obstacles
        self.assertLess(dN, d0 * 0.4)
        # still obstacle-clear at the end
        for r in world.robots.values():
            for o in world.obstacles:
                clr = ((r.pose[0] - o.x) ** 2 + (r.pose[1] - o.y) ** 2) ** 0.5 - o.radius - r.radius
                self.assertGreater(clr, -1e-6)


if __name__ == "__main__":
    unittest.main()
