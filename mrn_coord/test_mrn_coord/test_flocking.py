"""Tests for the Boids flocking step — one rule at a time."""

import math
import unittest

from mrn_coord.flocking import flock_velocities


class TestFlocking(unittest.TestCase):
    def test_separation_pushes_apart(self):
        # two stationary agents closer than `separation` -> steer away from each other
        pos = [(0.0, 0.0), (0.3, 0.0)]
        vel = [(0.0, 0.0), (0.0, 0.0)]
        out = flock_velocities(pos, vel, perception=3.0, separation=1.0)
        self.assertLess(out[0][0], 0.0)      # agent 0 moves -x
        self.assertGreater(out[1][0], 0.0)   # agent 1 moves +x

    def test_cohesion_pulls_toward_cluster(self):
        # lone agent with a cluster to its +x (beyond separation, within perception)
        pos = [(0.0, 0.0), (2.0, 0.0), (2.0, 0.5), (2.0, -0.5)]
        vel = [(0.0, 0.0)] * 4
        out = flock_velocities(pos, vel, perception=4.0, separation=1.0, w_coh=1.0)
        self.assertGreater(out[0][0], 0.0)   # steered toward the cluster (+x)

    def test_alignment_matches_neighbor_velocity(self):
        # symmetric neighbors (cohesion cancels) both moving +x -> agent gains +x
        pos = [(0.0, 0.0), (1.5, 0.0), (-1.5, 0.0)]
        vel = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
        out = flock_velocities(pos, vel, perception=3.0, separation=1.0,
                               w_ali=1.0, w_coh=1.0, inertia=0.0)
        self.assertGreater(out[0][0], 0.0)

    def test_isolated_agent_keeps_scaled_velocity(self):
        pos = [(0.0, 0.0), (50.0, 50.0)]   # second agent far outside perception
        vel = [(1.0, 0.0), (0.0, 0.0)]
        out = flock_velocities(pos, vel, perception=3.0, inertia=0.85)
        self.assertAlmostEqual(out[0][0], 0.85)
        self.assertAlmostEqual(out[0][1], 0.0)

    def test_speed_is_clamped(self):
        pos = [(0.0, 0.0), (0.05, 0.0)]   # extremely close -> huge separation push
        vel = [(0.0, 0.0), (0.0, 0.0)]
        out = flock_velocities(pos, vel, separation=1.0, w_sep=5.0, max_speed=2.0)
        for vx, vy in out:
            self.assertLessEqual(math.hypot(vx, vy), 2.0 + 1e-9)

    def test_empty_and_single(self):
        self.assertEqual(flock_velocities([], []), [])
        out = flock_velocities([(0.0, 0.0)], [(1.0, 1.0)], inertia=0.5)
        self.assertAlmostEqual(out[0][0], 0.5)
        self.assertAlmostEqual(out[0][1], 0.5)


if __name__ == "__main__":
    unittest.main()
