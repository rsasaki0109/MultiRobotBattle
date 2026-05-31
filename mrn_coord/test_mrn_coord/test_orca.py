"""Tests for the ORCA local collision-avoidance core."""

import math
import unittest

from mrn_coord.orca import orca_velocity


def _step_to_goal(starts, goals, *, radius=0.5, max_speed=1.0, dt=0.05, steps=400):
    """Simulate N reciprocal ORCA agents driving to their goals.

    Returns (positions_history_min_distance, final_positions). Each agent's
    preferred velocity points straight at its goal; ORCA reconciles them.
    """
    pos = [list(s) for s in starts]
    vel = [[0.0, 0.0] for _ in starts]
    min_dist = float("inf")
    for _ in range(steps):
        new_vel = []
        for i in range(len(pos)):
            gx, gy = goals[i]
            dx, dy = gx - pos[i][0], gy - pos[i][1]
            d = math.hypot(dx, dy)
            if d < 1e-6:
                pref = (0.0, 0.0)
            else:
                speed = min(max_speed, d / dt)
                pref = (dx / d * speed, dy / d * speed)
            neighbors = [((pos[j][0], pos[j][1]), (vel[j][0], vel[j][1]), radius)
                         for j in range(len(pos)) if j != i]
            new_vel.append(orca_velocity(
                (pos[i][0], pos[i][1]), (vel[i][0], vel[i][1]), pref, neighbors,
                radius=radius, max_speed=max_speed, time_step=dt))
        for i in range(len(pos)):
            vel[i] = list(new_vel[i])
            pos[i][0] += vel[i][0] * dt
            pos[i][1] += vel[i][1] * dt
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                min_dist = min(min_dist, math.hypot(pos[i][0] - pos[j][0],
                                                    pos[i][1] - pos[j][1]))
    return min_dist, pos


class TestOrca(unittest.TestCase):
    def test_no_neighbors_returns_preferred(self):
        v = orca_velocity((0.0, 0.0), (0.0, 0.0), (1.0, 0.0), max_speed=2.0)
        self.assertAlmostEqual(v[0], 1.0, places=6)
        self.assertAlmostEqual(v[1], 0.0, places=6)

    def test_preferred_capped_at_max_speed(self):
        v = orca_velocity((0.0, 0.0), (0.0, 0.0), (10.0, 0.0), max_speed=1.5)
        self.assertAlmostEqual(math.hypot(*v), 1.5, places=6)

    def test_head_on_passes_without_collision(self):
        # Slightly off-axis (real scenarios are never perfectly symmetric):
        # ORCA breaks the tie, the two slip past each other, both reach the far
        # side — and never come within their combined radius.
        radius = 0.5
        min_dist, final = _step_to_goal(
            [(-5.0, 0.0), (5.0, 0.1)], [(5.0, 0.1), (-5.0, 0.0)], radius=radius)
        self.assertGreaterEqual(min_dist, 2 * radius - 0.02)
        self.assertGreater(final[0][0], 3.0)
        self.assertLess(final[1][0], -3.0)

    def test_symmetric_crossing_stays_collision_free(self):
        # Four agents crossing through a common centre. Under perfect symmetry
        # ORCA cannot break the tie to converge (a documented property — a real
        # controller adds a tiny perturbation), but its core *guarantee* still
        # holds: they never collide. That guarantee is what we assert here.
        r = 0.5
        starts = [(-4.0, 0.0), (4.0, 0.0), (0.0, -4.0), (0.0, 4.0)]
        goals = [(4.0, 0.0), (-4.0, 0.0), (0.0, 4.0), (0.0, -4.0)]
        min_dist, _ = _step_to_goal(starts, goals, radius=r, steps=500)
        self.assertGreaterEqual(min_dist, 2 * r - 0.05)

    def test_static_obstacle_deflects(self):
        # Obstacle dead ahead: the chosen velocity must not aim straight at it.
        v = orca_velocity((0.0, 0.0), (1.0, 0.0), (1.0, 0.0),
                          obstacles=[(2.0, 0.0, 0.5)],
                          radius=0.5, max_speed=1.0, time_horizon_obst=2.0)
        self.assertGreater(abs(v[1]), 1e-3)


if __name__ == "__main__":
    unittest.main()
