"""Tests for footstep planning (Hornung et al., Humanoids 2012) and the
multi-humanoid footstep MAPF built on it.

The single-humanoid contracts: weighted A* with ``w = 1`` is optimal and ``w > 1``
is bounded-suboptimal (cost <= w*opt) while expanding far fewer states; the
stronger admissible heuristic finds the *same* optimum as the bare-Euclidean one
with fewer expansions; the anytime schedule's cost falls toward the optimum as
``w`` falls and ends optimal; and every foot in a plan is collision-free.

The multi-humanoid contracts: prioritized footstep MAPF returns body-collision-
free plans by construction; ``bodies_collision_free`` actually detects overlap;
a lower-priority humanoid yields on a crossing; and a symmetric head-on in a
1-wide corridor defeats the fixed priority order (a humanoid maps to ``None``),
the honest limitation of prioritized planning.
"""

import math
import unittest

from mrn_coord.mapf.footstep import (
    FootstepState,
    FootstepWorld,
    ara_star,
    plan_footsteps,
)
from mrn_coord.mapf.footstep_mapf import (
    bodies_collision_free,
    prioritized_footstep_mapf,
)

R = "R"


class TestFootstepWorld(unittest.TestCase):
    def test_collision_checking(self):
        world = FootstepWorld(2.0, 2.0, obstacles=((0.8, 0.8, 1.2, 1.2),))
        self.assertTrue(world.foot_collision_free(0.3, 0.3, 0.0))   # open
        self.assertFalse(world.foot_collision_free(1.0, 1.0, 0.0))  # on obstacle
        self.assertFalse(world.foot_collision_free(-0.1, 1.0, 0.0))  # out of bounds

    def test_from_grid(self):
        from mrn_coord.mapf import GridWorld
        grid = GridWorld(4, 4, blocked={(2, 2)})
        world = FootstepWorld.from_grid(grid, cell_size=0.25)
        self.assertAlmostEqual(world.width, 1.0)
        # the blocked cell (2,2) spans [0.5,0.75) in metric -> a foot there hits
        self.assertFalse(world.foot_collision_free(0.625, 0.625, 0.0))


class TestSingleHumanoid(unittest.TestCase):
    def setUp(self):
        self.world = FootstepWorld(2.0, 1.5)
        self.start = FootstepState(0.4, 0.75, 0.0, R)
        self.goal = (1.4, 0.75)

    def test_optimal_reachable_and_feet_clear(self):
        plan = plan_footsteps(self.world, self.start, self.goal, w=1.0)
        self.assertIsNotNone(plan)
        # every foot pose in the plan is collision-free and in bounds
        for s in plan.states:
            self.assertTrue(self.world.foot_collision_free(s.x, s.y, s.theta))
        # the last stance foot is within tolerance of the goal
        last = plan.states[-1]
        self.assertLessEqual(math.hypot(last.x - self.goal[0],
                                        last.y - self.goal[1]), 0.18 + 1e-9)

    def test_weighted_bound_and_fewer_expansions(self):
        popt, aopt = plan_footsteps(self.world, self.start, self.goal, w=1.0,
                                    return_stats=True)
        for w in (1.5, 2.0, 3.0):
            p, a = plan_footsteps(self.world, self.start, self.goal, w=w,
                                  return_stats=True)
            self.assertIsNotNone(p)
            self.assertLessEqual(p.cost, w * popt.cost + 1e-6)
        # w=2 expands strictly fewer states than optimal A*
        _, a2 = plan_footsteps(self.world, self.start, self.goal, w=2.0,
                               return_stats=True)
        self.assertLess(a2["expansions"], aopt["expansions"])

    def test_heuristic_informedness(self):
        ps, as_ = plan_footsteps(self.world, self.start, self.goal, w=1.0,
                                 heuristic="steps", return_stats=True)
        pe, ae = plan_footsteps(self.world, self.start, self.goal, w=1.0,
                                heuristic="euclid", return_stats=True)
        self.assertAlmostEqual(ps.cost, pe.cost, places=6)        # same optimum
        self.assertLess(as_["expansions"], ae["expansions"])      # but fewer

    def test_anytime_monotone_and_final_optimal(self):
        opt = plan_footsteps(self.world, self.start, self.goal, w=1.0).cost
        plans = ara_star(self.world, self.start, self.goal,
                         weights=(3.0, 2.0, 1.5, 1.0))
        costs = [p.cost for p in plans]
        self.assertTrue(all(costs[i] >= costs[i + 1] - 1e-9
                            for i in range(len(costs) - 1)))
        self.assertAlmostEqual(costs[-1], opt, places=6)

    def test_determinism(self):
        a = plan_footsteps(self.world, self.start, self.goal, w=2.0)
        b = plan_footsteps(self.world, self.start, self.goal, w=2.0)
        self.assertEqual([(s.x, s.y, s.theta, s.foot) for s in a.states],
                         [(s.x, s.y, s.theta, s.foot) for s in b.states])


class TestMultiHumanoidMAPF(unittest.TestCase):
    def test_crossing_collision_free_and_yields(self):
        world = FootstepWorld(3.0, 3.0)
        agents = {
            "A": (FootstepState(0.4, 1.5, 0.0, R), (2.6, 1.5)),
            "B": (FootstepState(1.5, 0.4, math.pi / 2, R), (1.5, 2.6)),
        }
        plans = prioritized_footstep_mapf(world, agents, w=2.0)
        self.assertTrue(all(p is not None for p in plans.values()))
        self.assertTrue(bodies_collision_free(plans))
        # the lower-priority humanoid pays more than it would alone (it yielded)
        solo = {k: plan_footsteps(world, s, g, w=2.0).cost
                for k, (s, g) in agents.items()}
        self.assertTrue(any(plans[k].cost > solo[k] + 1e-6 for k in agents))

    def test_bodies_collision_free_detects_overlap(self):
        # two humanoids planned independently onto the SAME lane collide
        world = FootstepWorld(3.0, 1.0)
        a = plan_footsteps(world, FootstepState(0.4, 0.5, 0.0, R), (2.6, 0.5),
                           w=2.0)
        b = plan_footsteps(world, FootstepState(2.6, 0.5, math.pi, R),
                           (0.4, 0.5), w=2.0)
        self.assertFalse(bodies_collision_free({"a": a, "b": b}))

    def test_headon_corridor_defeats_priority(self):
        walls = (tuple((x * 0.25, 0.0, x * 0.25 + 0.25, 0.5) for x in range(12))
                 + tuple((x * 0.25, 1.0, x * 0.25 + 0.25, 1.5)
                         for x in range(12)))
        corr = FootstepWorld(3.0, 1.5, obstacles=walls)
        agents = {
            "A": (FootstepState(0.4, 0.75, 0.0, R), (2.6, 0.75)),
            "B": (FootstepState(2.6, 0.75, math.pi, R), (0.4, 0.75)),
        }
        plans = prioritized_footstep_mapf(corr, agents, w=2.0, max_tick=30,
                                          max_expansions=5000)
        self.assertTrue(any(p is None for p in plans.values()))   # incomplete
        self.assertTrue(bodies_collision_free(plans))             # still safe


if __name__ == "__main__":
    unittest.main()
