#!/usr/bin/env python3
"""Record the Gazebo (gz sim) 3D navigation demo GIF — the real simulator on GPU.

Three robots cross an obstacle arena in the ``mrn_gazebo`` Gazebo Harmonic world
``worlds/multirobot_demo.sdf``, driven over ``cmd_vel`` by the repo's own
navigation stack — A* grid planning + pure-pursuit + reciprocal avoidance
(``mrn_sim.navigate`` primitives) closed over Gazebo's reported poses, realized
through the kinematic ``VelocityControl`` system. Each carries a 360° LiDAR whose
returns are overlaid on the render. Recorded fully offscreen (no GUI / desktop
window) by the shared :mod:`_gz_record` harness.

Media-generation only; not part of CI; not bit-for-bit deterministic — the 3D
counterpart to the deterministic 2D ``make_*_gif.py`` demos.

    python3 scripts/record_gazebo_gif.py
    python3 scripts/record_gazebo_gif.py --duration 14 --fps 15 --width 720
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir))
sys.path.insert(0, _HERE)                                  # _gz_record
sys.path.insert(0, os.path.join(_REPO, "mrn_sim"))
sys.path.insert(0, os.path.join(_REPO, "mrn_coord"))

from _gz_record import Cam, Scenario, add_cli_and_run  # noqa: E402
from mrn_sim import Obstacle, World  # noqa: E402
from mrn_sim.navigate import plan_world_path  # noqa: E402
from mrn_coord.flocking import (  # noqa: E402
    mutual_avoidance,
    obstacle_avoidance,
    velocity_to_unicycle,
)
from mrn_coord.mapf.path_follower import carrot_point  # noqa: E402

W, H = 12.0, 8.0
OBSTACLES = [(6.0, 4.0, 1.3), (3.0, 5.5, 0.8), (9.0, 5.5, 0.9), (8.5, 2.2, 0.7)]
IDS = ["robot_1", "robot_2", "robot_3"]
# Each robot ping-pongs between two waypoints; diverse routes keep them crossing.
GOALS = {
    "robot_1": [(11.0, 7.0), (1.0, 1.0)],   # main diagonal
    "robot_2": [(1.0, 7.0), (11.0, 1.0)],   # anti-diagonal
    "robot_3": [(11.0, 4.0), (1.0, 4.0)],   # horizontal sweep past the big obstacle
}
_PLAN_WORLD = World(W, H, {}, [Obstacle(*o) for o in OBSTACLES])
ROBOT_RGB = {"robot_1": (56, 189, 248), "robot_2": (244, 114, 182), "robot_3": (163, 230, 53)}

LOOKAHEAD = 1.0
NAV_SPEED = 2.6
W_OBST, OBST_INFLUENCE, OBST_STRENGTH = 1.2, 1.5, 2.0
W_MUTUAL, MUTUAL_R = 1.5, 1.4
MAX_V, MAX_W = 3.4, 3.0
GOAL_TOL = 0.5


def make_step(duration):
    goal_idx = {rid: 0 for rid in IDS}
    paths = {rid: None for rid in IDS}

    def step(poses, elapsed):
        positions = [(poses[r][0], poses[r][1]) for r in IDS]
        obs = obstacle_avoidance(positions, OBSTACLES,
                                 influence=OBST_INFLUENCE, strength=OBST_STRENGTH)
        mut = mutual_avoidance(positions, radius=MUTUAL_R)
        cmds = {}
        for i, rid in enumerate(IDS):
            pose = poses[rid]
            goal = GOALS[rid][goal_idx[rid]]
            if math.hypot(goal[0] - pose[0], goal[1] - pose[1]) <= GOAL_TOL:
                goal_idx[rid] = (goal_idx[rid] + 1) % len(GOALS[rid])
                paths[rid] = None
                cmds[rid] = (0.0, 0.0)
                continue
            if paths[rid] is None:
                paths[rid] = plan_world_path(_PLAN_WORLD, (pose[0], pose[1]), goal,
                                             cell_size=0.5, inflation=0.4)
            path = paths[rid] or [(pose[0], pose[1]), goal]
            cx, cy = carrot_point(pose, path, LOOKAHEAD)
            dx, dy = cx - pose[0], cy - pose[1]
            d = math.hypot(dx, dy) or 1.0
            vx = dx / d * NAV_SPEED + W_OBST * obs[i][0] + W_MUTUAL * mut[i][0]
            vy = dy / d * NAV_SPEED + W_OBST * obs[i][1] + W_MUTUAL * mut[i][1]
            cmds[rid] = velocity_to_unicycle(pose[2], vx, vy, max_v=MAX_V, max_omega=MAX_W)
        return cmds

    return step


SCENARIO = Scenario(
    world=os.path.join(_REPO, "mrn_gazebo", "worlds", "multirobot_demo.sdf"),
    bridge_cfg=os.path.join(_REPO, "mrn_gazebo", "config", "gz_bridge_demo.yaml"),
    ids=IDS,
    make_step=make_step,
    cam=Cam((6.0, -7.2, 9.2), yaw=1.5707, pitch=0.66, width=1000, height=700,
            hfov=0.92, lidar_z=0.44),
    crop=(0.10, 0.04),
    use_lidar=True,
    robot_rgb=ROBOT_RGB,
    lidar_step=2,
)

if __name__ == "__main__":
    add_cli_and_run(SCENARIO, "docs/media/gazebo_demo.gif")
