#!/usr/bin/env python3
"""Record the Gazebo (gz sim) 3D **ORCA crowd** demo GIF.

The 3D counterpart of ``make_orca_gif.py``: two interleaved streams of holonomic
robots (four heading right, four heading left) on near-head-on lanes walk into
each other in the ``mrn_gazebo`` world ``worlds/orca_demo.sdf`` and pass *through*
one another, collision-free, by reciprocal avoidance. Each is driven over
``cmd_vel`` by :func:`mrn_coord.orca.orca_velocity` closed over Gazebo's reported
poses; the holonomic ORCA velocity is realized as a unicycle command through the
kinematic ``VelocityControl`` system. Recorded fully offscreen by the shared
:mod:`_gz_record` harness. Media-generation only; not part of CI.

    python3 scripts/record_gazebo_orca_gif.py
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "mrn_sim"))
sys.path.insert(0, os.path.join(_REPO, "mrn_coord"))

from _gz_record import Cam, Scenario, add_cli_and_run  # noqa: E402
from mrn_coord.orca import orca_velocity  # noqa: E402
from mrn_coord.flocking import velocity_to_unicycle  # noqa: E402

# id -> (goal_x, lane_y); rightward stream spawns at x=1.5, leftward at x=14.5.
ROBOTS = {
    "r1": (14.5, 3.0), "r2": (14.5, 5.0), "r3": (14.5, 7.0), "r4": (14.5, 9.0),
    "l1": (1.5, 3.3), "l2": (1.5, 5.3), "l3": (1.5, 7.3), "l4": (1.5, 9.3),
}
IDS = list(ROBOTS)
ROBOT_R = 0.45
ORCA_SPEED = 2.6
TIME_HORIZON = 3.0
MAX_V, MAX_W = 3.6, 3.2
GOAL_TOL = 0.4


def make_step(duration):
    vels = {rid: (0.0, 0.0) for rid in IDS}

    def step(poses, elapsed):
        cmds = {}
        for rid in IDS:
            pose = poses[rid]
            gx, gy = ROBOTS[rid]
            dx, dy = gx - pose[0], gy - pose[1]
            dist = math.hypot(dx, dy)
            if dist <= GOAL_TOL:
                vels[rid] = (0.0, 0.0)
                cmds[rid] = (0.0, 0.0)
                continue
            speed = min(ORCA_SPEED, dist / 0.3)
            pref = (dx / dist * speed, dy / dist * speed)
            neighbors = [((poses[o][0], poses[o][1]), vels[o], ROBOT_R)
                         for o in IDS if o != rid]
            vx, vy = orca_velocity((pose[0], pose[1]), vels[rid], pref, neighbors,
                                   radius=ROBOT_R, max_speed=ORCA_SPEED,
                                   time_horizon=TIME_HORIZON, time_step=0.1)
            vels[rid] = (vx, vy)
            cmds[rid] = velocity_to_unicycle(pose[2], vx, vy, max_v=MAX_V, max_omega=MAX_W)
        return cmds

    return step


SCENARIO = Scenario(
    world=os.path.join(_REPO, "mrn_gazebo", "worlds", "orca_demo.sdf"),
    bridge_cfg=os.path.join(_REPO, "mrn_gazebo", "config", "gz_bridge_orca.yaml"),
    ids=IDS,
    make_step=make_step,
    cam=Cam((8.0, -3.5, 12.5), yaw=1.5707, pitch=0.80, width=1000, height=640, hfov=1.02),
    crop=(0.12, 0.05),
)

if __name__ == "__main__":
    add_cli_and_run(SCENARIO, "docs/media/gazebo_orca_demo.gif")
