#!/usr/bin/env python3
"""Record the Gazebo (gz sim) 3D **swarm** demo GIF.

The 3D counterpart of ``make_swarm_sim_gif.py``: twelve robots flock across the
``mrn_gazebo`` world ``worlds/swarm_demo.sdf``, migrating past obstacles to a far
goal under the repo's Boids rules (``mrn_coord.flocking.flock_velocities`` —
separation / alignment / cohesion) blended with a migration pull and obstacle
avoidance, closed over Gazebo's reported poses and realized through the kinematic
``VelocityControl`` system. Each carries a 360° LiDAR whose returns are overlaid
on the render. Recorded fully offscreen by the shared :mod:`_gz_record` harness.

Media-generation only; not part of CI.

    python3 scripts/record_gazebo_swarm_gif.py
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
from mrn_coord.flocking import (  # noqa: E402
    flock_velocities,
    goal_seek,
    obstacle_avoidance,
    velocity_to_unicycle,
)

IDS = [f"a{i}" for i in range(12)]
OBSTACLES = [(12.0, 5.0, 1.1), (13.5, 9.5, 0.9), (17.0, 6.5, 0.8)]
GOAL = (21.5, 7.0)
SPEED = 2.6
W_GOAL, W_OBST = 0.7, 1.4
MAX_V, MAX_W = 3.4, 3.2
_FLOCK_RGB = (56, 189, 248)


def _clamp(vx, vy, m):
    s = math.hypot(vx, vy)
    return (vx * m / s, vy * m / s) if s > m else (vx, vy)


def make_step(duration):
    vels = {rid: (0.0, 0.0) for rid in IDS}

    def step(poses, elapsed):
        positions = [(poses[r][0], poses[r][1]) for r in IDS]
        vlist = [vels[r] for r in IDS]
        fv = flock_velocities(positions, vlist, perception=3.5, separation=1.3,
                              w_sep=1.7, w_ali=1.0, w_coh=1.0, max_speed=SPEED)
        gv = goal_seek(positions, GOAL, gain=1.0, max_speed=SPEED)
        ov = obstacle_avoidance(positions, OBSTACLES, influence=2.0, strength=3.0)
        cmds = {}
        for i, rid in enumerate(IDS):
            vx = fv[i][0] + W_GOAL * gv[i][0] + W_OBST * ov[i][0]
            vy = fv[i][1] + W_GOAL * gv[i][1] + W_OBST * ov[i][1]
            vx, vy = _clamp(vx, vy, SPEED)
            vels[rid] = (vx, vy)
            cmds[rid] = velocity_to_unicycle(poses[rid][2], vx, vy,
                                             max_v=MAX_V, max_omega=MAX_W)
        return cmds

    return step


SCENARIO = Scenario(
    world=os.path.join(_REPO, "mrn_gazebo", "worlds", "swarm_demo.sdf"),
    bridge_cfg=os.path.join(_REPO, "mrn_gazebo", "config", "gz_bridge_swarm.yaml"),
    ids=IDS,
    make_step=make_step,
    cam=Cam((11.0, -3.5, 13.5), yaw=1.5707, pitch=0.82, width=1000, height=620,
            hfov=1.05, lidar_z=0.38),
    crop=(0.12, 0.05),
    use_lidar=True,
    robot_rgb={rid: _FLOCK_RGB for rid in IDS},
    lidar_rays=False,    # 12 overlapping fans are noise — show a clean point cloud
    lidar_step=2,
    colors=80,
)

if __name__ == "__main__":
    add_cli_and_run(SCENARIO, "docs/media/gazebo_swarm_demo.gif")
