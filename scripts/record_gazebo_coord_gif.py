#!/usr/bin/env python3
"""Record the Gazebo (gz sim) 3D **coordination** demo GIF.

The 3D counterpart of ``make_coordination_gif.py``: three robots funnel through a
one-cell doorway in a wall (a collision-free Conflict-Based-Search schedule) and
then assemble a triangle formation, in the ``mrn_gazebo`` world
``worlds/coord_demo.sdf``. The CBS doorway schedule and the formation assembly
are **precomputed by mrn_coord** (the same ``cbs`` + formation ``simulate`` the 2D
demo uses) and replayed in 3D: each robot tracks its precomputed waypoint over
``cmd_vel`` through the kinematic ``VelocityControl`` system. Each carries a 360°
LiDAR whose returns are overlaid on the render. Recorded fully offscreen by the
shared :mod:`_gz_record` harness. Media-generation only; not part of CI.

    python3 scripts/record_gazebo_coord_gif.py
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
from mrn_coord.mapf import GridWorld, cbs, pad_paths  # noqa: E402
from mrn_coord.formation import polygon_formation, simulate  # noqa: E402
from mrn_coord.flocking import velocity_to_unicycle  # noqa: E402

# Same scene as make_coordination_gif.py, scaled from grid cells to metres.
SCALE = 1.4
WIDTH, HEIGHT, DOORWAY_Y = 11, 7, 3
STARTS = {"r1": (1, 1), "r2": (1, 3), "r3": (1, 5)}
GOALS = {"r1": (8, 5), "r2": (8, 3), "r3": (8, 1)}
IDS = list(STARTS)
EDGES = [("r1", "r2"), ("r2", "r3"), ("r1", "r3")]
ROBOT_RGB = {"r1": (56, 189, 248), "r2": (244, 114, 182), "r3": (163, 230, 53)}

MAX_V, MAX_W = 3.2, 3.2
TRACK_GAIN, TRACK_SPEED = 2.4, 2.6


def _blocked():
    return {(5, y) for y in range(HEIGHT) if y != DOORWAY_Y}


def _lerp(a, b, s):
    return (a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s)


def _build_trajectory(sub=8, hold=10):
    """CBS doorway traversal + formation assembly as ``{id: (x_m, y_m)}`` list."""
    grid = GridWorld(WIDTH, HEIGHT, blocked=_blocked())
    sol = cbs(grid, {a: (STARTS[a], GOALS[a]) for a in STARTS}, max_expansions=50_000)
    if sol is None:
        raise RuntimeError("CBS failed to solve the demo scenario")
    paths = pad_paths(sol.paths)
    horizon = max(len(p) for p in paths.values())

    traj = []
    for t in range(horizon - 1):
        for k in range(sub):
            s = k / sub
            traj.append({a: _lerp(paths[a][t], paths[a][t + 1], s) for a in paths})
    arrived = {a: paths[a][-1] for a in paths}
    traj.extend(dict(arrived) for _ in range(hold))

    spec = polygon_formation(IDS, radius=1.4)
    start = {a: (float(arrived[a][0]), float(arrived[a][1])) for a in arrived}
    formation, _ = simulate(start, spec, EDGES, gain=1.4, dt=0.05, steps=160)
    traj.extend(formation[i] for i in range(0, len(formation), 3))
    traj.extend(dict(formation[-1]) for _ in range(hold * 2))

    return [{a: (p[a][0] * SCALE, p[a][1] * SCALE) for a in p} for p in traj]


def _clamp(vx, vy, m):
    s = math.hypot(vx, vy)
    return (vx * m / s, vy * m / s) if s > m else (vx, vy)


def make_step(duration):
    traj = _build_trajectory()
    T = len(traj)
    print(f"precomputed CBS + formation trajectory: {T} waypoints")

    def step(poses, elapsed):
        idx = min(T - 1, int(elapsed / duration * (T - 1)))
        target = traj[idx]
        cmds = {}
        for rid in IDS:
            px, py, yaw = poses[rid]
            tx, ty = target[rid]
            vx, vy = _clamp(TRACK_GAIN * (tx - px), TRACK_GAIN * (ty - py), TRACK_SPEED)
            cmds[rid] = velocity_to_unicycle(yaw, vx, vy, max_v=MAX_V, max_omega=MAX_W)
        return cmds

    return step


SCENARIO = Scenario(
    world=os.path.join(_REPO, "mrn_gazebo", "worlds", "coord_demo.sdf"),
    bridge_cfg=os.path.join(_REPO, "mrn_gazebo", "config", "gz_bridge_coord.yaml"),
    ids=IDS,
    make_step=make_step,
    cam=Cam((7.7, -4.6, 12.0), yaw=1.5707, pitch=0.78, width=1000, height=660,
            hfov=0.92, lidar_z=0.35),
    crop=(0.10, 0.05),
    use_lidar=True,
    robot_rgb=ROBOT_RGB,
    lidar_step=2,
    colors=80,
)

if __name__ == "__main__":
    add_cli_and_run(SCENARIO, "docs/media/gazebo_coord_demo.gif")
