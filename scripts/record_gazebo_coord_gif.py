#!/usr/bin/env python3
"""Record the Gazebo (gz sim) 3D **coordination** demo GIF.

The 3D counterpart of ``make_coordination_gif.py``: three robots funnel through a
one-cell doorway in a wall (a collision-free schedule from Conflict-Based Search)
and then assemble into a triangle formation, in the ``mrn_gazebo`` Gazebo
Harmonic world ``worlds/coord_demo.sdf``. The CBS doorway schedule and the
formation assembly are **precomputed by mrn_coord** (the same ``cbs`` + formation
``simulate`` the 2D demo uses) and replayed in 3D: each robot tracks its
precomputed waypoint over ``cmd_vel`` through the kinematic ``VelocityControl``
system. A static camera sensor renders the scene on the GPU and the frames are
bridged to ROS and encoded — entirely offscreen (no GUI / desktop window).

Media-generation only; not part of CI.

    python3 scripts/record_gazebo_coord_gif.py
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir))
_WORLD = os.path.join(_REPO, "mrn_gazebo", "worlds", "coord_demo.sdf")
_BRIDGE_CFG = os.path.join(_REPO, "mrn_gazebo", "config", "gz_bridge_coord.yaml")

sys.path.insert(0, os.path.join(_REPO, "mrn_sim"))
sys.path.insert(0, os.path.join(_REPO, "mrn_coord"))
from mrn_coord.mapf import GridWorld, cbs, pad_paths  # noqa: E402
from mrn_coord.formation import polygon_formation, simulate  # noqa: E402

# Same scene as make_coordination_gif.py, scaled from grid cells to meters.
SCALE = 1.4
WIDTH, HEIGHT, DOORWAY_Y = 11, 7, 3
STARTS = {"r1": (1, 1), "r2": (1, 3), "r3": (1, 5)}
GOALS = {"r1": (8, 5), "r2": (8, 3), "r3": (8, 1)}
IDS = list(STARTS)
EDGES = [("r1", "r2"), ("r2", "r3"), ("r1", "r3")]

MAX_V, MAX_W = 3.2, 3.2
TRACK_GAIN = 2.4      # P-gain for tracking the precomputed waypoint
TRACK_SPEED = 2.6     # holonomic tracking speed cap (VelocityControl ~0.6x)

_CROP_TOP, _CROP_BOTTOM = 0.10, 0.05


def _blocked():
    return {(5, y) for y in range(HEIGHT) if y != DOORWAY_Y}


def _lerp(a, b, s):
    return (a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s)


def _build_trajectory(sub=8, hold=10):
    """Precompute the CBS doorway traversal + formation assembly as a list of
    ``{id: (x_m, y_m)}`` waypoints in world metres."""
    grid = GridWorld(WIDTH, HEIGHT, blocked=_blocked())
    sol = cbs(grid, {a: (STARTS[a], GOALS[a]) for a in STARTS}, max_expansions=50_000)
    if sol is None:
        raise RuntimeError("CBS failed to solve the demo scenario")
    paths = pad_paths(sol.paths)
    horizon = max(len(p) for p in paths.values())

    traj = []
    for t in range(horizon - 1):                       # act 1: doorway traversal
        for k in range(sub):
            s = k / sub
            traj.append({a: _lerp(paths[a][t], paths[a][t + 1], s) for a in paths})
    arrived = {a: paths[a][-1] for a in paths}
    traj.extend(dict(arrived) for _ in range(hold))

    spec = polygon_formation(IDS, radius=1.4)            # act 2: formation
    start = {a: (float(arrived[a][0]), float(arrived[a][1])) for a in arrived}
    formation, _ = simulate(start, spec, EDGES, gain=1.4, dt=0.05, steps=160)
    traj.extend(formation[i] for i in range(0, len(formation), 3))
    traj.extend(dict(formation[-1]) for _ in range(hold * 2))

    return [{a: (p[a][0] * SCALE, p[a][1] * SCALE) for a in p} for p in traj]


def _env():
    env = dict(os.environ)
    env.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                   "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    env["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
    env.setdefault("ROS_DOMAIN_ID", "77")
    env.pop("DISPLAY", None)
    return env


def _clamp(vx, vy, m):
    s = math.hypot(vx, vy)
    return (vx * m / s, vy * m / s) if s > m else (vx, vy)


def _track_commands(poses, target):
    """P-control each robot toward its precomputed target -> (v, omega)."""
    from mrn_coord.flocking import velocity_to_unicycle
    cmds = {}
    for rid in IDS:
        px, py, yaw = poses[rid]
        tx, ty = target[rid]
        vx, vy = _clamp(TRACK_GAIN * (tx - px), TRACK_GAIN * (ty - py), TRACK_SPEED)
        cmds[rid] = velocity_to_unicycle(yaw, vx, vy, max_v=MAX_V, max_omega=MAX_W)
    return cmds


def run(output, duration, fps, width, settle):
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist, PoseStamped
    from sensor_msgs.msg import Image
    from PIL import Image as PImage

    traj = _build_trajectory()
    T = len(traj)
    print(f"precomputed CBS + formation trajectory: {T} waypoints")

    env = _env()
    for k in ("__EGL_VENDOR_LIBRARY_FILENAMES", "__GLX_VENDOR_LIBRARY_NAME",
              "RMW_IMPLEMENTATION", "ROS_DOMAIN_ID"):
        os.environ[k] = env[k]
    os.environ.pop("DISPLAY", None)

    procs = []

    def launch(cmd):
        procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL, preexec_fn=os.setsid))

    def shutdown():
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(1.0)
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    try:
        print("launching gz sim (headless) ...")
        launch(["gz", "sim", "-s", "-r", "--headless-rendering", _WORLD])
        print("launching ros_gz bridges ...")
        launch(["ros2", "run", "ros_gz_bridge", "parameter_bridge",
                "--ros-args", "-p", f"config_file:={_BRIDGE_CFG}"])
        launch(["ros2", "run", "ros_gz_image", "image_bridge", "/rec_camera"])

        rclpy.init()
        node = Node("gz_coord_recorder")
        poses = {}
        latest = {"img": None}

        def mk_pose_cb(rid):
            def cb(m):
                q = m.pose.orientation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                                 1 - 2 * (q.y * q.y + q.z * q.z))
                poses[rid] = (m.pose.position.x, m.pose.position.y, yaw)
            return cb

        cmd_pubs = {}
        for rid in IDS:
            node.create_subscription(PoseStamped, f"/model/{rid}/pose",
                                     mk_pose_cb(rid), 10)
            cmd_pubs[rid] = node.create_publisher(Twist, f"/model/{rid}/cmd_vel", 10)

        def img_cb(m):
            a = np.frombuffer(bytes(m.data), dtype=np.uint8)
            latest["img"] = a.reshape(m.height, m.width, 3).copy()
        node.create_subscription(Image, "/rec_camera", img_cb, 10)

        print("waiting for poses + first frame ...")
        t0 = time.time()
        while (len(poses) < len(IDS) or latest["img"] is None) and time.time() - t0 < 25:
            rclpy.spin_once(node, timeout_sec=0.1)
        if len(poses) < len(IDS) or latest["img"] is None:
            raise RuntimeError(f"missing inputs: poses={len(poses)}/{len(IDS)}")

        def target_at(elapsed):
            idx = min(T - 1, int(elapsed / duration * (T - 1)))
            return traj[idx]

        def drive(elapsed):
            if any(rid not in poses for rid in IDS):
                return
            cmds = _track_commands(poses, target_at(elapsed))
            for rid in IDS:
                v, w = cmds[rid]
                t = Twist()
                t.linear.x = float(v)
                t.angular.z = float(w)
                cmd_pubs[rid].publish(t)

        print(f"settling {settle}s ...")
        ts = time.time()
        while time.time() - ts < settle:
            drive(0.0)
            rclpy.spin_once(node, timeout_sec=0.02)

        print(f"recording {duration}s @ {fps}fps ...")
        frames = []
        period = 1.0 / fps
        start_t = time.time()
        next_t = start_t
        while True:
            elapsed = time.time() - start_t
            if elapsed >= duration:
                break
            drive(elapsed)
            rclpy.spin_once(node, timeout_sec=0.005)
            if time.time() >= next_t:
                frames.append(latest["img"])
                next_t += period
        for rid in IDS:
            cmd_pubs[rid].publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
        print(f"captured {len(frames)} frames; encoding ...")
        _encode(frames, output, fps, width, PImage)
    finally:
        shutdown()


def _encode(frames, output, fps, width, PImage):
    imgs = []
    for f in frames:
        if f is None:
            continue
        im = PImage.fromarray(f, "RGB")
        top = int(im.height * _CROP_TOP)
        bot = int(im.height * (1.0 - _CROP_BOTTOM))
        im = im.crop((0, top, im.width, bot))
        if width and im.width != width:
            h = int(round(im.height * width / im.width))
            im = im.resize((width, h), PImage.LANCZOS)
        imgs.append(im.quantize(colors=96, method=PImage.FASTOCTREE,
                                dither=PImage.Dither.NONE))
    if not imgs:
        raise RuntimeError("no frames captured")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    imgs[0].save(output, save_all=True, append_images=imgs[1:], optimize=True,
                 loop=0, duration=int(round(1000 / fps)))
    print(f"wrote {output} ({os.path.getsize(output) // 1024} KB, {len(imgs)} frames)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="docs/media/gazebo_coord_demo.gif")
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--settle", type=float, default=1.0)
    args = ap.parse_args()
    run(args.output, args.duration, args.fps, args.width, args.settle)


if __name__ == "__main__":
    main()
