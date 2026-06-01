#!/usr/bin/env python3
"""Record the Gazebo (gz sim) 3D demo GIF — the real simulator, rendered on GPU.

Unlike the other ``make_*_gif.py`` scripts (which animate the lightweight 2D
``mrn_sim`` core with matplotlib), this one drives the **actual Gazebo Harmonic
world** ``mrn_gazebo/worlds/multirobot_demo.sdf``: three robots cross an arena of
cylindrical obstacles in 3D. An offscreen camera sensor renders the scene on the
GPU (no GUI, no desktop window), the frames are bridged to ROS, and the robots
are driven over ``cmd_vel`` by the repo's own navigation stack — A* grid
planning + pure-pursuit + reciprocal avoidance (``mrn_sim.navigate`` primitives),
closed over Gazebo's reported poses. Each robot's body follows a commanded twist
via the kinematic ``VelocityControl`` system (stable and slip-free, unlike a
hand-tuned ``DiffDrive`` chassis). The captured frames are encoded to a GIF.

Because it runs Gazebo + ROS + GPU rendering, this is **media-generation only**,
not part of CI, and (being wall-clock-paced) not bit-for-bit deterministic — the
3D counterpart to the deterministic 2D demos, driven by the same algorithms.

Requirements: ``gz sim`` (Harmonic), ``ros_gz_bridge`` / ``ros_gz_image``, a GPU
with EGL, ``rclpy``, Pillow. Run with ROS 2 Jazzy sourced::

    python3 scripts/record_gazebo_gif.py
    python3 scripts/record_gazebo_gif.py --duration 12 --fps 16 --width 720
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
_WORLD = os.path.join(_REPO, "mrn_gazebo", "worlds", "multirobot_demo.sdf")
_BRIDGE_CFG = os.path.join(_REPO, "mrn_gazebo", "config", "gz_bridge_demo.yaml")

# Pull in the repo's real navigation stack (A* planner + pure-pursuit carrot +
# reciprocal avoidance) so the Gazebo robots are driven by the same algorithms
# the 2D demos use — purposeful routes around obstacles, not naive potential
# fields that stall in local minima.
sys.path.insert(0, os.path.join(_REPO, "mrn_sim"))
sys.path.insert(0, os.path.join(_REPO, "mrn_coord"))
from mrn_sim import Obstacle, World  # noqa: E402
from mrn_sim.navigate import plan_world_path  # noqa: E402
from mrn_coord.flocking import (  # noqa: E402
    mutual_avoidance,
    obstacle_avoidance,
    velocity_to_unicycle,
)
from mrn_coord.mapf.path_follower import carrot_point  # noqa: E402

# Arena + scene geometry (must match multirobot_demo.sdf).
W, H = 12.0, 8.0
ROBOT_R = 0.3
OBSTACLES = [(6.0, 4.0, 1.3), (3.0, 5.5, 0.8), (9.0, 5.5, 0.9), (8.5, 2.2, 0.7)]
IDS = ["robot_1", "robot_2", "robot_3"]
# Each robot ping-pongs between two opposite corners, so the three streams cross
# through the middle and must route around the obstacles (and each other).
GOALS = {
    "robot_1": [(11.0, 7.0), (1.0, 1.0)],   # main diagonal
    "robot_2": [(1.0, 7.0), (11.0, 1.0)],   # anti-diagonal
    "robot_3": [(11.0, 4.0), (1.0, 4.0)],   # horizontal sweep past the big obstacle
}
_PLAN_WORLD = World(W, H, {}, [Obstacle(*o) for o in OBSTACLES])

LOOKAHEAD = 1.0
NAV_SPEED = 2.6        # carrot pull magnitude (VelocityControl realizes ~0.6x)
W_OBST, OBST_INFLUENCE, OBST_STRENGTH = 1.2, 1.5, 2.0
W_MUTUAL, MUTUAL_R = 1.5, 1.4
MAX_V, MAX_W = 3.4, 3.0
GOAL_TOL = 0.5


def _env():
    """Environment that forces NVIDIA EGL (offscreen render) + CycloneDDS."""
    env = dict(os.environ)
    env.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                   "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    env["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
    env.setdefault("ROS_DOMAIN_ID", "77")
    env.pop("DISPLAY", None)  # ensure headless rendering, never the user's desktop
    return env


def _nav_commands(poses, goal_idx, paths):
    """One navigation step over the *Gazebo* poses -> per-robot ``(v, omega)``.

    This is ``mrn_sim.navigate.navigate_step`` unrolled to close the loop over
    Gazebo's reported poses (instead of stepping the 2D world): each robot
    follows the carrot on its A*-planned path while reciprocally avoiding the
    obstacles and the other robots. On reaching a goal it advances to the next
    in its cycle and replans. Mutates ``goal_idx`` / ``paths`` in place.
    """
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
        path = paths[rid] or [(pose[0], pose[1]), goal]  # straight-line fallback
        cx, cy = carrot_point(pose, path, LOOKAHEAD)
        dx, dy = cx - pose[0], cy - pose[1]
        d = math.hypot(dx, dy) or 1.0
        vx = dx / d * NAV_SPEED + W_OBST * obs[i][0] + W_MUTUAL * mut[i][0]
        vy = dy / d * NAV_SPEED + W_OBST * obs[i][1] + W_MUTUAL * mut[i][1]
        cmds[rid] = velocity_to_unicycle(pose[2], vx, vy, max_v=MAX_V, max_omega=MAX_W)
    return cmds


def _wait_topic(node, name, timeout=20.0):
    import rclpy
    t0 = time.time()
    while time.time() - t0 < timeout:
        names = [n for n, _ in node.get_topic_names_and_types()]
        if name in names:
            return True
        rclpy.spin_once(node, timeout_sec=0.2)
    return False


def run(output, duration, fps, width, settle):
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import Image
    from PIL import Image as PImage

    env = _env()
    for k in ("__EGL_VENDOR_LIBRARY_FILENAMES", "__GLX_VENDOR_LIBRARY_NAME",
              "RMW_IMPLEMENTATION", "ROS_DOMAIN_ID"):
        os.environ[k] = env[k]
    os.environ.pop("DISPLAY", None)

    procs = []

    def launch(cmd):
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
        procs.append(p)
        return p

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
        node = Node("gz_demo_recorder")
        if not _wait_topic(node, "/rec_camera", 25.0):
            raise RuntimeError("camera topic /rec_camera never appeared")

        poses = {}
        latest = {"img": None}

        def mk_pose_cb(rid):
            def cb(m):
                q = m.pose.orientation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                                 1 - 2 * (q.y * q.y + q.z * q.z))
                poses[rid] = (m.pose.position.x, m.pose.position.y, yaw)
            return cb

        scans = {}

        def mk_scan_cb(rid):
            def cb(m):
                scans[rid] = (m.angle_min, m.angle_increment, list(m.ranges))
            return cb

        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import LaserScan
        cmd_pubs = {}
        for rid in IDS:
            node.create_subscription(PoseStamped, f"/model/{rid}/pose",
                                     mk_pose_cb(rid), 10)
            node.create_subscription(LaserScan, f"/model/{rid}/scan",
                                     mk_scan_cb(rid), 10)
            cmd_pubs[rid] = node.create_publisher(Twist, f"/model/{rid}/cmd_vel", 10)

        def img_cb(m):
            a = np.frombuffer(bytes(m.data), dtype=np.uint8)
            latest["img"] = a.reshape(m.height, m.width, 3).copy()
        node.create_subscription(Image, "/rec_camera", img_cb, 10)

        print("waiting for poses + first frame ...")
        t0 = time.time()
        while (len(poses) < len(IDS) or latest["img"] is None) and time.time() - t0 < 20:
            rclpy.spin_once(node, timeout_sec=0.1)
        if len(poses) < len(IDS) or latest["img"] is None:
            raise RuntimeError(f"missing inputs: poses={list(poses)} img={latest['img'] is not None}")

        goal_idx = {rid: 0 for rid in IDS}
        paths = {rid: None for rid in IDS}

        def drive():
            if any(rid not in poses for rid in IDS):
                return
            cmds = _nav_commands(poses, goal_idx, paths)
            for rid in IDS:
                v, w = cmds[rid]
                t = Twist()
                t.linear.x = float(v)
                t.angular.z = float(w)
                cmd_pubs[rid].publish(t)

        # Settle: let the robots start moving before we begin capturing.
        print(f"settling {settle}s ...")
        ts = time.time()
        while time.time() - ts < settle:
            drive()
            rclpy.spin_once(node, timeout_sec=0.02)

        print(f"recording {duration}s @ {fps}fps ...")
        frames = []
        period = 1.0 / fps
        next_t = time.time()
        end_t = next_t + duration
        while time.time() < end_t:
            drive()
            rclpy.spin_once(node, timeout_sec=0.005)
            now = time.time()
            if now >= next_t:
                # snapshot the frame together with the poses + scans behind it
                frames.append((latest["img"], dict(poses),
                               {r: scans.get(r) for r in IDS}))
                next_t += period
        # stop the robots
        for rid in IDS:
            cmd_pubs[rid].publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
        print(f"captured {len(frames)} frames; encoding ...")
        _encode(frames, output, fps, width, PImage)
    finally:
        shutdown()


# Per-robot RGB (matches the chassis materials in multirobot_demo.sdf), used to
# tint each robot's overlaid laser scan.
ROBOT_RGB = {
    "robot_1": (56, 189, 248),    # cyan
    "robot_2": (244, 114, 182),   # pink
    "robot_3": (163, 230, 53),    # green
}

# --- Static camera projection (world -> pixel) -----------------------------
# These MUST match the rec_camera sensor in multirobot_demo.sdf. The camera is
# fixed, so one projection serves every frame; we use it to splat each robot's
# LiDAR returns back onto the rendered image as a live laser scan.
_CAM_POS = (6.0, -7.2, 9.2)
_CAM_YAW, _CAM_PITCH = 1.5707, 0.66
_CAM_W, _CAM_H, _CAM_HFOV = 1000, 700, 0.92
_LIDAR_Z = 0.44   # world height of the lidar ring (model z 0.1 + sensor z 0.34)
_F = (_CAM_W / 2.0) / math.tan(_CAM_HFOV / 2.0)


def _project(p):
    """World point ``(x, y, z)`` -> ``(u, v)`` pixel in the full 1000x700 frame,
    or ``None`` if behind the camera. Derived (and validated) for the gz camera
    convention: optical axis +X, up +Z, left +Y, with R = Rz(yaw)·Ry(pitch)."""
    dx, dy, dz = p[0] - _CAM_POS[0], p[1] - _CAM_POS[1], p[2] - _CAM_POS[2]
    cy, sy = math.cos(_CAM_YAW), math.sin(_CAM_YAW)
    cp, sp = math.cos(_CAM_PITCH), math.sin(_CAM_PITCH)
    px = cy * cp * dx + sy * cp * dy - sp * dz   # forward (depth)
    py = -sy * dx + cy * dy                       # left
    pz = cy * sp * dx + sy * sp * dy + cp * dz    # up
    if px <= 0.05:
        return None
    return (_CAM_W / 2.0 - _F * py / px, _CAM_H / 2.0 - _F * pz / px)


def _draw_lidar(draw, poses, scans):
    """Overlay each robot's laser scan onto the frame: faint rays + bright hits."""
    for rid, sc in scans.items():
        pose = poses.get(rid)
        if not sc or pose is None:
            continue
        amin, ainc, ranges = sc
        rx, ry, yaw = pose
        origin = _project((rx, ry, _LIDAR_Z))
        col = ROBOT_RGB.get(rid, (220, 220, 220))
        dim = tuple(int(c * 0.40) for c in col)
        for i, r in enumerate(ranges):
            if not r or not (r == r) or r >= 5.95:   # skip no-return / inf / NaN
                continue
            a = yaw + amin + i * ainc
            hit = _project((rx + r * math.cos(a), ry + r * math.sin(a), _LIDAR_Z))
            if hit is None:
                continue
            if origin is not None:
                draw.line([origin, hit], fill=dim, width=1)
            draw.ellipse([hit[0] - 1.6, hit[1] - 1.6, hit[0] + 1.6, hit[1] + 1.6],
                         fill=col)


# The camera frames a 12x8 arena isometrically, which leaves dead ground above
# the far edge and a sliver below the near edge — crop them off before encoding.
_CROP_TOP, _CROP_BOTTOM = 0.10, 0.04


def _encode(frames, output, fps, width, PImage):
    from PIL import ImageDraw
    imgs = []
    for f, fposes, fscans in frames:
        if f is None:
            continue
        im = PImage.fromarray(f, "RGB")
        if fscans and any(fscans.values()):     # splat the laser scans on first
            _draw_lidar(ImageDraw.Draw(im), fposes, fscans)
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
    ap.add_argument("--output", default="docs/media/gazebo_demo.gif")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--settle", type=float, default=1.5)
    args = ap.parse_args()
    run(args.output, args.duration, args.fps, args.width, args.settle)


if __name__ == "__main__":
    main()
