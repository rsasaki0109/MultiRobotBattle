#!/usr/bin/env python3
"""Record the Gazebo (gz sim) 3D **ORCA crowd** demo GIF.

The 3D counterpart of ``make_orca_gif.py``: two interleaved streams of holonomic
robots (four heading right, four heading left) walk straight at each other in the
``mrn_gazebo`` Gazebo Harmonic world ``worlds/orca_demo.sdf`` and pass *through*
one another, collision-free, by reciprocal avoidance. Each robot is driven over
``cmd_vel`` by :func:`mrn_coord.orca.orca_velocity` closed over Gazebo's reported
poses; the holonomic ORCA velocity is realized as a unicycle command through the
kinematic ``VelocityControl`` system. A static camera sensor renders the scene on
the GPU and the frames are bridged to ROS and encoded — entirely offscreen (no
GUI / desktop window).

Like its sibling ``record_gazebo_gif.py`` this is **media-generation only**, not
part of CI, and (wall-clock-paced) not bit-for-bit deterministic.

    python3 scripts/record_gazebo_orca_gif.py
    python3 scripts/record_gazebo_orca_gif.py --duration 13 --fps 14 --width 720
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
_WORLD = os.path.join(_REPO, "mrn_gazebo", "worlds", "orca_demo.sdf")
_BRIDGE_CFG = os.path.join(_REPO, "mrn_gazebo", "config", "gz_bridge_orca.yaml")

sys.path.insert(0, os.path.join(_REPO, "mrn_sim"))
sys.path.insert(0, os.path.join(_REPO, "mrn_coord"))
from mrn_coord.orca import orca_velocity  # noqa: E402
from mrn_coord.flocking import velocity_to_unicycle  # noqa: E402

# Robot id -> (goal_x, lane_y); rightward stream starts at x=1.5, leftward at
# x=14.5. Lanes are interleaved so the two streams must thread through each other.
ROBOTS = {
    "r1": (14.5, 3.0), "r2": (14.5, 5.0), "r3": (14.5, 7.0), "r4": (14.5, 9.0),
    "l1": (1.5, 3.3), "l2": (1.5, 5.3), "l3": (1.5, 7.3), "l4": (1.5, 9.3),
}
IDS = list(ROBOTS)
RIGHTWARD = {"r1", "r2", "r3", "r4"}
ROBOT_R = 0.45
ORCA_SPEED = 2.6       # preferred speed (VelocityControl realizes ~0.6x)
TIME_HORIZON = 3.0
MAX_V, MAX_W = 3.6, 3.2
GOAL_TOL = 0.4

# Camera projection params — MUST match rec_camera in orca_demo.sdf.
_CAM_POS = (8.0, -3.5, 12.5)
_CAM_YAW, _CAM_PITCH = 1.5707, 0.80
_CAM_W, _CAM_H, _CAM_HFOV = 1000, 640, 1.02
_CROP_TOP, _CROP_BOTTOM = 0.12, 0.05


def _env():
    env = dict(os.environ)
    env.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                   "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    env["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
    env.setdefault("ROS_DOMAIN_ID", "77")
    env.pop("DISPLAY", None)
    return env


def _orca_commands(poses, vels):
    """Per-robot ``(v, omega)`` from one ORCA step over the Gazebo poses."""
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


def run(output, duration, fps, width, settle):
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist, PoseStamped
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
        node = Node("gz_orca_recorder")
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

        vels = {rid: (0.0, 0.0) for rid in IDS}

        def drive():
            if any(rid not in poses for rid in IDS):
                return
            cmds = _orca_commands(poses, vels)
            for rid in IDS:
                v, w = cmds[rid]
                t = Twist()
                t.linear.x = float(v)
                t.angular.z = float(w)
                cmd_pubs[rid].publish(t)

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
    ap.add_argument("--output", default="docs/media/gazebo_orca_demo.gif")
    ap.add_argument("--duration", type=float, default=13.0)
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--settle", type=float, default=1.0)
    args = ap.parse_args()
    run(args.output, args.duration, args.fps, args.width, args.settle)


if __name__ == "__main__":
    main()
