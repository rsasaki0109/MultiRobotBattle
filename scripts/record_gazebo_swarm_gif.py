#!/usr/bin/env python3
"""Record the Gazebo (gz sim) 3D **swarm** demo GIF.

The 3D counterpart of ``make_swarm_sim_gif.py``: twelve robots flock across the
``mrn_gazebo`` Gazebo Harmonic world ``worlds/swarm_demo.sdf``, migrating past a
few obstacles to a far goal. Each robot is driven over ``cmd_vel`` by the repo's
Boids rules (``mrn_coord.flocking.flock_velocities`` — separation / alignment /
cohesion) blended with a migration pull and obstacle avoidance, closed over
Gazebo's reported poses and realized through the kinematic ``VelocityControl``
system. A static camera sensor renders the scene on the GPU and the frames are
bridged to ROS and encoded — entirely offscreen (no GUI / desktop window).

Media-generation only; not part of CI; not bit-for-bit deterministic.

    python3 scripts/record_gazebo_swarm_gif.py
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
_WORLD = os.path.join(_REPO, "mrn_gazebo", "worlds", "swarm_demo.sdf")
_BRIDGE_CFG = os.path.join(_REPO, "mrn_gazebo", "config", "gz_bridge_swarm.yaml")

sys.path.insert(0, os.path.join(_REPO, "mrn_sim"))
sys.path.insert(0, os.path.join(_REPO, "mrn_coord"))
from mrn_coord.flocking import (  # noqa: E402
    flock_velocities,
    goal_seek,
    obstacle_avoidance,
    velocity_to_unicycle,
)

IDS = [f"a{i}" for i in range(12)]
OBSTACLES = [(12.0, 5.0, 1.1), (13.5, 9.5, 0.9), (17.0, 6.5, 0.8)]
GOAL = (21.5, 7.0)
SPEED = 2.6            # flock max speed (VelocityControl realizes ~0.6x)
W_GOAL, W_OBST = 0.7, 1.4
MAX_V, MAX_W = 3.4, 3.2

# Camera projection params — MUST match rec_camera in swarm_demo.sdf.
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


def _clamp(vx, vy, m):
    s = math.hypot(vx, vy)
    return (vx * m / s, vy * m / s) if s > m else (vx, vy)


def _swarm_commands(poses, vels):
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
        node = Node("gz_swarm_recorder")
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
            cmds = _swarm_commands(poses, vels)
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
    ap.add_argument("--output", default="docs/media/gazebo_swarm_demo.gif")
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--settle", type=float, default=1.0)
    args = ap.parse_args()
    run(args.output, args.duration, args.fps, args.width, args.settle)


if __name__ == "__main__":
    main()
