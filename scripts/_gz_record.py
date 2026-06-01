#!/usr/bin/env python3
"""Shared offscreen-recording harness for the Gazebo (gz sim) demo GIFs.

Every ``record_gazebo_*_gif.py`` demo is the same machine: launch a headless
``gz sim`` server + the ros_gz bridges, drive the robots over ``cmd_vel`` from an
rclpy control loop closed over Gazebo's reported poses, capture an offscreen
camera-sensor stream, and encode a GIF — all on the GPU with no GUI / desktop
window. Only the *world*, the *controller*, and the *camera framing* differ, so
that machinery lives here and each demo is a thin controller plus a
:class:`Scenario`.

Optionally overlays each robot's 360° LiDAR returns back onto the render
(projected through the fixed camera), so a demo can look like a live laser scan.

Media-generation only; not part of CI. Requires gz sim (Harmonic), ros_gz, a GPU
with EGL, rclpy, Pillow — run with ROS 2 Jazzy sourced.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field


@dataclass
class Cam:
    """Static camera placement — MUST match the rec_camera in the world SDF.

    The gz camera convention is optical axis +X, up +Z, left +Y, with the world
    rotation ``R = Rz(yaw)·Ry(pitch)`` (roll 0). Used to project world points to
    pixels for the LiDAR overlay.
    """

    pos: tuple
    yaw: float
    pitch: float
    width: int = 1000
    height: int = 640
    hfov: float = 0.95
    lidar_z: float = 0.44

    def project(self, p):
        """World ``(x, y, z)`` -> ``(u, v)`` pixel, or ``None`` if behind."""
        dx, dy, dz = p[0] - self.pos[0], p[1] - self.pos[1], p[2] - self.pos[2]
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        px = cy * cp * dx + sy * cp * dy - sp * dz   # forward (depth)
        py = -sy * dx + cy * dy                       # left
        pz = cy * sp * dx + sy * sp * dy + cp * dz    # up
        if px <= 0.05:
            return None
        f = (self.width / 2.0) / math.tan(self.hfov / 2.0)
        return (self.width / 2.0 - f * py / px, self.height / 2.0 - f * pz / px)


@dataclass
class Scenario:
    """Everything that distinguishes one Gazebo demo recording from another."""

    world: str
    bridge_cfg: str
    ids: list
    make_step: callable          # (duration) -> step(poses, elapsed) -> {id: (v, omega)}
    cam: Cam
    crop: tuple = (0.10, 0.04)    # (top, bottom) fraction trimmed before encoding
    use_lidar: bool = False
    robot_rgb: dict = field(default_factory=dict)   # id -> (r, g, b) for scan tint
    lidar_rays: bool = True       # draw the rays (False = a cleaner point cloud)
    lidar_step: int = 1           # draw every Nth return (thins clutter / file size)
    colors: int = 96              # GIF palette size


def _env():
    """Force NVIDIA EGL (offscreen render) + CycloneDDS; never touch the display."""
    env = dict(os.environ)
    env.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                   "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    env["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
    env.setdefault("ROS_DOMAIN_ID", "77")
    env.pop("DISPLAY", None)
    return env


def _draw_lidar(draw, poses, scans, cam, robot_rgb, rays=True, step=1):
    """Overlay each robot's laser scan: optional faint rays + bright hit points."""
    for rid, sc in scans.items():
        pose = poses.get(rid)
        if not sc or pose is None:
            continue
        amin, ainc, ranges = sc
        rx, ry, yaw = pose
        origin = cam.project((rx, ry, cam.lidar_z)) if rays else None
        col = robot_rgb.get(rid, (220, 220, 220))
        dim = tuple(int(c * 0.40) for c in col)
        for i in range(0, len(ranges), max(1, step)):
            r = ranges[i]
            if not r or not (r == r) or r >= 5.95:    # skip no-return / inf / NaN
                continue
            a = yaw + amin + i * ainc
            hit = cam.project((rx + r * math.cos(a), ry + r * math.sin(a), cam.lidar_z))
            if hit is None:
                continue
            if rays and origin is not None:
                draw.line([origin, hit], fill=dim, width=1)
            draw.ellipse([hit[0] - 1.6, hit[1] - 1.6, hit[0] + 1.6, hit[1] + 1.6],
                         fill=col)


def _encode(frames, output, fps, width, scen, PImage):
    from PIL import ImageDraw
    top_f, bot_f = scen.crop
    imgs = []
    for img, fposes, fscans in frames:
        if img is None:
            continue
        im = PImage.fromarray(img, "RGB")
        if scen.use_lidar and fscans and any(fscans.values()):
            _draw_lidar(ImageDraw.Draw(im), fposes, fscans, scen.cam, scen.robot_rgb,
                        rays=scen.lidar_rays, step=scen.lidar_step)
        top = int(im.height * top_f)
        bot = int(im.height * (1.0 - bot_f))
        im = im.crop((0, top, im.width, bot))
        if width and im.width != width:
            h = int(round(im.height * width / im.width))
            im = im.resize((width, h), PImage.LANCZOS)
        imgs.append(im.quantize(colors=scen.colors, method=PImage.FASTOCTREE,
                                dither=PImage.Dither.NONE))
    if not imgs:
        raise RuntimeError("no frames captured")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    imgs[0].save(output, save_all=True, append_images=imgs[1:], optimize=True,
                 loop=0, duration=int(round(1000 / fps)))
    print(f"wrote {output} ({os.path.getsize(output) // 1024} KB, {len(imgs)} frames)")


def record(scen, output, *, duration=14.0, fps=14, width=720, settle=1.0):
    """Run the scenario end-to-end and write ``output`` (a GIF)."""
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist, PoseStamped
    from sensor_msgs.msg import Image, LaserScan
    from PIL import Image as PImage

    ids = scen.ids
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
        launch(["gz", "sim", "-s", "-r", "--headless-rendering", scen.world])
        print("launching ros_gz bridges ...")
        launch(["ros2", "run", "ros_gz_bridge", "parameter_bridge",
                "--ros-args", "-p", f"config_file:={scen.bridge_cfg}"])
        launch(["ros2", "run", "ros_gz_image", "image_bridge", "/rec_camera"])

        rclpy.init()
        node = Node("gz_recorder")
        poses, scans = {}, {}
        latest = {"img": None}

        def mk_pose_cb(rid):
            def cb(m):
                q = m.pose.orientation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                                 1 - 2 * (q.y * q.y + q.z * q.z))
                poses[rid] = (m.pose.position.x, m.pose.position.y, yaw)
            return cb

        def mk_scan_cb(rid):
            def cb(m):
                scans[rid] = (m.angle_min, m.angle_increment, list(m.ranges))
            return cb

        cmd_pubs = {}
        for rid in ids:
            node.create_subscription(PoseStamped, f"/model/{rid}/pose", mk_pose_cb(rid), 10)
            cmd_pubs[rid] = node.create_publisher(Twist, f"/model/{rid}/cmd_vel", 10)
            if scen.use_lidar:
                node.create_subscription(LaserScan, f"/model/{rid}/scan", mk_scan_cb(rid), 10)

        def img_cb(m):
            a = np.frombuffer(bytes(m.data), dtype=np.uint8)
            latest["img"] = a.reshape(m.height, m.width, 3).copy()
        node.create_subscription(Image, "/rec_camera", img_cb, 10)

        print("waiting for poses + first frame ...")
        t0 = time.time()
        while (len(poses) < len(ids) or latest["img"] is None) and time.time() - t0 < 25:
            rclpy.spin_once(node, timeout_sec=0.1)
        if len(poses) < len(ids) or latest["img"] is None:
            raise RuntimeError(f"missing inputs: poses={len(poses)}/{len(ids)}")

        step = scen.make_step(duration)

        def drive(elapsed):
            if any(rid not in poses for rid in ids):
                return
            cmds = step(poses, elapsed)
            for rid in ids:
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
                frames.append((latest["img"], dict(poses),
                               {r: scans.get(r) for r in ids} if scen.use_lidar else {}))
                next_t += period
        for rid in ids:
            cmd_pubs[rid].publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
        print(f"captured {len(frames)} frames; encoding ...")
        _encode(frames, output, fps, width, scen, PImage)
    finally:
        shutdown()


def add_cli_and_run(scen, default_output):
    """Standard CLI (--output/--duration/--fps/--width/--settle) -> record()."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=default_output)
    ap.add_argument("--duration", type=float, default=14.0)
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--settle", type=float, default=1.0)
    a = ap.parse_args()
    record(scen, a.output, duration=a.duration, fps=a.fps, width=a.width, settle=a.settle)
