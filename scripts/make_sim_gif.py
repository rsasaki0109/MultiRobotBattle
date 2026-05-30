#!/usr/bin/env python3
"""Generate the mrn_sim demo GIF, driven by the real 2D world simulator.

Three robots roam a bounded world with circular obstacles, exchanging V2V links
when in range. The motion is produced by the actual ``mrn_sim`` step (unicycle
kinematics + collision), with a simple deterministic waypoint-seeking controller
in this script — so it is honest and reproducible, no ROS or running stack.

Usage::

    python3 scripts/make_sim_gif.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "mrn_sim"))

from mrn_sim import Obstacle, Robot, World, normalize_angle, step  # noqa: E402

BG = "#0b0e14"
PANEL = "#0d1117"
GRID = "#1b2130"
WALL = "#4b5566"
WALL_EDGE = "#6b7689"
INK = "#c9d1d9"
MUTED = "#6b7689"
LINK = "#e2e8f0"
COLORS = ["#38bdf8", "#f472b6", "#a3e635"]

W, H = 12.0, 8.0
SENSE = 5.0


def _initial_world():
    robots = {
        "1": Robot("1", (1.0, 1.0, 0.0), 0.28),
        "2": Robot("2", (11.0, 1.0, math.pi), 0.28),
        "3": Robot("3", (1.0, 7.0, -math.pi / 2), 0.28),
    }
    obstacles = [
        Obstacle(6.0, 4.0, 1.3),
        Obstacle(3.0, 5.5, 0.8),
        Obstacle(9.0, 5.5, 0.9),
        Obstacle(8.5, 2.2, 0.7),
    ]
    return World(W, H, robots, obstacles)


# Each robot cycles through a set of waypoints (opposite corners / centerish).
WAYPOINTS = {
    "1": [(11.0, 7.0), (11.0, 1.0), (1.0, 7.0), (1.0, 1.0)],
    "2": [(1.0, 7.0), (1.0, 1.0), (11.0, 7.0), (11.0, 1.0)],
    "3": [(11.0, 1.0), (11.0, 7.0), (1.0, 1.0), (1.0, 7.0)],
}


def _controller(world, wp_index, dt):
    """Deterministic potential-field waypoint-seeking with obstacle avoidance.

    The desired heading combines an attractive pull toward the current waypoint
    with repulsion from nearby obstacles and walls, so robots flow around the
    obstacles instead of stalling against them.
    """
    commands = {}
    new_index = dict(wp_index)
    for rid, robot in world.robots.items():
        x, y, theta = robot.pose
        wps = WAYPOINTS[rid]
        idx = wp_index[rid]
        tx, ty = wps[idx]
        if math.hypot(tx - x, ty - y) < 0.7:
            new_index[rid] = (idx + 1) % len(wps)
            tx, ty = wps[new_index[rid]]

        # attractive unit vector toward the waypoint
        ax, ay = tx - x, ty - y
        an = math.hypot(ax, ay) or 1.0
        fx, fy = ax / an, ay / an

        # repulsion from obstacles (influence falls off with clearance)
        for o in world.obstacles:
            dx, dy = x - o.x, y - o.y
            d = math.hypot(dx, dy)
            clearance = d - o.radius - robot.radius
            if clearance < 2.0:
                w = 2.2 / (max(clearance, 0.15) ** 2)
                fx += (dx / (d or 1.0)) * w
                fy += (dy / (d or 1.0)) * w

        # repulsion from the four walls
        for dist, vx, vy in (
            (x, 1.0, 0.0), (W - x, -1.0, 0.0),
            (y, 0.0, 1.0), (H - y, 0.0, -1.0),
        ):
            if dist < 1.5:
                w = 1.5 / (max(dist, 0.2) ** 2)
                fx += vx * w
                fy += vy * w

        desired = math.atan2(fy, fx)
        heading_err = normalize_angle(desired - theta)
        omega = max(-2.5, min(2.5, 2.5 * heading_err))
        v = 1.7 * max(0.0, math.cos(heading_err))
        commands[rid] = (v, omega)
    return commands, new_index


def _build_frames(steps, dt):
    world = _initial_world()
    wp_index = {rid: 0 for rid in world.robots}
    frames = [world]
    for _ in range(steps):
        commands, wp_index = _controller(world, wp_index, dt)
        world = step(world, commands, dt)
        frames.append(world)
    return frames


def render(output: str, steps: int = 150, dt: float = 0.12, fps: int = 20) -> None:
    frames = _build_frames(steps, dt)
    trail_len = 24
    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=100)
    fig.patch.set_facecolor(BG)
    ids = list(frames[0].robots)

    def draw(i):
        ax.clear()
        world = frames[i]
        ax.set_facecolor(PANEL)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)

        for o in world.obstacles:
            ax.add_patch(Circle((o.x, o.y), o.radius, facecolor=WALL,
                                edgecolor=WALL_EDGE, lw=1.4, zorder=1))

        # V2V links within sensing range
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                pa = world.robots[ids[a]].pose
                pb = world.robots[ids[b]].pose
                if math.hypot(pa[0] - pb[0], pa[1] - pb[1]) <= SENSE:
                    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=LINK, lw=1.2,
                            alpha=0.4, linestyle=(0, (4, 3)), zorder=2)

        for k, rid in enumerate(ids):
            color = COLORS[k % len(COLORS)]
            # trail
            lo = max(0, i - trail_len)
            xs = [frames[j].robots[rid].pose[0] for j in range(lo, i + 1)]
            ys = [frames[j].robots[rid].pose[1] for j in range(lo, i + 1)]
            for j in range(1, len(xs)):
                ax.plot(xs[j - 1:j + 1], ys[j - 1:j + 1], color=color,
                        lw=2.0, alpha=0.12 + 0.5 * (j / len(xs)),
                        solid_capstyle="round", zorder=2)
            x, y, theta = world.robots[rid].pose
            # soft glow
            for gk in range(4):
                ax.add_patch(Circle((x, y), world.robots[rid].radius * (1.6 + 0.7 * gk),
                                    facecolor=color, edgecolor="none",
                                    alpha=0.10 * (1 - gk / 4), zorder=3))
            ax.add_patch(Circle((x, y), world.robots[rid].radius, facecolor=color,
                                edgecolor=BG, lw=1.4, zorder=5))
            # heading arrow
            ax.plot([x, x + 0.55 * math.cos(theta)], [y, y + 0.55 * math.sin(theta)],
                    color=BG, lw=2.2, zorder=6)
            ax.text(x, y - world.robots[rid].radius - 0.18, rid, color=color,
                    fontsize=8, ha="center", va="top", weight="bold", zorder=6)

        ax.text(0.2, H - 0.25, "2D Multi-Robot World (mrn_sim)", color=INK,
                fontsize=12, weight="bold", va="top", zorder=7)
        ax.text(0.2, H - 0.75, "unicycle kinematics · obstacles · V2V sensing",
                color=MUTED, fontsize=9, va="top", zorder=7)
        return ()

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps, blit=False)
    anim.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)
    _optimize_gif(output, fps)


def _optimize_gif(path: str, fps: int, colors: int = 96) -> None:
    from PIL import Image

    src = Image.open(path)
    out = []
    try:
        while True:
            out.append(src.copy().convert("RGB").quantize(
                colors=colors, method=Image.FASTOCTREE, dither=Image.Dither.NONE))
            src.seek(src.tell() + 1)
    except EOFError:
        pass
    out[0].save(path, save_all=True, append_images=out[1:], optimize=True,
                loop=0, duration=int(round(1000 / fps)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/media/sim_demo.gif")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    render(args.output, steps=args.steps, fps=args.fps)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
