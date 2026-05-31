#!/usr/bin/env python3
"""Generate the navigation GIF: point-to-point planning + following.

Each robot plans a shortest path around the obstacles on an occupancy grid
(``mrn_sim.navigate.plan_world_path``, i.e. grid A*) and follows it with
pure pursuit through the continuous, collision-aware ``mrn_sim`` world to its
own goal. The classic navigation pipeline, deterministic and no ROS.

Usage::

    python3 scripts/make_nav_gif.py
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
sys.path.insert(0, os.path.join(_HERE, os.pardir, "mrn_coord"))

from mrn_coord.mapf.path_follower import pure_pursuit  # noqa: E402
from mrn_sim import Obstacle, Robot, World  # noqa: E402
from mrn_sim.navigate import plan_world_path  # noqa: E402
from mrn_sim.world import step  # noqa: E402

BG = "#0b0e14"
PANEL = "#0d1117"
GRID = "#1b2130"
WALL = "#4b5566"
WALL_EDGE = "#6b7689"
INK = "#c9d1d9"
MUTED = "#6b7689"
COLORS = ["#38bdf8", "#f472b6", "#a3e635", "#fbbf24"]

W, H = 20.0, 14.0
DT = 0.12


def _build_frames(steps):
    obstacles = [Obstacle(7.0, 7.0, 1.8), Obstacle(13.0, 9.5, 1.6),
                 Obstacle(13.0, 4.5, 1.6), Obstacle(10.0, 11.5, 1.0)]
    starts = {"1": (1.5, 2.0), "2": (1.5, 6.0), "3": (1.5, 9.0), "4": (1.5, 12.0)}
    goals = {"1": (18.5, 12.0), "2": (18.5, 8.5), "3": (18.5, 5.0), "4": (18.5, 1.5)}
    robots = {a: Robot(a, (p[0], p[1], 0.0), 0.25) for a, p in starts.items()}
    world = World(W, H, robots, obstacles)
    # plan each robot's path once (grid A* on the inflated occupancy)
    paths = {a: plan_world_path(world, starts[a], goals[a], cell_size=0.4, inflation=0.45)
             for a in starts}
    paths = {a: p for a, p in paths.items() if p is not None}

    frames = []
    for _ in range(steps):
        frames.append(world)
        commands = {}
        for a in world.robots:
            p = paths.get(a)
            if not p:
                continue
            v, omega, reached = pure_pursuit(
                world.robots[a].pose, p, lookahead=0.9, v_nominal=1.6,
                goal_tolerance=0.3)
            commands[a] = (0.0, 0.0) if reached else (v, omega)
        world = step(world, commands, DT)
    frames.append(world)
    return frames, obstacles, paths, goals


def render(output: str, steps: int = 150, fps: int = 20) -> None:
    frames, obstacles, paths, goals = _build_frames(steps)
    ids = list(frames[0].robots)
    fig, ax = plt.subplots(figsize=(7.8, 5.4), dpi=100)
    fig.patch.set_facecolor(BG)
    trail = 16

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
        for o in obstacles:
            ax.add_patch(Circle((o.x, o.y), o.radius, facecolor=WALL,
                                edgecolor=WALL_EDGE, lw=1.4, zorder=1))

        for k, a in enumerate(ids):
            color = COLORS[k % len(COLORS)]
            p = paths.get(a)
            if p:
                ax.plot([w[0] for w in p], [w[1] for w in p], color=color,
                        lw=1.1, alpha=0.35, linestyle=(0, (4, 3)), zorder=2)
            gx, gy = goals[a]
            ax.scatter([gx], [gy], marker="*", s=160, color=color, alpha=0.9,
                       edgecolor=BG, linewidth=0.6, zorder=3)
            # trail
            lo = max(0, i - trail)
            xs = [frames[j].robots[a].pose[0] for j in range(lo, i + 1)]
            ys = [frames[j].robots[a].pose[1] for j in range(lo, i + 1)]
            ax.plot(xs, ys, color=color, lw=2.0, alpha=0.5, solid_capstyle="round",
                    zorder=4)
            x, y, th = world.robots[a].pose
            ax.add_patch(Circle((x, y), 0.28, facecolor=color, edgecolor=BG,
                                lw=1.2, zorder=5))
            ax.plot([x, x + 0.5 * math.cos(th)], [y, y + 0.5 * math.sin(th)],
                    color=BG, lw=2.0, zorder=6)

        ax.text(0.3, H - 0.3, "Point-to-point navigation — plan (A*) + follow (pure pursuit)",
                color=INK, fontsize=11, weight="bold", va="top", zorder=7)
        ax.text(0.3, H - 0.85, "occupancy grid · grid A* · pure pursuit (mrn_sim + mrn_coord)",
                color=MUTED, fontsize=8.5, va="top", zorder=7)
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
    parser.add_argument("--output", default="docs/media/nav_demo.gif")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    render(args.output, steps=args.steps, fps=args.fps)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
