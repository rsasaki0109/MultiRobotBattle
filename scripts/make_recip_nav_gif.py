#!/usr/bin/env python3
"""Generate the reciprocal-avoidance navigation GIF.

Robots in parallel lanes plus counter-flow robots each plan an A* path and
follow it (``mrn_sim.navigate``), but with reciprocal avoidance: each treats the
others as moving obstacles, so independent navigators sidestep one another (and
the static obstacles) instead of passing through. Deterministic, no ROS.

Usage::

    python3 scripts/make_recip_nav_gif.py
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

from mrn_sim import Obstacle, Robot, World  # noqa: E402
from mrn_sim.navigate import navigate_step, plan_world_path  # noqa: E402

BG = "#0b0e14"
PANEL = "#0d1117"
GRID = "#1b2130"
WALL = "#4b5566"
WALL_EDGE = "#6b7689"
INK = "#c9d1d9"
MUTED = "#6b7689"
COLORS = ["#38bdf8", "#f472b6", "#a3e635", "#fbbf24", "#c084fc", "#34d399"]

W, H = 22.0, 12.0
DT = 0.12


def _build_frames(steps):
    obstacles = [Obstacle(11.0, 6.0, 1.4), Obstacle(6.0, 9.0, 1.0),
                 Obstacle(16.0, 3.0, 1.0)]
    # three lanes left->right, two counter-flow right->left that weave between
    starts = {"1": (1.5, 2.5), "2": (1.5, 6.0), "3": (1.5, 9.5),
              "4": (20.5, 4.0), "5": (20.5, 8.0)}
    goals = {"1": (20.5, 2.5), "2": (20.5, 6.0), "3": (20.5, 9.5),
             "4": (1.5, 4.0), "5": (1.5, 8.0)}
    robots = {a: Robot(a, (p[0], p[1], 0.0 if p[0] < W / 2 else math.pi), 0.25)
              for a, p in starts.items()}
    world = World(W, H, robots, obstacles)
    paths = {a: plan_world_path(world, starts[a], goals[a], cell_size=0.5, inflation=0.4)
             for a in starts}
    paths = {a: p for a, p in paths.items() if p is not None}

    frames = [world]
    for _ in range(steps):
        world, reached = navigate_step(world, paths, dt=DT, w_mutual=1.8,
                                       mutual_radius=1.7, max_speed=1.8, max_v=1.8)
        frames.append(world)
    return frames, obstacles, paths, goals


def render(output: str, steps: int = 150, fps: int = 20) -> None:
    frames, obstacles, paths, goals = _build_frames(steps)
    ids = list(frames[0].robots)
    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=100)
    fig.patch.set_facecolor(BG)
    trail = 14

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
            gx, gy = goals[a]
            ax.scatter([gx], [gy], marker="*", s=150, color=color, alpha=0.85,
                       edgecolor=BG, linewidth=0.6, zorder=3)
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

        ax.text(0.3, H - 0.3, "Reciprocal-avoidance navigation",
                color=INK, fontsize=11.5, weight="bold", va="top", zorder=7)
        ax.text(0.3, H - 0.85,
                "A* plan + pursuit + mutual avoidance (mrn_sim + mrn_coord)",
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
    parser.add_argument("--output", default="docs/media/recip_nav_demo.gif")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    render(args.output, steps=args.steps, fps=args.fps)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
