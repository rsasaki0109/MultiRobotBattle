#!/usr/bin/env python3
"""Generate the dynamic-obstacle + replanning navigation GIF.

A robot navigates to its goal (A* plan + pure-pursuit follow). A moving obstacle
slides onto its path; the robot detects the block (``path_blocked``) and replans
around it (``plan_world_path`` on the current obstacles), still reaching the
goal. Deterministic, no ROS.

Usage::

    python3 scripts/make_replan_gif.py
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
from mrn_sim.navigate import path_blocked, plan_world_path  # noqa: E402
from mrn_sim.world import step  # noqa: E402

BG = "#0b0e14"
PANEL = "#0d1117"
GRID = "#1b2130"
WALL = "#4b5566"
WALL_EDGE = "#6b7689"
MOVING = "#fb7185"
INK = "#c9d1d9"
MUTED = "#6b7689"
COL = "#38bdf8"

W, H = 22.0, 10.0
DT = 0.12
GOAL = (20.0, 5.0)


def _build_frames(steps):
    static = [Obstacle(7.0, 7.5, 1.0), Obstacle(14.0, 2.5, 1.0)]
    world = World(W, H, {"r": Robot("r", (1.5, 5.0, 0.0), 0.25)}, list(static))
    path = plan_world_path(world, (1.5, 5.0), GOAL, cell_size=0.5, inflation=0.45)
    frames = []
    replanned = False
    for k in range(steps):
        # a moving obstacle slides down onto the path at (11, 5) and settles
        my = max(5.0, 10.5 - 0.22 * k)
        obstacles = static + [Obstacle(11.0, my, 1.3)]
        world = World(W, H, world.robots, obstacles)
        obs_t = [(o.x, o.y, o.radius) for o in obstacles]
        did_replan = False
        if path is None or path_blocked(obs_t, path, clearance=0.4):
            pose = world.robots["r"].pose
            np_ = plan_world_path(world, (pose[0], pose[1]), GOAL,
                                  cell_size=0.5, inflation=0.45)
            if np_ is not None:
                path = np_
            did_replan = True
            replanned = True
        frames.append((world, path, my, did_replan, replanned))
        pose = world.robots["r"].pose
        v, omega, done = pure_pursuit(pose, path, lookahead=0.9, v_nominal=1.5,
                                      goal_tolerance=0.3) if path else (0, 0, False)
        if done:
            frames.append((world, path, my, False, replanned))
            break
        world = step(world, {"r": (v, omega)}, DT)
    return frames, static


def render(output: str, steps: int = 150, fps: int = 20) -> None:
    frames, static = _build_frames(steps)
    fig, ax = plt.subplots(figsize=(7.8, 4.2), dpi=100)
    fig.patch.set_facecolor(BG)
    trail = 18

    def draw(i):
        ax.clear()
        world, path, my, did_replan, replanned = frames[i]
        ax.set_facecolor(PANEL)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)
        for o in static:
            ax.add_patch(Circle((o.x, o.y), o.radius, facecolor=WALL,
                                edgecolor=WALL_EDGE, lw=1.4, zorder=1))
        # moving obstacle
        ax.add_patch(Circle((11.0, my), 1.3, facecolor=MOVING, alpha=0.85,
                            edgecolor=BG, lw=1.2, zorder=2))
        # current plan
        if path:
            ax.plot([w[0] for w in path], [w[1] for w in path], color=COL,
                    lw=1.1, alpha=0.5, linestyle=(0, (4, 3)), zorder=3)
        ax.scatter([GOAL[0]], [GOAL[1]], marker="*", s=200, color=COL,
                   edgecolor=BG, linewidth=0.6, zorder=3)
        lo = max(0, i - trail)
        xs = [frames[j][0].robots["r"].pose[0] for j in range(lo, i + 1)]
        ys = [frames[j][0].robots["r"].pose[1] for j in range(lo, i + 1)]
        ax.plot(xs, ys, color=COL, lw=2.2, alpha=0.6, solid_capstyle="round", zorder=4)
        x, y, th = world.robots["r"].pose
        ax.add_patch(Circle((x, y), 0.3, facecolor=COL, edgecolor=BG, lw=1.2, zorder=5))
        ax.plot([x, x + 0.6 * math.cos(th)], [y, y + 0.6 * math.sin(th)],
                color=BG, lw=2.2, zorder=6)

        ax.text(0.3, H - 0.3, "Dynamic obstacle + replanning",
                color=INK, fontsize=11.5, weight="bold", va="top", zorder=7)
        msg = "replanning around the moving obstacle" if did_replan else \
              "A* plan + pure pursuit (mrn_sim)"
        ax.text(0.3, H - 0.8, msg, color=(MOVING if did_replan else MUTED),
                fontsize=8.5, weight="bold" if did_replan else "normal",
                va="top", zorder=7)
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
    parser.add_argument("--output", default="docs/media/replan_demo.gif")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    render(args.output, steps=args.steps, fps=args.fps)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
