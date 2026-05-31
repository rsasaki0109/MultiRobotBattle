#!/usr/bin/env python3
"""Generate the swarm-in-world GIF: flocking unicycle robots avoiding obstacles.

Driven by the real, deterministic loop ``mrn_sim.swarm.flock_in_world`` (Boids +
obstacle avoidance -> velocity_to_unicycle -> unicycle world step). This is the
verifiable 2D twin of the Gazebo swarm: a flock of differential-drive robots
flowing around obstacles inside a bounded world. No ROS, seeded, reproducible.

Usage::

    python3 scripts/make_swarm_sim_gif.py
"""

from __future__ import annotations

import argparse
import math
import os
import random
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
from mrn_sim.swarm import flock_in_world  # noqa: E402

BG = "#0b0e14"
PANEL = "#0d1117"
GRID = "#1b2130"
WALL = "#4b5566"
WALL_EDGE = "#6b7689"
INK = "#c9d1d9"
MUTED = "#6b7689"

W, H = 24.0, 15.0
N = 28
DT = 0.12


def _build_frames(steps):
    rng = random.Random(5)
    robots = {}
    for i in range(N):
        x = rng.uniform(1.5, 7.0)
        y = rng.uniform(1.5, H - 1.5)
        robots[f"r{i}"] = Robot(f"r{i}", (x, y, rng.uniform(0, 2 * math.pi)), 0.25)
    obstacles = [Obstacle(12.0, 7.5, 2.0), Obstacle(17.5, 4.0, 1.3),
                 Obstacle(18.0, 11.0, 1.5)]
    world = World(W, H, robots, obstacles)
    vel = [(1.2, 0.0)] * N
    goal = (W - 2.5, H - 2.5)        # migrate across the obstacle field
    frames = [world]
    for _ in range(steps):
        world, vel = flock_in_world(
            world, vel, dt=DT, perception=4.5, separation=1.5, max_speed=2.2,
            w_obstacle=1.2, obstacle_influence=2.2, obstacle_strength=2.5,
            goal=goal, w_goal=0.9, max_v=2.2, max_omega=3.0)
        frames.append(world)
    return frames, obstacles, goal


def render(output: str, steps: int = 150, fps: int = 20) -> None:
    frames, obstacles, goal = _build_frames(steps)
    ids = list(frames[0].robots)
    fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=100)
    fig.patch.set_facecolor(BG)
    trail = 12

    def draw(i):
        ax.clear()
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
        # migration goal
        ax.scatter([goal[0]], [goal[1]], marker="*", s=320, color="#fbbf24",
                   edgecolor=BG, linewidth=0.8, zorder=4)

        lo = max(0, i - trail)
        xs, ys, us, vs, cols = [], [], [], [], []
        for rid in ids:
            tx = [frames[j].robots[rid].pose[0] for j in range(lo, i + 1)]
            ty = [frames[j].robots[rid].pose[1] for j in range(lo, i + 1)]
            ax.plot(tx, ty, color="#38bdf8", lw=1.0, alpha=0.18, zorder=2)
            x, y, th = frames[i].robots[rid].pose
            xs.append(x)
            ys.append(y)
            us.append(0.5 * math.cos(th))
            vs.append(0.5 * math.sin(th))
            cols.append((0.5 + 0.5 * math.cos(th), 0.62, 0.5 + 0.5 * math.sin(th)))
        ax.quiver(xs, ys, us, vs, color=cols, scale=20, width=0.005,
                  headwidth=4, zorder=5)

        ax.text(0.3, H - 0.35, f"Swarm migration — {N} robots flock to the goal",
                color=INK, fontsize=11.5, weight="bold", va="top", zorder=7)
        ax.text(0.3, H - 0.9,
                "Boids + obstacle avoidance + migration -> unicycle (mrn_sim + mrn_coord)",
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
    parser.add_argument("--output", default="docs/media/swarm_sim_demo.gif")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    render(args.output, steps=args.steps, fps=args.fps)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
