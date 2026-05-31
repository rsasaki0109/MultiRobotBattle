#!/usr/bin/env python3
"""Generate the swarm-mission GIF: a multi-phase scenario in the 2D world.

A flock carries out a small mission, all driven by the real, deterministic
``mrn_sim.swarm.flock_in_world``:

1. **regroup**  — scattered robots flock together;
2. **migrate**  — the flock follows a sequence of waypoints across the obstacle
   field toward the final goal;
3. **evade**    — a predator appears mid-mission and the flock scatters away;
4. **recover**  — the predator leaves and the flock regroups and reaches the goal.

Synthetic, seeded, no ROS. See ``make_swarm_sim_gif.py`` / ``make_predator_gif.py``
for the single-behavior demos.

Usage::

    python3 scripts/make_mission_gif.py
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
PRED = "#fb7185"
WP = "#fbbf24"

W, H = 26.0, 16.0
N = 24
DT = 0.12
WAYPOINTS = [(8.0, 12.0), (16.0, 4.0), (23.0, 13.0)]
PREDATOR_WINDOW = (55, 95)   # steps during which the predator is active


def _centroid(world):
    xs = [r.pose[0] for r in world.robots.values()]
    ys = [r.pose[1] for r in world.robots.values()]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _build_frames(steps):
    rng = random.Random(3)
    robots = {}
    for i in range(N):
        robots[f"r{i}"] = Robot(
            f"r{i}", (rng.uniform(1.5, 5.0), rng.uniform(2.0, 8.0),
                      rng.uniform(0, 2 * math.pi)), 0.25)
    obstacles = [Obstacle(12.0, 8.5, 1.8), Obstacle(19.0, 8.5, 1.4),
                 Obstacle(6.0, 12.0, 1.0)]
    world = World(W, H, robots, obstacles)
    vel = [(0.0, 0.0)] * N
    wp_idx = 0
    frames = []
    for k in range(steps):
        c = _centroid(world)
        goal = WAYPOINTS[wp_idx]
        if math.hypot(c[0] - goal[0], c[1] - goal[1]) < 1.6 and wp_idx < len(WAYPOINTS) - 1:
            wp_idx += 1
            goal = WAYPOINTS[wp_idx]
        predator = None
        if PREDATOR_WINDOW[0] <= k < PREDATOR_WINDOW[1]:
            # predator lunges in from below the flock
            predator = (c[0] + 1.0, c[1] - 4.5)
        phase = ("regroup" if k < 12 else
                 "evade" if predator is not None else
                 "migrate" if wp_idx < len(WAYPOINTS) - 1 else "reach goal")
        frames.append((world, goal, predator, phase, wp_idx))
        # phase 1 is just flocking (no goal pull yet) so they bunch up first
        active_goal = goal if k >= 12 else None
        world, vel = flock_in_world(
            world, vel, dt=DT, perception=4.5, separation=1.5, max_speed=2.4,
            w_obstacle=1.3, obstacle_influence=2.3, obstacle_strength=2.6,
            goal=active_goal, w_goal=1.0,
            predator=predator, w_predator=2.2, predator_influence=6.5,
            max_v=2.4, max_omega=3.0)
    frames.append((world, WAYPOINTS[-1], None, "reach goal", len(WAYPOINTS) - 1))
    return frames, obstacles


def render(output: str, steps: int = 150, fps: int = 20) -> None:
    frames, obstacles = _build_frames(steps)
    ids = list(frames[0][0].robots)
    fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=100)
    fig.patch.set_facecolor(BG)
    trail = 10

    def draw(i):
        ax.clear()
        world, goal, predator, phase, wp_idx = frames[i]
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
        # waypoints (faint), current target highlighted
        for j, (wx, wy) in enumerate(WAYPOINTS):
            done = j < wp_idx
            ax.scatter([wx], [wy], marker="*", s=110 if j != wp_idx else 240,
                       color=WP, alpha=0.3 if done else (1.0 if j == wp_idx else 0.5),
                       edgecolor=BG, linewidth=0.6, zorder=4)

        lo = max(0, i - trail)
        xs, ys, us, vs, cols = [], [], [], [], []
        for rid in ids:
            tx = [frames[j][0].robots[rid].pose[0] for j in range(lo, i + 1)]
            ty = [frames[j][0].robots[rid].pose[1] for j in range(lo, i + 1)]
            ax.plot(tx, ty, color="#38bdf8", lw=1.0, alpha=0.16, zorder=2)
            x, y, th = world.robots[rid].pose
            xs.append(x)
            ys.append(y)
            us.append(0.5 * math.cos(th))
            vs.append(0.5 * math.sin(th))
            cols.append((0.5 + 0.5 * math.cos(th), 0.62, 0.5 + 0.5 * math.sin(th)))
        ax.quiver(xs, ys, us, vs, color=cols, scale=22, width=0.005,
                  headwidth=4, zorder=5)

        if predator is not None:
            for gk in range(4):
                ax.add_patch(Circle(predator, 0.4 * (1.4 + 0.6 * gk), facecolor=PRED,
                                    edgecolor="none", alpha=0.12 * (1 - gk / 4), zorder=3))
            ax.scatter([predator[0]], [predator[1]], marker="X", s=180, color=PRED,
                       edgecolor=BG, linewidth=1.0, zorder=6)

        accent = PRED if phase == "evade" else WP if phase != "regroup" else INK
        ax.text(0.3, H - 0.35, "Swarm mission (mrn_sim + mrn_coord)",
                color=INK, fontsize=11.5, weight="bold", va="top", zorder=7)
        ax.text(0.3, H - 0.95, f"phase: {phase}", color=accent, fontsize=10,
                weight="bold", va="top", zorder=7)
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
    parser.add_argument("--output", default="docs/media/mission_demo.gif")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    render(args.output, steps=args.steps, fps=args.fps)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
