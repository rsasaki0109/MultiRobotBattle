#!/usr/bin/env python3
"""Generate the swarm GIF, driven by the real Boids step.

Tens of agents flock in a bounded box under the actual
``mrn_coord.flocking.flock_velocities`` rules (separation / alignment /
cohesion), with a soft wall-turn so they stay in frame. Synthetic,
deterministic (seeded), no ROS — it showcases that the simulation foundation
scales from a handful of robots to a swarm.

Usage::

    python3 scripts/make_swarm_gif.py
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "mrn_coord"))

from mrn_coord.flocking import flock_velocities  # noqa: E402

BG = "#0b0e14"
PANEL = "#0d1117"
GRID = "#1b2130"
INK = "#c9d1d9"
MUTED = "#6b7689"

W, H = 24.0, 16.0
N = 70
MAX_SPEED = 2.6
DT = 0.12


def _wall_turn(p, v, margin=2.5, push=1.4):
    """Add an inward velocity component near the walls."""
    vx, vy = v
    x, y = p
    if x < margin:
        vx += push * (margin - x) / margin
    elif x > W - margin:
        vx -= push * (x - (W - margin)) / margin
    if y < margin:
        vy += push * (margin - y) / margin
    elif y > H - margin:
        vy -= push * (y - (H - margin)) / margin
    return (vx, vy)


def _build_frames(steps):
    rng = random.Random(7)
    pos = [(rng.uniform(2, W - 2), rng.uniform(2, H - 2)) for _ in range(N)]
    ang = [rng.uniform(0, 2 * math.pi) for _ in range(N)]
    vel = [(math.cos(a) * 1.5, math.sin(a) * 1.5) for a in ang]
    frames = [list(pos)]
    headings = [list(ang)]
    for _ in range(steps):
        vel = flock_velocities(
            pos, vel, perception=4.0, separation=1.6,
            w_sep=1.7, w_ali=1.0, w_coh=0.9, inertia=0.9, max_speed=MAX_SPEED,
        )
        vel = [_wall_turn(pos[i], vel[i]) for i in range(N)]
        pos = [(pos[i][0] + vel[i][0] * DT, pos[i][1] + vel[i][1] * DT)
               for i in range(N)]
        # clamp inside the box as a hard backstop
        pos = [(min(max(x, 0.3), W - 0.3), min(max(y, 0.3), H - 0.3))
               for (x, y) in pos]
        frames.append(list(pos))
        headings.append([math.atan2(v[1], v[0]) for v in vel])
    return frames, headings


def render(output: str, steps: int = 150, fps: int = 20) -> None:
    frames, headings = _build_frames(steps)
    fig, ax = plt.subplots(figsize=(7.8, 5.2), dpi=100)
    fig.patch.set_facecolor(BG)
    trail = 10

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

        # trails
        lo = max(0, i - trail)
        for k in range(N):
            xs = [frames[j][k][0] for j in range(lo, i + 1)]
            ys = [frames[j][k][1] for j in range(lo, i + 1)]
            ax.plot(xs, ys, color="#38bdf8", lw=1.0, alpha=0.22,
                    solid_capstyle="round", zorder=1)

        xs = [p[0] for p in frames[i]]
        ys = [p[1] for p in frames[i]]
        us = [0.45 * math.cos(h) for h in headings[i]]
        vs = [0.45 * math.sin(h) for h in headings[i]]
        # color by heading for a lively look
        colors = [(0.5 + 0.5 * math.cos(h), 0.6, 0.5 + 0.5 * math.sin(h))
                  for h in headings[i]]
        ax.quiver(xs, ys, us, vs, color=colors, scale=18, width=0.004,
                  headwidth=4, zorder=5)

        ax.text(0.4, H - 0.4, f"Swarm flocking — {N} agents (mrn_coord.flocking)",
                color=INK, fontsize=12, weight="bold", va="top", zorder=7)
        ax.text(0.4, H - 1.0, "separation · alignment · cohesion",
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
    parser.add_argument("--output", default="docs/media/swarm_demo.gif")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    render(args.output, steps=args.steps, fps=args.fps)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
