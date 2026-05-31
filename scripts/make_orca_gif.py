#!/usr/bin/env python3
"""Generate the ORCA reciprocal-avoidance GIF.

Two crowds walk straight at each other — five agents heading right, five
heading left, on interleaved lanes — so every agent must pass through the
oncoming stream. With :func:`mrn_coord.orca.orca_velocity` each one picks, every
tick, the velocity closest to its goal that is provably collision-free given the
others; because every agent reasons reciprocally, the two streams interleave and
pass through each other smoothly, without a single collision, and reassemble on
the far side. Deterministic, no ROS.

Usage::

    python3 scripts/make_orca_gif.py
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
sys.path.insert(0, os.path.join(_HERE, os.pardir, "mrn_coord"))

from mrn_coord.orca import orca_velocity  # noqa: E402

BG = "#0b0e14"
PANEL = "#0d1117"
GRID = "#1b2130"
INK = "#c9d1d9"
MUTED = "#6b7689"
RIGHTWARD = "#38bdf8"   # crowd A: left -> right
LEFTWARD = "#fbbf24"    # crowd B: right -> left

W, H = 16.0, 11.0
ROBOT_R = 0.4
MAX_SPEED = 1.5
DT = 0.1
HORIZON = 3.0

# Two interleaved streams that must pass through each other.
_YS_L = [2.0, 4.0, 6.0, 8.0, 10.0]   # start y of the rightbound crowd
_YS_R = [3.0, 5.0, 7.0, 9.0, 1.5]    # start y of the leftbound crowd


def _build_frames(steps):
    starts = [(1.5, y) for y in _YS_L] + [(14.5, y) for y in _YS_R]
    goals = [(14.5, y) for y in _YS_L] + [(1.5, y) for y in _YS_R]
    rightward = [True] * len(_YS_L) + [False] * len(_YS_R)
    n = len(starts)
    pos = [list(p) for p in starts]
    vel = [[0.0, 0.0] for _ in range(n)]

    frames = [[(p[0], p[1], 0.0 if rightward[i] else math.pi) for i, p in enumerate(pos)]]
    for _ in range(steps):
        new_vel = []
        for i in range(n):
            gx, gy = goals[i]
            dx, dy = gx - pos[i][0], gy - pos[i][1]
            d = math.hypot(dx, dy)
            if d < 0.12:
                new_vel.append((0.0, 0.0))
                continue
            speed = min(MAX_SPEED, d / DT)
            pref = (dx / d * speed, dy / d * speed)
            neighbors = [((pos[j][0], pos[j][1]), (vel[j][0], vel[j][1]), ROBOT_R)
                         for j in range(n) if j != i]
            new_vel.append(orca_velocity(
                (pos[i][0], pos[i][1]), (vel[i][0], vel[i][1]), pref, neighbors,
                radius=ROBOT_R, max_speed=MAX_SPEED, time_horizon=HORIZON, time_step=DT))
        for i in range(n):
            vel[i] = list(new_vel[i])
            pos[i][0] += vel[i][0] * DT
            pos[i][1] += vel[i][1] * DT
        th = [math.atan2(v[1], v[0]) if (v[0] or v[1]) else
              (0.0 if rightward[i] else math.pi) for i, v in enumerate(vel)]
        frames.append([(pos[i][0], pos[i][1], th[i]) for i in range(n)])
    return frames, goals, rightward


def render(output: str, steps: int = 130, fps: int = 20) -> None:
    frames, goals, rightward = _build_frames(steps)
    n = len(frames[0])
    fig, ax = plt.subplots(figsize=(6.6, 4.7), dpi=100)
    fig.patch.set_facecolor(BG)
    trail = 18

    def color(k):
        return RIGHTWARD if rightward[k] else LEFTWARD

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
        snap = frames[i]
        for k in range(n):
            c = color(k)
            gx, gy = goals[k]
            ax.scatter([gx], [gy], marker="*", s=90, color=c, alpha=0.55,
                       edgecolor=BG, linewidth=0.6, zorder=2)
            lo = max(0, i - trail)
            xs = [frames[j][k][0] for j in range(lo, i + 1)]
            ys = [frames[j][k][1] for j in range(lo, i + 1)]
            ax.plot(xs, ys, color=c, lw=2.0, alpha=0.45, solid_capstyle="round",
                    zorder=3)
            x, y, th = snap[k]
            ax.add_patch(Circle((x, y), ROBOT_R, facecolor=c, edgecolor=BG,
                                lw=1.2, zorder=4))
            ax.plot([x, x + 0.55 * math.cos(th)], [y, y + 0.55 * math.sin(th)],
                    color=BG, lw=2.0, zorder=5)

        ax.text(0.3, H - 0.25, "ORCA reciprocal collision avoidance",
                color=INK, fontsize=11.5, weight="bold", va="top", zorder=7)
        ax.text(0.3, H - 0.78,
                "two crowds pass through each other — collision-free (mrn_coord.orca)",
                color=MUTED, fontsize=8.0, va="top", zorder=7)
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
    parser.add_argument("--output", default="docs/media/orca_demo.gif")
    parser.add_argument("--steps", type=int, default=130)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    render(args.output, steps=args.steps, fps=args.fps)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
