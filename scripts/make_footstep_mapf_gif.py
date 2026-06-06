#!/usr/bin/env python3
"""Animate multi-humanoid footstep MAPF — bodies cross without colliding.

Several humanoids plan footsteps to their goals; prioritized footstep MAPF
(`mrn_coord.mapf.footstep_mapf`) deconflicts their **bodies** tick by tick, so
they cross a shared area without touching — a lower-priority humanoid waits or
detours. The animation draws, from the real planner:

- each humanoid's planned **footsteps** (oriented rectangles, one colour each),
- its **body disc** (the thing that is kept clear) sliding along the plan,
- a goal ring and a short trail.

    python3 scripts/make_footstep_mapf_gif.py --out out/footstep_mapf.gif

Deterministic, headless (Agg), no ROS.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "mrn_coord"))

from mrn_coord.mapf.footstep import (  # noqa: E402
    FOOT_LENGTH,
    FOOT_WIDTH,
    FootstepState,
    FootstepWorld,
    _foot_corners,
)
from mrn_coord.mapf.footstep_mapf import (  # noqa: E402
    DEFAULT_BODY_RADIUS,
    prioritized_footstep_mapf,
)

BG = "#0b0e14"
GRID = "#1b2230"
INK = "#c9d1d9"
MUTED = "#8b95a7"
COLORS = ["#5b8cff", "#ff8b5b", "#33d6a6", "#c792ea"]
# Bodies are deconflicted per footstep (at integer ticks); between ticks the
# discs are interpolated for smooth motion, so they are drawn a hair under the
# planning radius to stay visibly clear through the interpolation too.
DRAW_R = DEFAULT_BODY_RADIUS - 0.015


def _scenario(name):
    """A team of humanoids that must cross a shared area."""
    world = FootstepWorld(3.0, 3.0)
    if name == "crossing":
        agents = {
            "A": (FootstepState(0.4, 1.5, 0.0, "R"), (2.6, 1.5)),
            "B": (FootstepState(1.5, 0.4, math.pi / 2, "R"), (1.5, 2.6)),
        }
    else:  # three-way
        agents = {
            "A": (FootstepState(0.4, 1.5, 0.0, "R"), (2.6, 1.5)),
            "B": (FootstepState(1.5, 0.4, math.pi / 2, "R"), (1.5, 2.6)),
            "C": (FootstepState(0.4, 0.4, math.pi / 4, "R"), (2.6, 2.6)),
        }
    return world, agents


def _at(states, tc):
    """Interpolated (x, y) along a per-tick stance sequence at continuous tick tc."""
    L = len(states)
    t0 = min(int(math.floor(tc)), L - 1)
    t1 = min(t0 + 1, L - 1)
    frac = tc - math.floor(tc)
    a, b = states[t0], states[t1]
    return (a.x + (b.x - a.x) * frac, a.y + (b.y - a.y) * frac)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/footstep_mapf.gif")
    ap.add_argument("--scenario", default="threeway",
                    choices=["crossing", "threeway"])
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--frames-per-tick", type=int, default=6)
    args = ap.parse_args()

    world, agents = _scenario(args.scenario)
    plans = prioritized_footstep_mapf(world, agents, w=2.0)
    ids = [k for k in agents if plans.get(k) is not None]
    horizon = max(len(plans[k].states) for k in ids)
    tcs = [t / args.frames_per_tick
           for t in range(int((horizon - 1) * args.frames_per_tick) + 1)]
    tcs += [horizon - 1] * (args.fps // 2)   # hold at the end

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_title("multi-humanoid footstep MAPF — bodies stay clear",
                 color=INK, fontsize=12, pad=10)

    discs, com_dots, trails = {}, {}, {}
    for i, k in enumerate(ids):
        col = COLORS[i % len(COLORS)]
        # all footsteps, faint
        for s in plans[k].states:
            ax.add_patch(Polygon(_foot_corners(s.x, s.y, s.theta, FOOT_LENGTH,
                                               FOOT_WIDTH), closed=True,
                                 facecolor=col, edgecolor=col, alpha=0.16,
                                 linewidth=0.8))
        # goal ring
        gx, gy = agents[k][1][0], agents[k][1][1]
        ax.add_patch(Circle((gx, gy), 0.12, fill=False, edgecolor=col,
                            linewidth=2.0, alpha=0.8))
        disc = Circle((0, 0), DRAW_R, facecolor=col, edgecolor=BG,
                      linewidth=1.5, alpha=0.45, zorder=4)
        ax.add_patch(disc)
        discs[k] = disc
        com_dots[k], = ax.plot([], [], "o", color=col, ms=7, mec=BG, mew=1.0,
                               zorder=6)
        trails[k], = ax.plot([], [], color=col, lw=1.4, alpha=0.6, zorder=3)
        ax.plot([], [], color=col, lw=6, label=f"humanoid {k}")
    ax.legend(loc="upper center", ncol=len(ids), fontsize=9, framealpha=0.0,
              labelcolor=INK, bbox_to_anchor=(0.5, -0.01))

    hist = {k: ([], []) for k in ids}

    def update(fi):
        tc = tcs[fi]
        artists = []
        for k in ids:
            x, y = _at(plans[k].states, tc)
            discs[k].center = (x, y)
            com_dots[k].set_data([x], [y])
            xs, ys = hist[k]
            xs.append(x)
            ys.append(y)
            trails[k].set_data(xs, ys)
            artists += [discs[k], com_dots[k], trails[k]]
        return artists

    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=len(tcs),
                         interval=1000 / args.fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=100,
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    _optimize_gif(args.out, args.fps)
    print(f"wrote {args.out}  ({len(ids)} humanoids, horizon {horizon} steps)")


def _optimize_gif(path, fps):
    try:
        from PIL import Image, ImageSequence
    except Exception:
        return
    im = Image.open(path)
    frames = [f.convert("RGB").quantize(colors=48, method=Image.Quantize.FASTOCTREE)
              for f in ImageSequence.Iterator(im)]
    if not frames:
        return
    frames[0].save(path, save_all=True, append_images=frames[1:], optimize=True,
                   duration=int(1000 / fps), loop=0, disposal=2)


if __name__ == "__main__":
    main()
