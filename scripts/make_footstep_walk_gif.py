#!/usr/bin/env python3
"""Animate a humanoid footstep plan turned into a dynamically stable walk.

Plans footsteps with the search-based footstep planner (Hornung et al. 2012),
then generates the center-of-mass trajectory with ZMP preview control (Kajita
et al. 2003) and renders both, driven by the real `mrn_coord.mapf` code:

- **Left** — a top-down floor view: each planned footstep is an oriented
  rectangle (left / right foot in two colours), the current **support foot**
  lights up, the **CoM** (cyan) traces a swaying path forward, and the induced
  **ZMP** (orange) hugs the support foot — the visual proof of stability.
- **Right** — the lateral story over time: the stepped reference ZMP, the
  induced ZMP tracking it, and the CoM swaying smoothly between, with a moving
  time cursor.

    python3 scripts/make_footstep_walk_gif.py --out out/footstep_walk.gif

Deterministic, headless (Agg), no ROS.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "mrn_coord"))

from mrn_coord.mapf.footstep import (  # noqa: E402
    FOOT_LENGTH,
    FOOT_WIDTH,
    FootstepState,
    FootstepWorld,
    LEFT,
    _foot_corners,
    plan_footsteps,
)
from mrn_coord.mapf.lipm_walk import generate_walk  # noqa: E402

# Palette (dark, matches the repo's other GIFs).
BG = "#0b0e14"
GRID = "#1b2230"
INK = "#c9d1d9"
MUTED = "#8b95a7"
LEFT_FOOT = "#5b8cff"
RIGHT_FOOT = "#ff8b5b"
COM = "#33d6e6"
ZMP = "#ffd166"


def _scenario(turn):
    """A footstep plan: a straight forward walk, or one that curves."""
    if turn:
        world = FootstepWorld(3.0, 3.0)
        start = FootstepState(0.4, 0.5, 0.0, "R")
        goal = (2.4, 2.4)
    else:
        world = FootstepWorld(3.2, 1.4, collision_res=0.05)
        start = FootstepState(0.4, 0.7, 0.0, "R")
        goal = (2.8, 0.7)
    plan = plan_footsteps(world, start, goal, w=2.0)
    return world, plan


def _animate(world, plan, out, fps, step_duration, dt, stride):
    wp = generate_walk(plan.states, step_duration=step_duration, dt=dt)
    n = len(wp.com_x)
    times = [k * dt for k in range(n)]
    # stop the animation shortly after the last footfall (drop the static hold)
    cutoff = min(n, int((len(plan.states) * step_duration + 0.7) / dt))
    frames_idx = list(range(0, cutoff, stride)) + [cutoff - 1]

    fig, (axf, axp) = plt.subplots(
        1, 2, figsize=(11.0, 4.6), gridspec_kw={"width_ratios": [1.7, 1.0]})
    fig.patch.set_facecolor(BG)
    for ax in (axf, axp):
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=7)

    # --- left: top-down floor ---
    axf.set_aspect("equal")
    axf.set_xlim(0, world.width)
    axf.set_ylim(0, world.height)
    axf.set_title("footstep plan → ZMP-stable walk", color=INK, fontsize=11, pad=8)
    for (xmin, ymin, xmax, ymax) in world.obstacles:
        axf.add_patch(Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax),
                               (xmin, ymax)], closed=True, facecolor="#2b3344",
                              edgecolor="none"))
    # all footprints, faint
    foot_patches = []
    for s in plan.states:
        col = LEFT_FOOT if s.foot == LEFT else RIGHT_FOOT
        corners = _foot_corners(s.x, s.y, s.theta, FOOT_LENGTH, FOOT_WIDTH)
        p = Polygon(corners, closed=True, facecolor=col, edgecolor=col,
                    alpha=0.18, linewidth=1.0)
        axf.add_patch(p)
        foot_patches.append(p)
    support_hi = Polygon([(0, 0)], closed=True, facecolor="none",
                         edgecolor=INK, linewidth=2.0, alpha=0.0)
    axf.add_patch(support_hi)
    com_trail, = axf.plot([], [], color=COM, lw=1.8, alpha=0.9)
    zmp_trail, = axf.plot([], [], color=ZMP, lw=1.2, alpha=0.7)
    com_dot, = axf.plot([], [], "o", color=COM, ms=11, mec=BG, mew=1.5, zorder=5)
    zmp_dot, = axf.plot([], [], "o", color=ZMP, ms=6, mec=BG, mew=1.0, zorder=6)
    axf.plot([], [], color=COM, lw=6, label="CoM")
    axf.plot([], [], color=ZMP, lw=6, label="ZMP")
    axf.legend(loc="upper left", fontsize=8, framealpha=0.0, labelcolor=INK)

    # --- right: lateral sway over time ---
    axp.set_xlim(0, times[-1])
    ys = wp.ref_y + wp.zmp_y + wp.com_y
    axp.set_ylim(min(ys) - 0.05, max(ys) + 0.05)
    axp.set_title("lateral: ZMP tracks the stepped reference, CoM sways",
                  color=INK, fontsize=9, pad=8)
    axp.set_xlabel("time [s]", color=MUTED, fontsize=8)
    axp.set_ylabel("y [m]", color=MUTED, fontsize=8)
    axp.plot(times, wp.ref_y, color=MUTED, lw=1.0, ls="--", label="ZMP ref")
    axp.plot(times, wp.zmp_y, color=ZMP, lw=1.4, label="ZMP")
    axp.plot(times, wp.com_y, color=COM, lw=1.6, label="CoM")
    axp.legend(loc="upper right", fontsize=7, framealpha=0.0, labelcolor=INK)
    cursor = axp.axvline(0, color=INK, lw=1.0, alpha=0.7)

    def update(fi):
        k = frames_idx[fi]
        com_trail.set_data(wp.com_x[:k + 1], wp.com_y[:k + 1])
        zmp_trail.set_data(wp.zmp_x[:k + 1], wp.zmp_y[:k + 1])
        com_dot.set_data([wp.com_x[k]], [wp.com_y[k]])
        zmp_dot.set_data([wp.zmp_x[k]], [wp.zmp_y[k]])
        si = wp.support[k]
        fx, fy, ft = wp.foot_poses[si]
        support_hi.set_xy(_foot_corners(fx, fy, ft, FOOT_LENGTH, FOOT_WIDTH))
        support_hi.set_alpha(0.95)
        cursor.set_xdata([times[k], times[k]])
        return (com_trail, zmp_trail, com_dot, zmp_dot, support_hi, cursor)

    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=len(frames_idx),
                         interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=98,
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    _optimize_gif(out, fps)
    return wp


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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/footstep_walk.gif")
    ap.add_argument("--turn", action="store_true",
                    help="a curving walk instead of straight forward")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--step-duration", type=float, default=0.7)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--stride", type=int, default=3,
                    help="render every Nth trajectory sample")
    args = ap.parse_args()

    world, plan = _scenario(args.turn)
    if plan is None:
        print("no footstep plan found")
        return
    wp = _animate(world, plan, args.out, args.fps, args.step_duration, args.dt,
                  args.stride)
    rms = 1000.0 * wp.zmp_rms_error()
    print(f"wrote {args.out}  ({len(plan.states)} steps, ZMP rms {rms:.1f} mm)")


if __name__ == "__main__":
    main()
