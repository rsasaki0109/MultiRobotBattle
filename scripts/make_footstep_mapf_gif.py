#!/usr/bin/env python3
"""Animate multi-humanoid footstep MAPF — a *real* ZMP-preview walk per humanoid.

Several humanoids plan footsteps to their goals; prioritized footstep MAPF
(`mrn_coord.mapf.footstep_mapf`) deconflicts their **bodies** tick by tick, so
they cross a shared area without touching. Each deconflicted footstep plan is
then turned into a **dynamically stable walk** by the same ZMP preview-control
simulator used for the single-humanoid GIF (`mrn_coord.mapf.lipm_walk`,
Kajita et al. 2003) — so the motion you see is the genuine center-of-mass
trajectory, not a disc sliding along footstep centres.

For every humanoid the animation draws, all from the real code:

- its planned **footsteps** (faint oriented rectangles, one colour each),
- the current **support foot**, lit up (the foot the ZMP must stay over),
- the **ZMP** (small dot) hugging that support foot — the proof of stability,
- the swaying **center of mass** (bright dot + trail), driven by preview control,
- a **torso disc** centred on the CoM — the body the coordinator keeps clear,
- a goal ring.

All humanoids share one ``step_duration`` / ``dt`` clock, so footfalls line up
with the MAPF ticks at which the bodies were deconflicted; the torsos are drawn
a hair under the planning radius so the realised CoM sway stays provably clear
(the script asserts the minimum inter-torso distance stays positive).

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
from mrn_coord.mapf.lipm_walk import generate_walk  # noqa: E402

BG = "#0b0e14"
GRID = "#1b2230"
INK = "#c9d1d9"
MUTED = "#8b95a7"
ZMP = "#ffd166"
COLORS = ["#5b8cff", "#ff8b5b", "#33d6a6", "#c792ea"]
# The coordinator reserves a disc of DEFAULT_BODY_RADIUS at every footstep tick.
# The torso we *draw* (and track with the simulated CoM) is a hair smaller, so
# the lateral CoM sway between ticks stays clear too; verified at render time.
TORSO_R = 0.19


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


def _min_torso_clearance(walks, ids):
    """Smallest inter-humanoid CoM distance over the whole aligned walk."""
    n = max(len(walks[k].com_x) for k in ids)
    worst = float("inf")
    for i in range(n):
        pts = []
        for k in ids:
            w = walks[k]
            j = min(i, len(w.com_x) - 1)
            pts.append((w.com_x[j], w.com_y[j]))
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                (ax, ay), (bx, by) = pts[a], pts[b]
                worst = min(worst, math.hypot(ax - bx, ay - by))
    return worst


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/footstep_mapf.gif")
    ap.add_argument("--scenario", default="threeway",
                    choices=["crossing", "threeway"])
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--step-duration", type=float, default=0.7)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--stride", type=int, default=4,
                    help="render every Nth trajectory sample")
    args = ap.parse_args()

    world, agents = _scenario(args.scenario)
    plans = prioritized_footstep_mapf(world, agents, w=2.0)
    ids = [k for k in agents if plans.get(k) is not None]

    # Turn each deconflicted footstep plan into a real ZMP-preview walk. One
    # shared clock (step_duration, dt) keeps the footfalls aligned across
    # humanoids, so the MAPF tick-level deconfliction still holds.
    walks = {k: generate_walk(plans[k].states, step_duration=args.step_duration,
                              dt=args.dt) for k in ids}
    nsamp = max(len(walks[k].com_x) for k in ids)

    clearance = _min_torso_clearance(walks, ids)
    assert clearance > 2 * TORSO_R, (
        f"torsos overlap: min CoM dist {clearance:.3f} <= {2 * TORSO_R}")

    # sample indices, padded with a short hold at the end
    frames_idx = list(range(0, nsamp, args.stride)) + [nsamp - 1]
    frames_idx += [nsamp - 1] * (args.fps // 2)

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_title("multi-humanoid footstep MAPF → ZMP-preview walks, bodies stay clear",
                 color=INK, fontsize=10.5, pad=10)

    torsos, com_dots, com_trails, zmp_dots, support_hi = {}, {}, {}, {}, {}
    for i, k in enumerate(ids):
        col = COLORS[i % len(COLORS)]
        # all footsteps, faint
        for s in plans[k].states:
            ax.add_patch(Polygon(_foot_corners(s.x, s.y, s.theta, FOOT_LENGTH,
                                               FOOT_WIDTH), closed=True,
                                 facecolor=col, edgecolor=col, alpha=0.13,
                                 linewidth=0.8))
        # goal ring
        gx, gy = agents[k][1][0], agents[k][1][1]
        ax.add_patch(Circle((gx, gy), 0.12, fill=False, edgecolor=col,
                            linewidth=2.0, alpha=0.8))
        # support-foot highlight (lit, on top of the faint prints)
        support_hi[k] = Polygon([(0, 0)], closed=True, facecolor=col,
                                edgecolor=INK, linewidth=1.6, alpha=0.0,
                                zorder=3)
        ax.add_patch(support_hi[k])
        # torso disc (the deconflicted body), tracked by the simulated CoM
        torso = Circle((0, 0), TORSO_R, facecolor=col, edgecolor=BG,
                       linewidth=1.5, alpha=0.32, zorder=4)
        ax.add_patch(torso)
        torsos[k] = torso
        com_trails[k], = ax.plot([], [], color=col, lw=1.5, alpha=0.7, zorder=5)
        com_dots[k], = ax.plot([], [], "o", color=col, ms=8, mec=BG, mew=1.2,
                               zorder=7)
        zmp_dots[k], = ax.plot([], [], "o", color=ZMP, ms=4.5, mec=BG, mew=0.6,
                               zorder=8)
        ax.plot([], [], color=col, lw=6, label=f"humanoid {k}")
    ax.plot([], [], "o", color=ZMP, ms=6, mec=BG, mew=0.6, label="ZMP")
    ax.legend(loc="upper center", ncol=len(ids) + 1, fontsize=8.5,
              framealpha=0.0, labelcolor=INK, bbox_to_anchor=(0.5, -0.01))

    def update(fi):
        k_sample = frames_idx[fi]
        artists = []
        for k in ids:
            w = walks[k]
            j = min(k_sample, len(w.com_x) - 1)
            cx, cy = w.com_x[j], w.com_y[j]
            torsos[k].center = (cx, cy)
            com_dots[k].set_data([cx], [cy])
            zmp_dots[k].set_data([w.zmp_x[j]], [w.zmp_y[j]])
            # the CoM trail is a pure function of this frame's sample index, so
            # it is immune to any frame-callback ordering quirks
            com_trails[k].set_data(w.com_x[:j + 1], w.com_y[:j + 1])
            fx, fy, ft = w.foot_poses[w.support[j]]
            support_hi[k].set_xy(_foot_corners(fx, fy, ft, FOOT_LENGTH,
                                               FOOT_WIDTH))
            support_hi[k].set_alpha(0.5)
            artists += [torsos[k], com_dots[k], zmp_dots[k], com_trails[k],
                        support_hi[k]]
        return artists

    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=len(frames_idx),
                         interval=1000 / args.fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=98,
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    _optimize_gif(args.out, args.fps)
    print(f"wrote {args.out}  ({len(ids)} humanoids, {nsamp} samples, "
          f"min inter-torso clearance {clearance - 2 * TORSO_R:.3f} m)")


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
