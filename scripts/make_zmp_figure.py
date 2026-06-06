#!/usr/bin/env python3
"""A figure about the Zero-Moment Point: why preview control keeps a biped up.

Plans footsteps (Hornung et al. 2012), then generates the walk twice with ZMP
preview control (Kajita et al. 2003) — once with the full preview gains, once
with the preview term switched off (feedback only) — and draws the difference:

- **Left** — top-down floor with the planned footsteps (the support polygons).
  The **preview** ZMP (green) threads through every support foot — the
  dynamic-stability criterion satisfied. The **feedback-only** ZMP (red) cannot
  anticipate the footfalls, overshoots, and leaves the feet: the robot would
  tip over. Same feet, same plan, only the preview term differs.
- **Right** — the ZMP tracking its stepped reference over time, forward (top)
  and lateral (bottom), with the CoM that produces it.

    python3 scripts/make_zmp_figure.py --out docs/media/zmp_stability.png

Deterministic, headless (Agg), no ROS.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
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
from mrn_coord.mapf.lipm_walk import (  # noqa: E402
    _cart_table_system,
    _dot,
    generate_walk,
    preview_gains,
    zmp_stability,
)


def _feedback_track(zmp_ref, gains):
    """A no-preview baseline: state feedback that tracks only the *current*
    reference, ``v = -K (x - [ref_k, 0, 0])``. It cannot anticipate the
    footfalls, so the induced ZMP overshoots at every step transition."""
    A, b, c = _cart_table_system(gains.z_h, gains.dt, gains.g)
    K = gains.K
    x = [zmp_ref[0], 0.0, 0.0]
    zmp = []
    for r in zmp_ref:
        v = -_dot(K, [x[0] - r, x[1], x[2]])
        zmp.append(_dot(c, x))
        x = [A[0][0] * x[0] + A[0][1] * x[1] + A[0][2] * x[2] + b[0] * v,
             A[1][1] * x[1] + A[1][2] * x[2] + b[1] * v,
             A[2][2] * x[2] + b[2] * v]
    return zmp


BG = "#0b0e14"
GRID = "#1b2230"
INK = "#c9d1d9"
MUTED = "#8b95a7"
LEFT_FOOT = "#5b8cff"
RIGHT_FOOT = "#ff8b5b"
GOOD = "#33d6a6"      # preview ZMP — stays in support
BAD = "#ff5b6e"       # feedback-only ZMP — leaves support
COM = "#33d6e6"
REF = "#8b95a7"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/media/zmp_stability.png")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--step-duration", type=float, default=0.7)
    args = ap.parse_args()

    world = FootstepWorld(3.2, 1.4, collision_res=0.05)
    plan = plan_footsteps(world, FootstepState(0.4, 0.7, 0.0, "R"), (2.8, 0.7),
                          w=2.0)

    gains = preview_gains(z_h=0.8, dt=args.dt, preview_steps=80, Q=1.0, R=1e-8)
    wp = generate_walk(plan.states, gains=gains, step_duration=args.step_duration,
                       dt=args.dt)
    fb_x = _feedback_track(wp.ref_x, gains)
    fb_y = _feedback_track(wp.ref_y, gains)
    frac, _ = zmp_stability(wp, foot_length=FOOT_LENGTH, foot_width=FOOT_WIDTH)
    # how often the no-preview ZMP is inside its support foot
    from mrn_coord.mapf.lipm_walk import _point_in_foot
    fb_in = sum(
        _point_in_foot(fb_x[k], fb_y[k], *wp.foot_poses[wp.support[k]],
                       length=FOOT_LENGTH, width=FOOT_WIDTH)
        for k in range(len(fb_x)))
    frac0 = fb_in / len(fb_x)
    n_walk = int((len(plan.states) * args.step_duration + 0.4) / args.dt)
    n_walk = min(n_walk, len(wp.zmp_x))

    fig = plt.figure(figsize=(12.0, 5.0))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1.0], height_ratios=[1, 1],
                          hspace=0.45, wspace=0.22)
    axf = fig.add_subplot(gs[:, 0])
    axx = fig.add_subplot(gs[0, 1])
    axy = fig.add_subplot(gs[1, 1])
    for ax in (axf, axx, axy):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=7)

    # --- left: support polygons + the two ZMP trajectories ---
    axf.set_aspect("equal")
    axf.set_xlim(0, world.width)
    axf.set_ylim(0.25, 1.15)
    axf.set_title("the ZMP must stay in the support polygon — only preview keeps it there",
                  color=INK, fontsize=11, pad=8)
    for s in plan.states:
        col = LEFT_FOOT if s.foot == LEFT else RIGHT_FOOT
        axf.add_patch(Polygon(_foot_corners(s.x, s.y, s.theta, FOOT_LENGTH,
                                            FOOT_WIDTH), closed=True,
                              facecolor=col, edgecolor=col, alpha=0.25,
                              linewidth=1.2))
    axf.plot(fb_x[:n_walk], fb_y[:n_walk], color=BAD, lw=1.8, alpha=0.9,
             label=f"no preview — overshoots, leaves the feet ({100 * frac0:.0f}% in)")
    axf.plot(wp.zmp_x[:n_walk], wp.zmp_y[:n_walk], color=GOOD, lw=2.4,
             label=f"preview control — stays in support ({100 * frac:.0f}% in)")
    axf.plot(wp.com_x[:n_walk], wp.com_y[:n_walk], color=COM, lw=1.3, ls=":",
             alpha=0.8, label="CoM")
    axf.legend(loc="lower center", fontsize=8.5, framealpha=0.0, labelcolor=INK,
               ncol=1)

    # --- right: ZMP tracking its reference, forward and lateral ---
    times = [k * args.dt for k in range(n_walk)]
    axx.set_title("forward ZMP tracks its reference", color=INK, fontsize=9, pad=6)
    axx.plot(times, wp.ref_x[:n_walk], color=REF, lw=1.0, ls="--", label="ref")
    axx.plot(times, wp.zmp_x[:n_walk], color=GOOD, lw=1.6, label="ZMP")
    axx.plot(times, wp.com_x[:n_walk], color=COM, lw=1.2, label="CoM")
    axx.set_ylabel("x [m]", color=MUTED, fontsize=8)
    axx.legend(loc="lower right", fontsize=7, framealpha=0.0, labelcolor=INK)

    axy.set_title("lateral ZMP tracks the stepped reference (the sway)",
                  color=INK, fontsize=9, pad=6)
    axy.plot(times, wp.ref_y[:n_walk], color=REF, lw=1.0, ls="--", label="ref")
    axy.plot(times, wp.zmp_y[:n_walk], color=GOOD, lw=1.6, label="ZMP")
    axy.plot(times, wp.com_y[:n_walk], color=COM, lw=1.2, label="CoM")
    axy.set_xlabel("time [s]", color=MUTED, fontsize=8)
    axy.set_ylabel("y [m]", color=MUTED, fontsize=8)
    axy.legend(loc="upper right", fontsize=7, framealpha=0.0, labelcolor=INK)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}  (preview in-support {100 * frac:.0f}%, "
          f"feedback-only {100 * frac0:.0f}%)")


if __name__ == "__main__":
    main()
