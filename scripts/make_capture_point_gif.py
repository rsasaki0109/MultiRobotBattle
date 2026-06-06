#!/usr/bin/env python3
"""Animate Capture Point push recovery (Pratt et al. 2006).

A humanoid (a Linear Inverted Pendulum — point mass at height z_h over a foot)
is pushed. Three side-by-side copies take the *same* push but step the foot to a
different place: short of the capture point, exactly at it, and past it. Only the
one that steps to the **Capture Point** xi = x + v/omega0 comes back upright; the
other two topple. Driven by the real `mrn_coord.mapf.capture_point` rollouts.

    python3 scripts/make_capture_point_gif.py --out out/capture_point.gif

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "mrn_coord"))

from mrn_coord.mapf.capture_point import (  # noqa: E402
    capture_point,
    simulate_lipm,
)

BG = "#0b0e14"
GRID = "#1b2230"
INK = "#c9d1d9"
GOOD = "#33d6a6"
BAD = "#ff5b6e"
CP = "#ffd166"
GROUND = "#2b3344"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/capture_point.gif")
    ap.add_argument("--z", type=float, default=0.8, help="CoM height z_h [m]")
    ap.add_argument("--push", type=float, default=0.7, help="push velocity [m/s]")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--duration", type=float, default=1.4)
    ap.add_argument("--dt", type=float, default=0.02)
    args = ap.parse_args()

    z = args.z
    v = args.push
    xi = capture_point(0.0, v, z)
    cases = [
        ("step short", 0.6 * xi, BAD),
        ("step to capture point", xi, GOOD),
        ("step long", 1.4 * xi, BAD),
    ]
    trajs = [simulate_lipm(0.0, v, foot, z, dt=args.dt, duration=args.duration)
             for _, foot, _ in cases]
    nframes = len(trajs[0].x)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.4))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Capture Point push recovery — push {v:.1f} m/s, "
                 f"step to xi = x + v/omega0 = {xi:.2f} m",
                 color=INK, fontsize=12, y=0.99)

    span = 0.9
    poles, masses, titles = [], [], []
    for ax, (name, foot, col), traj in zip(axes, cases, trajs):
        ax.set_facecolor(BG)
        ax.set_aspect("equal")
        ax.set_xlim(-span, span)
        ax.set_ylim(-0.08, z + 0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.axhline(0, color=GROUND, lw=3)
        ax.plot([xi], [0], "v", color=CP, ms=9, mec=BG, mew=0.8)
        ax.text(xi, 0.05, " xi", color=CP, fontsize=8, ha="left")
        ax.plot([foot], [0], "s", color=col, ms=9, mec=BG, mew=0.8)
        pole, = ax.plot([], [], color=col, lw=3, solid_capstyle="round")
        mass, = ax.plot([], [], "o", color=col, ms=20, mec=BG, mew=1.5)
        t = ax.set_title(name, color=col, fontsize=10, pad=6)
        poles.append((pole, foot))
        masses.append(mass)
        titles.append((t, name, traj))

    def update(fi):
        arts = []
        for (pole, foot), mass, (t, name, traj) in zip(poles, masses, titles):
            x = traj.x[fi]
            xc = max(-span + 0.05, min(span - 0.05, x))
            pole.set_data([foot, xc], [0, z])
            mass.set_data([xc], [z])
            fell = abs(x - foot) > z
            t.set_text(name + ("  — fell" if fell else
                               ("  — captured" if fi == len(traj.x) - 1 else "")))
            arts += [pole, mass]
        return arts

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    anim = FuncAnimation(fig, update, frames=nframes,
                         interval=1000 / args.fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=100,
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    _optimize_gif(args.out, args.fps)
    print(f"wrote {args.out}  (xi={xi:.3f} m, captured: {trajs[1].captured()})")


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
