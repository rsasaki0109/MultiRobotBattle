#!/usr/bin/env python3
"""Headline demo — hill vs domination on the same contested centre.

Side-by-side: consecutive hold (KOTH) vs cumulative zone control.

    python3 scripts/make_objective_gif.py --out docs/media/objective_duel.gif
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "mrn_coord"))

from mrn_coord.battle import (  # noqa: E402
    BLUE,
    RED,
    TEAM_NAMES,
    battle_scenario,
    simulate,
)
from mrn_coord.battle_objectives import objective_zone  # noqa: E402

BG = "#0b0e14"
INK = "#e6edf3"
MUTED = "#8b95a7"
ZONE = "#f5cc4d"
TEAM = {RED: "#ff5b5b", BLUE: "#5b8cff"}


def _draw_zone(ax, zone):
    if not zone:
        return
    cx, cy, r = zone
    ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=ZONE,
                        linewidth=1.6, linestyle=(0, (5, 4)), alpha=0.85))


def _progress_pct(res, cfg, fi):
    if not res.objective_progress or fi >= len(res.objective_progress):
        return 0
    prog = res.objective_progress[fi]
    if not prog:
        return 0
    hold = max(1, cfg.objective_hold_ticks)
    return min(100, int(100 * max(prog.values()) / hold))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media/objective_duel.gif")
    ap.add_argument("--fps", type=int, default=14)
    args = ap.parse_args()

    panels = []
    for name in ("hill", "domination"):
        bots, cfg, title = battle_scenario(name)
        res = simulate(bots, cfg, max_ticks=650)
        panels.append((name, title, cfg, res))

    nframes = max(len(p[3].frames) for p in panels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=BG)
    fig.suptitle("OBJECTIVE MODES — fight for the centre, not just annihilation",
                 color=INK, fontsize=12, fontweight="bold", y=0.98)

    scat, fire, prog_txt, banners = [], [], [], []
    for ax, (mode, title, cfg, _) in zip(axes, panels):
        ax.set_facecolor(BG)
        ax.set_xlim(0, cfg.width)
        ax.set_ylim(0, cfg.height)
        ax.set_aspect("equal")
        ax.set_title(title, color=INK, fontsize=10, pad=6)
        ax.tick_params(colors=MUTED, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(MUTED)
        _draw_zone(ax, objective_zone(cfg))
        scat.append(ax.scatter([], [], s=32, c=[], alpha=0.9))
        lc = LineCollection([], linewidths=0.7, alpha=0.4)
        ax.add_collection(lc)
        fire.append(lc)
        prog_txt.append(ax.text(cfg.width / 2, cfg.height - 1.2, "",
                                ha="center", color=ZONE, fontsize=9))
        banners.append(ax.text(cfg.width / 2, cfg.height / 2, "",
                               ha="center", va="center", fontsize=18,
                               fontweight="bold", alpha=0.0, zorder=9))

    def frame(t):
        artists = []
        for k, (ax, (mode, title, cfg, res), s, lc, pt, bn) in enumerate(
                zip(axes, panels, scat, fire, prog_txt, banners)):
            fi = min(t, len(res.frames) - 1)
            fr = res.frames[fi]
            xs, ys, cols = [], [], []
            for (x, y, team, hp, alive, kind) in fr:
                if not alive:
                    continue
                xs.append(x)
                ys.append(y)
                cols.append(TEAM[team])
            s.set_offsets(list(zip(xs, ys)) if xs else [(0, 0)])
            s.set_color(cols if cols else [MUTED])
            segs = []
            if fi < len(res.shots):
                for (x0, y0, x1, y1, team) in res.shots[fi]:
                    segs.append([(x0, y0), (x1, y1)])
            lc.set_segments(segs)
            pct = _progress_pct(res, cfg, fi)
            pt.set_text(f"{mode} {pct}%")
            if res.winner is not None and fi >= len(res.frames) - 2:
                bn.set_text(f"{TEAM_NAMES[res.winner].upper()} WINS")
                bn.set_color(TEAM[res.winner])
                bn.set_alpha(0.92)
            artists.extend([s, lc, pt, bn])
        return artists

    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.05, wspace=0.08)
    anim = FuncAnimation(fig, frame, frames=nframes + 16,
                         interval=1000 // args.fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=96,
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    _optimize_gif(args.out, args.fps)
    for mode, _, _, res in panels:
        print(f"  {mode}: {TEAM_NAMES.get(res.winner, 'draw')} in {res.ticks} ticks")
    print(f"wrote {args.out}")


def _optimize_gif(path, fps):
    try:
        from PIL import Image, ImageSequence
    except Exception:
        return
    im = Image.open(path)
    frames = [f.convert("RGB").quantize(colors=64, method=Image.Quantize.FASTOCTREE)
              for f in ImageSequence.Iterator(im)]
    if not frames:
        return
    frames[0].save(path, save_all=True, append_images=frames[1:], optimize=True,
                   duration=int(1000 / fps), loop=0, disposal=2)


if __name__ == "__main__":
    main()
