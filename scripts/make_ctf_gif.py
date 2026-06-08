#!/usr/bin/env python3
"""Headline demo — capture the flag: grab the centre flag, fight home.

    python3 scripts/make_ctf_gif.py --out docs/media/ctf_duel.gif
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

from mrn_coord.battle import RED, BLUE, TEAM_NAMES, battle_scenario, simulate  # noqa: E402

BG = "#0b0e14"
INK = "#e6edf3"
MUTED = "#8b95a7"
FLAG = "#f5cc4d"
TEAM = {RED: "#ff5b5b", BLUE: "#5b8cff"}


def _draw_zones(ax, zones):
    for z in zones:
        if z[0] == "flag":
            _, fx, fy, fr = z
            ax.add_patch(Circle((fx, fy), fr, fill=False, edgecolor=FLAG,
                                linewidth=1.4, linestyle=(0, (4, 4)), alpha=0.8))
        elif z[0] == "base":
            _, team, bx, by, br = z
            col = TEAM.get(team, MUTED)
            ax.add_patch(Circle((bx, by), br, fill=False, edgecolor=col,
                                linewidth=1.6, alpha=0.75))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media/ctf_duel.gif")
    ap.add_argument("--fps", type=int, default=14)
    args = ap.parse_args()

    bots, cfg, title = battle_scenario("ctf")
    res = simulate(bots, cfg, max_ticks=900)
    zones = list(res.objective_zone)
    nframes = len(res.frames)

    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=BG)
    fig.suptitle("CAPTURE THE FLAG — fight for the centre, score at home",
                 color=INK, fontsize=12, fontweight="bold", y=0.98)
    ax.set_facecolor(BG)
    ax.set_xlim(0, cfg.width)
    ax.set_ylim(0, cfg.height)
    ax.set_aspect("equal")
    ax.set_title(title, color=INK, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(MUTED)
    _draw_zones(ax, zones)

    scat = ax.scatter([], [], s=34, c=[], alpha=0.9)
    lc = LineCollection([], linewidths=0.7, alpha=0.42)
    ax.add_collection(lc)
    flag_scat = ax.scatter([], [], s=90, c=FLAG, marker="D", edgecolors="#fff",
                           linewidths=0.5, zorder=8)
    banner = ax.text(cfg.width / 2, cfg.height / 2, "", ha="center", va="center",
                     fontsize=20, fontweight="bold", alpha=0.0, zorder=9)

    def frame(t):
        fi = min(t, len(res.frames) - 1)
        xs, ys, cols = [], [], []
        for (x, y, team, hp, alive, kind) in res.frames[fi]:
            if not alive:
                continue
            xs.append(x)
            ys.append(y)
            cols.append(TEAM[team])
        scat.set_offsets(list(zip(xs, ys)) if xs else [(0, 0)])
        scat.set_color(cols if cols else [MUTED])
        segs = []
        if fi < len(res.shots):
            for (x0, y0, x1, y1, team) in res.shots[fi]:
                segs.append([(x0, y0), (x1, y1)])
        lc.set_segments(segs)
        prog = (res.objective_progress or [{}])[min(fi, len(res.objective_progress) - 1)]
        fx, fy = prog.get("flag", [cfg.width / 2, cfg.height / 2])
        flag_scat.set_offsets([[fx, fy]])
        if res.winner is not None and fi >= len(res.frames) - 2:
            banner.set_text(f"{TEAM_NAMES[res.winner].upper()} CAPTURES")
            banner.set_color(TEAM[res.winner])
            banner.set_alpha(0.92)
        return scat, lc, flag_scat, banner

    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.06)
    anim = FuncAnimation(fig, frame, frames=nframes + 16,
                         interval=1000 // args.fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=96,
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    _optimize_gif(args.out, args.fps)
    print(f"wrote {args.out}  winner={TEAM_NAMES.get(res.winner)} ticks={res.ticks}")


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
