#!/usr/bin/env python3
"""Headline demo — capture the flag: grab the centre flag, fight home.

RoboMaster-style chassis on a competition grid.

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
from matplotlib.patches import Circle  # noqa: E402

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, os.pardir, "mrn_coord"))
sys.path.insert(0, _SCRIPT_DIR)

import _battle_gif_render as render  # noqa: E402

from mrn_coord.battle import TEAM_NAMES, battle_scenario, simulate  # noqa: E402

FLAG = "#f5cc4d"


def _draw_zones(ax, zones):
    for z in zones:
        if z[0] == "flag":
            _, fx, fy, fr = z
            ax.add_patch(Circle((fx, fy), fr, fill=False, edgecolor=FLAG,
                                linewidth=1.4, linestyle=(0, (4, 4)), alpha=0.8, zorder=2))
        elif z[0] == "base":
            _, team, bx, by, br = z
            col = render.TEAM_HEX.get(team, render.MUTED)
            ax.add_patch(Circle((bx, by), br, fill=False, edgecolor=col,
                                linewidth=1.6, alpha=0.75, zorder=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media/ctf_duel.gif")
    ap.add_argument("--fps", type=int, default=14)
    args = ap.parse_args()

    bots, cfg, title = battle_scenario("ctf")
    res = simulate(bots, cfg, max_ticks=900)
    zones = list(res.objective_zone)
    nframes = len(res.frames)
    deaths = render.collect_deaths(res.frames)
    ticks = list(range(nframes)) + [nframes - 1] * args.fps

    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.patch.set_facecolor(render.BG)
    fig.suptitle("CAPTURE THE FLAG — fight for the centre, score at home",
                 color=render.INK, fontsize=12, fontweight="bold", y=0.98)
    ax.set_xlim(0, cfg.width)
    ax.set_ylim(0, cfg.height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(render.MUTED)
    ax.set_title(title, color=render.INK, fontsize=10)
    render.draw_arena(ax, cfg, minimal=True)
    _draw_zones(ax, zones)

    robots = render.RobotLayers(ax, flash_life=7)
    flag_scat = ax.scatter([], [], s=90, c=FLAG, marker="D", edgecolors="#fff",
                           linewidths=0.5, zorder=11)
    banner = ax.text(cfg.width / 2, cfg.height / 2, "", ha="center", va="center",
                     fontsize=20, fontweight="bold", alpha=0.0, zorder=12)

    def update(fi):
        t = ticks[fi]
        frame = res.frames[t]
        prev = res.frames[t - 1] if t > 0 else None
        shots = res.shots[t] if t < len(res.shots) else ()
        robots.update(frame, prev, shots, deaths, t)

        prog = (res.objective_progress or [{}])[min(t, len(res.objective_progress) - 1)]
        fx, fy = prog.get("flag", [cfg.width / 2, cfg.height / 2])
        flag_scat.set_offsets([[fx, fy]])

        if res.winner is not None and t >= nframes - 1:
            banner.set_text(f"{TEAM_NAMES[res.winner].upper()} CAPTURES")
            banner.set_color(render.TEAM_HEX[res.winner])
            banner.set_alpha(0.92)
        return []

    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.06)
    anim = FuncAnimation(fig, update, frames=len(ticks),
                         interval=1000 // args.fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=96,
              savefig_kwargs={"facecolor": render.BG})
    plt.close(fig)
    render.optimize_gif(args.out, args.fps, colors=64)
    print(f"wrote {args.out}  winner={TEAM_NAMES.get(res.winner)} ticks={res.ticks}")


if __name__ == "__main__":
    main()
