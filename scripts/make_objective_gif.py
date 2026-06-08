#!/usr/bin/env python3
"""Headline demo — hill vs domination on the same contested centre.

Side-by-side: consecutive hold (KOTH) vs cumulative zone control.
RoboMaster-style chassis on a competition grid.

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
from matplotlib.patches import Circle  # noqa: E402

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, os.pardir, "mrn_coord"))
sys.path.insert(0, _SCRIPT_DIR)

import _battle_gif_render as render  # noqa: E402

from mrn_coord.battle import TEAM_NAMES, battle_scenario, simulate  # noqa: E402
from mrn_coord.battle_objectives import objective_zone  # noqa: E402

ZONE = "#f5cc4d"


def _draw_zone(ax, zone):
    if not zone:
        return
    cx, cy, r = zone
    ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=ZONE,
                        linewidth=1.6, linestyle=(0, (5, 4)), alpha=0.85, zorder=2))


def _progress_pct(res, cfg, fi):
    if not res.objective_progress or fi >= len(res.objective_progress):
        return 0
    prog = res.objective_progress[fi]
    if not prog:
        return 0
    hold = max(1, cfg.objective_hold_ticks)
    return min(100, int(100 * max(prog.values()) / hold))


class ObjectivePanel:
    def __init__(self, ax, name):
        bots, cfg, title = battle_scenario(name)
        self.mode = name
        self.cfg = cfg
        self.res = simulate(bots, cfg, max_ticks=650)
        self.deaths = render.collect_deaths(self.res.frames)

        ax.set_facecolor(render.FIELD)
        ax.set_xlim(0, cfg.width)
        ax.set_ylim(0, cfg.height)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(render.MUTED)
        ax.set_title(title, color=render.INK, fontsize=10, pad=6)
        render.draw_arena(ax, cfg, minimal=True)
        _draw_zone(ax, objective_zone(cfg))
        self.robots = render.RobotLayers(ax, flash_life=7)
        self.prog_txt = ax.text(cfg.width / 2, cfg.height - 1.0, "",
                                ha="center", color=ZONE, fontsize=9, zorder=10)
        self.banner = ax.text(cfg.width / 2, cfg.height / 2, "",
                              ha="center", va="center", fontsize=18,
                              fontweight="bold", alpha=0.0, zorder=12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media/objective_duel.gif")
    ap.add_argument("--fps", type=int, default=14)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.patch.set_facecolor(render.BG)
    fig.suptitle("OBJECTIVE MODES — fight for the centre, not just annihilation",
                 color=render.INK, fontsize=12, fontweight="bold", y=0.98)

    panels = [ObjectivePanel(ax, name) for ax, name in zip(axes, ("hill", "domination"))]
    nframes = max(len(p.res.frames) for p in panels)
    ticks = list(range(nframes)) + [nframes - 1] * args.fps

    def update(fi):
        t = ticks[fi]
        for p in panels:
            fi_local = min(t, len(p.res.frames) - 1)
            frame = p.res.frames[fi_local]
            prev = p.res.frames[fi_local - 1] if fi_local > 0 else None
            shots = p.res.shots[fi_local] if fi_local < len(p.res.shots) else ()
            p.robots.update(frame, prev, shots, p.deaths, fi_local)
            p.prog_txt.set_text(f"{p.mode} {_progress_pct(p.res, p.cfg, fi_local)}%")
            if p.res.winner is not None and fi_local >= len(p.res.frames) - 1:
                p.banner.set_text(f"{TEAM_NAMES[p.res.winner].upper()} WINS")
                p.banner.set_color(render.TEAM_HEX[p.res.winner])
                p.banner.set_alpha(0.92)
        return []

    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.05, wspace=0.08)
    anim = FuncAnimation(fig, update, frames=len(ticks),
                         interval=1000 // args.fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=96,
              savefig_kwargs={"facecolor": render.BG})
    plt.close(fig)
    render.optimize_gif(args.out, args.fps, colors=64)
    for p in panels:
        print(f"  {p.mode}: {TEAM_NAMES.get(p.res.winner, 'draw')} in {p.res.ticks} ticks")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
