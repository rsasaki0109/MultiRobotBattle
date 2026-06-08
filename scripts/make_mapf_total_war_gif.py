#!/usr/bin/env python3
"""Headline demo — MAPF stack vs local rules on a king-of-the-hill total-war contest.

Same 18 vs 18 spawn fighting for the centre; red runs Hungarian+greedy (left)
or CBS-TA+prioritized MAPF (right). RoboMaster-style chassis art.

    python3 scripts/make_mapf_total_war_gif.py --out docs/media/mapf_total_war.gif
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

from mrn_coord.battle import (  # noqa: E402
    TEAM_NAMES,
    clone_bots,
    mapf_total_war_pair,
    simulate,
)

ZONE = "#f5cc4d"


def _run_pair(n=18, seed=8):
    spawn, cfg_local, cfg_mapf, _ = mapf_total_war_pair(n=n, seed=seed)
    res_local = simulate(clone_bots(spawn), cfg_local, max_ticks=650)
    res_mapf = simulate(clone_bots(spawn), cfg_mapf, max_ticks=650)
    return cfg_local, res_local, res_mapf


def _zone(cfg):
    cx, cy = cfg.objective_center or (cfg.width / 2, cfg.height / 2)
    return cx, cy, cfg.objective_radius


def _progress(res, hold_ticks, fi):
    if not res.objective_progress or fi >= len(res.objective_progress):
        return 0
    prog = res.objective_progress[fi]
    if not prog:
        return 0
    return min(100, int(100 * max(prog.values()) / max(1, hold_ticks)))


class MapfPanel:
    def __init__(self, ax, cfg, res, subtitle):
        self.cfg = cfg
        self.res = res
        self.subtitle = subtitle
        self.deaths = render.collect_deaths(res.frames)

        ax.set_facecolor(render.FIELD)
        ax.set_xlim(0, cfg.width)
        ax.set_ylim(0, cfg.height)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(render.MUTED)
        ax.set_title(subtitle, color=render.INK, fontsize=10)
        render.draw_arena(ax, cfg, minimal=True)
        zx, zy, zr = _zone(cfg)
        ax.add_patch(Circle((zx, zy), zr, fill=False, edgecolor=ZONE,
                            linewidth=1.6, linestyle=(0, (5, 4)), alpha=0.85, zorder=2))
        self.robots = render.RobotLayers(ax, flash_life=7)
        self.prog = ax.text(zx, zy, "", ha="center", va="center",
                            color=ZONE, fontsize=10, fontweight="bold", zorder=10)
        self.banner = ax.text(cfg.width / 2, cfg.height / 2, "",
                              ha="center", va="center", fontsize=18,
                              fontweight="bold", alpha=0.0, zorder=12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media/mapf_total_war.gif")
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--fps", type=int, default=14)
    args = ap.parse_args()

    cfg, res_local, res_mapf = _run_pair(args.n, args.seed)
    nframes = max(len(res_local.frames), len(res_mapf.frames))
    ticks = list(range(nframes)) + [nframes - 1] * args.fps

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    fig.patch.set_facecolor(render.BG)
    fig.suptitle(f"MAPF TOTAL WAR — {args.n * 2} robots, king of the hill",
                 color=render.INK, fontsize=12, fontweight="bold", y=0.98)

    panels = [
        MapfPanel(axes[0], cfg, res_local, "Hungarian + greedy"),
        MapfPanel(axes[1], cfg, res_mapf, "CBS-TA + prioritized MAPF"),
    ]

    def update(fi):
        t = ticks[fi]
        for p in panels:
            fi_local = min(t, len(p.res.frames) - 1)
            frame = p.res.frames[fi_local]
            prev = p.res.frames[fi_local - 1] if fi_local > 0 else None
            shots = p.res.shots[fi_local] if fi_local < len(p.res.shots) else ()
            p.robots.update(frame, prev, shots, p.deaths, fi_local)
            p.prog.set_text(f"{_progress(p.res, cfg.objective_hold_ticks, fi_local)}%")
            w = p.res.winner
            if w is not None and fi_local >= len(p.res.frames) - 1:
                p.banner.set_text(f"{TEAM_NAMES[w].upper()} WINS")
                p.banner.set_color(render.TEAM_HEX[w])
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
    print(f"wrote {args.out}  local={TEAM_NAMES.get(res_local.winner)} "
          f"mapf={TEAM_NAMES.get(res_mapf.winner)}")


if __name__ == "__main__":
    main()
