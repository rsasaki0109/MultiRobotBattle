#!/usr/bin/env python3
"""Headline demo — MAPF stack vs local rules on capture-the-flag.

Same 10 vs 10 spawn; red runs Hungarian+greedy (left) or CBS-TA+prioritized MAPF
(right). RoboMaster-style chassis art.

    python3 scripts/make_ctf_mapf_gif.py --out docs/media/ctf_mapf.gif
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
    ctf_mapf_pair,
    simulate,
)

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


class CtfMapfPanel:
    def __init__(self, ax, cfg, res, subtitle, zones):
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
        _draw_zones(ax, zones)
        self.robots = render.RobotLayers(ax, flash_life=7)
        self.flag_scat = ax.scatter([], [], s=80, c=FLAG, marker="D",
                                    edgecolors="#fff", linewidths=0.5, zorder=11)
        self.banner = ax.text(cfg.width / 2, cfg.height / 2, "",
                              ha="center", va="center", fontsize=18,
                              fontweight="bold", alpha=0.0, zorder=12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media/ctf_mapf.gif")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--fps", type=int, default=14)
    args = ap.parse_args()

    spawn, cfg_local, cfg_mapf, titles = ctf_mapf_pair(n=args.n, seed=args.seed)
    res_local = simulate(clone_bots(spawn), cfg_local, max_ticks=900)
    res_mapf = simulate(clone_bots(spawn), cfg_mapf, max_ticks=900)
    zones = list(res_local.objective_zone)
    nframes = max(len(res_local.frames), len(res_mapf.frames))
    ticks = list(range(nframes)) + [nframes - 1] * args.fps

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    fig.patch.set_facecolor(render.BG)
    fig.suptitle(f"CTF × MAPF — {args.n * 2} robots, grab the flag and run home",
                 color=render.INK, fontsize=12, fontweight="bold", y=0.98)

    panels = [
        CtfMapfPanel(axes[0], cfg_local, res_local, titles[0], zones),
        CtfMapfPanel(axes[1], cfg_mapf, res_mapf, titles[1], zones),
    ]

    def update(fi):
        t = ticks[fi]
        for p in panels:
            fi_local = min(t, len(p.res.frames) - 1)
            frame = p.res.frames[fi_local]
            prev = p.res.frames[fi_local - 1] if fi_local > 0 else None
            shots = p.res.shots[fi_local] if fi_local < len(p.res.shots) else ()
            p.robots.update(frame, prev, shots, p.deaths, fi_local)
            prog = (p.res.objective_progress or [{}])[
                min(fi_local, len(p.res.objective_progress) - 1)]
            fx, fy = prog.get("flag", [p.cfg.width / 2, p.cfg.height / 2])
            p.flag_scat.set_offsets([[fx, fy]])
            w = p.res.winner
            if w is not None and fi_local >= len(p.res.frames) - 1:
                p.banner.set_text(f"{TEAM_NAMES[w].upper()} CAPTURES")
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
