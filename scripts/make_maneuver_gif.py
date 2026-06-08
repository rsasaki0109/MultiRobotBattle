#!/usr/bin/env python3
"""Headline demo — swap only the movement layer on the chokepoint.

Runs the same soldiers-on-chokepoint setup four times: red gets greedy / A* /
prioritized / CBS maneuver while blue stays greedy; Hungarian assignment and
wedge formation are held fixed. Renders a 2×2 GIF with RoboMaster chassis art.

    python3 scripts/make_maneuver_gif.py --out docs/media/maneuver_layers.gif
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, os.pardir, "mrn_coord"))
sys.path.insert(0, _SCRIPT_DIR)

import _battle_gif_render as render  # noqa: E402

from mrn_coord.battle import (  # noqa: E402
    MANEUVER_HEADLINE_LABELS,
    MANEUVER_HEADLINE_MODES,
    RED,
    maneuver_headline_duel,
    simulate,
)


def _run_all(seed=11, max_ticks=700):
    runs = []
    for mode in MANEUVER_HEADLINE_MODES:
        bots, cfg, title = maneuver_headline_duel(mode, seed=seed, n=10)
        res = simulate(bots, cfg, max_ticks=max_ticks)
        runs.append((cfg, res, MANEUVER_HEADLINE_LABELS[mode]))
    return runs


class ManeuverPanel:
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
        ax.set_title(subtitle, color=render.INK, fontsize=9)
        render.draw_arena(ax, cfg, minimal=True)
        render.draw_terrain(ax, cfg)
        self.robots = render.RobotLayers(ax, flash_life=7)
        self.banner = ax.text(cfg.width / 2, cfg.height / 2, "",
                              ha="center", va="center", fontsize=16,
                              fontweight="bold", alpha=0.0, zorder=12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media/maneuver_layers.gif")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--max-ticks", type=int, default=700)
    args = ap.parse_args()

    runs = _run_all(seed=args.seed, max_ticks=args.max_ticks)
    nframes = max(len(r[1].frames) for r in runs)
    ticks = list(range(nframes)) + [nframes - 1] * args.fps

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.2))
    fig.patch.set_facecolor(render.BG)
    fig.suptitle(
        "Headline demo — swap only the movement layer (red MAPF vs blue greedy)",
        color=render.INK, fontsize=12, fontweight="bold", y=0.98,
    )

    panels = []
    for ax, (cfg, res, title) in zip(axes.flat, runs):
        panels.append(ManeuverPanel(ax, cfg, res, title))

    def update(fi):
        t = ticks[fi]
        for p in panels:
            fi_local = min(t, len(p.res.frames) - 1)
            frame = p.res.frames[fi_local]
            prev = p.res.frames[fi_local - 1] if fi_local > 0 else None
            shots = p.res.shots[fi_local] if fi_local < len(p.res.shots) else ()
            projs = (p.res.projectiles[fi_local]
                     if fi_local < len(getattr(p.res, "projectiles", ())) else ())
            p.robots.update(frame, prev, shots, projs, deaths=p.deaths, t=fi_local)
            w = p.res.winner
            if w is not None and fi_local >= len(p.res.frames) - 1:
                p.banner.set_text(f"{'RED' if w == RED else 'BLUE'} WINS")
                p.banner.set_color(render.TEAM_HEX[w])
                p.banner.set_alpha(0.92)
        return []

    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.04,
                        hspace=0.12, wspace=0.08)
    anim = FuncAnimation(fig, update, frames=len(ticks),
                         interval=1000 // args.fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=96,
              savefig_kwargs={"facecolor": render.BG})
    plt.close(fig)
    render.optimize_gif(args.out, args.fps, colors=64)
    winners = [r[1].winner for r in runs]
    print(f"wrote {args.out}  winners={winners}")


if __name__ == "__main__":
    main()
