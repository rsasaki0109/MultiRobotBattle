#!/usr/bin/env python3
"""Headline demo — greedy vs MAPF-planned maneuver on the chokepoint.

Runs the same soldiers-on-chokepoint setup twice (greedy pursuit vs prioritized
maneuver + Hungarian assignment) and renders a side-by-side GIF with RoboMaster
chassis art.

    python3 scripts/make_maneuver_gif.py --out docs/media/maneuver_duel.gif
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
    RED,
    BattleConfig,
    make_company,
    simulate,
)
import random  # noqa: E402


from mrn_coord.battle_terrain import chokepoint_terrain  # noqa: E402


def _make_pair(seed=11):
    base = dict(**chokepoint_terrain(), tactics="count_aware", formation="wedge",
                assignment="hungarian", maneuver_replan_ticks=15)
    rng = random.Random(seed)
    center_r = (BattleConfig().width * 0.13, BattleConfig().height * 0.5)
    center_b = (BattleConfig().width * 0.87, BattleConfig().height * 0.5)
    roster = [("soldier", 10)]

    cfg_g = BattleConfig(**base, maneuver="greedy")
    cfg_p = BattleConfig(**base, maneuver="prioritized")
    bots_g = (make_company(cfg_g, RED, center_r, roster, rng, jitter=2.8) +
              make_company(cfg_g, 1, center_b, roster, random.Random(seed + 1),
                           jitter=2.8))
    bots_p = (make_company(cfg_p, RED, center_r, roster, random.Random(seed + 2),
                           jitter=2.8) +
              make_company(cfg_p, 1, center_b, roster, random.Random(seed + 3),
                           jitter=2.8))
    return (cfg_g, simulate(bots_g, cfg_g, max_ticks=700),
            simulate(bots_p, cfg_p, max_ticks=700))


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
        ax.set_title(subtitle, color=render.INK, fontsize=10)
        render.draw_arena(ax, cfg, minimal=True)
        render.draw_terrain(ax, cfg)
        self.robots = render.RobotLayers(ax, flash_life=7)
        self.banner = ax.text(cfg.width / 2, cfg.height / 2, "",
                              ha="center", va="center", fontsize=18,
                              fontweight="bold", alpha=0.0, zorder=12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media/maneuver_duel.gif")
    ap.add_argument("--fps", type=int, default=14)
    args = ap.parse_args()

    cfg, res_g, res_p = _make_pair()
    nframes = max(len(res_g.frames), len(res_p.frames))
    ticks = list(range(nframes)) + [nframes - 1] * args.fps

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    fig.patch.set_facecolor(render.BG)
    fig.suptitle("Chokepoint duel — greedy pursuit vs prioritized MAPF maneuver",
                 color=render.INK, fontsize=12, fontweight="bold", y=0.98)

    panels = [
        ManeuverPanel(axes[0], cfg, res_g, "Greedy pursuit"),
        ManeuverPanel(axes[1], cfg, res_p, "Prioritized MAPF maneuver"),
    ]

    def update(fi):
        t = ticks[fi]
        for p in panels:
            fi_local = min(t, len(p.res.frames) - 1)
            frame = p.res.frames[fi_local]
            prev = p.res.frames[fi_local - 1] if fi_local > 0 else None
            shots = p.res.shots[fi_local] if fi_local < len(p.res.shots) else ()
            p.robots.update(frame, prev, shots, p.deaths, fi_local)
            w = p.res.winner
            if w is not None and fi_local >= len(p.res.frames) - 1:
                p.banner.set_text(f"{'RED' if w == RED else 'BLUE'} WINS")
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
    print(f"wrote {args.out}  greedy={res_g.winner} mapf={res_p.winner}")


if __name__ == "__main__":
    main()
