#!/usr/bin/env python3
"""Headline demo — Hungarian+greedy vs CBS-TA+prioritized on the chokepoint.

Same soldiers and terrain; only the MAPF layers change (assignment + maneuver).
Side-by-side GIF with RoboMaster chassis art for README / docs/media.

    python3 scripts/make_mapf_stack_gif.py --out docs/media/mapf_stack_duel.gif
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
    Bot,
    make_company,
    simulate,
)
import random  # noqa: E402

from mrn_coord.battle_terrain import chokepoint_terrain  # noqa: E402

BASE_KW = dict(
    **chokepoint_terrain(),
    tactics="count_aware",
    formation="wedge",
    maneuver_replan_ticks=15,
    assignment_replan_ticks=15,
)


def _clone_bots(bots):
    return [Bot(b.x, b.y, b.vx, b.vy, b.team, b.hp, b.max_hp, alive=b.alive,
                dps=b.dps, attack_range=b.attack_range, max_speed=b.max_speed,
                kind=b.kind)
            for b in bots]


def _spawn(seed=11, n=10):
    rng = random.Random(seed)
    center_r = (BattleConfig().width * 0.13, BattleConfig().height * 0.5)
    center_b = (BattleConfig().width * 0.87, BattleConfig().height * 0.5)
    roster = [("soldier", n)]
    cfg = BattleConfig(**BASE_KW)
    red = make_company(cfg, RED, center_r, roster, rng, jitter=2.8)
    blue = make_company(cfg, 1, center_b, roster, random.Random(seed + 1),
                        jitter=2.8)
    return red + blue


def _run_pair(seed=11):
    spawn = _spawn(seed)
    cfg_local = BattleConfig(
        **BASE_KW,
        assignment="none",
        assignment_by_team={RED: "hungarian"},
        maneuver="greedy",
        maneuver_by_team={1: "greedy"},
    )
    cfg_mapf = BattleConfig(
        **BASE_KW,
        assignment="none",
        assignment_by_team={RED: "cbs_ta"},
        maneuver="greedy",
        maneuver_by_team={RED: "prioritized", 1: "greedy"},
    )
    res_local = simulate(_clone_bots(spawn), cfg_local, max_ticks=700)
    res_mapf = simulate(_clone_bots(spawn), cfg_mapf, max_ticks=700)
    return cfg_local, res_local, res_mapf


class MapfStackPanel:
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
    ap.add_argument("--out", default="docs/media/mapf_stack_duel.gif")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--fps", type=int, default=14)
    args = ap.parse_args()

    cfg, res_local, res_mapf = _run_pair(args.seed)
    nframes = max(len(res_local.frames), len(res_mapf.frames))
    ticks = list(range(nframes)) + [nframes - 1] * args.fps

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    fig.patch.set_facecolor(render.BG)
    fig.suptitle("MAPF stack duel — Hungarian+greedy vs CBS-TA+prioritized",
                 color=render.INK, fontsize=12, fontweight="bold", y=0.98)

    panels = [
        MapfStackPanel(axes[0], cfg, res_local, "Hungarian + greedy"),
        MapfStackPanel(axes[1], cfg, res_mapf, "CBS-TA + prioritized MAPF"),
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
    print(f"wrote {args.out}  local={res_local.winner} mapf={res_mapf.winner}")


if __name__ == "__main__":
    main()
