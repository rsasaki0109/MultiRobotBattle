#!/usr/bin/env python3
"""Animate the headline swarm battle — total-war allied campaign.

RoboMaster-style custom chassis, turrets, and tracers on a competition arena.
Eight echelons (infantry, tanks, scouts, snipers × two alliances) clash until one
coalition is wiped out.

    python3 scripts/make_battle_gif.py --out docs/media/battle.gif

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
from matplotlib.patches import Rectangle  # noqa: E402

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, os.pardir, "mrn_coord"))
sys.path.insert(0, _SCRIPT_DIR)

import _battle_gif_render as render  # noqa: E402

from mrn_coord.battle import (  # noqa: E402
    ALLIANCE_NAMES,
    TEAM_NAMES,
    battle_scenario,
    simulate,
)


def _alliance_totals(counts, teams, alliances):
    totals = {}
    for team, c in zip(teams, counts):
        aid = alliances.get(team, team)
        totals[aid] = totals.get(aid, 0) + c
    return totals


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/media/battle.gif")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--stride", type=int, default=5,
                    help="record every Nth simulation tick")
    ap.add_argument("--max-ticks", type=int, default=1000)
    args = ap.parse_args()

    bots, cfg, title = battle_scenario("grand_alliance")
    res = simulate(bots, cfg, max_ticks=args.max_ticks, frame_stride=args.stride)
    nframes = len(res.frames)
    teams = res.teams
    alliances = res.alliances
    n0 = {t: sum(1 for b in bots if b.team == t) for t in teams}
    n_total = len(bots)
    win_alliance = res.winning_alliance
    win_name = (ALLIANCE_NAMES.get(win_alliance, "draw").upper()
                if win_alliance is not None else "DRAW")
    deaths = render.collect_deaths(res.frames)

    ticks = list(range(0, nframes, 1))
    ticks += [nframes - 1] * args.fps

    fig, ax = plt.subplots(figsize=(16.2, 7.6))
    fig.patch.set_facecolor(render.BG)
    ax.set_xlim(-2, cfg.width + 2)
    ax.set_ylim(-3.8, cfg.height + 2.8)
    ax.set_aspect("equal")
    ax.axis("off")
    render.draw_arena(ax, cfg)
    render.draw_terrain(ax, cfg)

    ax.text(cfg.width / 2, cfg.height + 1.55,
            f"ROBOMASTER TOTAL WAR  —  {n_total} custom bots · four armies · two alliances",
            color=render.INK, fontsize=13.5, fontweight="bold", ha="center", va="bottom")
    ax.text(cfg.width / 2, cfg.height + 0.42, title,
            color=render.MUTED, fontsize=8.5, ha="center", va="bottom")

    bar_y, bar_h = -2.55, 1.05
    half = cfg.width / 2 - 2.5
    west_lab = ax.text(2.5, bar_y + bar_h + 0.18, "", color=render.ALLIANCE_HEX[0],
                       fontsize=11, fontweight="bold", ha="left", va="bottom")
    east_lab = ax.text(cfg.width - 2.5, bar_y + bar_h + 0.18, "",
                       color=render.ALLIANCE_HEX[1], fontsize=11, fontweight="bold",
                       ha="right", va="bottom")
    cas_lab = ax.text(cfg.width / 2, bar_y + bar_h + 0.18, "", color=render.MUTED,
                      fontsize=9, ha="center", va="bottom", family="monospace")
    ax.add_patch(Rectangle((2.5, bar_y), half, bar_h, fill=False,
                           edgecolor="#2a3242", lw=1.0))
    ax.add_patch(Rectangle((cfg.width / 2 + 2.5, bar_y), half, bar_h, fill=False,
                           edgecolor="#2a3242", lw=1.0))
    west_bar = Rectangle((2.5 + half, bar_y), 0.0, bar_h,
                         facecolor=render.ALLIANCE_HEX[0], edgecolor="none", zorder=8)
    east_bar = Rectangle((cfg.width / 2 + 2.5, bar_y), 0.0, bar_h,
                         facecolor=render.ALLIANCE_HEX[1], edgecolor="none", zorder=8)
    ax.add_patch(west_bar)
    ax.add_patch(east_bar)
    team_lab = ax.text(cfg.width / 2, bar_y - 0.65, "", color=render.MUTED,
                       fontsize=7.5, ha="center", va="top", family="monospace")
    robots = render.RobotLayers(ax)
    banner = ax.text(cfg.width / 2, cfg.height / 2, "", ha="center", va="center",
                     fontsize=34, fontweight="bold", zorder=12, alpha=0.0)

    n0_west = sum(n0[t] for t in teams if alliances.get(t, t) == 0)
    n0_east = sum(n0[t] for t in teams if alliances.get(t, t) == 1)

    def update(fi):
        t = ticks[fi]
        frame = res.frames[t]
        prev = res.frames[t - 1] if t > 0 else None
        projs = res.projectiles[t] if t < len(res.projectiles) else ()
        robots.update(frame, prev, res.shots[t], projs, deaths, t)

        counts = res.counts[min(t, len(res.counts) - 1)]
        totals = _alliance_totals(counts, teams, alliances)
        west = totals.get(0, 0)
        east = totals.get(1, 0)
        west_bar.set_width(half * west / max(n0_west, 1))
        west_bar.set_x(2.5 + half - half * west / max(n0_west, 1))
        east_bar.set_width(half * east / max(n0_east, 1))
        west_lab.set_text(f"WESTERN  {west}")
        east_lab.set_text(f"{east}  EASTERN")
        cas_lab.set_text(f"KIA  {n_total - west - east}")
        parts = [f"{TEAM_NAMES[team]} {c}" for team, c in zip(teams, counts)]
        team_lab.set_text("  ·  ".join(parts))

        if t >= nframes - 1 and win_alliance is not None:
            banner.set_text(f"{win_name}  VICTORY")
            banner.set_color(render.ALLIANCE_HEX[win_alliance])
            banner.set_alpha(0.94)
        return []

    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01)
    anim = FuncAnimation(fig, update, frames=len(ticks),
                         interval=1000 / args.fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=102,
              savefig_kwargs={"facecolor": render.BG})
    plt.close(fig)
    render.optimize_gif(args.out, args.fps)
    print(f"wrote {args.out}  ({n_total} bots, {res.ticks} ticks -> {win_name})")


if __name__ == "__main__":
    main()
