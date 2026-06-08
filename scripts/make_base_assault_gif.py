#!/usr/bin/env python3
"""Headline demo — base assault: push into the enemy HQ and hold to win.

RoboMaster-style chassis, home-base rings, and capture progress.

    python3 scripts/make_base_assault_gif.py --out docs/media/base_assault.gif
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


def _assault_pct(res, cfg, fi):
    if not res.objective_progress or fi >= len(res.objective_progress):
        return 0
    prog = res.objective_progress[fi]
    hold = max(1, cfg.objective_hold_ticks)
    vals = [v for k, v in prog.items() if str(k).startswith("assault_")]
    if not vals:
        vals = [v for v in prog.values() if isinstance(v, (int, float))]
    if not vals:
        return 0
    return min(100, int(100 * max(vals) / hold))


def _draw_bases(ax, zones):
    for z in zones:
        if z[0] != "base":
            continue
        _, team, bx, by, br = z
        col = render.TEAM_HEX.get(team, render.MUTED)
        ax.add_patch(Circle((bx, by), br, fill=False, edgecolor=col,
                            linewidth=1.8, alpha=0.85, zorder=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media/base_assault.gif")
    ap.add_argument("--fps", type=int, default=14)
    args = ap.parse_args()

    bots, cfg, title = battle_scenario("base_assault")
    res = simulate(bots, cfg, max_ticks=900)
    zones = list(res.objective_zone)
    nframes = len(res.frames)
    deaths = render.collect_deaths(res.frames)
    ticks = list(range(nframes)) + [nframes - 1] * args.fps

    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.patch.set_facecolor(render.BG)
    fig.suptitle("BASE ASSAULT — breach the enemy HQ and hold to capture",
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
    render.draw_terrain(ax, cfg)
    _draw_bases(ax, zones)

    robots = render.RobotLayers(ax, flash_life=7)
    pct_text = ax.text(cfg.width / 2, cfg.height * 0.08, "", ha="center",
                       color="#f5cc4d", fontsize=11, fontweight="bold", zorder=12)
    banner = ax.text(cfg.width / 2, cfg.height / 2, "", ha="center", va="center",
                     fontsize=20, fontweight="bold", alpha=0.0, zorder=12)

    def update(fi):
        t = ticks[fi]
        frame = res.frames[t]
        prev = res.frames[t - 1] if t > 0 else None
        shots = res.shots[t] if t < len(res.shots) else ()
        projs = res.projectiles[t] if t < len(res.projectiles) else ()
        robots.update(frame, prev, shots, projs, deaths, t)
        pct = _assault_pct(res, cfg, t)
        pct_text.set_text(f"capture {pct}%" if pct else "")
        if res.winner is not None and t >= nframes - 1:
            banner.set_text(f"{TEAM_NAMES[res.winner].upper()} CAPTURES HQ")
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
