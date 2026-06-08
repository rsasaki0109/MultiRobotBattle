#!/usr/bin/env python3
"""Headline demo — artillery barrage: splash rounds vs clustered infantry.

RoboMaster-style chassis, arcing rounds, and splash rings on detonation.

    python3 scripts/make_artillery_gif.py --out docs/media/artillery_barrage.gif
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

from mrn_coord.battle import RED, battle_scenario, simulate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media/artillery_barrage.gif")
    ap.add_argument("--fps", type=int, default=14)
    args = ap.parse_args()

    bots, cfg, title = battle_scenario("artillery_barrage")
    res = simulate(bots, cfg, max_ticks=900)
    nframes = len(res.frames)
    deaths = render.collect_deaths(res.frames)
    ticks = list(range(nframes)) + [nframes - 1] * args.fps
    splash_patches = []

    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.patch.set_facecolor(render.BG)
    fig.suptitle("ARTILLERY BARRAGE — splash rounds vs line infantry",
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

    robots = render.RobotLayers(ax, flash_life=7)
    banner = ax.text(cfg.width / 2, cfg.height / 2, "", ha="center", va="center",
                     fontsize=20, fontweight="bold", alpha=0.0, zorder=12)

    def update(fi):
        t = ticks[fi]
        frame = res.frames[t]
        prev = res.frames[t - 1] if t > 0 else None
        shots = res.shots[t] if t < len(res.shots) else ()
        projs = res.projectiles[t] if t < len(res.projectiles) else ()
        expls = res.explosions[t] if t < len(res.explosions) else ()
        robots.update(frame, prev, shots, projs, deaths, t)

        for p in splash_patches:
            p.remove()
        splash_patches.clear()
        for ex, ey, radius, team in expls:
            col = render.TEAM_HEX.get(team, "#f5cc4d")
            patch = Circle((ex, ey), radius, fill=True, facecolor=col,
                             edgecolor=col, linewidth=1.2, alpha=0.22, zorder=7)
            ax.add_patch(patch)
            splash_patches.append(patch)
            inner = Circle((ex, ey), radius * 0.35, fill=True, facecolor="#fff3a8",
                           edgecolor="none", alpha=0.35, zorder=8)
            ax.add_patch(inner)
            splash_patches.append(inner)

        if res.winner is not None and t >= res.ticks - 2:
            banner.set_text("RED WINS" if res.winner == RED else "BLUE WINS")
            banner.set_color(render.TEAM_HEX.get(res.winner, render.INK))
            banner.set_alpha(0.95)
        else:
            banner.set_alpha(0.0)
        return []

    anim = FuncAnimation(fig, update, frames=len(ticks), interval=1000 / args.fps,
                         blit=False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    render.optimize_gif(args.out, args.fps)
    print("wrote", args.out, f"({res.ticks} ticks, winner={res.winner})")


if __name__ == "__main__":
    main()
