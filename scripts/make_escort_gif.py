#!/usr/bin/env python3
"""Headline demo — escort: push the payload through terrain to the enemy HQ.

RoboMaster-style chassis, spawn/goal rings, and delivery progress.

    python3 scripts/make_escort_gif.py --out docs/media/escort.gif
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, os.pardir, "mrn_coord"))
sys.path.insert(0, _SCRIPT_DIR)

import _battle_gif_render as render  # noqa: E402

from mrn_coord.battle import TEAM_NAMES, battle_scenario, simulate  # noqa: E402


def _escort_pct(res, fi):
    if not res.objective_progress or fi >= len(res.objective_progress):
        return 0
    prog = res.objective_progress[fi]
    if "escort_pct" in prog:
        return int(prog["escort_pct"])
    gp = prog.get("goal_progress")
    return int(round(100 * gp)) if gp is not None else 0


def _draw_zones(ax, zones):
    for z in zones:
        tag = z[0]
        if tag == "spawn":
            _, team, bx, by, br = z
            col = render.TEAM_HEX.get(team, render.MUTED)
            ax.add_patch(Circle((bx, by), br, fill=False, edgecolor=col,
                                linewidth=1.6, alpha=0.75, linestyle="--", zorder=2))
        elif tag in ("goal", "base"):
            _, team, bx, by, br = z
            col = render.TEAM_HEX.get(team, render.MUTED)
            lw = 2.2 if tag == "goal" else 1.4
            ax.add_patch(Circle((bx, by), br, fill=False, edgecolor=col,
                                linewidth=lw, alpha=0.9, zorder=2))


def _draw_payload(ax, x, y, team):
    col = render.TEAM_HEX.get(team, "#f5cc4d")
    ax.add_patch(FancyBboxPatch((x - 1.1, y - 0.75), 2.2, 1.5,
                                boxstyle="round,pad=0.08",
                                facecolor=col, edgecolor="#f5cc4d",
                                linewidth=1.4, alpha=0.92, zorder=8))
    ax.plot([x - 0.55, x + 0.55], [y, y], color="#0f1320", linewidth=2.2, zorder=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media/escort.gif")
    ap.add_argument("--fps", type=int, default=14)
    args = ap.parse_args()

    bots, cfg, title = battle_scenario("escort")
    res = simulate(bots, cfg, max_ticks=900)
    zones = list(res.objective_zone)
    escort_team = cfg.escort_team if cfg.escort_team is not None else 0
    nframes = len(res.frames)
    deaths = render.collect_deaths(res.frames)
    ticks = list(range(nframes)) + [nframes - 1] * args.fps

    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.patch.set_facecolor(render.BG)
    fig.suptitle("ESCORT — push the payload to the enemy HQ",
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
    _draw_zones(ax, zones)

    robots = render.RobotLayers(ax, flash_life=7)
    pct_text = ax.text(cfg.width / 2, cfg.height * 0.08, "", ha="center",
                       color="#f5cc4d", fontsize=11, fontweight="bold", zorder=12)
    banner = ax.text(cfg.width / 2, cfg.height / 2, "", ha="center", va="center",
                     fontsize=20, fontweight="bold", alpha=0.0, zorder=12)
    payload_art = {"patch": None, "line": None}

    def update(fi):
        t = ticks[fi]
        frame = res.frames[t]
        prev = res.frames[t - 1] if t > 0 else None
        shots = res.shots[t] if t < len(res.shots) else ()
        projs = res.projectiles[t] if t < len(res.projectiles) else ()
        robots.update(frame, prev, shots, projs, deaths, t)
        prog = res.objective_progress[t] if t < len(res.objective_progress) else {}
        payload = prog.get("payload") or []
        if len(payload) == 2:
            px, py = payload
            if payload_art["patch"] is not None:
                payload_art["patch"].remove()
                payload_art["line"][0].remove()
            payload_art["patch"] = FancyBboxPatch(
                (px - 1.1, py - 0.75), 2.2, 1.5, boxstyle="round,pad=0.08",
                facecolor=render.TEAM_HEX.get(escort_team, "#f5cc4d"),
                edgecolor="#f5cc4d", linewidth=1.4, alpha=0.92, zorder=8)
            ax.add_patch(payload_art["patch"])
            payload_art["line"] = ax.plot([px - 0.55, px + 0.55], [py, py],
                                          color="#0f1320", linewidth=2.2, zorder=9)
        pct = _escort_pct(res, t)
        pct_text.set_text(f"delivery {pct}%" if pct else "")
        if res.winner is not None and t >= nframes - 1:
            banner.set_text(f"{TEAM_NAMES[res.winner].upper()} DELIVERS PAYLOAD")
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
