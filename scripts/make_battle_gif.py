#!/usr/bin/env python3
"""Animate the headline swarm battle — total-war allied campaign.

Eight echelons (infantry, tanks, scouts, snipers × two alliances) clash on a
wide field until one coalition is wiped out.

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
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "mrn_coord"))

from mrn_coord.battle import (  # noqa: E402
    ALLIANCE_NAMES,
    BLUE,
    GREEN,
    RED,
    TEAM_NAMES,
    YELLOW,
    battle_scenario,
    simulate,
)

BG = "#080a10"
FIELD = "#0d111a"
INK = "#e6edf3"
MUTED = "#8b95a7"
FRONT = "#3d4a63"
TEAM_COLOR = {
    RED: (1.00, 0.36, 0.36),
    BLUE: (0.36, 0.55, 1.00),
    GREEN: (0.36, 0.85, 0.45),
    YELLOW: (0.96, 0.80, 0.30),
}
ALLIANCE_HEX = {0: "#ff6b5b", 1: "#5b9cff"}
# Dense battlefield — many small discs read as massed formations.
KIND_SIZE = {"": 28, "scout": 14, "soldier": 26, "tank": 52, "sniper": 20}
FLASH_LIFE = 9


def _rgba(team, hp):
    r, g, b = TEAM_COLOR[team]
    a = 0.35 + 0.65 * max(0.0, min(1.0, hp))
    return (r, g, b, a)


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

    deaths = {}
    for t in range(nframes - 1):
        for (b0, b1) in zip(res.frames[t], res.frames[t + 1]):
            if b0[4] and not b1[4]:
                deaths.setdefault(t + 1, []).append((b1[0], b1[1], b1[2]))

    ticks = list(range(0, nframes, 1))
    ticks += [nframes - 1] * args.fps

    fig, ax = plt.subplots(figsize=(16.0, 7.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(FIELD)
    ax.set_xlim(0, cfg.width)
    ax.set_ylim(-3.2, cfg.height + 2.4)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(Rectangle((0, 0), cfg.width, cfg.height, fill=False,
                           edgecolor="#1e2838", lw=2.0))
    # No-man's-land stripe at centre
    mid = cfg.width / 2
    ax.add_patch(Rectangle((mid - 1.2, 0), 2.4, cfg.height, fill=True,
                           facecolor="#121820", edgecolor="none", zorder=0))
    for y in range(0, int(cfg.height), 6):
        ax.plot([mid - 0.4, mid + 0.4], [y, y + 3], color=FRONT, lw=0.6,
                alpha=0.35, zorder=0)

    ax.text(cfg.width / 2, cfg.height + 1.2,
            f"TOTAL WAR  —  {n_total} robots, four armies, two allied fronts",
            color=INK, fontsize=14, fontweight="bold", ha="center", va="bottom")
    ax.text(cfg.width / 2, cfg.height + 0.35, title,
            color=MUTED, fontsize=9, ha="center", va="bottom")

    bar_y, bar_h = -2.4, 1.1
    half = cfg.width / 2 - 2.5
    west_lab = ax.text(2.5, bar_y + bar_h + 0.15, "", color=ALLIANCE_HEX[0],
                       fontsize=11, fontweight="bold", ha="left", va="bottom")
    east_lab = ax.text(cfg.width - 2.5, bar_y + bar_h + 0.15, "",
                       color=ALLIANCE_HEX[1], fontsize=11, fontweight="bold",
                       ha="right", va="bottom")
    cas_lab = ax.text(cfg.width / 2, bar_y + bar_h + 0.15, "", color=MUTED,
                      fontsize=9, ha="center", va="bottom", family="monospace")
    ax.add_patch(Rectangle((2.5, bar_y), half, bar_h, fill=False,
                           edgecolor="#2a3242", lw=1.0))
    ax.add_patch(Rectangle((cfg.width / 2 + 2.5, bar_y), half, bar_h, fill=False,
                           edgecolor="#2a3242", lw=1.0))
    west_bar = Rectangle((2.5 + half, bar_y), 0.0, bar_h,
                         facecolor=ALLIANCE_HEX[0], edgecolor="none")
    east_bar = Rectangle((cfg.width / 2 + 2.5, bar_y), 0.0, bar_h,
                          facecolor=ALLIANCE_HEX[1], edgecolor="none")
    ax.add_patch(west_bar)
    ax.add_patch(east_bar)
    team_lab = ax.text(cfg.width / 2, bar_y - 0.6, "", color=MUTED,
                       fontsize=7.5, ha="center", va="top", family="monospace")

    bots_scatter = ax.scatter([], [], s=150, zorder=5, edgecolors="#080a10",
                              linewidths=0.35)
    fire = LineCollection([], linewidths=0.7, zorder=4, alpha=0.55)
    ax.add_collection(fire)
    flash_scatter = ax.scatter([], [], s=[], facecolors="none", linewidths=1.2,
                               zorder=6)
    banner = ax.text(cfg.width / 2, cfg.height / 2, "", ha="center", va="center",
                     fontsize=32, fontweight="bold", zorder=10, alpha=0.0)

    n0_west = sum(n0[t] for t in teams if alliances.get(t, t) == 0)
    n0_east = sum(n0[t] for t in teams if alliances.get(t, t) == 1)

    def update(fi):
        t = ticks[fi]
        frame = res.frames[t]
        xs, ys, cols, sizes = [], [], [], []
        for (x, y, team, hp, alive, kind) in frame:
            if not alive:
                continue
            xs.append(x)
            ys.append(y)
            cols.append(_rgba(team, hp))
            sizes.append(KIND_SIZE.get(kind, 26))
        bots_scatter.set_offsets(list(zip(xs, ys)) if xs else [(0, 0)])
        bots_scatter.set_facecolors(cols if cols else [(0, 0, 0, 0)])
        bots_scatter.set_sizes(sizes if sizes else [0])

        segs, fcols = [], []
        for (ax0, ay0, bx, by, team) in res.shots[t]:
            segs.append([(ax0, ay0), (bx, by)])
            r, g, b = TEAM_COLOR[team]
            fcols.append((r, g, b, 0.65))
        fire.set_segments(segs)
        fire.set_color(fcols if fcols else "none")

        fx, fy, fs, fc = [], [], [], []
        for dt in range(FLASH_LIFE):
            born = t - dt
            for (x, y, team) in deaths.get(born, []):
                age = dt / FLASH_LIFE
                fx.append(x)
                fy.append(y)
                fs.append(60 + 700 * age)
                r, g, b = TEAM_COLOR[team]
                fc.append((r, g, b, 0.95 * (1.0 - age)))
        flash_scatter.set_offsets(list(zip(fx, fy)) if fx else [(0, 0)])
        flash_scatter.set_sizes(fs if fs else [0])
        flash_scatter.set_edgecolors(fc if fc else [(0, 0, 0, 0)])

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

        parts = []
        for team, c in zip(teams, counts):
            parts.append(f"{TEAM_NAMES[team]} {c}")
        team_lab.set_text("  ·  ".join(parts))

        if t >= nframes - 1 and win_alliance is not None:
            banner.set_text(f"{win_name}  VICTORY")
            banner.set_color(ALLIANCE_HEX[win_alliance])
            banner.set_alpha(0.94)
        return [bots_scatter, fire, flash_scatter, west_bar, east_bar, banner]

    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)
    anim = FuncAnimation(fig, update, frames=len(ticks),
                         interval=1000 / args.fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=100,
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    _optimize_gif(args.out, args.fps)
    print(f"wrote {args.out}  ({n_total} bots, {res.ticks} ticks -> {win_name})")


def _optimize_gif(path, fps):
    try:
        from PIL import Image, ImageSequence
    except Exception:
        return
    im = Image.open(path)
    frames = [f.convert("RGB").quantize(colors=72, method=Image.Quantize.FASTOCTREE)
              for f in ImageSequence.Iterator(im)]
    if not frames:
        return
    frames[0].save(path, save_all=True, append_images=frames[1:], optimize=True,
                   duration=int(1000 / fps), loop=0, disposal=2)


if __name__ == "__main__":
    main()
