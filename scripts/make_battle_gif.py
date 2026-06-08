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
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, os.pardir, "mrn_coord"))
sys.path.insert(0, _SCRIPT_DIR)

import _battle_robot_art as art  # noqa: E402

from mrn_coord.battle import (  # noqa: E402
    ALLIANCE_NAMES,
    battle_scenario,
    simulate,
)

BG = "#06080d"
FIELD = "#0a0e16"
GRID = "#141c28"
INK = "#eef2f8"
MUTED = "#8b95a7"
FRONT = "#2a3548"
ALLIANCE_HEX = {0: "#ff5b4a", 1: "#4d8cff"}
FLASH_LIFE = 10
WHEEL_RGB = (0.12, 0.13, 0.16, 0.85)


def _alliance_totals(counts, teams, alliances):
    totals = {}
    for team, c in zip(teams, counts):
        aid = alliances.get(team, team)
        totals[aid] = totals.get(aid, 0) + c
    return totals


def _draw_arena(ax, cfg):
    ax.set_facecolor(FIELD)
    ax.add_patch(Rectangle((0, 0), cfg.width, cfg.height, fill=False,
                           edgecolor="#243044", lw=2.4, zorder=0))
    # Competition floor grid
    for x in np.arange(0, cfg.width + 0.1, 8):
        ax.plot([x, x], [0, cfg.height], color=GRID, lw=0.35, alpha=0.55, zorder=0)
    for y in np.arange(0, cfg.height + 0.1, 8):
        ax.plot([0, cfg.width], [y, y], color=GRID, lw=0.35, alpha=0.55, zorder=0)
    mid = cfg.width / 2
    ax.add_patch(Rectangle((mid - 1.4, 0), 2.8, cfg.height, fill=True,
                           facecolor="#0c1018", edgecolor="#1a2434", lw=0.8, zorder=0))
    for y in np.arange(2, cfg.height, 7):
        ax.plot([mid - 0.55, mid + 0.55], [y, y + 2.5], color=FRONT, lw=0.7,
                alpha=0.45, zorder=0)
    # Corner pillars (RoboMaster arena cues)
    for cx, cy in ((6, 6), (cfg.width - 6, 6),
                   (6, cfg.height - 6), (cfg.width - 6, cfg.height - 6)):
        ax.add_patch(Circle((cx, cy), 1.8, fill=False, edgecolor="#2e3d54",
                            lw=1.2, alpha=0.6, zorder=0))


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

    fig, ax = plt.subplots(figsize=(16.2, 7.6))
    fig.patch.set_facecolor(BG)
    ax.set_xlim(-2, cfg.width + 2)
    ax.set_ylim(-3.8, cfg.height + 2.8)
    ax.set_aspect("equal")
    ax.axis("off")
    _draw_arena(ax, cfg)

    ax.text(cfg.width / 2, cfg.height + 1.55,
            f"ROBOMASTER TOTAL WAR  —  {n_total} custom bots · four armies · two alliances",
            color=INK, fontsize=13.5, fontweight="bold", ha="center", va="bottom")
    ax.text(cfg.width / 2, cfg.height + 0.42, title,
            color=MUTED, fontsize=8.5, ha="center", va="bottom")

    bar_y, bar_h = -2.55, 1.05
    half = cfg.width / 2 - 2.5
    west_lab = ax.text(2.5, bar_y + bar_h + 0.18, "", color=ALLIANCE_HEX[0],
                       fontsize=11, fontweight="bold", ha="left", va="bottom")
    east_lab = ax.text(cfg.width - 2.5, bar_y + bar_h + 0.18, "",
                       color=ALLIANCE_HEX[1], fontsize=11, fontweight="bold",
                       ha="right", va="bottom")
    cas_lab = ax.text(cfg.width / 2, bar_y + bar_h + 0.18, "", color=MUTED,
                      fontsize=9, ha="center", va="bottom", family="monospace")
    ax.add_patch(Rectangle((2.5, bar_y), half, bar_h, fill=False,
                           edgecolor="#2a3242", lw=1.0))
    ax.add_patch(Rectangle((cfg.width / 2 + 2.5, bar_y), half, bar_h, fill=False,
                           edgecolor="#2a3242", lw=1.0))
    west_bar = Rectangle((2.5 + half, bar_y), 0.0, bar_h,
                         facecolor=ALLIANCE_HEX[0], edgecolor="none", zorder=8)
    east_bar = Rectangle((cfg.width / 2 + 2.5, bar_y), 0.0, bar_h,
                          facecolor=ALLIANCE_HEX[1], edgecolor="none", zorder=8)
    ax.add_patch(west_bar)
    ax.add_patch(east_bar)
    team_lab = ax.text(cfg.width / 2, bar_y - 0.65, "", color=MUTED,
                       fontsize=7.5, ha="center", va="top", family="monospace")

    hulls = PolyCollection([], closed=True, linewidths=0.45, edgecolors="#080a10",
                           zorder=4)
    stripes = PolyCollection([], closed=True, linewidths=0.0, zorder=5)
    wheels = PolyCollection([], closed=True, linewidths=0.0, zorder=3)
    barrels = LineCollection([], linewidths=1.1, colors="#1a1d24", zorder=6,
                             capstyle="round")
    fire_glow = LineCollection([], linewidths=2.8, zorder=7, alpha=0.22,
                               capstyle="round")
    fire = LineCollection([], linewidths=1.0, zorder=7, alpha=0.75,
                          capstyle="round")
    flash = LineCollection([], linewidths=1.4, zorder=9, alpha=0.9)
    ax.add_collection(wheels)
    ax.add_collection(hulls)
    ax.add_collection(stripes)
    ax.add_collection(barrels)
    ax.add_collection(fire_glow)
    ax.add_collection(fire)
    ax.add_collection(flash)

    banner = ax.text(cfg.width / 2, cfg.height / 2, "", ha="center", va="center",
                     fontsize=34, fontweight="bold", zorder=12, alpha=0.0)

    n0_west = sum(n0[t] for t in teams if alliances.get(t, t) == 0)
    n0_east = sum(n0[t] for t in teams if alliances.get(t, t) == 1)

    def _frame_robots(frame, prev):
        hpoly, hcol, spoly, scol, wpoly, wcol, blines = [], [], [], [], [], [], []
        for i, (x, y, team, hp, alive, kind) in enumerate(frame):
            if not alive:
                continue
            k = kind or "soldier"
            if prev:
                px, py = prev[i][0], prev[i][1]
            else:
                px, py = x, y
            hd = art.infer_heading(x, y, px, py, team)
            hpoly.append(art.hull_polygon(x, y, hd, k))
            hcol.append(art.hull_face_rgba(team, hp))
            spoly.append(art.stripe_polygon(x, y, hd, k))
            scol.append(art.stripe_rgba(team, hp))
            wr = art.wheel_radius(k)
            if wr > 0:
                for wx, wy in art.wheel_offsets(x, y, hd, k):
                    wpoly.append([
                        (wx - wr, wy - wr * 0.55), (wx + wr, wy - wr * 0.55),
                        (wx + wr, wy + wr * 0.55), (wx - wr, wy + wr * 0.55),
                    ])
                    wcol.append(WHEEL_RGB)
            blines.append(art.barrel_segment(x, y, hd, k))
        return hpoly, hcol, spoly, scol, wpoly, wcol, blines

    def update(fi):
        t = ticks[fi]
        frame = res.frames[t]
        prev = res.frames[t - 1] if t > 0 else None
        hpoly, hcol, spoly, scol, wpoly, wcol, blines = _frame_robots(frame, prev)
        hulls.set_verts(hpoly if hpoly else np.zeros((0, 4, 2)))
        hulls.set_facecolors(hcol if hcol else [(0, 0, 0, 0)])
        stripes.set_verts(spoly if spoly else np.zeros((0, 4, 2)))
        stripes.set_facecolors(scol if scol else [(0, 0, 0, 0)])
        wheels.set_verts(wpoly if wpoly else np.zeros((0, 4, 2)))
        wheels.set_facecolors(wcol if wcol else [(0, 0, 0, 0)])
        barrels.set_segments(blines if blines else [])
        barrels.set_color("#22262e" if blines else "none")

        segs, glow, fcols = [], [], []
        for (ax0, ay0, bx, by, team) in res.shots[t]:
            segs.append([(ax0, ay0), (bx, by)])
            r, g, b = art.TEAM_RGB.get(team, (0.9, 0.9, 0.9))
            glow.append((r, g, b, 0.35))
            fcols.append((min(1, r + 0.25), min(1, g + 0.25), min(1, b + 0.2), 0.9))
        fire.set_segments(segs)
        fire.set_color(fcols if fcols else "none")
        fire_glow.set_segments(segs)
        fire_glow.set_color(glow if glow else "none")

        fsegs, fcols2 = [], []
        for dt in range(FLASH_LIFE):
            born = t - dt
            for (x, y, team) in deaths.get(born, []):
                age = dt / FLASH_LIFE
                r = 0.35 + 1.1 * age
                fsegs.extend([
                    [(x - r, y), (x + r, y)],
                    [(x, y - r), (x, y + r)],
                ])
                rr, gg, bb = art.TEAM_RGB.get(team, (1, 1, 1))
                fcols2.extend([(rr, gg, bb, 0.85 * (1 - age))] * 2)
        flash.set_segments(fsegs)
        flash.set_color(fcols2 if fcols2 else "none")

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

        from mrn_coord.battle import TEAM_NAMES
        parts = [f"{TEAM_NAMES[team]} {c}" for team, c in zip(teams, counts)]
        team_lab.set_text("  ·  ".join(parts))

        if t >= nframes - 1 and win_alliance is not None:
            banner.set_text(f"{win_name}  VICTORY")
            banner.set_color(ALLIANCE_HEX[win_alliance])
            banner.set_alpha(0.94)
        return []

    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01)
    anim = FuncAnimation(fig, update, frames=len(ticks),
                         interval=1000 / args.fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=102,
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
    frames = [f.convert("RGB").quantize(colors=96, method=Image.Quantize.FASTOCTREE)
              for f in ImageSequence.Iterator(im)]
    if not frames:
        return
    frames[0].save(path, save_all=True, append_images=frames[1:], optimize=True,
                   duration=int(1000 / fps), loop=0, disposal=2)


if __name__ == "__main__":
    main()
