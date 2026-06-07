#!/usr/bin/env python3
"""Headline demo — greedy vs MAPF-planned maneuver on the chokepoint.

Runs the same soldiers-on-chokepoint setup twice (greedy pursuit vs prioritized
maneuver + Hungarian assignment) and renders a side-by-side GIF.

    python3 scripts/make_maneuver_gif.py --out out/maneuver_duel.gif
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "mrn_coord"))

from mrn_coord.battle import (  # noqa: E402
    BLUE,
    RED,
    BattleConfig,
    battle_scenario,
    make_company,
    simulate,
)
import random  # noqa: E402

BG = "#0b0e14"
INK = "#e6edf3"
MUTED = "#8b95a7"
TEAM = {RED: "#ff5b5b", BLUE: "#5b8cff"}


def _make_pair(seed=11):
    obstacles = ((20.0, 4.5, 2.6), (20.0, 12.0, 2.6), (20.0, 19.5, 2.6))
    base = dict(obstacles=obstacles, tactics="count_aware", formation="wedge",
                assignment="hungarian", maneuver_replan_ticks=15)
    rng = random.Random(seed)
    center_r = (BattleConfig().width * 0.13, BattleConfig().height * 0.5)
    center_b = (BattleConfig().width * 0.87, BattleConfig().height * 0.5)
    roster = [("soldier", 10)]

    cfg_g = BattleConfig(**base, maneuver="greedy")
    cfg_p = BattleConfig(**base, maneuver="prioritized")
    bots_g = (make_company(cfg_g, RED, center_r, roster, rng, jitter=2.8) +
              make_company(cfg_g, BLUE, center_b, roster, random.Random(seed + 1),
                           jitter=2.8))
    bots_p = (make_company(cfg_p, RED, center_r, roster, random.Random(seed + 2),
                           jitter=2.8) +
              make_company(cfg_p, BLUE, center_b, roster, random.Random(seed + 3),
                           jitter=2.8))
    return (bots_g, cfg_g, simulate(bots_g, cfg_g, max_ticks=700),
            bots_p, cfg_p, simulate(bots_p, cfg_p, max_ticks=700))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/maneuver_duel.gif")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    _, cfg, res_g, _, _, res_p = _make_pair()
    nframes = max(len(res_g.frames), len(res_p.frames))
    obstacles = cfg.obstacles

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
    for ax, title, res in zip(axes,
                               ("Greedy pursuit", "Prioritized MAPF maneuver"),
                               (res_g, res_p)):
        ax.set_facecolor(BG)
        ax.set_xlim(0, cfg.width)
        ax.set_ylim(0, cfg.height)
        ax.set_aspect("equal")
        ax.set_title(title, color=INK, fontsize=11)
        ax.tick_params(colors=MUTED, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(MUTED)
        for (ox, oy, r) in obstacles:
            ax.add_patch(plt.Circle((ox, oy), r, color="#334155", alpha=0.85))

    scat = [ax.scatter([], [], s=28, c=[], alpha=0.9) for ax in axes]
    fire = [LineCollection([], colors=MUTED, linewidths=0.6, alpha=0.35)
            for _ in axes]
    for ax, lc in zip(axes, fire):
        ax.add_collection(lc)
    banner = fig.text(0.5, 0.02, "", ha="center", color=INK, fontsize=10)

    def frame(t):
        artists = []
        labels = []
        for k, (res, ax, s, lc) in enumerate(zip((res_g, res_p), axes, scat, fire)):
            fi = min(t, len(res.frames) - 1)
            fr = res.frames[fi]
            xs, ys, cols = [], [], []
            for (x, y, team, hp, alive, kind) in fr:
                if not alive:
                    continue
                xs.append(x)
                ys.append(y)
                cols.append(TEAM[team])
            s.set_offsets(list(zip(xs, ys)) if xs else [(0, 0)])
            s.set_color(cols if cols else [MUTED])
            segs = []
            if fi < len(res.shots):
                for (x0, y0, x1, y1, team) in res.shots[fi]:
                    segs.append([(x0, y0), (x1, y1)])
            lc.set_segments(segs)
            w = res.winner
            if w is not None and fi >= len(res.frames) - 2:
                labels.append(f"{'Greedy' if k == 0 else 'MAPF'}: "
                              f"{'Red' if w == RED else 'Blue'} wins")
            artists.extend([s, lc])
        banner.set_text("  |  ".join(labels) if labels else "")
        artists.append(banner)
        return artists

    anim = FuncAnimation(fig, frame, frames=nframes + 20, interval=1000 // args.fps,
                         blit=False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
