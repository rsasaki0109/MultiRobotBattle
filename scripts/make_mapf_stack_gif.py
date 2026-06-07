#!/usr/bin/env python3
"""Headline demo — Hungarian+greedy vs CBS-TA+prioritized on the chokepoint.

Same soldiers and terrain; only the MAPF layers change (assignment + maneuver).
Side-by-side GIF for README / docs/media.

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
from matplotlib.collections import LineCollection  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "mrn_coord"))

from mrn_coord.battle import (  # noqa: E402
    BLUE,
    RED,
    BattleConfig,
    Bot,
    make_company,
    simulate,
)
import random  # noqa: E402

BG = "#0b0e14"
INK = "#e6edf3"
MUTED = "#8b95a7"
TEAM = {RED: "#ff5b5b", BLUE: "#5b8cff"}

OBSTACLES = ((20.0, 4.5, 2.6), (20.0, 12.0, 2.6), (20.0, 19.5, 2.6))
BASE_KW = dict(
    obstacles=OBSTACLES,
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
    blue = make_company(cfg, BLUE, center_b, roster, random.Random(seed + 1),
                        jitter=2.8)
    return red + blue


def _run_pair(seed=11):
    spawn = _spawn(seed)
    cfg_local = BattleConfig(
        **BASE_KW,
        assignment="none",
        assignment_by_team={RED: "hungarian"},
        maneuver="greedy",
        maneuver_by_team={BLUE: "greedy"},
    )
    cfg_mapf = BattleConfig(
        **BASE_KW,
        assignment="none",
        assignment_by_team={RED: "cbs_ta"},
        maneuver="greedy",
        maneuver_by_team={RED: "prioritized", BLUE: "greedy"},
    )
    res_local = simulate(_clone_bots(spawn), cfg_local, max_ticks=700)
    res_mapf = simulate(_clone_bots(spawn), cfg_mapf, max_ticks=700)
    return cfg_local, res_local, res_mapf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media/mapf_stack_duel.gif")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    cfg, res_local, res_mapf = _run_pair(args.seed)
    nframes = max(len(res_local.frames), len(res_mapf.frames))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
    panels = (
        (res_local, "Hungarian + greedy"),
        (res_mapf, "CBS-TA + prioritized MAPF"),
    )
    for ax, (_, title) in zip(axes, panels):
        ax.set_facecolor(BG)
        ax.set_xlim(0, cfg.width)
        ax.set_ylim(0, cfg.height)
        ax.set_aspect("equal")
        ax.set_title(title, color=INK, fontsize=10)
        ax.tick_params(colors=MUTED, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(MUTED)
        for (ox, oy, r) in OBSTACLES:
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
        for k, ((res, short), ax, s, lc) in enumerate(
                zip(panels, axes, scat, fire)):
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
                labels.append(f"{short}: {'Red' if w == RED else 'Blue'} wins")
            artists.extend([s, lc])
        banner.set_text("  |  ".join(labels) if labels else "")
        artists.append(banner)
        return artists

    anim = FuncAnimation(fig, frame, frames=nframes + 20, interval=1000 // args.fps,
                         blit=False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    print(f"wrote {args.out}  local={res_local.winner} mapf={res_mapf.winner}")


if __name__ == "__main__":
    main()
