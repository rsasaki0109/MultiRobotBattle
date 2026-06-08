#!/usr/bin/env python3
"""Headline demo — MAPF stack vs local rules on a king-of-the-hill total-war contest.

Same 18 vs 18 spawn fighting for the centre; red runs Hungarian+greedy (left)
or CBS-TA+prioritized MAPF (right).

    python3 scripts/make_mapf_total_war_gif.py --out docs/media/mapf_total_war.gif
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
from matplotlib.patches import Circle  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "mrn_coord"))

from mrn_coord.battle import (  # noqa: E402
    BLUE,
    RED,
    Bot,
    BattleConfig,
    TEAM_NAMES,
    make_contest_armies,
    simulate,
)

BG = "#0b0e14"
INK = "#e6edf3"
MUTED = "#8b95a7"
ZONE = "#f5cc4d"
TEAM = {RED: "#ff5b5b", BLUE: "#5b8cff"}

HILL_KW = dict(
    objective="hill",
    objective_radius=4.8,
    objective_hold_ticks=120,
    tactics="count_aware",
    formation="wedge",
    maneuver_replan_ticks=12,
    assignment_replan_ticks=12,
)


def _clone_bots(bots):
    return [Bot(b.x, b.y, b.vx, b.vy, b.team, b.hp, b.max_hp, alive=b.alive,
                dps=b.dps, attack_range=b.attack_range, max_speed=b.max_speed,
                kind=b.kind)
            for b in bots]


def _spawn(n=18, seed=8):
    cfg = BattleConfig(**HILL_KW)
    return make_contest_armies(n, cfg, seed=seed), cfg


def _run_pair(n=18, seed=8):
    spawn, base = _spawn(n, seed)
    cfg_local = BattleConfig(
        **HILL_KW,
        assignment="none",
        assignment_by_team={RED: "hungarian"},
        maneuver="greedy",
        maneuver_by_team={BLUE: "greedy"},
    )
    cfg_mapf = BattleConfig(
        **HILL_KW,
        assignment="none",
        assignment_by_team={RED: "cbs_ta"},
        maneuver="greedy",
        maneuver_by_team={RED: "prioritized", BLUE: "greedy"},
    )
    res_local = simulate(_clone_bots(spawn), cfg_local, max_ticks=650)
    res_mapf = simulate(_clone_bots(spawn), cfg_mapf, max_ticks=650)
    return cfg_local, res_local, res_mapf


def _zone(cfg):
    cx, cy = cfg.objective_center or (cfg.width / 2, cfg.height / 2)
    return cx, cy, cfg.objective_radius


def _progress(res, hold_ticks, fi):
    if not res.objective_progress or fi >= len(res.objective_progress):
        return 0
    prog = res.objective_progress[fi]
    if not prog:
        return 0
    return min(100, int(100 * max(prog.values()) / max(1, hold_ticks)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media/mapf_total_war.gif")
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--fps", type=int, default=14)
    args = ap.parse_args()

    cfg, res_local, res_mapf = _run_pair(args.n, args.seed)
    nframes = max(len(res_local.frames), len(res_mapf.frames))
    zx, zy, zr = _zone(cfg)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), facecolor=BG)
    fig.suptitle(f"MAPF TOTAL WAR — {args.n * 2} robots, king of the hill",
                 color=INK, fontsize=12, fontweight="bold", y=0.98)

    panels = (
        (res_local, "Hungarian + greedy"),
        (res_mapf, "CBS-TA + prioritized MAPF"),
    )
    scat, fire, prog, banners = [], [], [], []
    for ax, (_, title) in zip(axes, panels):
        ax.set_facecolor(BG)
        ax.set_xlim(0, cfg.width)
        ax.set_ylim(0, cfg.height)
        ax.set_aspect("equal")
        ax.set_title(title, color=INK, fontsize=10)
        ax.tick_params(colors=MUTED, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(MUTED)
        ax.add_patch(Circle((zx, zy), zr, fill=False, edgecolor=ZONE,
                            linewidth=1.6, linestyle=(0, (5, 4)), alpha=0.85))
        scat.append(ax.scatter([], [], s=30, c=[], alpha=0.9))
        lc = LineCollection([], linewidths=0.7, alpha=0.42)
        ax.add_collection(lc)
        fire.append(lc)
        prog.append(ax.text(zx, zy, "", ha="center", va="center",
                            color=ZONE, fontsize=10, fontweight="bold"))
        banners.append(ax.text(cfg.width / 2, cfg.height / 2, "",
                               ha="center", va="center", fontsize=18,
                               fontweight="bold", alpha=0.0, zorder=9))

    def frame(t):
        artists = []
        labels = []
        for k, ((res, short), s, lc, pr, bn) in enumerate(
                zip(panels, scat, fire, prog, banners)):
            fi = min(t, len(res.frames) - 1)
            xs, ys, cols = [], [], []
            for (x, y, team, hp, alive, kind) in res.frames[fi]:
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
            pr.set_text(f"{_progress(res, cfg.objective_hold_ticks, fi)}%")
            w = res.winner
            if w is not None and fi >= len(res.frames) - 2:
                labels.append(f"{short}: {TEAM_NAMES[w]} wins")
                bn.set_text(f"{TEAM_NAMES[w].upper()} WINS")
                bn.set_color(TEAM[w])
                bn.set_alpha(0.92)
            artists.extend([s, lc, pr, bn])
        return artists

    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.05, wspace=0.08)
    anim = FuncAnimation(fig, frame, frames=nframes + 16,
                         interval=1000 // args.fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=96,
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    _optimize_gif(args.out, args.fps)
    print(f"wrote {args.out}  local={TEAM_NAMES.get(res_local.winner)} "
          f"mapf={TEAM_NAMES.get(res_mapf.winner)}")


def _optimize_gif(path, fps):
    try:
        from PIL import Image, ImageSequence
    except Exception:
        return
    im = Image.open(path)
    frames = [f.convert("RGB").quantize(colors=64, method=Image.Quantize.FASTOCTREE)
              for f in ImageSequence.Iterator(im)]
    if not frames:
        return
    frames[0].save(path, save_all=True, append_images=frames[1:], optimize=True,
                   duration=int(1000 / fps), loop=0, disposal=2)


if __name__ == "__main__":
    main()
