#!/usr/bin/env python3
"""Animate a swarm battle — two flocking armies fight to the last robot.

Drives the real :mod:`mrn_coord.battle` simulation: red and blue armies, each a
decentralized Boids flock, advance to contact and trade fire. Damage is
per-attacker, so robots that get locally outnumbered melt fast — focus fire
emerges from the local rules. The animation shows, from the simulation:

- each robot as a team-coloured disc that **fades as its health drops**,
- a **laser line** from every robot firing on its nearest in-range enemy,
- an **expanding flash** wherever a robot is eliminated,
- a live **tally** of the two armies and, at the end, the winner.

    python3 scripts/make_battle_gif.py --out out/battle.gif --seed 19

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
    BLUE,
    RED,
    TEAM_NAMES,
    BattleConfig,
    run_battle,
)

BG = "#0b0e14"
PANEL = "#141a26"
INK = "#e6edf3"
MUTED = "#8b95a7"
TEAM_COLOR = {RED: (1.00, 0.36, 0.36), BLUE: (0.36, 0.55, 1.00)}
TEAM_HEX = {RED: "#ff5b5b", BLUE: "#5b8cff"}
FLASH_LIFE = 7   # frames a death flash lives


def _rgba(team, hp):
    r, g, b = TEAM_COLOR[team]
    a = 0.28 + 0.72 * max(0.0, min(1.0, hp))   # wounded fade out
    return (r, g, b, a)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/battle.gif")
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--n", type=int, default=14, help="robots per team")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--stride", type=int, default=2,
                    help="render every Nth simulation tick")
    args = ap.parse_args()

    cfg = BattleConfig()
    res = run_battle(args.n, cfg, seed=args.seed)
    nframes = len(res.frames)
    win_name = TEAM_NAMES.get(res.winner, "draw")

    # detect eliminations: a bot alive at tick t but dead at t+1 → flash at t+1
    deaths = {}   # tick -> list of (x, y, team)
    for t in range(nframes - 1):
        cur, nxt = res.frames[t], res.frames[t + 1]
        for (b0, b1) in zip(cur, nxt):
            if b0[4] and not b1[4]:
                deaths.setdefault(t + 1, []).append((b1[0], b1[1], b1[2]))

    ticks = list(range(0, nframes, args.stride))
    if ticks[-1] != nframes - 1:
        ticks.append(nframes - 1)
    ticks += [nframes - 1] * (args.fps)   # hold on the result

    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, cfg.width)
    ax.set_ylim(-2.6, cfg.height + 2.4)  # headroom for the HUD (bottom) + title (top)
    ax.set_aspect("equal")
    ax.axis("off")

    # battlefield frame
    ax.add_patch(Rectangle((0, 0), cfg.width, cfg.height, fill=False,
                           edgecolor="#222a38", lw=1.5))

    # HUD: title + two tally bars just under the arena floor
    ax.text(cfg.width / 2, cfg.height + 0.7,
            "SWARM  BATTLE  —  two flocking armies, last robot standing",
            color=INK, fontsize=13, fontweight="bold", ha="center", va="bottom")
    bar_y, bar_h, half = -1.9, 1.1, cfg.width / 2 - 1.5
    red_lab = ax.text(1.5, bar_y + bar_h + 0.15, "", color=TEAM_HEX[RED],
                      fontsize=12, fontweight="bold", ha="left", va="bottom")
    blue_lab = ax.text(cfg.width - 1.5, bar_y + bar_h + 0.15, "",
                       color=TEAM_HEX[BLUE], fontsize=12, fontweight="bold",
                       ha="right", va="bottom")
    ax.add_patch(Rectangle((1.5, bar_y), half, bar_h, fill=False,
                           edgecolor="#2a3242", lw=1.0))
    ax.add_patch(Rectangle((cfg.width / 2 + 1.5, bar_y), half, bar_h, fill=False,
                           edgecolor="#2a3242", lw=1.0))
    red_bar = Rectangle((1.5 + half, bar_y), 0.0, bar_h,
                        facecolor=TEAM_HEX[RED], edgecolor="none")
    blue_bar = Rectangle((cfg.width / 2 + 1.5, bar_y), 0.0, bar_h,
                         facecolor=TEAM_HEX[BLUE], edgecolor="none")
    ax.add_patch(red_bar)
    ax.add_patch(blue_bar)

    bots_scatter = ax.scatter([], [], s=150, zorder=5, edgecolors="#0b0e14",
                              linewidths=0.8)
    fire = LineCollection([], linewidths=1.1, zorder=3)
    ax.add_collection(fire)
    flash_scatter = ax.scatter([], [], s=[], facecolors="none", linewidths=1.6,
                               zorder=6)
    banner = ax.text(cfg.width / 2, cfg.height / 2, "", ha="center", va="center",
                     fontsize=30, fontweight="bold", zorder=10, alpha=0.0)

    n0 = args.n

    def update(fi):
        t = ticks[fi]
        frame = res.frames[t]
        xs, ys, cols = [], [], []
        for (x, y, team, hp, alive) in frame:
            if not alive:
                continue
            xs.append(x)
            ys.append(y)
            cols.append(_rgba(team, hp))
        bots_scatter.set_offsets(list(zip(xs, ys)) if xs else [(0, 0)])
        bots_scatter.set_facecolors(cols if cols else [(0, 0, 0, 0)])

        # firing lines for this tick
        segs, fcols = [], []
        for (ax0, ay0, bx, by, team) in res.shots[t]:
            segs.append([(ax0, ay0), (bx, by)])
            r, g, b = TEAM_COLOR[team]
            fcols.append((r, g, b, 0.5))
        fire.set_segments(segs)
        fire.set_color(fcols if fcols else "none")

        # active death flashes (expanding fading rings)
        fx, fy, fs, fc = [], [], [], []
        for dt in range(FLASH_LIFE):
            born = t - dt
            for (x, y, team) in deaths.get(born, []):
                age = dt / FLASH_LIFE
                fx.append(x)
                fy.append(y)
                fs.append(150 + 950 * age)
                r, g, b = TEAM_COLOR[team]
                fc.append((r, g, b, 0.9 * (1.0 - age)))
        flash_scatter.set_offsets(list(zip(fx, fy)) if fx else [(0, 0)])
        flash_scatter.set_sizes(fs if fs else [0])
        flash_scatter.set_edgecolors(fc if fc else [(0, 0, 0, 0)])

        red, blue = res.counts[min(t, len(res.counts) - 1)]
        red_bar.set_width(half * red / n0)
        red_bar.set_x(1.5 + half - half * red / n0)   # grows from the centre out
        blue_bar.set_width(half * blue / n0)
        red_lab.set_text(f"RED  {red}")
        blue_lab.set_text(f"{blue}  BLUE")

        # winner banner once the result is in
        if t >= nframes - 1 and res.winner is not None:
            banner.set_text(f"{win_name.upper()}  WINS")
            banner.set_color(TEAM_HEX[res.winner])
            banner.set_alpha(0.92)
        return [bots_scatter, fire, flash_scatter, red_bar, blue_bar, banner]

    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.02)
    anim = FuncAnimation(fig, update, frames=len(ticks),
                         interval=1000 / args.fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=96,
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    _optimize_gif(args.out, args.fps)
    print(f"wrote {args.out}  (seed {args.seed}: {win_name} wins in {res.ticks} "
          f"ticks, survivors red={res.survivors[RED]} blue={res.survivors[BLUE]})")


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
