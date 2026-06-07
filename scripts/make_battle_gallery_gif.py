#!/usr/bin/env python3
"""Animate a gallery of swarm battles — four kinds of fight, side by side.

Renders the four showcase scenarios from :func:`mrn_coord.battle.battle_scenario`
in a 2x2 grid, each running its own real battle simultaneously:

- **Duel** — the classic two flocking armies, 14 vs 14;
- **Free-for-all** — three armies, every team fights every other;
- **Quality vs quantity** — 5 heavy tanks against 16 fast scouts (unit classes);
- **Chokepoint** — terrain: a row of obstacles splits the battlefield.

Each panel is driven by the actual simulation, holds on its own result, and
shows a winner banner when it finishes.

    python3 scripts/make_battle_gallery_gif.py --out out/battle_gallery.gif

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
from matplotlib.patches import Circle  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "mrn_coord"))

from mrn_coord.battle import (  # noqa: E402
    BLUE,
    GREEN,
    RED,
    YELLOW,
    TEAM_NAMES,
    battle_scenario,
    simulate,
)

BG = "#0b0e14"
INK = "#e6edf3"
MUTED = "#8b95a7"
TEAM_COLOR = {RED: (1.00, 0.36, 0.36), BLUE: (0.36, 0.55, 1.00),
              GREEN: (0.36, 0.85, 0.45), YELLOW: (0.96, 0.80, 0.30)}
TEAM_HEX = {RED: "#ff5b5b", BLUE: "#5b8cff", GREEN: "#5bd96f", YELLOW: "#f5cc4d"}
# disc area per unit class (tanks big, scouts small)
KIND_SIZE = {"": 70, "scout": 34, "soldier": 64, "tank": 150, "sniper": 58}
FLASH_LIFE = 7
SCENARIOS = ("duel", "free_for_all", "quality_vs_quantity", "chokepoint")


def _rgba(team, hp):
    r, g, b = TEAM_COLOR[team]
    return (r, g, b, 0.28 + 0.72 * max(0.0, min(1.0, hp)))


class Panel:
    def __init__(self, ax, name):
        bots, cfg, title = battle_scenario(name)
        self.cfg = cfg
        self.title = title
        self.res = simulate(bots, cfg, max_ticks=900)
        self.teams = self.res.teams
        self.n0 = {t: sum(1 for b in bots if b.team == t) for t in self.teams}
        self.nframes = len(self.res.frames)

        # eliminations -> flashes
        self.deaths = {}
        for t in range(self.nframes - 1):
            for (a, b) in zip(self.res.frames[t], self.res.frames[t + 1]):
                if a[4] and not b[4]:
                    self.deaths.setdefault(t + 1, []).append((b[0], b[1], b[2]))

        ax.set_facecolor(BG)
        ax.set_xlim(0, cfg.width)
        ax.set_ylim(0, cfg.height)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#222a38")
        ax.set_title(title, color=INK, fontsize=10, pad=5)

        for (ox, oy, r) in cfg.obstacles:
            ax.add_patch(Circle((ox, oy), r, facecolor="#222a32",
                                edgecolor="#33405a", lw=1.0, zorder=1))

        self.scatter = ax.scatter([], [], zorder=5, edgecolors="#0b0e14",
                                  linewidths=0.5)
        self.fire = LineCollection([], linewidths=0.8, zorder=3)
        ax.add_collection(self.fire)
        self.flash = ax.scatter([], [], facecolors="none", linewidths=1.3,
                                zorder=6)
        self.tally = ax.text(cfg.width / 2, cfg.height - 0.7, "", color=INK,
                             fontsize=9, ha="center", va="top", fontweight="bold")
        self.banner = ax.text(cfg.width / 2, cfg.height / 2, "", ha="center",
                              va="center", fontsize=20, fontweight="bold",
                              zorder=9, alpha=0.0)
        self.ax = ax

    def draw(self, gt):
        t = min(gt, self.nframes - 1)
        frame = self.res.frames[t]
        xs, ys, cols, sizes = [], [], [], []
        for (x, y, team, hp, alive, kind) in frame:
            if not alive:
                continue
            xs.append(x)
            ys.append(y)
            cols.append(_rgba(team, hp))
            sizes.append(KIND_SIZE.get(kind, 70))
        self.scatter.set_offsets(list(zip(xs, ys)) if xs else [(0, 0)])
        self.scatter.set_facecolors(cols if cols else [(0, 0, 0, 0)])
        self.scatter.set_sizes(sizes if sizes else [0])

        segs, fcols = [], []
        for (ax0, ay0, bx, by, team) in self.res.shots[t]:
            segs.append([(ax0, ay0), (bx, by)])
            r, g, b = TEAM_COLOR[team]
            fcols.append((r, g, b, 0.45))
        self.fire.set_segments(segs)
        self.fire.set_color(fcols if fcols else "none")

        fx, fy, fs, fc = [], [], [], []
        for dt in range(FLASH_LIFE):
            for (x, y, team) in self.deaths.get(t - dt, []):
                age = dt / FLASH_LIFE
                fx.append(x)
                fy.append(y)
                fs.append(80 + 520 * age)
                r, g, b = TEAM_COLOR[team]
                fc.append((r, g, b, 0.9 * (1.0 - age)))
        self.flash.set_offsets(list(zip(fx, fy)) if fx else [(0, 0)])
        self.flash.set_sizes(fs if fs else [0])
        self.flash.set_edgecolors(fc if fc else [(0, 0, 0, 0)])

        counts = self.res.counts[min(t, len(self.res.counts) - 1)]
        parts = []
        for team, c in zip(self.teams, counts):
            parts.append(f"$\\bf{{{c}}}$ {TEAM_NAMES[team]}")
        self.tally.set_text("    ".join(parts))

        if t >= self.nframes - 1 and self.res.winner is not None:
            self.banner.set_text(f"{TEAM_NAMES[self.res.winner].upper()} WINS")
            self.banner.set_color(TEAM_HEX[self.res.winner])
            self.banner.set_alpha(0.92)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/battle_gallery.gif")
    ap.add_argument("--fps", type=int, default=22)
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.4))
    fig.patch.set_facecolor(BG)
    panels = [Panel(ax, name) for ax, name in zip(axes.flat, SCENARIOS)]
    fig.suptitle("SWARM  BATTLE  —  four kinds of fight, same local rules",
                 color=INK, fontsize=14, fontweight="bold", y=0.99)

    G = max(p.nframes for p in panels)
    ticks = list(range(0, G, args.stride))
    if ticks[-1] != G - 1:
        ticks.append(G - 1)
    ticks += [G - 1] * args.fps   # hold on the results

    def update(fi):
        gt = ticks[fi]
        for p in panels:
            p.draw(gt)
        return []

    fig.subplots_adjust(left=0.015, right=0.985, top=0.92, bottom=0.02,
                        wspace=0.06, hspace=0.12)
    anim = FuncAnimation(fig, update, frames=len(ticks),
                         interval=1000 / args.fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=92,
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    _optimize_gif(args.out, args.fps)
    for p, name in zip(panels, SCENARIOS):
        w = TEAM_NAMES.get(p.res.winner, "draw")
        print(f"  {name:20s} {p.res.ticks:3d} ticks -> {w} wins")
    print(f"wrote {args.out}  ({len(ticks)} frames)")


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
