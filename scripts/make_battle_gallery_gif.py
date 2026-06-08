#!/usr/bin/env python3
"""Animate a gallery of swarm battles — four kinds of fight, side by side.

Renders the four showcase scenarios from :func:`mrn_coord.battle.battle_scenario`
in a 2x2 grid with RoboMaster-style chassis art:

- **Duel** — the classic two flocking armies, 14 vs 14;
- **Free-for-all** — three armies, every team fights every other;
- **Quality vs quantity** — 5 heavy tanks against 16 fast scouts (unit classes);
- **Chokepoint** — terrain: a row of obstacles splits the battlefield.

    python3 scripts/make_battle_gallery_gif.py --out docs/media/battle_gallery.gif

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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, os.pardir, "mrn_coord"))
sys.path.insert(0, _SCRIPT_DIR)

import _battle_gif_render as render  # noqa: E402

from mrn_coord.battle import TEAM_NAMES, battle_scenario, simulate  # noqa: E402

SCENARIOS = ("duel", "free_for_all", "quality_vs_quantity", "chokepoint")


class Panel:
    def __init__(self, ax, name):
        bots, cfg, title = battle_scenario(name)
        self.cfg = cfg
        self.title = title
        self.res = simulate(bots, cfg, max_ticks=900)
        self.teams = self.res.teams
        self.n0 = {t: sum(1 for b in bots if b.team == t) for t in self.teams}
        self.nframes = len(self.res.frames)
        self.deaths = render.collect_deaths(self.res.frames)

        ax.set_facecolor(render.FIELD)
        ax.set_xlim(0, cfg.width)
        ax.set_ylim(0, cfg.height)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#222a38")
        ax.set_title(title, color=render.INK, fontsize=10, pad=5)

        render.draw_arena(ax, cfg, minimal=True)
        render.draw_terrain(ax, cfg)

        self.robots = render.RobotLayers(ax, flash_life=7, fire_alpha=0.55)
        self.tally = ax.text(cfg.width / 2, cfg.height - 0.7, "", color=render.INK,
                             fontsize=9, ha="center", va="top", fontweight="bold")
        self.banner = ax.text(cfg.width / 2, cfg.height / 2, "", ha="center",
                              va="center", fontsize=20, fontweight="bold",
                              zorder=12, alpha=0.0)

    def draw(self, gt):
        t = min(gt, self.nframes - 1)
        frame = self.res.frames[t]
        prev = self.res.frames[t - 1] if t > 0 else None
        shots = self.res.shots[t] if t < len(self.res.shots) else ()
        self.robots.update(frame, prev, shots, self.deaths, t)

        counts = self.res.counts[min(t, len(self.res.counts) - 1)]
        parts = [f"$\\bf{{{c}}}$ {TEAM_NAMES[team]}" for team, c in zip(self.teams, counts)]
        self.tally.set_text("    ".join(parts))

        if t >= self.nframes - 1 and self.res.winner is not None:
            self.banner.set_text(f"{TEAM_NAMES[self.res.winner].upper()} WINS")
            self.banner.set_color(render.TEAM_HEX[self.res.winner])
            self.banner.set_alpha(0.92)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/media/battle_gallery.gif")
    ap.add_argument("--fps", type=int, default=22)
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.4))
    fig.patch.set_facecolor(render.BG)
    panels = [Panel(ax, name) for ax, name in zip(axes.flat, SCENARIOS)]
    fig.suptitle("SWARM  BATTLE  —  four kinds of fight, RoboMaster chassis",
                 color=render.INK, fontsize=14, fontweight="bold", y=0.99)

    G = max(p.nframes for p in panels)
    ticks = list(range(0, G, args.stride))
    if ticks[-1] != G - 1:
        ticks.append(G - 1)
    ticks += [G - 1] * args.fps

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
              savefig_kwargs={"facecolor": render.BG})
    plt.close(fig)
    render.optimize_gif(args.out, args.fps, colors=64)
    for p, name in zip(panels, SCENARIOS):
        w = TEAM_NAMES.get(p.res.winner, "draw")
        print(f"  {name:20s} {p.res.ticks:3d} ticks -> {w} wins")
    print(f"wrote {args.out}  ({len(ticks)} frames)")


if __name__ == "__main__":
    main()
