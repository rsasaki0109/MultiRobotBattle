#!/usr/bin/env python3
"""Headline demo — hill, domination, and CTF in one RoboMaster-style GIF.

Three objective modes side-by-side on the same engine.

    python3 scripts/make_objective_triple_gif.py --out docs/media/objective_triple.gif
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

from mrn_coord.battle import TEAM_NAMES, battle_scenario, simulate  # noqa: E402
from mrn_coord.battle_objectives import objective_zone  # noqa: E402

ZONE = "#f5cc4d"
FLAG = "#f5cc4d"


def _progress_pct(res, cfg, fi):
    if not res.objective_progress or fi >= len(res.objective_progress):
        return 0
    prog = res.objective_progress[fi]
    if not prog:
        return 0
    hold = max(1, cfg.objective_hold_ticks)
    numeric = {k: v for k, v in prog.items() if isinstance(v, (int, float))}
    if not numeric:
        return 0
    return min(100, int(100 * max(numeric.values()) / hold))


class ObjectivePanel:
    def __init__(self, ax, name):
        bots, cfg, title = battle_scenario(name)
        self.mode = name
        self.cfg = cfg
        self.res = simulate(bots, cfg, max_ticks=900 if name == "ctf" else 650)
        self.deaths = render.collect_deaths(self.res.frames)
        self.zones = list(self.res.objective_zone) if name == "ctf" else None

        ax.set_facecolor(render.FIELD)
        ax.set_xlim(0, cfg.width)
        ax.set_ylim(0, cfg.height)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(render.MUTED)
        ax.set_title(title, color=render.INK, fontsize=9, pad=5)
        render.draw_arena(ax, cfg, minimal=True)

        if name == "ctf":
            for z in self.zones or []:
                if z[0] == "flag":
                    _, fx, fy, fr = z
                    ax.add_patch(Circle((fx, fy), fr, fill=False, edgecolor=FLAG,
                                        linewidth=1.4, linestyle=(0, (4, 4)),
                                        alpha=0.8, zorder=2))
                elif z[0] == "base":
                    _, team, bx, by, br = z
                    ax.add_patch(Circle((bx, by), br, fill=False,
                                        edgecolor=render.TEAM_HEX.get(team, render.MUTED),
                                        linewidth=1.6, alpha=0.75, zorder=2))
        else:
            zone = objective_zone(cfg)
            if zone:
                cx, cy, r = zone
                ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=ZONE,
                                    linewidth=1.6, linestyle=(0, (5, 4)),
                                    alpha=0.85, zorder=2))

        self.robots = render.RobotLayers(ax, flash_life=7)
        self.prog_txt = ax.text(cfg.width / 2, cfg.height - 0.9, "",
                                ha="center", color=ZONE, fontsize=8, zorder=10)
        self.flag_scat = None
        if name == "ctf":
            self.flag_scat = ax.scatter([], [], s=70, c=FLAG, marker="D",
                                        edgecolors="#fff", linewidths=0.5, zorder=11)
        self.banner = ax.text(cfg.width / 2, cfg.height / 2, "",
                              ha="center", va="center", fontsize=14,
                              fontweight="bold", alpha=0.0, zorder=12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media/objective_triple.gif")
    ap.add_argument("--fps", type=int, default=14)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
    fig.patch.set_facecolor(render.BG)
    fig.suptitle("OBJECTIVE MODES — hill · domination · capture the flag",
                 color=render.INK, fontsize=12, fontweight="bold", y=0.98)

    panels = [ObjectivePanel(ax, name) for ax, name in zip(axes, ("hill", "domination", "ctf"))]
    nframes = max(len(p.res.frames) for p in panels)
    ticks = list(range(nframes)) + [nframes - 1] * args.fps

    def update(fi):
        t = ticks[fi]
        for p in panels:
            fi_local = min(t, len(p.res.frames) - 1)
            frame = p.res.frames[fi_local]
            prev = p.res.frames[fi_local - 1] if fi_local > 0 else None
            shots = p.res.shots[fi_local] if fi_local < len(p.res.shots) else ()
            p.robots.update(frame, prev, shots, p.deaths, fi_local)
            if p.mode == "ctf":
                prog = (p.res.objective_progress or [{}])[
                    min(fi_local, len(p.res.objective_progress) - 1)]
                fx, fy = prog.get("flag", [p.cfg.width / 2, p.cfg.height / 2])
                if p.flag_scat is not None:
                    p.flag_scat.set_offsets([[fx, fy]])
                p.prog_txt.set_text("ctf")
            else:
                p.prog_txt.set_text(f"{p.mode} {_progress_pct(p.res, p.cfg, fi_local)}%")
            if p.res.winner is not None and fi_local >= len(p.res.frames) - 1:
                label = "CAPTURES" if p.mode == "ctf" else "WINS"
                p.banner.set_text(f"{TEAM_NAMES[p.res.winner].upper()} {label}")
                p.banner.set_color(render.TEAM_HEX[p.res.winner])
                p.banner.set_alpha(0.92)
        return []

    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.06, wspace=0.06)
    anim = FuncAnimation(fig, update, frames=len(ticks),
                         interval=1000 // args.fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=96,
              savefig_kwargs={"facecolor": render.BG})
    plt.close(fig)
    render.optimize_gif(args.out, args.fps, colors=64)
    for p in panels:
        print(f"  {p.mode}: {TEAM_NAMES.get(p.res.winner, 'draw')} in {p.res.ticks} ticks")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
