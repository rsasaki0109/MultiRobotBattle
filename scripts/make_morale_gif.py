#!/usr/bin/env python3
"""Headline demo — morale rout: collapsing scouts flee off-field.

    python3 scripts/make_morale_gif.py --out docs/media/morale_rout.gif
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, os.pardir, "mrn_coord"))
sys.path.insert(0, _SCRIPT_DIR)

import _battle_gif_render as render  # noqa: E402

from mrn_coord.battle import RED, BLUE, TEAM_NAMES, battle_scenario, simulate  # noqa: E402


class MoralePanel:
    def __init__(self, ax, cfg, res, subtitle):
        self.cfg = cfg
        self.res = res
        self.subtitle = subtitle
        self.deaths = render.collect_deaths(res.frames)
        self.bars = {}

        ax.set_facecolor(render.FIELD)
        ax.set_xlim(0, cfg.width)
        ax.set_ylim(0, cfg.height)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(render.MUTED)
        ax.set_title(subtitle, color=render.INK, fontsize=10)
        render.draw_arena(ax, cfg, minimal=True)
        render.draw_terrain(ax, cfg)
        self.robots = render.RobotLayers(ax, flash_life=7)
        self.banner = ax.text(cfg.width / 2, cfg.height / 2, "",
                              ha="center", va="center", fontsize=18,
                              fontweight="bold", alpha=0.0, zorder=12)
        bar_h = 0.9
        for team, y0 in ((RED, cfg.height - 1.2), (BLUE, 0.3)):
            col = render.TEAM_HEX[team]
            bg = Rectangle((1.0, y0), cfg.width - 2.0, bar_h,
                           facecolor="#1a1f2e", edgecolor=render.MUTED,
                           linewidth=0.8, zorder=15)
            fg = Rectangle((1.0, y0), cfg.width - 2.0, bar_h,
                           facecolor=col, edgecolor="none", alpha=0.85, zorder=16)
            ax.add_patch(bg)
            ax.add_patch(fg)
            lbl = ax.text(1.4, y0 + bar_h * 0.5, TEAM_NAMES[team].upper(),
                          va="center", ha="left", fontsize=7, color="#0d1117",
                          fontweight="bold", zorder=17)
            self.bars[team] = (fg, lbl)

    def _update_bars(self, fi):
        prog = (self.res.morale_progress or [{}])[min(fi, len(self.res.morale_progress) - 1)]
        span = self.cfg.width - 2.0
        for team, (fg, lbl) in self.bars.items():
            frac = prog.get(team, 1.0)
            fg.set_width(max(0.02, span * frac))
            lbl.set_text(f"{TEAM_NAMES[team].upper()} {frac:.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media/morale_rout.gif")
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--max-ticks", type=int, default=900)
    args = ap.parse_args()

    bots, cfg, title = battle_scenario("morale_duel")
    res = simulate(bots, cfg, max_ticks=args.max_ticks)
    nframes = len(res.frames)
    ticks = list(range(nframes)) + [nframes - 1] * args.fps

    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    fig.patch.set_facecolor(render.BG)
    fig.suptitle(title, color=render.INK, fontsize=12, fontweight="bold", y=0.98)
    panel = MoralePanel(ax, cfg, res, "Team strength — rout below 38%")

    def update(fi):
        t = ticks[fi]
        fi_local = min(t, len(res.frames) - 1)
        frame = res.frames[fi_local]
        prev = res.frames[fi_local - 1] if fi_local > 0 else None
        shots = res.shots[fi_local] if fi_local < len(res.shots) else ()
        projs = (res.projectiles[fi_local]
                 if fi_local < len(getattr(res, "projectiles", ())) else ())
        panel.robots.update(frame, prev, shots, projs, deaths=panel.deaths, t=fi_local)
        panel._update_bars(fi_local)
        w = res.winner
        if w is not None and fi_local >= len(res.frames) - 1:
            panel.banner.set_text(f"{'RED' if w == RED else 'BLUE'} WINS")
            panel.banner.set_color(render.TEAM_HEX[w])
            panel.banner.set_alpha(0.92)
        return []

    fig.subplots_adjust(left=0.04, right=0.96, top=0.90, bottom=0.06)
    anim = FuncAnimation(fig, update, frames=len(ticks),
                         interval=1000 // args.fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=96,
              savefig_kwargs={"facecolor": render.BG})
    plt.close(fig)
    render.optimize_gif(args.out, args.fps, colors=64)
    print(f"wrote {args.out}  winner={res.winner} ticks={res.ticks}")


if __name__ == "__main__":
    main()
