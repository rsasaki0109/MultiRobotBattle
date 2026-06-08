#!/usr/bin/env python3
"""Kingdom-scale battle GIF — two battle lines clash on a wide field.

RoboMaster-style chassis on a competition grid; 80 vs 80 line clash.

    python3 scripts/make_kingdom_gif.py --out docs/media/kingdom_clash.gif
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media/kingdom_clash.gif")
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--stride", type=int, default=2,
                    help="simulate recording stride (2 = half the frames)")
    args = ap.parse_args()

    bots, cfg, title = battle_scenario("kingdom")
    res = simulate(bots, cfg, max_ticks=1000, frame_stride=args.stride)
    nframes = len(res.frames)
    deaths = render.collect_deaths(res.frames)
    ticks = list(range(nframes)) + [nframes - 1] * args.fps

    fig, ax = plt.subplots(figsize=(14.5, 6.4))
    fig.patch.set_facecolor(render.BG)
    ax.set_xlim(-1, cfg.width + 1)
    ax.set_ylim(-1.5, cfg.height + 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    render.draw_arena(ax, cfg, minimal=True)

    ax.text(cfg.width / 2, cfg.height + 0.75,
            "KINGDOM CLASH  —  80 vs 80 battle lines",
            color=render.INK, fontsize=12, fontweight="bold", ha="center", va="bottom")
    ax.text(cfg.width / 2, cfg.height + 0.15, title,
            color=render.MUTED, fontsize=8, ha="center", va="bottom")

    robots = render.RobotLayers(ax)
    tally = ax.text(cfg.width / 2, -0.85, "", ha="center", va="top",
                    color=render.INK, fontsize=10, family="monospace")
    banner = ax.text(cfg.width / 2, cfg.height / 2, "", ha="center", va="center",
                     fontsize=28, fontweight="bold", zorder=12, alpha=0.0)

    def update(fi):
        t = ticks[fi]
        frame = res.frames[t]
        prev = res.frames[t - 1] if t > 0 else None
        shots = res.shots[t] if t < len(res.shots) else ()
        robots.update(frame, prev, shots, deaths, t)

        counts = res.counts[min(t, len(res.counts) - 1)]
        tally.set_text("  ·  ".join(f"{TEAM_NAMES.get(tm, tm)} {n}"
                                     for tm, n in zip(res.teams, counts)))
        if res.winner is not None and t >= nframes - 1:
            banner.set_text(f"{TEAM_NAMES[res.winner].upper()} WINS")
            banner.set_color(render.TEAM_HEX[res.winner])
            banner.set_alpha(0.94)
        return []

    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.04)
    anim = FuncAnimation(fig, update, frames=len(ticks),
                         interval=1000 // args.fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=96,
              savefig_kwargs={"facecolor": render.BG})
    plt.close(fig)
    render.optimize_gif(args.out, args.fps, colors=80)
    print(f"wrote {args.out}  winner={res.winner} ticks={res.ticks}")


if __name__ == "__main__":
    main()
