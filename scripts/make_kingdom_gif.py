#!/usr/bin/env python3
"""Kingdom-scale battle GIF — two battle lines clash on a wide field.

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "mrn_coord"))

from mrn_coord.battle import (  # noqa: E402
    BLUE,
    RED,
    TEAM_NAMES,
    battle_scenario,
    simulate,
)

BG = "#0b0e14"
INK = "#e6edf3"
MUTED = "#8b95a7"
TEAM = {RED: "#ff5b5b", BLUE: "#5b8cff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media/kingdom_clash.gif")
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--stride", type=int, default=2,
                    help="simulate recording stride (2 = half the frames)")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cols", type=int, default=10)
    args = ap.parse_args()

    bots, cfg, title = battle_scenario("kingdom")
    res = simulate(bots, cfg, max_ticks=1000, frame_stride=args.stride)
    nframes = len(res.frames)

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, cfg.width)
    ax.set_ylim(0, cfg.height)
    ax.set_aspect("equal")
    ax.set_title(title, color=INK, fontsize=12)
    ax.tick_params(colors=MUTED, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(MUTED)

    scat = ax.scatter([], [], s=12, c=[], alpha=0.85, linewidths=0)
    tally = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                    color=INK, fontsize=10, family="monospace")
    banner = fig.text(0.5, 0.02, "", ha="center", color=INK, fontsize=11)

    def frame(t):
        fi = min(t, nframes - 1)
        fr = res.frames[fi]
        xs, ys, cols, sizes = [], [], [], []
        for (x, y, team, hp, alive, kind) in fr:
            if not alive:
                continue
            xs.append(x)
            ys.append(y)
            cols.append(TEAM[team])
            sizes.append(14 if kind == "tank" else 10)
        scat.set_offsets(list(zip(xs, ys)) if xs else [(0, 0)])
        scat.set_color(cols if cols else [MUTED])
        scat.set_sizes(sizes if sizes else [10])
        counts = res.counts[fi]
        tally.set_text(
            "  ".join(f"{TEAM_NAMES.get(t, t)}:{n}"
                      for t, n in zip(res.teams, counts)))
        if res.winner is not None and fi >= nframes - 2:
            banner.set_text(f"{TEAM_NAMES[res.winner].upper()} WINS — "
                            f"{sum(res.survivors.values())} survivors")
        else:
            banner.set_text("")
        return scat, tally, banner

    anim = FuncAnimation(fig, frame, frames=nframes + 15,
                         interval=1000 // args.fps, blit=False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    print(f"wrote {args.out}  winner={res.winner} ticks={res.ticks}")


if __name__ == "__main__":
    main()
