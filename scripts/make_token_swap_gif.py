#!/usr/bin/env python3
"""Generate the Token-Swapping comparison GIF, driven by the real algorithms.

The motion is produced by the actual :mod:`mrn_coord.mapf.token_swapping` code,
not a hand-drawn loop. The *same* scrambled rainbow (the reverse permutation of
seven tokens) is sorted on three graph topologies side by side, each by its
**optimal** swap sequence:

- **Path P₇** — only adjacent swaps are legal, so the optimum is the inversion
  count (``path_swaps``): 21 swaps.
- **Cycle C₇** — the wrap-around edge helps; exact BFS optimum (``optimal_swaps``):
  9 swaps.
- **Complete K₇** — every transposition is legal, so the optimum is ``n − cycles``
  (``complete_swaps``): 3 swaps.

That is the paper's headline in one picture: the graph, not the permutation,
decides the cost. The complete graph finishes in three swaps and waits while the
path grinds through twenty-one.

Usage::

    PYTHONPATH=mrn_coord python3 scripts/make_token_swap_gif.py \
        --out docs/media/token_swap.gif
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "mrn_coord"))

from mrn_coord.mapf import token_swapping as ts  # noqa: E402

N = 7
FRAMES_PER_SWAP = 6
GAP_FRAMES = 2
END_HOLD = 18
DISC_R = 0.30


def _path_coords(n):
    # horizontal line centred on the panel
    return {i: (i - (n - 1) / 2.0, 0.0) for i in range(n)}


def _circle_coords(n, radius=1.25):
    # start at top, go clockwise so the rainbow reads naturally
    out = {}
    for i in range(n):
        a = math.pi / 2 - 2 * math.pi * i / n
        out[i] = (radius * math.cos(a), radius * math.sin(a))
    return out


def _placements(initial, swaps):
    """Sequence of placements: state *before* each swap, plus the final state."""
    states = [dict(initial)]
    p = dict(initial)
    for u, v in swaps:
        p = ts.apply_swap(p, u, v)
        states.append(dict(p))
    return states


class Panel:
    def __init__(self, ax, title, coords, edges, swaps, initial, color):
        self.ax = ax
        self.coords = coords
        self.swaps = swaps
        self.states = _placements(initial, swaps)
        self.color = color
        self.title = title
        self.span = FRAMES_PER_SWAP + GAP_FRAMES

        ax.set_aspect("equal")
        ax.axis("off")
        xs = [c[0] for c in coords.values()]
        ys = [c[1] for c in coords.values()]
        # uniform vertical extent so all three titles line up across panels
        ax.set_xlim(min(xs) - 0.55, max(xs) + 0.55)
        ax.set_ylim(-2.25, 2.15)

        # static graph edges
        for u, v in edges:
            x0, y0 = coords[u]
            x1, y1 = coords[v]
            ax.plot([x0, x1], [y0, y1], color="#d0d3d8", lw=1.4, zorder=1,
                    solid_capstyle="round")

        # token discs (one per label) + label texts
        self.discs = {}
        self.texts = {}
        for lab in range(N):
            c = Circle((0, 0), DISC_R, zorder=3,
                       facecolor=color(lab), edgecolor="white", lw=1.8)
            ax.add_patch(c)
            self.discs[lab] = c
            self.texts[lab] = ax.text(0, 0, str(lab), ha="center", va="center",
                                      fontsize=9, fontweight="bold",
                                      color="white", zorder=4)
        self.title_txt = ax.set_title(title, fontsize=12, fontweight="bold",
                                      pad=8)
        self.count_txt = ax.text(
            0.5, -0.02, "", transform=ax.transAxes, ha="center", va="top",
            fontsize=11, color="#333")

    def n_swaps(self):
        return len(self.swaps)

    def label_positions(self, frame):
        """Return {label: (x, y)} and the number of completed swaps at frame."""
        n = self.n_swaps()
        s = frame // self.span               # which swap window
        local = frame % self.span

        if s >= n or local >= FRAMES_PER_SWAP:
            # settled state: swap `s` (if any) has already committed
            done = min(s + 1, n) if s < n else n
            state = self.states[done]
            return {lab: self.coords[v] for v, lab in state.items()}, done

        # mid-swap interpolation of the two involved tokens
        u, v = self.swaps[s]
        before = self.states[s]
        pos = {lab: self.coords[vert] for vert, lab in before.items()}
        t = (local + 1) / FRAMES_PER_SWAP
        ease = 0.5 - 0.5 * math.cos(math.pi * t)   # smoothstep
        lu, lv = before[u], before[v]
        Pu, Pv = self.coords[u], self.coords[v]
        dx, dy = Pv[0] - Pu[0], Pv[1] - Pu[1]
        d = math.hypot(dx, dy) or 1.0
        px, py = -dy / d, dx / d                   # perpendicular
        amp = 0.22 * min(d, 1.4)
        bow = amp * math.sin(math.pi * t)
        pos[lu] = (Pu[0] + dx * ease + px * bow, Pu[1] + dy * ease + py * bow)
        pos[lv] = (Pv[0] - dx * ease - px * bow, Pv[1] - dy * ease - py * bow)
        return pos, s

    def draw(self, frame):
        pos, done = self.label_positions(frame)
        for lab, (x, y) in pos.items():
            self.discs[lab].center = (x, y)
            self.texts[lab].set_position((x, y))
        self.count_txt.set_text(f"swaps: {done} / {self.n_swaps()}")
        return done == self.n_swaps()


def build(out_path, fps):
    initial = {i: N - 1 - i for i in range(N)}    # start from the reversed rainbow
    target = {i: i for i in range(N)}             # sort to a clean 0..6 rainbow

    cmap = plt.get_cmap("turbo")
    color = lambda lab: cmap(0.07 + 0.86 * lab / (N - 1))

    gp = ts.make_path_graph(N)
    gc = ts.make_cycle_graph(N)
    gk = ts.make_complete_graph(N)

    path_sw = ts.path_swaps(initial, target)
    cyc_sw = ts.optimal_swaps(gc, initial, target).swaps
    comp_sw = ts.complete_swaps(initial, target)

    def edges(g):
        return {(u, v) for u in g for v in g[u] if u < v}

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.3),
                             gridspec_kw={"width_ratios": [1.7, 1.0, 1.0]})
    fig.patch.set_facecolor("white")

    panels = [
        Panel(axes[0], f"Path  P₇  —  {len(path_sw)} swaps", _path_coords(N),
              edges(gp), path_sw, initial, color),
        Panel(axes[1], f"Cycle  C₇  —  {len(cyc_sw)} swaps", _circle_coords(N),
              edges(gc), cyc_sw, initial, color),
        Panel(axes[2], f"Complete  K₇  —  {len(comp_sw)} swaps",
              _circle_coords(N), edges(gk), comp_sw, initial, color),
    ]
    fig.suptitle(
        "Token Swapping — same scramble, the graph decides the cost "
        "(Yamanaka et al.)",
        fontsize=13.5, fontweight="bold", y=0.99)
    fig.text(0.5, 0.045,
             "no blanks · the only move is an adjacent swap · "
             "minimise the total number of swaps",
             ha="center", fontsize=9.5, color="#666")

    total = max(p.n_swaps() for p in panels) * panels[0].span + END_HOLD

    def update(frame):
        for p in panels:
            p.draw(frame)
        return []

    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.12,
                        wspace=0.06)
    anim = FuncAnimation(fig, update, frames=total, interval=1000 / fps,
                         blit=False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=92)
    plt.close(fig)
    print(f"wrote {out_path}  ({total} frames @ {fps}fps)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/media/token_swap.gif")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()
    build(args.out, args.fps)


if __name__ == "__main__":
    main()
