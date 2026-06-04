#!/usr/bin/env python3
"""Animate a MAPF solution — agents flowing through a grid, collision-free.

Run any of the repo's solvers on a random (or seeded) instance and render the
result as a GIF: each agent is a coloured disc that slides along its planned
path, leaving a short trail, with a ring marking its goal. The title carries the
solver, the sum-of-costs, and the makespan — the same numbers the benchmark gate
checks, now made visible.

The **gallery** mode is the comparison that sells the collection: the *same*
start/goal instance solved side-by-side by several algorithms at once, so you can
watch CBS find the optimum while PIBT and LaCAM flow greedily and prioritized
planning lets later agents wait.

Examples::

    # one solver, one GIF
    python3 scripts/animate_mapf.py --solver lacam --agents 12 --out out/lacam.gif

    # the side-by-side comparison
    python3 scripts/animate_mapf.py --gallery cbs,prioritized,pibt_swap,lacam \\
        --width 12 --height 12 --agents 14 --seed 7 --out out/gallery.gif

Deterministic (seeded), headless (Agg), no ROS.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "mrn_coord"))

from mrn_coord.mapf import GridWorld  # noqa: E402
from mrn_coord.mapf.cbs import cbs  # noqa: E402
from mrn_coord.mapf.cbsh import cbsh  # noqa: E402
from mrn_coord.mapf.ecbs import ecbs  # noqa: E402
from mrn_coord.mapf.fecbs import fecbs  # noqa: E402
from mrn_coord.mapf.lacam import lacam  # noqa: E402
from mrn_coord.mapf.pbs import pbs  # noqa: E402
from mrn_coord.mapf.pibt_swap import pibt_swap  # noqa: E402
from mrn_coord.mapf.prioritized import prioritized_planning  # noqa: E402
from mrn_coord.mapf.rmstar import rmstar  # noqa: E402
from mrn_coord.mapf.solution import makespan, pad_paths, sum_of_costs  # noqa: E402
from mrn_coord.mapf.whca import whca_star  # noqa: E402

# Palette (dark, matches the repo's other GIFs).
BG = "#0b0e14"
PANEL = "#0d1117"
GRID = "#1b2230"
WALL = "#2b3344"
WALL_EDGE = "#3a4459"
INK = "#c9d1d9"
MUTED = "#8b95a7"


def _to_paths(result):
    """Normalise a solver's return value to a ``{agent: [cell, ...]}`` dict."""
    if result is None:
        return None
    if hasattr(result, "paths"):
        return result.paths
    if isinstance(result, dict):
        return result.get("paths", result)
    return None


# Each entry: name -> (label, callable(grid, agents) -> paths-or-None).
SOLVERS = {
    "cbs": ("CBS (optimal)", lambda g, a: cbs(g, a, max_expansions=20000)),
    "cbsh": ("CBSH (optimal, WDG)", lambda g, a: cbsh(g, a, max_expansions=20000)),
    "ecbs": ("ECBS (w=1.5)", lambda g, a: ecbs(g, a, w=1.5, max_expansions=40000)),
    "fecbs": ("FECBS (flex, w=1.5)",
              lambda g, a: fecbs(g, a, w=1.5, max_expansions=40000)),
    "rmstar": ("rM* (recursive M*)", lambda g, a: rmstar(g, a, max_expansions=80000)),
    "pbs": ("PBS (priority search)", lambda g, a: pbs(g, a)),
    "prioritized": ("Prioritized planning", lambda g, a: prioritized_planning(g, a)),
    "whca": ("WHCA* (windowed)", lambda g, a: whca_star(g, a)),
    "pibt_swap": ("PIBT (+swap)", lambda g, a: pibt_swap(g, a)),
    "lacam": ("LaCAM", lambda g, a: lacam(g, a)),
}


def _random_instance(w, h, n, seed, obstacles):
    rng = random.Random(seed)
    blocked = frozenset((x, y) for x in range(w) for y in range(h)
                        if rng.random() < obstacles)
    grid = GridWorld(w, h, blocked)
    free = [(x, y) for x in range(w) for y in range(h) if grid.is_free((x, y))]
    rng.shuffle(free)
    if len(free) < 2 * n:
        raise SystemExit(f"not enough free cells for {n} agents on {w}x{h} "
                         f"at obstacle density {obstacles}")
    agents = {i: (free[i], free[n + i]) for i in range(n)}
    return grid, agents


def _agent_colors(n):
    cmap = plt.get_cmap("turbo")
    return [cmap(0.06 + 0.88 * (i / max(1, n - 1))) for i in range(n)]


def _interp_pos(path, t):
    """Position at continuous time ``t`` (cells, lerped between integer steps)."""
    import math
    ti = int(math.floor(t))
    frac = t - ti

    def at(k):
        return path[k] if k < len(path) else path[-1]
    a = at(ti)
    b = at(ti + 1)
    return (a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac)


def _draw_static(ax, grid, agents, colors, label):
    ax.set_facecolor(PANEL)
    ax.set_xlim(-0.5, grid.width - 0.5)
    ax.set_ylim(-0.5, grid.height - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()                       # (0,0) at top-left, grid-style
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)
    # faint grid lines
    for x in range(grid.width + 1):
        ax.plot([x - 0.5, x - 0.5], [-0.5, grid.height - 0.5],
                color=GRID, lw=0.5, zorder=0)
    for y in range(grid.height + 1):
        ax.plot([-0.5, grid.width - 0.5], [y - 0.5, y - 0.5],
                color=GRID, lw=0.5, zorder=0)
    # walls
    for x in range(grid.width):
        for y in range(grid.height):
            if not grid.is_free((x, y)):
                ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1,
                                       facecolor=WALL, edgecolor=WALL_EDGE,
                                       lw=0.8, zorder=1))
    # goals: hollow rings in each agent's colour
    for i, (_s, gcell) in agents.items():
        ax.add_patch(Circle(gcell, 0.30, fill=False, edgecolor=colors[i],
                            lw=1.8, alpha=0.9, zorder=2))
    if label:
        ax.set_title(label, color=INK, fontsize=11, pad=8)


def _animate(grids_solvers, out, w, h, agents, steps_per_cell, fps, title):
    """grids_solvers: list of (label, grid, agents, paths). Renders all panels
    in one figure, advancing the same clock."""
    n_panels = len(grids_solvers)
    cols = min(n_panels, 2 if n_panels <= 4 else 3)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 5.2 * rows))
    fig.patch.set_facecolor(BG)
    axes = [axes] if n_panels == 1 else list(axes.flat)

    horizon = max(makespan(p) for (_l, _g, _a, p) in grids_solvers)
    colors = _agent_colors(len(agents))

    discs = []          # per panel: list of Circle
    trails = []         # per panel: LineCollection
    counters = []       # per panel: text
    for ax, (label, grid, ag, paths) in zip(axes, grids_solvers):
        _draw_static(ax, grid, ag, colors, label)
        pd = pad_paths(paths)
        panel_discs = []
        for i in sorted(ag):
            c = Circle(pd[i][0], 0.33, facecolor=colors[i], edgecolor="white",
                       lw=0.7, zorder=5)
            ax.add_patch(c)
            panel_discs.append(c)
        discs.append(panel_discs)
        lc = LineCollection([], colors=colors, linewidths=2.4, alpha=0.5,
                            zorder=3)
        ax.add_collection(lc)
        trails.append(lc)
        soc = sum_of_costs(paths)
        ms = makespan(paths)
        counters.append(ax.text(
            0.5, -0.03, f"SOC {soc}   makespan {ms}", transform=ax.transAxes,
            ha="center", va="top", color=MUTED, fontsize=9))
    for ax in axes[n_panels:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, color=INK, fontsize=13, y=0.995)
    top = 0.94 if title else 0.97
    fig.subplots_adjust(left=0.02, right=0.98, top=top, bottom=0.04,
                        wspace=0.08, hspace=0.30)

    tail = 6
    hold = steps_per_cell * 3                       # linger on the final frame
    total_frames = horizon * steps_per_cell + hold

    def update(frame):
        t = min(frame / steps_per_cell, horizon)
        artists = []
        for (label, grid, ag, paths), panel_discs, lc in zip(
                grids_solvers, discs, trails):
            pd = pad_paths(paths)
            segs, seg_colors = [], []
            for idx, i in enumerate(sorted(ag)):
                pos = _interp_pos(pd[i], t)
                panel_discs[idx].center = pos
                artists.append(panel_discs[idx])
                # trail: a few sub-step samples behind the agent
                pts = []
                k = 0
                while k <= tail:
                    tt = t - k / steps_per_cell
                    if tt < 0:
                        break
                    pts.append(_interp_pos(pd[i], tt))
                    k += 1
                for s in range(len(pts) - 1):
                    segs.append([pts[s], pts[s + 1]])
                    seg_colors.append(colors[i])
            lc.set_segments(segs)
            lc.set_color(seg_colors)
            artists.append(lc)
        return artists

    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps,
                         blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    dpi = 92 if n_panels <= 2 else 84
    anim.save(out, writer=PillowWriter(fps=fps), dpi=dpi,
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    _optimize_gif(out, fps)


def _optimize_gif(path, fps):
    """Re-encode the GIF with an adaptive palette — the dark flat-shaded frames
    quantise nearly losslessly and the file shrinks ~2-4x, README-friendly."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver", default="lacam", choices=sorted(SOLVERS),
                    help="single-solver mode (default: lacam)")
    ap.add_argument("--gallery", default=None,
                    help="comma-separated solver names for a side-by-side panel")
    ap.add_argument("--width", type=int, default=12)
    ap.add_argument("--height", type=int, default=12)
    ap.add_argument("--agents", type=int, default=12)
    ap.add_argument("--obstacles", type=float, default=0.12,
                    help="fraction of cells blocked (default 0.12)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps-per-cell", type=int, default=6,
                    help="interpolated frames per grid move (smoothness)")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default="out/mapf.gif")
    args = ap.parse_args()

    grid, agents = _random_instance(args.width, args.height, args.agents,
                                    args.seed, args.obstacles)

    names = (args.gallery.split(",") if args.gallery else [args.solver])
    panels = []
    for name in names:
        name = name.strip()
        if name not in SOLVERS:
            raise SystemExit(f"unknown solver {name!r}; choose from "
                             f"{', '.join(sorted(SOLVERS))}")
        label, fn = SOLVERS[name]
        paths = _to_paths(fn(grid, agents))
        if paths is None:
            print(f"  {name}: no solution (skipped)")
            continue
        print(f"  {name}: SOC {sum_of_costs(paths)}, makespan {makespan(paths)}")
        panels.append((label, grid, agents, paths))
    if not panels:
        raise SystemExit("no solver produced a solution")

    title = (f"MAPF — {args.width}x{args.height}, {args.agents} agents, seed "
             f"{args.seed}") if len(panels) > 1 else None
    _animate(panels, args.out, args.width, args.height, agents,
             args.steps_per_cell, args.fps, title)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
