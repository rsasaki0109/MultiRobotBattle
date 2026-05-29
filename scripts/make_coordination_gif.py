#!/usr/bin/env python3
"""Generate the coordination-layer demo GIF, driven by the real algorithms.

Unlike a hand-drawn loop, the motion here is produced by the actual
``mrn_coord`` code: Conflict-Based Search plans the collision-free paths, and
the displacement-based formation controller drives the assembly. It is a
synthetic, deterministic, no-ROS animation — reproducible in CI — that shows the
two landed coordination pillars in one story:

1. **MAPF / CBS** — three robots funnel through a one-cell doorway without ever
   colliding; CBS sequences them through.
2. **Formation** — having crossed, they converge into a triangle via the
   relative-measurement consensus law.

Usage::

    PYTHONPATH=mrn_coord python3 scripts/make_coordination_gif.py
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse, Rectangle

# Make `mrn_coord` importable when run from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "mrn_coord"))

from mrn_coord.formation import polygon_formation, simulate  # noqa: E402
from mrn_coord.mapf import GridWorld, cbs, pad_paths  # noqa: E402

# --- palette (GitHub dark friendly) ----------------------------------------
BG = "#0b0e14"
PANEL = "#0d1117"
GRID = "#1b2130"
WALL = "#39414f"
INK = "#c9d1d9"
MUTED = "#6b7689"
COLORS = {"1": "#38bdf8", "2": "#f472b6", "3": "#a3e635"}
LINK = "#e2e8f0"

WIDTH, HEIGHT = 11, 7
DOORWAY_Y = 3
STARTS = {"1": (1, 1), "2": (1, 3), "3": (1, 5)}
GOALS = {"1": (8, 5), "2": (8, 3), "3": (8, 1)}


def _blocked():
    return {(5, y) for y in range(HEIGHT) if y != DOORWAY_Y}


def _lerp(a, b, s):
    return (a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s)


def _build_frames(sub: int):
    """Precompute per-frame positions for both acts as (act, positions)."""
    grid = GridWorld(WIDTH, HEIGHT, blocked=_blocked())
    agents = {a: (STARTS[a], GOALS[a]) for a in STARTS}
    solution = cbs(grid, agents, max_expansions=50_000)
    if solution is None:
        raise RuntimeError("CBS failed to solve the demo scenario")
    paths = pad_paths(solution.paths)
    horizon = max(len(p) for p in paths.values())

    frames = []
    # Act 1: MAPF traversal, interpolated `sub` frames per timestep.
    for t in range(horizon - 1):
        for k in range(sub):
            s = k / sub
            positions = {a: _lerp(paths[a][t], paths[a][t + 1], s) for a in paths}
            frames.append(("mapf", positions))
    arrived = {a: paths[a][-1] for a in paths}
    for _ in range(sub):                       # brief hold at the goals
        frames.append(("mapf", dict(arrived)))

    # Act 2: formation assembly from the arrival positions.
    spec = polygon_formation(list(STARTS), radius=1.4)
    edges = [("1", "2"), ("2", "3"), ("1", "3")]
    start = {a: (float(arrived[a][0]), float(arrived[a][1])) for a in arrived}
    trajectory, _ = simulate(start, spec, edges, gain=1.4, dt=0.05, steps=160)
    for idx in range(0, len(trajectory), 3):
        frames.append(("formation", trajectory[idx]))
    for _ in range(sub * 2):                   # hold the final formation
        frames.append(("formation", trajectory[-1]))

    return frames, edges


def _glow(ax, xy, color, *, base=0.42, layers=5):
    for k in range(layers):
        r = base * (1.0 + 0.6 * k)
        ax.add_patch(Ellipse(
            xy, 2 * r, 2 * r, facecolor=color, edgecolor="none",
            alpha=0.30 * (1 - k / layers) ** 1.6, zorder=3,
        ))


def render(output: str, sub: int = 6, fps: int = 20) -> None:
    frames, edges = _build_frames(sub)
    blocked = _blocked()
    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=100)
    fig.patch.set_facecolor(BG)

    def draw(i):
        act, positions = frames[i]
        ax.clear()
        ax.set_facecolor(PANEL)
        ax.set_xlim(-0.6, WIDTH - 0.4)
        ax.set_ylim(-0.6, HEIGHT - 0.4)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)

        for gx in range(WIDTH):
            ax.axvline(gx, color=GRID, lw=0.5, alpha=0.5, zorder=0)
        for gy in range(HEIGHT):
            ax.axhline(gy, color=GRID, lw=0.5, alpha=0.5, zorder=0)
        for (bx, by) in blocked:
            ax.add_patch(Rectangle(
                (bx - 0.5, by - 0.5), 1, 1, facecolor=WALL,
                edgecolor=BG, lw=1.0, zorder=1,
            ))

        if act == "mapf":
            title, accent = "Conflict-Based Search — collision-free paths", COLORS["1"]
            for a, g in GOALS.items():            # goal markers
                ax.add_patch(Rectangle(
                    (g[0] - 0.34, g[1] - 0.34), 0.68, 0.68, facecolor="none",
                    edgecolor=COLORS[a], lw=1.8, alpha=0.8, zorder=2,
                ))
        else:
            title, accent = "Formation control — assume formation", COLORS["2"]
            ids = list(positions)
            for a, b in edges:                     # formation links
                pa, pb = positions[a], positions[b]
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=LINK,
                        lw=1.6, alpha=0.5, linestyle=(0, (4, 3)), zorder=2)

        for a, p in positions.items():
            _glow(ax, p, COLORS[a])
            ax.scatter([p[0]], [p[1]], s=150, color=COLORS[a],
                       edgecolor=BG, linewidth=1.5, zorder=5)
            ax.text(p[0], p[1] - 0.55, a, color=COLORS[a], fontsize=8.5,
                    ha="center", va="top", weight="bold", zorder=6)

        ax.text(-0.4, HEIGHT - 0.55, "Multi-Robot Coordination", color=INK,
                fontsize=13, weight="bold", va="top", zorder=7)
        ax.text(-0.4, HEIGHT - 1.15, title, color=accent, fontsize=10.5,
                weight="bold", va="top", zorder=7)
        # progress bar
        prog = i / (len(frames) - 1)
        ax.add_patch(Rectangle((-0.4, -0.5), (WIDTH - 0.2), 0.1,
                               color=GRID, zorder=7))
        ax.add_patch(Rectangle((-0.4, -0.5), (WIDTH - 0.2) * prog, 0.1,
                               color=accent, zorder=8))
        ax.text(WIDTH - 0.5, -0.5, "mrn_coord", color=MUTED, fontsize=8,
                ha="right", va="bottom", zorder=7)
        return ()

    anim = FuncAnimation(fig, draw, frames=len(frames),
                         interval=1000 / fps, blit=False)
    anim.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)
    _optimize_gif(output, fps)


def _optimize_gif(path: str, fps: int, colors: int = 96) -> None:
    from PIL import Image

    src = Image.open(path)
    frames_list = []
    try:
        while True:
            frames_list.append(src.copy().convert("RGB").quantize(
                colors=colors, method=Image.FASTOCTREE, dither=Image.Dither.NONE))
            src.seek(src.tell() + 1)
    except EOFError:
        pass
    frames_list[0].save(
        path, save_all=True, append_images=frames_list[1:],
        optimize=True, loop=0, duration=int(round(1000 / fps)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/media/coordination_demo.gif")
    parser.add_argument("--sub", type=int, default=6)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    render(args.output, sub=args.sub, fps=args.fps)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
