#!/usr/bin/env python3
"""Generate the warehouse AMR-fleet GIF (lifelong MAPF, PIBT).

A shelf-and-aisle warehouse filled with autonomous mobile robots (AMRs) that
never stop: each one is handed an endless stream of pickup/dropoff tasks, and
the moment it reaches a station it is given the next. Every per-timestep move is
the collision-free configuration computed by **PIBT** (priority inheritance with
backtracking) over the obstacle-aware distance gradient, with a cost-aware
allocator matching idle robots to near tasks. The fleet flows around the
shelving without ever colliding or deadlocking, and the running counter shows
the throughput (tasks served per timestep) climb — the metric that actually
matters for a warehouse AMR fleet. Deterministic, no ROS.

Usage::

    python3 scripts/make_warehouse_gif.py
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
from matplotlib.patches import Circle, FancyBboxPatch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "mrn_coord"))

from mrn_coord.lifelong.lifelong import (  # noqa: E402
    TaskStream, make_warehouse, run_lifelong)

BG = "#0b0e14"
PANEL = "#0d1117"
SHELF = "#243044"
SHELF_EDGE = "#30405c"
STATION = "#1f6f6b"
INK = "#c9d1d9"
MUTED = "#6b7689"
ACCENT = "#38bdf8"

ROWS, COLS, AISLE = 3, 5, 1
AGENTS = 12
STEPS = 130
SUBSTEPS = 2          # interpolated frames between integer cells (smooth motion)
ROBOT_R = 0.34


def _robot_colors(n):
    """A palette of distinct hues that read on the dark panel."""
    import colorsys
    out = []
    for k in range(n):
        h = (0.58 + k / n) % 1.0           # spread around the wheel, start at cyan
        r, g, b = colorsys.hsv_to_rgb(h, 0.62, 0.98)
        out.append((r, g, b))
    return out


def _simulate():
    """Run the warehouse lifelong sim; return everything the render needs."""
    grid, endpoints = make_warehouse(ROWS, COLS, aisle=AISLE)

    # Distinct start cells drawn from the stations.
    starts, used = {}, set()
    for i in range(AGENTS):
        for cell in endpoints:
            if cell not in used:
                starts[i] = cell
                used.add(cell)
                break

    stream = TaskStream(pool=endpoints)
    result = run_lifelong(grid, starts, stream, max_steps=STEPS,
                          keep_history=True, allocator="hungarian")

    # A goal changes exactly when a task is completed and the next is assigned,
    # so cumulative goal-changes give the tasks-served timeline (per integer step).
    gh = result.goal_history
    served = [0]
    for t in range(1, len(gh)):
        delta = sum(1 for a in gh[t] if gh[t][a] != gh[t - 1][a])
        served.append(served[-1] + delta)

    return grid, endpoints, result, served


def _interp_frames(history):
    """Expand integer-cell history into smooth sub-stepped (x, y) frames.

    PIBT forbids vertex and swap conflicts, so linear interpolation between two
    consecutive collision-free configurations never makes two robots overlap.
    Returns ``(frames, step_of_frame)`` where each frame is ``{id: (x, y)}``.
    """
    frames, step_of = [], []
    ids = sorted(history[0])
    for t in range(len(history) - 1):
        a0, a1 = history[t], history[t + 1]
        for s in range(SUBSTEPS):
            f = s / SUBSTEPS
            frames.append({i: (a0[i][0] + (a1[i][0] - a0[i][0]) * f,
                               a0[i][1] + (a1[i][1] - a0[i][1]) * f) for i in ids})
            step_of.append(t)
    frames.append({i: (c[0], c[1]) for i, c in history[-1].items()})
    step_of.append(len(history) - 1)
    return frames, step_of


def render(output: str, fps: int = 20) -> None:
    grid, endpoints, result, served = _simulate()
    frames, step_of = _interp_frames(result.history)
    goal_hist = result.goal_history
    ids = sorted(result.history[0])
    colors = dict(zip(ids, _robot_colors(len(ids))))
    W, H = grid.width, grid.height
    trail = 10

    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=96)
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.1)

    def draw(fi):
        ax.clear()
        ax.set_facecolor(PANEL)
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(-0.5, H - 0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()                       # row 0 at top, warehouse-style
        for spine in ax.spines.values():
            spine.set_color(SHELF_EDGE)

        # Shelving (one rounded tile per blocked cell — reads as racking units).
        for (x, y) in grid.blocked:
            ax.add_patch(FancyBboxPatch(
                (x - 0.42, y - 0.42), 0.84, 0.84,
                boxstyle="round,pad=0,rounding_size=0.16",
                facecolor=SHELF, edgecolor=SHELF_EDGE, lw=0.8, zorder=1))
        # Pick/drop stations.
        for (x, y) in endpoints:
            ax.scatter([x], [y], marker="s", s=26, facecolor="none",
                       edgecolor=STATION, lw=1.1, zorder=2)

        snap = frames[fi]
        step = step_of[fi]
        goals = goal_hist[min(step, len(goal_hist) - 1)]
        for k in ids:
            c = colors[k]
            # faint marker at this robot's current task station
            gx, gy = goals[k]
            ax.scatter([gx], [gy], marker="*", s=55, color=c, alpha=0.4,
                       edgecolor=BG, linewidth=0.4, zorder=3)
            # short trail
            lo = max(0, fi - trail)
            xs = [frames[j][k][0] for j in range(lo, fi + 1)]
            ys = [frames[j][k][1] for j in range(lo, fi + 1)]
            ax.plot(xs, ys, color=c, lw=1.8, alpha=0.4, solid_capstyle="round",
                    zorder=4)
            x, y = snap[k]
            ax.add_patch(Circle((x, y), ROBOT_R, facecolor=c, edgecolor=BG,
                                lw=1.0, zorder=5))

        # title + live throughput share the top margin; subtitle sits below.
        ax.text(-0.3, -0.78, "Lifelong MAPF — warehouse AMR fleet",
                color=INK, fontsize=12, weight="bold", va="bottom", zorder=7)
        n_served = served[min(step, len(served) - 1)]
        tput = n_served / max(1, step) if step else 0.0
        ax.text(W - 0.2, -0.78,
                f"AMRs {len(ids)}   served {n_served}   {tput:.2f}/step",
                color=ACCENT, fontsize=8.5, va="bottom", ha="right", zorder=7)
        ax.text(-0.3, H - 0.1,
                "endless pick/drop tasks, collision-free via PIBT "
                "(mrn_coord.lifelong)",
                color=MUTED, fontsize=8.0, va="top", zorder=7)
        return ()

    anim = FuncAnimation(fig, draw, frames=len(frames),
                         interval=1000 / fps, blit=False)
    anim.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)
    _optimize_gif(output, fps)
    print(f"throughput {result.throughput:.3f} tasks/step over {result.steps} "
          f"steps, {result.completed} tasks served, "
          f"avg service {result.avg_service_time:.2f} steps")


def _optimize_gif(path: str, fps: int, colors: int = 80) -> None:
    from PIL import Image

    src = Image.open(path)
    out = []
    try:
        while True:
            out.append(src.copy().convert("RGB").quantize(
                colors=colors, method=Image.FASTOCTREE, dither=Image.Dither.NONE))
            src.seek(src.tell() + 1)
    except EOFError:
        pass
    out[0].save(path, save_all=True, append_images=out[1:], optimize=True,
                loop=0, duration=int(round(1000 / fps)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/media/warehouse_demo.gif")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    render(args.output, fps=args.fps)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
