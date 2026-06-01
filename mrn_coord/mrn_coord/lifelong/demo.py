"""``mrn_lifelong_demo``: run lifelong MAPF on a warehouse and report throughput.

A no-argument, no-ROS entry point: it fills a small shelf-and-aisle warehouse
with agents that pick up an endless stream of tasks (PIBT stepping), prints the
throughput / service-time metrics, and renders a few frames of the warehouse so
you can watch the robots flow around the shelves — the runnable counterpart to
the unit tests.
"""

from __future__ import annotations

import argparse

from .lifelong import TaskStream, make_warehouse, run_lifelong


def _render(grid, pos) -> list:
    """ASCII frame: ``#`` shelf, digit/letter = agent, ``.`` free."""
    occ = {c: a for a, c in pos.items()}
    glyph = "0123456789abcdefghijklmnopqrstuvwxyz"
    ids = sorted({a for a in pos})
    label = {a: glyph[i % len(glyph)] for i, a in enumerate(ids)}
    lines = []
    for y in range(grid.height):
        row = []
        for x in range(grid.width):
            cell = (x, y)
            if cell in occ:
                row.append(label[occ[cell]])
            elif not grid.is_free(cell):
                row.append("#")
            else:
                row.append(".")
        lines.append("".join(row))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2, help="shelf rows")
    parser.add_argument("--cols", type=int, default=3, help="shelf columns")
    parser.add_argument("--agents", type=int, default=6, help="number of robots")
    parser.add_argument("--steps", type=int, default=120, help="timesteps to run")
    parser.add_argument("--frames", type=int, default=6,
                        help="ASCII frames to print (0 to skip)")
    args = parser.parse_args()

    grid, endpoints = make_warehouse(rows=args.rows, cols=args.cols)
    n = min(args.agents, len(endpoints))
    starts = {f"r{i}": endpoints[i] for i in range(n)}
    stream = TaskStream(list(endpoints))

    res = run_lifelong(grid, starts, stream, max_steps=args.steps,
                       keep_history=args.frames > 0)

    print(f"=== lifelong MAPF — {grid.width}x{grid.height} warehouse, "
          f"{n} agents, {args.steps} steps (PIBT) ===")
    print(f"  tasks completed : {res.completed}")
    print(f"  throughput      : {res.throughput:.3f} tasks/step "
          f"({res.throughput * n:.3f} per robot-step is the ceiling)")
    print(f"  avg service time: {res.avg_service_time:.1f} steps/task")
    print(f"  max wait        : {res.max_wait} steps")
    print(f"  per agent       : {res.per_agent}")

    if args.frames > 0 and res.history:
        stride = max(1, len(res.history) // args.frames)
        print("\n  frames (# shelf, letters = robots):")
        for t in range(0, len(res.history), stride):
            print(f"  t={t}")
            for line in _render(grid, res.history[t]):
                print(f"    {line}")


if __name__ == "__main__":
    main()
