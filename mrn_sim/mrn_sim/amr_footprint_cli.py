"""``mrn_amr_footprint``: replay a warehouse lifelong plan as bodied AMRs.

A no-ROS entry point that runs lifelong MAPF (PIBT) on a small warehouse, then
executes the resulting plan with :func:`mrn_sim.amr_footprint.execute_amr` — a
differential-drive robot with a rectangular footprint that must turn in place
before it drives — across a sweep of aisle widths. It prints the two gaps the
discrete plan hides: the turning-cost makespan stretch, and the footprint
clearance going negative once the aisle narrows toward the body size.

    ros2 run mrn_sim mrn_amr_footprint
    ros2 run mrn_sim mrn_amr_footprint --rows 3 --cols 4 --agents 8
"""

from __future__ import annotations

import argparse

from .amr_footprint import Footprint, execute_amr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2, help="shelf rows")
    parser.add_argument("--cols", type=int, default=3, help="shelf columns")
    parser.add_argument("--agents", type=int, default=6)
    parser.add_argument("--steps", type=int, default=30, help="lifelong steps to replay")
    parser.add_argument("--length", type=float, default=0.7, help="footprint length (m)")
    parser.add_argument("--width", type=float, default=0.45, help="footprint width (m)")
    parser.add_argument("--max-v", type=float, default=1.0)
    parser.add_argument("--cells", type=float, nargs="+",
                        default=[1.6, 1.3, 1.0, 0.8],
                        help="aisle cell sizes (m) to sweep")
    args = parser.parse_args()

    from mrn_coord.lifelong.lifelong import (TaskStream, make_warehouse,
                                             run_lifelong)

    grid, endpoints = make_warehouse(args.rows, args.cols, aisle=1)
    starts, used = {}, set()
    for i in range(args.agents):
        for cell in endpoints:
            if cell not in used:
                starts[i] = cell
                used.add(cell)
                break
    result = run_lifelong(grid, starts, TaskStream(pool=endpoints),
                          max_steps=args.steps, keep_history=True,
                          allocator="hungarian")
    paths = {a: [result.history[t][a] for t in range(len(result.history))]
             for a in starts}
    fp = Footprint(args.length, args.width)

    print(f"warehouse {grid.width}x{grid.height} cells, {args.agents} AMRs, "
          f"footprint {args.length}x{args.width} m, "
          f"lifelong plan makespan {len(result.history) - 1} steps\n")
    print(f"{'cell(m)':>7} {'turn_frac':>9} {'makespan_s':>10} {'stretch':>7} "
          f"{'min_shelf':>9} {'min_robot':>9} {'collisions':>10}")
    for cs in args.cells:
        res = execute_amr(paths, grid.blocked, cell_size=cs, footprint=fp,
                          max_v=args.max_v)
        ideal = res.discrete_makespan * cs / args.max_v
        stretch = res.makespan_sec / ideal if ideal else 0.0
        print(f"{cs:>7.2f} {res.turn_time_frac:>9.3f} {res.makespan_sec:>10.2f} "
              f"{stretch:>6.2f}x {res.min_shelf_clearance:>9.3f} "
              f"{res.min_robot_clearance:>9.3f} {res.footprint_collisions:>10}")
    print("\nThe discrete plan is collision-free on the grid; a bodied AMR pays a "
          "turning stretch and, below ~1 m aisles, its rectangle overlaps "
          "(clearance < 0) where the point guarantee said it was safe.")


if __name__ == "__main__":
    main()
