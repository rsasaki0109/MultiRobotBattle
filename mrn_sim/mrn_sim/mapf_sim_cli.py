"""``mrn_mapf_sim``: execute a discrete MAPF plan in the continuous world.

Solves a small built-in 4-way crossing with a MAPF solver, then runs the *same*
plan through the deterministic 2D world three ways — free-running pursuit (no
schedule), Temporal-Plan-Graph execution (schedule-gated), and reactive DWA —
and prints the plan-vs-reality metrics side by side. The runnable counterpart to
``test_mapf_exec`` and the comparison report.
"""

from __future__ import annotations

import argparse


def main() -> None:
    from mrn_coord.mapf import GridWorld

    from .mapf_exec import execute_mapf_plan

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", default="lacam",
                        choices=["cbs", "ecbs", "lacam", "lns", "prioritized"])
    parser.add_argument("--size", type=int, default=7, help="grid is NxN")
    parser.add_argument("--cell-size", type=float, default=1.0)
    parser.add_argument("--robot-radius", type=float, default=0.2)
    args = parser.parse_args()

    n = args.size
    mid = n // 2
    agents = {
        "0": ((0, mid), (n - 1, mid)),
        "1": ((n - 1, mid), (0, mid)),
        "2": ((mid, 0), (mid, n - 1)),
        "3": ((mid, n - 1), (mid, 0)),
    }
    grid = GridWorld(n, n)

    print(f"=== MAPF plan execution — {n}x{n} crossing, {len(agents)} robots, "
          f"{args.solver} plan ===")
    print(f"  {'execution':10} {'success':>7} {'coll':>5} {'disc':>5} "
          f"{'cont':>5} {'makespan_s':>11} {'dev_m':>6}")
    for controller in ("pursuit", "tpg", "dwa"):
        r = execute_mapf_plan(grid, agents, solver=args.solver,
                              controller=controller, cell_size=args.cell_size,
                              robot_radius=args.robot_radius)
        d = r.as_dict()
        print(f"  {controller:10} {str(d['success']):>7} "
              f"{d['robot_collisions']:>5} {d['discrete_makespan']:>5} "
              f"{d['continuous_steps']:>5} {d['makespan_sec']:>11.1f} "
              f"{d['max_path_deviation']:>6.2f}")


if __name__ == "__main__":
    main()
