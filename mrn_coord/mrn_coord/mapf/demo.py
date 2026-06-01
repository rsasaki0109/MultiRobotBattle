"""``mrn_mapf_demo``: solve a small built-in MAPF instance and print it.

A no-argument, no-ROS entry point that runs Conflict-Based Search on a couple
of illustrative scenarios and renders the resulting collision-free paths as an
ASCII timeline — the runnable counterpart to the unit tests.
"""

from __future__ import annotations

import argparse

from .cbs import cbs
from .ecbs import ecbs
from .grid import GridWorld
from .prioritized import prioritized_planning
from .solution import pad_paths, render_ascii


def _crossing_scenario():
    # Two agents whose straight-line paths cross at the center cell.
    grid = GridWorld(5, 5)
    agents = {
        "1": ((0, 2), (4, 2)),
        "2": ((2, 0), (2, 4)),
    }
    return "crossing (2 agents)", grid, agents


def _swap_scenario():
    # Three agents that must reorder through a 3-wide corridor with a passing row.
    grid = GridWorld(5, 2)
    agents = {
        "1": ((0, 0), (4, 0)),
        "2": ((4, 0), (0, 0)),
        "3": ((2, 1), (2, 0)),
    }
    return "swap + reorder (3 agents)", grid, agents


def _print_solution(title, grid, agents, solver_name, solution):
    print(f"\n=== {title} — {solver_name} ===")
    if solution is None:
        print("  no solution found")
        return
    print(f"  sum-of-costs={solution.cost}  makespan={solution.makespan}")
    padded = pad_paths(solution.paths)
    horizon = max(len(p) for p in padded.values())
    for t in range(horizon):
        print(f"  t={t}")
        for line in render_ascii(grid, padded, t).splitlines():
            print(f"    {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solver", choices=["cbs", "ecbs", "prioritized"], default="cbs",
        help="high-level solver to run",
    )
    parser.add_argument(
        "-w", "--weight", type=float, default=1.5,
        help="ECBS suboptimality factor (cost <= w * optimal)",
    )
    args = parser.parse_args()
    solvers = {
        "cbs": cbs,
        "ecbs": lambda grid, agents: ecbs(grid, agents, w=args.weight),
        "prioritized": prioritized_planning,
    }
    solve = solvers[args.solver]

    for builder in (_crossing_scenario, _swap_scenario):
        title, grid, agents = builder()
        _print_solution(title, grid, agents, args.solver, solve(grid, agents))


if __name__ == "__main__":
    main()
