"""``mrn_formation_demo``: drive scattered agents into a formation and show it.

A no-ROS entry point that simulates the displacement-based controller pulling
three agents from arbitrary start positions into an equilateral triangle, and
prints the formation error decaying to (near) zero.
"""

from __future__ import annotations

import argparse

from .simulate import simulate
from .spec import polygon_formation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument(
        "--leader", default="", help="agent id to hold as a fixed leader",
    )
    args = parser.parse_args()

    agents = ["1", "2", "3"]
    spec = polygon_formation(agents, radius=2.0)
    # scattered, non-formation start
    start = {"1": (0.0, 0.0), "2": (5.0, 1.0), "3": (1.0, 4.0)}
    edges = [("1", "2"), ("2", "3"), ("1", "3")]  # complete (connected) graph
    leader = args.leader or None

    trajectory, errors = simulate(
        start, spec, edges,
        gain=args.gain, dt=args.dt, steps=args.steps, leader=leader,
    )

    print(f"agents={agents} edges={edges} leader={leader}")
    print(f"initial formation error: {errors[0]:.4f}")
    n = len(errors)
    for idx in (0, n // 4, n // 2, 3 * n // 4, n - 1):
        print(f"  step {idx:4d}: error={errors[idx]:.5f}")
    print(f"final formation error:   {errors[-1]:.6f}")
    print("final positions:")
    for agent, p in trajectory[-1].items():
        print(f"  {agent}: ({p[0]:+.3f}, {p[1]:+.3f})")


if __name__ == "__main__":
    main()
