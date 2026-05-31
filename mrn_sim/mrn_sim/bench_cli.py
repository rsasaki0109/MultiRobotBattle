"""``mrn_sim_bench``: run a scenario with a policy and print the metrics.

A one-command benchmark runner. By default it uses the built-in
``navigate_policy`` (A* + pursuit + reciprocal avoidance) so it works out of the
box; point it at any scenario YAML (your own or one from ``mrn_sim/scenarios``).
The same ``run_scenario`` is the entry point for plugging in your own policy
from Python.
"""

from __future__ import annotations

import argparse
import json
import os

from .benchmark import load_scenario, navigate_policy, run_scenario


def _builtin_scenario_path(name: str) -> str:
    fname = name if name.endswith(".yaml") else name + ".yaml"
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("mrn_sim"), "scenarios", fname)
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, "scenarios", fname)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario",
        help="path to a scenario YAML, or a built-in name "
        "(around_obstacle / crossing / doorway)",
    )
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--dt", type=float, default=0.1)
    args = parser.parse_args()

    path = args.scenario if os.path.exists(args.scenario) else _builtin_scenario_path(args.scenario)
    scenario = load_scenario(path)
    result = run_scenario(scenario, navigate_policy(scenario),
                          dt=args.dt, max_steps=args.max_steps)
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
