"""``mrn_mapf_bench``: run a MAPF solver on a MovingAI benchmark and report.

Point it at a MovingAI ``.map`` / ``.scen`` pair (downloaded from
movingai.com or the bundled ``example``) and a solver; it prints solve metrics.

    ros2 run mrn_coord mrn_mapf_bench                       # bundled example
    ros2 run mrn_coord mrn_mapf_bench my.map my.scen -n 4   # first 4 agents
"""

from __future__ import annotations

import argparse
import json
import os

from .movingai import load_map, load_scen, run_mapf_benchmark


def _bundled(name: str) -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("mrn_coord"), "benchmarks", name)
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, os.pardir, os.pardir, "benchmarks", name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", nargs="?", help="MovingAI .map (default: bundled example)")
    parser.add_argument("scen", nargs="?", help="MovingAI .scen (default: bundled example)")
    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument("--solver", choices=["cbs", "prioritized"], default="cbs")
    parser.add_argument("--max-expansions", type=int, default=100000)
    args = parser.parse_args()

    map_path = args.map or _bundled("example.map")
    scen_path = args.scen or _bundled("example.scen")
    grid = load_map(map_path)
    tasks = load_scen(scen_path)
    result = run_mapf_benchmark(grid, tasks, num_agents=args.num_agents,
                                solver=args.solver, max_expansions=args.max_expansions)
    result["map"] = os.path.basename(map_path)
    result["grid"] = f"{grid.width}x{grid.height}"
    result["tasks_available"] = len(tasks)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
