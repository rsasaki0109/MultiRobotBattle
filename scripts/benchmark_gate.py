#!/usr/bin/env python3
"""Scenario-driven benchmark regression gate.

Runs the bundled benchmarks (deterministic) and compares their metrics against
checked-in expectations in ``benchmarks/expected_metrics/``. Exits non-zero on
any regression, so CI fails if a change degrades planning/control/allocation —
the benchmarks become a guarded contract, not decoration.

    python3 scripts/benchmark_gate.py            # check (CI gate)
    python3 scripts/benchmark_gate.py --update   # rewrite the expectations

Pure and deterministic: no ROS daemon, no external data. Discrete metrics
(success, collisions, makespan steps, solved, sum-of-costs) are compared
exactly; floats within a small tolerance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir))
sys.path.insert(0, os.path.join(_REPO, "mrn_sim"))
sys.path.insert(0, os.path.join(_REPO, "mrn_coord"))

_EXPECTED_DIR = os.path.join(_REPO, "benchmarks", "expected_metrics")
_FLOAT_TOL = 0.05


def _run_sim_scenario(name: str, policy: str = "navigate") -> dict:
    from mrn_sim.benchmark import (
        load_scenario,
        navigate_policy,
        orca_policy,
        run_scenario,
    )

    builders = {"navigate": navigate_policy, "orca": orca_policy}
    scenario = load_scenario(os.path.join(_REPO, "mrn_sim", "scenarios", name + ".yaml"))
    result = run_scenario(scenario, builders[policy](scenario), dt=0.1, max_steps=600)
    out = result.as_dict()
    out["policy"] = policy
    return out


def _run_mapf_example(solver: str) -> dict:
    from mrn_coord.mapf.movingai import load_map, load_scen, run_mapf_benchmark

    bench = os.path.join(_REPO, "mrn_coord", "benchmarks")
    grid = load_map(os.path.join(bench, "example.map"))
    tasks = load_scen(os.path.join(bench, "example.scen"))
    res = run_mapf_benchmark(grid, tasks, solver=solver, max_expansions=50_000)
    res["case"] = "mapf_example_" + solver
    return res


# (case name, producer) — each returns a flat metrics dict.
SUITE = [
    ("sim_around_obstacle", lambda: _run_sim_scenario("around_obstacle")),
    ("sim_crossing", lambda: _run_sim_scenario("crossing")),
    ("sim_doorway", lambda: _run_sim_scenario("doorway")),
    ("sim_crossing_orca", lambda: _run_sim_scenario("crossing", "orca")),
    ("sim_doorway_orca", lambda: _run_sim_scenario("doorway", "orca")),
    ("mapf_example_cbs", lambda: _run_mapf_example("cbs")),
    ("mapf_example_prioritized", lambda: _run_mapf_example("prioritized")),
]


def _compare(expected: dict, actual: dict) -> list:
    """Return a list of human-readable mismatch strings (empty == pass)."""
    diffs = []
    for key, exp in expected.items():
        act = actual.get(key, "<missing>")
        if isinstance(exp, bool) or isinstance(exp, int) or isinstance(exp, str):
            if act != exp:
                diffs.append(f"{key}: expected {exp!r}, got {act!r}")
        elif isinstance(exp, float):
            if not isinstance(act, (int, float)) or abs(act - exp) > _FLOAT_TOL:
                diffs.append(f"{key}: expected {exp} ± {_FLOAT_TOL}, got {act}")
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true",
                        help="rewrite the expected metrics from current results")
    args = parser.parse_args()
    os.makedirs(_EXPECTED_DIR, exist_ok=True)

    failures = 0
    for case, run in SUITE:
        actual = run()
        path = os.path.join(_EXPECTED_DIR, case + ".json")
        if args.update:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(actual, fh, indent=2, sort_keys=True)
            print(f"updated {case}")
            continue
        if not os.path.exists(path):
            print(f"FAIL {case}: no expected metrics ({path})")
            failures += 1
            continue
        with open(path, "r", encoding="utf-8") as fh:
            expected = json.load(fh)
        diffs = _compare(expected, actual)
        if diffs:
            failures += 1
            print(f"FAIL {case}:")
            for d in diffs:
                print(f"    {d}")
        else:
            print(f"ok   {case}")

    if args.update:
        return 0
    print(f"\n{len(SUITE) - failures}/{len(SUITE)} benchmark cases within expectation")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
