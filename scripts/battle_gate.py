#!/usr/bin/env python3
"""Win-rate regression gate for swarm battle tactics.

Runs deterministic matchups and compares metrics against pinned expectations in
``benchmarks/expected_metrics/battle_gate.json``.

    python3 scripts/battle_gate.py
    python3 scripts/battle_gate.py --update
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir))
sys.path.insert(0, os.path.join(_REPO, "mrn_coord"))

_EXPECTED = os.path.join(_REPO, "benchmarks", "expected_metrics", "battle_gate.json")
_FLOAT_TOL = 0.08


def _run_matchup(red_tactics, blue_tactics, *, seeds, n_per_team=12,
                 red_maneuver=None, blue_maneuver=None, max_ticks=900,
                 red_assignment=None, blue_assignment=None):
    from mrn_coord.battle import BLUE, RED, BattleConfig, make_armies, simulate

    wins = {RED: 0, BLUE: 0, None: 0}
    for seed in seeds:
        kw = dict(tactics="nearest",
                  tactics_by_team={RED: red_tactics, BLUE: blue_tactics})
        if red_maneuver or blue_maneuver:
            kw["maneuver"] = "greedy"
            kw["maneuver_by_team"] = {}
            if red_maneuver:
                kw["maneuver_by_team"][RED] = red_maneuver
            if blue_maneuver:
                kw["maneuver_by_team"][BLUE] = blue_maneuver
        if red_assignment or blue_assignment:
            kw["assignment"] = "none"
            kw["assignment_by_team"] = {}
            if red_assignment:
                kw["assignment_by_team"][RED] = red_assignment
            if blue_assignment:
                kw["assignment_by_team"][BLUE] = blue_assignment
        cfg = BattleConfig(**kw)
        bots = make_armies(n_per_team, cfg, seed=seed)
        res = simulate(bots, cfg, max_ticks=max_ticks)
        wins[res.winner if res.winner is not None else None] += 1
    n = len(seeds)
    return {
        "red_tactics": red_tactics,
        "blue_tactics": blue_tactics,
        "n_seeds": n,
        "red_win_rate": wins[RED] / n,
        "blue_win_rate": wins[BLUE] / n,
        "draw_rate": wins[None] / n,
        "decisive_rate": 1.0 - wins[None] / n,
    }


def _run_decisive(tactics, *, seeds, n_per_team=12):
    from mrn_coord.battle import BattleConfig, run_battle

    decisive = 0
    for seed in seeds:
        res = run_battle(n_per_team, BattleConfig(tactics=tactics), seed=seed,
                         max_ticks=900)
        if res.winner is not None:
            decisive += 1
    n = len(seeds)
    return {"tactics": tactics, "decisive_rate": decisive / n, "n_seeds": n}


def _run_chokepoint_matchup(*, seeds, n_per_team=8, max_ticks=650,
                            red_assignment=None, blue_assignment=None):
    """Chokepoint with terrain — assignment layers matter more than open field."""
    from mrn_coord.battle import BLUE, RED, BattleConfig, make_company, simulate
    import random

    obstacles = ((20.0, 4.5, 2.6), (20.0, 12.0, 2.6), (20.0, 19.5, 2.6))
    wins = {RED: 0, BLUE: 0, None: 0}
    for seed in seeds:
        cfg = BattleConfig(
            obstacles=obstacles,
            tactics="count_aware",
            formation="wedge",
            assignment="none",
            assignment_by_team={},
        )
        if red_assignment:
            cfg.assignment_by_team[RED] = red_assignment
        if blue_assignment:
            cfg.assignment_by_team[BLUE] = blue_assignment
        rng = random.Random(seed)
        red = make_company(cfg, RED, (cfg.width * 0.13, cfg.height * 0.5),
                           [("soldier", n_per_team)], rng, jitter=2.8)
        blue = make_company(cfg, BLUE, (cfg.width * 0.87, cfg.height * 0.5),
                            [("soldier", n_per_team)], random.Random(seed + 1),
                            jitter=2.8)
        res = simulate(red + blue, cfg, max_ticks=max_ticks)
        wins[res.winner if res.winner is not None else None] += 1
    n = len(seeds)
    return {
        "n_seeds": n,
        "red_win_rate": wins[RED] / n,
        "blue_win_rate": wins[BLUE] / n,
        "draw_rate": wins[None] / n,
        "decisive_rate": 1.0 - wins[None] / n,
    }


def collect_metrics(*, seeds):
    metrics = {}
    for tactics in ("nearest", "count_aware", "transformer"):
        metrics[f"decisive_{tactics}"] = _run_decisive(tactics, seeds=seeds)
    metrics["count_aware_vs_nearest"] = _run_matchup(
        "count_aware", "nearest", seeds=seeds)
    metrics["transformer_vs_nearest"] = _run_matchup(
        "transformer", "nearest", seeds=seeds)
    metrics["astar_maneuver_vs_greedy"] = _run_matchup(
        "count_aware", "count_aware", seeds=seeds, n_per_team=8,
        red_maneuver="astar", blue_maneuver="greedy", max_ticks=600)
    metrics["prioritized_maneuver_vs_greedy"] = _run_matchup(
        "count_aware", "count_aware", seeds=seeds, n_per_team=8,
        red_maneuver="prioritized", blue_maneuver="greedy", max_ticks=600)
    metrics["hungarian_vs_local"] = _run_matchup(
        "count_aware", "count_aware", seeds=seeds, n_per_team=8,
        max_ticks=600, red_assignment="hungarian", blue_assignment="none")
    metrics["cbs_ta_vs_hungarian_chokepoint"] = _run_chokepoint_matchup(
        seeds=seeds, red_assignment="cbs_ta", blue_assignment="hungarian")
    return metrics


def _flatten_metrics(metrics):
    out = {}
    for key, block in metrics.items():
        if "decisive_rate" in block:
            out[f"{key}_rate"] = block["decisive_rate"]
        if "red_win_rate" in block:
            out[f"{key}_red_win_rate"] = block["red_win_rate"]
            out[f"{key}_decisive_rate"] = block["decisive_rate"]
    return out


def _compare(expected, actual):
    failures = []
    for key, exp in expected.items():
        if key not in actual:
            failures.append(f"missing metric {key}")
            continue
        if abs(float(actual[key]) - float(exp)) > _FLOAT_TOL:
            failures.append(f"{key}: expected {exp}, got {actual[key]}")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--seeds", type=int, default=12)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    metrics = collect_metrics(seeds=seeds)
    flat = _flatten_metrics(metrics)

    if args.update:
        os.makedirs(os.path.dirname(_EXPECTED), exist_ok=True)
        with open(_EXPECTED, "w", encoding="utf-8") as fh:
            json.dump(flat, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"updated {_EXPECTED}")
        return 0

    with open(_EXPECTED, "r", encoding="utf-8") as fh:
        expected = json.load(fh)
    failures = _compare(expected, flat)
    if failures:
        print("battle_gate FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("battle_gate OK")
    for key in sorted(flat):
        print(f"  {key}: {flat[key]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
