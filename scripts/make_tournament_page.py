#!/usr/bin/env python3
"""Build docs/tournament.json from pinned battle_gate metrics.

    python3 scripts/make_tournament_page.py
"""

from __future__ import annotations

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir))
_GATE = os.path.join(_REPO, "benchmarks", "expected_metrics", "battle_gate.json")
_OUT = os.path.join(_REPO, "docs", "tournament.json")

MATCHUPS = (
    ("count_aware_vs_nearest", "Count-aware vs nearest", "Open field tactics"),
    ("transformer_vs_nearest", "Transformer vs nearest", "Distilled policy vs baseline"),
    ("astar_maneuver_vs_greedy", "A* maneuver vs greedy", "Planned red movement"),
    ("prioritized_maneuver_vs_greedy", "Prioritized MAPF vs greedy", "MAPF maneuver red"),
    ("hungarian_vs_local", "Hungarian assignment vs local", "Combat matching"),
    ("cbs_ta_vs_hungarian_chokepoint", "CBS-TA vs Hungarian", "Chokepoint assignment"),
    ("mapf_stack_vs_local_chokepoint", "Full MAPF stack vs local", "CBS-TA + prioritized"),
    ("mapf_total_war_local", "MAPF total war (local)", "Hill — Hungarian + greedy"),
    ("mapf_total_war_mapf", "MAPF total war (MAPF)", "Hill — CBS-TA + MAPF"),
    ("ctf_mapf_local", "CTF × MAPF (local)", "CTF — Hungarian + greedy"),
    ("ctf_mapf_mapf", "CTF × MAPF (MAPF)", "CTF — CBS-TA + MAPF"),
)


def _elo_from_win_rate(p):
    """Rough ELO offset from a 50/50 baseline (400 scale, no draws)."""
    p = min(max(float(p), 0.02), 0.98)
    import math
    return round(400.0 * math.log10(p / (1.0 - p)))


def main():
    with open(_GATE, "r", encoding="utf-8") as fh:
        gate = json.load(fh)
    rows = []
    for key, label, note in MATCHUPS:
        red = gate.get(f"{key}_red_win_rate")
        dec = gate.get(f"{key}_decisive_rate")
        if red is None:
            continue
        rows.append({
            "id": key,
            "label": label,
            "note": note,
            "red_win_rate": red,
            "blue_win_rate": round(1.0 - red, 3) if dec and dec >= 0.99 else None,
            "decisive_rate": dec,
            "red_elo_delta": _elo_from_win_rate(red),
        })
    rows.sort(key=lambda r: r["red_elo_delta"], reverse=True)
    payload = {
        "source": "benchmarks/expected_metrics/battle_gate.json",
        "matchups": rows,
    }
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    with open(_OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {_OUT} ({len(rows)} matchups)")


if __name__ == "__main__":
    main()
