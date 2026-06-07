"""Browser battle demo — run a showcase scenario and return JSON for the canvas.

Pure Python, ROS-free; runs under CPython and Pyodide. Imports only the
``mapf-zoo`` wheel (which ships ``mrn_coord.battle`` and its dependencies).
"""

from __future__ import annotations

import json

from mrn_coord.battle import RED, TEAM_NAMES, battle_scenario, simulate

# Subsample frames so the JSON payload stays small in the browser tab.
FRAME_STRIDE = 2

SCENARIOS = {
    "duel": 600,
    "chokepoint": 650,
    "maneuver_duel": 550,
}


def _pack_frame(snapshot):
    """Alive bots only: ``[x, y, team, hp_frac, kind]``."""
    out = []
    for x, y, team, hp, alive, kind in snapshot:
        if not alive:
            continue
        out.append([round(x, 2), round(y, 2), team, round(hp, 3), kind or ""])
    return out


def _pack_shots(shots):
    return [[round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2), team]
            for (x0, y0, x1, y1, team) in shots]


def run(scenario_name: str) -> str:
    """Simulate ``scenario_name`` and return a JSON string for the animator."""
    try:
        if scenario_name not in SCENARIOS:
            return json.dumps({"ok": False, "error": "unknown scenario: %s" % scenario_name})
        bots, cfg, title = battle_scenario(scenario_name)
        res = simulate(bots, cfg, max_ticks=SCENARIOS[scenario_name])
        frames = [_pack_frame(fr) for fr in res.frames[::FRAME_STRIDE]]
        shots = [_pack_shots(sh) for sh in res.shots[::FRAME_STRIDE]]
        winner = None if res.winner is None else TEAM_NAMES.get(res.winner, str(res.winner))
        return json.dumps(
            {
                "ok": True,
                "title": title,
                "scenario": scenario_name,
                "width": cfg.width,
                "height": cfg.height,
                "obstacles": [list(o) for o in cfg.obstacles],
                "frames": frames,
                "shots": shots,
                "counts": res.counts[::FRAME_STRIDE],
                "teams": list(res.teams),
                "winner": winner,
                "ticks": res.ticks,
                "stride": FRAME_STRIDE,
            }
        )
    except Exception as exc:
        return json.dumps({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)})
