"""Browser battle demo — run a showcase scenario and return JSON for the canvas.

Pure Python, ROS-free; runs under CPython and Pyodide. Imports only the
``mapf-zoo`` wheel (which ships ``mrn_coord.battle`` and its dependencies).
"""

from __future__ import annotations

import json

from mrn_coord.battle import ALLIANCE_NAMES, TEAM_NAMES, battle_scenario, simulate

# Per-scenario limits — lite scenarios subsample more to keep Pyodide payloads small.
SCENARIOS = {
    "duel": {"max_ticks": 600, "frame_stride": 2},
    "chokepoint": {"max_ticks": 650, "frame_stride": 2},
    "maneuver_duel": {"max_ticks": 550, "frame_stride": 2},
    "mapf_stack_duel": {"max_ticks": 550, "frame_stride": 2},
    "grand_alliance_lite": {"max_ticks": 800, "frame_stride": 4},
    "kingdom_lite": {"max_ticks": 800, "frame_stride": 4},
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
        opts = SCENARIOS[scenario_name]
        stride = opts["frame_stride"]
        bots, cfg, title = battle_scenario(scenario_name)
        res = simulate(bots, cfg, max_ticks=opts["max_ticks"], frame_stride=stride)
        frames = [_pack_frame(fr) for fr in res.frames]
        shots = [_pack_shots(sh) for sh in res.shots]
        winner = None if res.winner is None else TEAM_NAMES.get(res.winner, str(res.winner))
        win_alliance = None
        if res.winning_alliance is not None:
            win_alliance = ALLIANCE_NAMES.get(res.winning_alliance,
                                              str(res.winning_alliance))
        alliances = {str(k): v for k, v in (cfg.alliances or {}).items()}
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
                "counts": res.counts,
                "teams": list(res.teams),
                "alliances": alliances,
                "winner": winner,
                "winning_alliance": win_alliance,
                "n_bots": len(bots),
                "ticks": res.ticks,
                "stride": stride,
            }
        )
    except Exception as exc:
        return json.dumps({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)})
