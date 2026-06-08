"""Browser battle demo — run a showcase scenario and return JSON for the canvas.

Pure Python, ROS-free; runs under CPython and Pyodide. Imports only the
``mapf-zoo`` wheel (which ships ``mrn_coord.battle`` and its dependencies).
"""

from __future__ import annotations

import json

from mrn_coord.battle import (
    ALLIANCE_NAMES,
    TEAM_NAMES,
    battle_scenario,
    clone_bots,
    mapf_total_war_pair,
    simulate,
)

# Per-scenario limits — lite scenarios subsample more to keep Pyodide payloads small.
SCENARIOS = {
    "duel": {"max_ticks": 600, "frame_stride": 2},
    "chokepoint": {"max_ticks": 650, "frame_stride": 2},
    "maneuver_duel": {"max_ticks": 550, "frame_stride": 2},
    "mapf_stack_duel": {"max_ticks": 550, "frame_stride": 2},
    "mapf_total_war": {"max_ticks": 650, "frame_stride": 2, "dual": True},
    "grand_alliance_lite": {"max_ticks": 800, "frame_stride": 4},
    "kingdom_lite": {"max_ticks": 800, "frame_stride": 4},
    "hill": {"max_ticks": 650, "frame_stride": 2},
    "domination": {"max_ticks": 700, "frame_stride": 2},
    "ctf": {"max_ticks": 900, "frame_stride": 2},
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


def _pack_zones(zone):
    if not zone:
        return []
    if len(zone) == 3 and isinstance(zone[0], (int, float)):
        return [round(v, 2) if isinstance(v, float) else v for v in zone]
    out = []
    for z in zone:
        if isinstance(z[0], str):
            out.append([z[0]] + [round(v, 2) if isinstance(v, float) else v for v in z[1:]])
        else:
            out.append([round(v, 2) if isinstance(v, float) else v for v in z])
    return out


def _pack_result(bots, cfg, res, *, scenario_name, title, stride):
    winner = None if res.winner is None else TEAM_NAMES.get(res.winner, str(res.winner))
    win_alliance = None
    if res.winning_alliance is not None:
        win_alliance = ALLIANCE_NAMES.get(res.winning_alliance,
                                          str(res.winning_alliance))
    return {
        "title": title,
        "scenario": scenario_name,
        "width": cfg.width,
        "height": cfg.height,
        "obstacles": [list(o) for o in cfg.obstacles],
        "frames": [_pack_frame(fr) for fr in res.frames],
        "shots": [_pack_shots(sh) for sh in res.shots],
        "counts": res.counts,
        "teams": list(res.teams),
        "alliances": {str(k): v for k, v in (cfg.alliances or {}).items()},
        "objective": res.objective,
        "objective_zone": _pack_zones(res.objective_zone),
        "objective_progress": res.objective_progress,
        "objective_hold_ticks": cfg.objective_hold_ticks,
        "winner": winner,
        "winning_alliance": win_alliance,
        "n_bots": len(bots),
        "ticks": res.ticks,
        "stride": stride,
    }


def _run_dual_mapf_total_war(opts):
    stride = opts["frame_stride"]
    spawn, cfg_local, cfg_mapf, titles = mapf_total_war_pair()
    panels = []
    for cfg, short in ((cfg_local, titles[0]), (cfg_mapf, titles[1])):
        res = simulate(clone_bots(spawn), cfg, max_ticks=opts["max_ticks"],
                       frame_stride=stride)
        panels.append(_pack_result(spawn, cfg, res, scenario_name="mapf_total_war",
                                   title=short, stride=stride))
    return {
        "ok": True,
        "dual": True,
        "title": f"MAPF total war — {len(spawn)} robots, king of the hill",
        "scenario": "mapf_total_war",
        "panels": panels,
    }


def run(scenario_name: str) -> str:
    """Simulate ``scenario_name`` and return a JSON string for the animator."""
    try:
        if scenario_name not in SCENARIOS:
            return json.dumps({"ok": False, "error": "unknown scenario: %s" % scenario_name})
        opts = SCENARIOS[scenario_name]
        if opts.get("dual"):
            return json.dumps(_run_dual_mapf_total_war(opts))
        stride = opts["frame_stride"]
        bots, cfg, title = battle_scenario(scenario_name)
        res = simulate(bots, cfg, max_ticks=opts["max_ticks"], frame_stride=stride)
        payload = _pack_result(bots, cfg, res, scenario_name=scenario_name,
                               title=title, stride=stride)
        payload["ok"] = True
        return json.dumps(payload)
    except Exception as exc:
        return json.dumps({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)})
