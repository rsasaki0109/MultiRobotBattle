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
    ctf_mapf_pair,
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
    "ctf_mapf": {"max_ticks": 900, "frame_stride": 2, "dual": True},
    "grand_alliance_lite": {"max_ticks": 800, "frame_stride": 4},
    "kingdom_lite": {"max_ticks": 800, "frame_stride": 4},
    "hill": {"max_ticks": 650, "frame_stride": 2},
    "domination": {"max_ticks": 700, "frame_stride": 2},
    "ctf": {"max_ticks": 900, "frame_stride": 2},
    "base_assault": {"max_ticks": 900, "frame_stride": 2},
    "escort": {"max_ticks": 900, "frame_stride": 2},
    "fog_ambush": {"max_ticks": 900, "frame_stride": 2},
    "artillery_barrage": {"max_ticks": 900, "frame_stride": 2},
    "fog_artillery": {"max_ticks": 900, "frame_stride": 2},
    "morale_duel": {"max_ticks": 900, "frame_stride": 2},
    "orca_charge_duel": {"max_ticks": 700, "frame_stride": 2},
}


def _pack_frame(snapshot, *, fog_visible=None, view_team=0):
    """Alive bots only: ``[x, y, team, hp_frac, kind]`` or with trailing ``visible``."""
    out = []
    vis = set(fog_visible or [])
    for i, (x, y, team, hp, alive, kind) in enumerate(snapshot):
        if not alive:
            continue
        row = [round(x, 2), round(y, 2), team, round(hp, 3), kind or ""]
        if fog_visible is not None:
            row.append(team == view_team or i in vis)
        out.append(row)
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
        "walls": [list(w) for w in getattr(cfg, "walls", ())],
        "elevation": [list(z) for z in getattr(cfg, "elevation", ())],
        "frames": [
            _pack_frame(
                fr,
                fog_visible=res.fog_visible[i] if cfg.fog_of_war else None,
                view_team=cfg.fog_view_team if cfg.fog_view_team is not None else 0,
            )
            for i, fr in enumerate(res.frames)
        ],
        "shots": [_pack_shots(sh) for sh in res.shots],
        "projectiles": [
            [[round(x, 2), round(y, 2), team] for (x, y, team) in fr]
            for fr in getattr(res, "projectiles", [])
        ],
        "explosions": [
            [[round(x, 2), round(y, 2), round(r, 2), team]
             for (x, y, r, team) in fr]
            for fr in getattr(res, "explosions", [])
        ],
        "counts": res.counts,
        "teams": list(res.teams),
        "alliances": {str(k): v for k, v in (cfg.alliances or {}).items()},
        "objective": res.objective,
        "objective_zone": _pack_zones(res.objective_zone),
        "objective_progress": res.objective_progress,
        "objective_hold_ticks": cfg.objective_hold_ticks,
        "fog_of_war": bool(getattr(cfg, "fog_of_war", False)),
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


def _run_dual_ctf_mapf(opts):
    stride = opts["frame_stride"]
    spawn, cfg_local, cfg_mapf, titles = ctf_mapf_pair()
    panels = []
    for cfg, short in ((cfg_local, titles[0]), (cfg_mapf, titles[1])):
        res = simulate(clone_bots(spawn), cfg, max_ticks=opts["max_ticks"],
                       frame_stride=stride)
        panels.append(_pack_result(spawn, cfg, res, scenario_name="ctf_mapf",
                                   title=short, stride=stride))
    return {
        "ok": True,
        "dual": True,
        "title": f"CTF × MAPF — {len(spawn)} robots, capture the flag",
        "scenario": "ctf_mapf",
        "panels": panels,
    }


def run(scenario_name: str) -> str:
    """Simulate ``scenario_name`` and return a JSON string for the animator."""
    try:
        if scenario_name not in SCENARIOS:
            return json.dumps({"ok": False, "error": "unknown scenario: %s" % scenario_name})
        opts = SCENARIOS[scenario_name]
        if scenario_name == "ctf_mapf":
            return json.dumps(_run_dual_ctf_mapf(opts))
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
