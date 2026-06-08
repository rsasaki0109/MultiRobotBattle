"""Collision-free charge — ORCA / BVC as the battle movement filter.

After flocking + pursue steering produces a preferred velocity, the charge layer
replaces it with a collision-free velocity from the MAPF zoo's decentralized
avoidance primitives (reciprocal ORCA or buffered Voronoi cells).
"""

from __future__ import annotations

import math

from .mapf.bvc import step_bvc
from .orca import orca_velocity

CHARGE_MODES = ("none", "orca", "bvc")


def charge_for_team(team, cfg) -> str:
    return (cfg.charge_by_team or {}).get(team, cfg.charge or "none")


def _needs_charge(cfg) -> bool:
    if cfg.charge not in (None, "", "none"):
        return True
    by = cfg.charge_by_team or {}
    return any(m in ("orca", "bvc") for m in by.values())


def _pref_from_desired(dx, dy, max_speed):
    d = math.hypot(dx, dy)
    if d <= 1e-9:
        return (0.0, 0.0)
    s = min(max_speed, d)
    return (dx / d * s, dy / d * s)


def apply_orca_charge(live, desired, cfg):
    """ORCA-filter preferred velocities for all live bots."""
    radius = cfg.charge_radius
    horizon = cfg.charge_time_horizon
    dt = cfg.dt
    n = len(live)
    new_des = [[0.0, 0.0] for _ in range(n)]
    for i, b in enumerate(live):
        mspeed = b.max_speed if b.max_speed is not None else cfg.max_speed
        pref = _pref_from_desired(desired[i][0], desired[i][1], mspeed)
        neighbors = []
        for j, other in enumerate(live):
            if j == i:
                continue
            neighbors.append(((other.x, other.y), (other.vx, other.vy), radius))
        obstacles = [(ox, oy, r) for ox, oy, r in cfg.obstacles]
        vx, vy = orca_velocity(
            (b.x, b.y), (b.vx, b.vy), pref, neighbors, obstacles,
            radius=radius, max_speed=mspeed, time_horizon=horizon,
            time_step=dt,
        )
        new_des[i][0] = vx
        new_des[i][1] = vy
    for i in range(n):
        desired[i][0] = new_des[i][0]
        desired[i][1] = new_des[i][1]


def apply_bvc_charge(live, desired, cfg):
    """One BVC step toward preferred goal points."""
    radius = cfg.charge_radius
    step = cfg.dt * max(
        (b.max_speed if b.max_speed is not None else cfg.max_speed) for b in live)
    positions = [(b.x, b.y) for b in live]
    goals = []
    for i, b in enumerate(live):
        mspeed = b.max_speed if b.max_speed is not None else cfg.max_speed
        d = math.hypot(desired[i][0], desired[i][1])
        if d <= 1e-9:
            goals.append((b.x, b.y))
        else:
            scale = min(1.0, mspeed / d)
            goals.append((b.x + desired[i][0] * scale * cfg.dt * 3.0,
                          b.y + desired[i][1] * scale * cfg.dt * 3.0))
    nxt = step_bvc(positions, goals, radius, step_size=step)
    for i, b in enumerate(live):
        dx, dy = nxt[i][0] - b.x, nxt[i][1] - b.y
        desired[i][0] = dx / cfg.dt
        desired[i][1] = dy / cfg.dt


def apply_charge(live, desired, cfg):
    """Apply per-team ORCA or BVC charge filters to ``desired`` velocities."""
    if not _needs_charge(cfg):
        return
    modes = {charge_for_team(b.team, cfg) for b in live}
    if modes == {"none"}:
        return
    if "bvc" in modes and "orca" in modes:
        # Mixed teams — apply per bot in sequence (orca first, then bvc subset)
        orca_idx = [i for i, b in enumerate(live) if charge_for_team(b.team, cfg) == "orca"]
        bvc_idx = [i for i, b in enumerate(live) if charge_for_team(b.team, cfg) == "bvc"]
        if orca_idx:
            sub_live = [live[i] for i in orca_idx]
            sub_des = [[desired[i][0], desired[i][1]] for i in orca_idx]
            apply_orca_charge(sub_live, sub_des, cfg)
            for k, i in enumerate(orca_idx):
                desired[i][0] = sub_des[k][0]
                desired[i][1] = sub_des[k][1]
        if bvc_idx:
            sub_live = [live[i] for i in bvc_idx]
            sub_des = [[desired[i][0], desired[i][1]] for i in bvc_idx]
            apply_bvc_charge(sub_live, sub_des, cfg)
            for k, i in enumerate(bvc_idx):
                desired[i][0] = sub_des[k][0]
                desired[i][1] = sub_des[k][1]
        return
    mode = next(m for m in modes if m != "none")
    if mode == "orca":
        apply_orca_charge(live, desired, cfg)
    elif mode == "bvc":
        apply_bvc_charge(live, desired, cfg)
