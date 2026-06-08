"""Objective zones — hill hold and domination control for swarm battles."""

from __future__ import annotations

import math

from .battle_teams import alliance_of

OBJECTIVE_MODES = ("annihilation", "hill", "domination")


def objective_zone(cfg):
    """Return ``(cx, cy, radius)`` for the control zone."""
    cx, cy = cfg.objective_center or (cfg.width / 2.0, cfg.height / 2.0)
    return (cx, cy, cfg.objective_radius)


def zone_leader(bots, cfg, teams):
    """Alliance or team id controlling the zone, or ``None`` if empty / tied."""
    cx, cy, radius = objective_zone(cfg)
    buckets = {}
    for b in bots:
        if not b.alive:
            continue
        if math.hypot(b.x - cx, b.y - cy) > radius:
            continue
        key = alliance_of(cfg.alliances, b.team) if cfg.alliances else b.team
        buckets[key] = buckets.get(key, 0) + 1
    if not buckets:
        return None
    best_n = max(buckets.values())
    leaders = [k for k, n in buckets.items() if n == best_n]
    if len(leaders) != 1:
        return None
    return leaders[0]


class ObjectiveTracker:
    """Track hill (consecutive) or domination (cumulative) control progress."""

    def __init__(self, cfg):
        self.mode = cfg.objective or "annihilation"
        self.hold_ticks = max(1, cfg.objective_hold_ticks)
        self.consecutive = {}
        self.cumulative = {}

    def tick(self, leader):
        if self.mode == "annihilation":
            return None
        if leader is None:
            if self.mode == "hill":
                self.consecutive.clear()
            return None
        if self.mode == "hill":
            self.consecutive[leader] = self.consecutive.get(leader, 0) + 1
            for k in list(self.consecutive):
                if k != leader:
                    del self.consecutive[k]
            if self.consecutive[leader] >= self.hold_ticks:
                return leader
            return None
        # domination — cumulative control time
        self.cumulative[leader] = self.cumulative.get(leader, 0) + 1
        if self.cumulative[leader] >= self.hold_ticks:
            return leader
        return None

    def snapshot(self):
        if self.mode == "hill":
            return dict(self.consecutive)
        if self.mode == "domination":
            return dict(self.cumulative)
        return {}


def winner_from_objective(key, cfg, teams):
    """Map an objective winner key to ``(team_id, alliance_id)``."""
    if cfg.alliances:
        for t in teams:
            if alliance_of(cfg.alliances, t) == key:
                return t, key
        return None, key
    return key, None
