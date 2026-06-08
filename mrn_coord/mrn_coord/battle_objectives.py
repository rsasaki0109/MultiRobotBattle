"""Objective zones — hill, domination, and capture-the-flag for swarm battles."""

from __future__ import annotations

import math

from .battle_teams import RED, BLUE, alliance_of

OBJECTIVE_MODES = ("annihilation", "hill", "domination", "ctf", "base_assault")


def objective_zone(cfg):
    """Return ``(cx, cy, radius)`` for the control zone."""
    cx, cy = cfg.objective_center or (cfg.width / 2.0, cfg.height / 2.0)
    return (cx, cy, cfg.objective_radius)


def flag_spawn(cfg):
    """Neutral flag start position."""
    cx, cy = cfg.objective_center or (cfg.width / 2.0, cfg.height / 2.0)
    return (cx, cy)


def team_bases(cfg, teams):
    """Home bases for CTF — red west, blue east."""
    w, h = cfg.width, cfg.height
    r = getattr(cfg, "base_radius", None) or cfg.objective_radius
    bases = {}
    for t in sorted(teams):
        if t == RED:
            bases[t] = (w * 0.12, h * 0.5, r)
        elif t == BLUE:
            bases[t] = (w * 0.88, h * 0.5, r)
        else:
            angle = 2.0 * math.pi * t / max(4, len(teams))
            bases[t] = (w * 0.5 + 0.34 * w * math.cos(angle),
                        h * 0.5 + 0.34 * h * math.sin(angle), r)
    return bases


def ctf_render_zones(cfg, teams):
    """Zones for animators: flag spawn + home bases."""
    fx, fy = flag_spawn(cfg)
    pickup = cfg.objective_radius
    out = [["flag", fx, fy, pickup]]
    for team, (bx, by, br) in team_bases(cfg, teams).items():
        out.append(["base", team, bx, by, br])
    return out


def base_assault_render_zones(cfg, teams):
    """Home bases to assault — same layout as CTF without the centre flag."""
    out = []
    for team, (bx, by, br) in team_bases(cfg, teams).items():
        out.append(["base", team, bx, by, br])
    return out


def base_capture_leader(bots, cfg, teams, base_owner_team):
    """Alliance id capturing ``base_owner_team``'s base, or ``None``."""
    bases = team_bases(cfg, teams)
    if base_owner_team not in bases:
        return None
    bx, by, br = bases[base_owner_team]
    buckets = {}
    for b in bots:
        if not b.alive:
            continue
        if math.hypot(b.x - bx, b.y - by) > br:
            continue
        key = alliance_of(cfg.alliances, b.team) if cfg.alliances else b.team
        buckets[key] = buckets.get(key, 0) + 1
    if not buckets:
        return None
    owner_key = (alliance_of(cfg.alliances, base_owner_team)
                 if cfg.alliances else base_owner_team)
    attackers = {k: n for k, n in buckets.items() if k != owner_key}
    if not attackers:
        return None
    defender_n = buckets.get(owner_key, 0)
    best_n = max(attackers.values())
    leaders = [k for k, n in attackers.items() if n == best_n]
    if len(leaders) != 1 or best_n <= defender_n:
        return None
    return leaders[0]


class BaseAssaultTracker:
    """Hold the enemy HQ — consecutive ticks inside their base zone wins."""

    def __init__(self, cfg, teams):
        self.hold_ticks = max(1, cfg.objective_hold_ticks)
        self.teams = list(teams)
        self.bases = team_bases(cfg, teams)
        self.progress = {}

    def tick(self, bots, cfg):
        active = set()
        for base_owner in self.teams:
            captor = base_capture_leader(bots, cfg, self.teams, base_owner)
            if captor is not None:
                active.add((captor, base_owner))
        for key in list(self.progress):
            if key not in active:
                del self.progress[key]
        for key in active:
            self.progress[key] = self.progress.get(key, 0) + 1
            if self.progress[key] >= self.hold_ticks:
                return key[0]
        return None

    def snapshot(self):
        snap = {}
        for (captor, base_owner), n in self.progress.items():
            snap[f"{captor}:{base_owner}"] = n
            snap[f"assault_{base_owner}"] = n
        return snap


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


class CtfTracker:
    """Capture-the-flag — pickup at centre, score by returning to home base."""

    def __init__(self, cfg, teams):
        self.pickup_r = cfg.objective_radius
        self.bases = team_bases(cfg, teams)
        fx, fy = flag_spawn(cfg)
        self.flag_x, self.flag_y = fx, fy
        self.carrier_idx = None

    def _winner_key(self, team, cfg):
        return alliance_of(cfg.alliances, team) if cfg.alliances else team

    def _in_base(self, bot):
        base = self.bases.get(bot.team)
        if base is None:
            return False
        bx, by, br = base
        return math.hypot(bot.x - bx, bot.y - by) <= br

    def tick(self, bots, cfg):
        if self.carrier_idx is not None:
            carrier = bots[self.carrier_idx]
            if not carrier.alive:
                self.flag_x, self.flag_y = carrier.x, carrier.y
                self.carrier_idx = None
            else:
                self.flag_x, self.flag_y = carrier.x, carrier.y
                if self._in_base(carrier):
                    return self._winner_key(carrier.team, cfg)
                return None

        best_i, best_d = None, self.pickup_r + 1.0
        for i, b in enumerate(bots):
            if not b.alive:
                continue
            d = math.hypot(b.x - self.flag_x, b.y - self.flag_y)
            if d <= self.pickup_r and d < best_d:
                best_d, best_i = d, i
        if best_i is not None:
            self.carrier_idx = best_i
            b = bots[best_i]
            self.flag_x, self.flag_y = b.x, b.y
        return None

    def snapshot(self, bots, cfg):
        carrier = None
        if self.carrier_idx is not None:
            b = bots[self.carrier_idx]
            if b.alive:
                carrier = self._winner_key(b.team, cfg)
        return {
            "flag": [round(self.flag_x, 2), round(self.flag_y, 2)],
            "carrier": carrier,
        }


def winner_from_objective(key, cfg, teams):
    """Map an objective winner key to ``(team_id, alliance_id)``."""
    if cfg.alliances:
        for t in teams:
            if alliance_of(cfg.alliances, t) == key:
                return t, key
        return None, key
    return key, None
