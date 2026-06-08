"""Objective zones — hill, domination, and capture-the-flag for swarm battles."""

from __future__ import annotations

import math

from .battle_teams import RED, BLUE, alliance_of, teams_are_enemies
from .battle_terrain import push_out_of_walls

OBJECTIVE_MODES = (
    "annihilation", "hill", "domination", "ctf", "base_assault", "escort",
)


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


def escort_goal_team(cfg, teams, escort_team):
    """Enemy home base the payload must reach."""
    for t in teams:
        if t != escort_team and teams_are_enemies(cfg.alliances, escort_team, t):
            return t
    return BLUE if escort_team == RED else RED


def escort_render_zones(cfg, teams, escort_team):
    """Spawn base, delivery goal, and any other team bases."""
    out = []
    goal_team = escort_goal_team(cfg, teams, escort_team)
    for team, (bx, by, br) in team_bases(cfg, teams).items():
        if team == escort_team:
            out.append(["spawn", team, bx, by, br])
        elif team == goal_team:
            out.append(["goal", team, bx, by, br])
        else:
            out.append(["base", team, bx, by, br])
    return out


def _push_out_of_obstacles(x, y, obstacles, *, body: float = 0.55):
    for ox, oy, r in obstacles:
        d = math.hypot(x - ox, y - oy)
        need = r + body
        if d < need:
            if d < 1e-9:
                x += need
            else:
                push = (need - d) / d
                x += push * (x - ox)
                y += push * (y - oy)
    return x, y


def _escort_key(cfg, team):
    return alliance_of(cfg.alliances, team) if cfg.alliances else team


class EscortTracker:
    """Escort a payload from home to the enemy HQ — allies push, enemies block."""

    def __init__(self, cfg, teams):
        self.escort_team = cfg.escort_team if cfg.escort_team is not None else RED
        self.bases = team_bases(cfg, teams)
        self.escort_radius = getattr(cfg, "escort_radius", 5.5)
        self.payload_speed = getattr(cfg, "payload_speed", 1.4)
        self.payload_body = getattr(cfg, "payload_radius", 1.0)
        self.goal_team = escort_goal_team(cfg, teams, self.escort_team)
        sx, sy, _ = self.bases[self.escort_team]
        gx, gy, gr = self.bases[self.goal_team]
        self.payload_x, self.payload_y = sx, sy
        self.start_x, self.start_y = sx, sy
        self.goal_x, self.goal_y = gx, gy
        self.goal_radius = getattr(cfg, "base_radius", None) or gr
        self.total_dist = max(1.0, math.hypot(gx - sx, gy - sy))
        self.escort_key = _escort_key(cfg, self.escort_team)

    def _nearby_counts(self, bots, cfg):
        ally_n, enemy_n = 0, 0
        for b in bots:
            if not b.alive:
                continue
            if math.hypot(b.x - self.payload_x, b.y - self.payload_y) > self.escort_radius:
                continue
            key = _escort_key(cfg, b.team)
            if key == self.escort_key:
                ally_n += 1
            elif teams_are_enemies(cfg.alliances, self.escort_team, b.team):
                enemy_n += 1
        return ally_n, enemy_n

    def _advance_payload(self, cfg, speed):
        px, py = self.payload_x, self.payload_y
        gx, gy = self.goal_x, self.goal_y
        dx, dy = gx - px, gy - py
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return
        base = math.atan2(dy, dx)
        body = self.payload_body
        best = None
        best_score = -1e18
        for off in (0.0, 0.45, -0.45, 0.9, -0.9, 1.35, -1.35):
            ang = base + off
            nx = px + speed * math.cos(ang)
            ny = py + speed * math.sin(ang)
            if cfg.obstacles:
                nx, ny = _push_out_of_obstacles(nx, ny, cfg.obstacles, body=body)
            if cfg.walls:
                nx, ny = push_out_of_walls(nx, ny, cfg.walls, body=body)
            moved = math.hypot(nx - px, ny - py)
            if moved < 0.015:
                continue
            remain = math.hypot(nx - gx, ny - gy)
            score = dist - remain - 0.05 * abs(off)
            if score > best_score:
                best_score = score
                best = (nx, ny)
        if best is not None:
            self.payload_x, self.payload_y = best
        elif speed >= dist:
            self.payload_x, self.payload_y = gx, gy

    def _move_payload(self, bots, cfg):
        ally_n, enemy_n = self._nearby_counts(bots, cfg)
        if ally_n <= 0:
            return
        speed_scale = max(0.22, (ally_n - enemy_n + 1) / (ally_n + enemy_n + 2))
        speed = self.payload_speed * cfg.dt * speed_scale
        self._advance_payload(cfg, speed)

    def tick(self, bots, cfg):
        self._move_payload(bots, cfg)
        if math.hypot(self.payload_x - self.goal_x, self.payload_y - self.goal_y) <= self.goal_radius:
            return self.escort_key
        return None

    def snapshot(self):
        dist = math.hypot(self.payload_x - self.goal_x, self.payload_y - self.goal_y)
        progress = max(0.0, min(1.0, 1.0 - dist / self.total_dist))
        return {
            "payload": [round(self.payload_x, 2), round(self.payload_y, 2)],
            "escort_team": self.escort_team,
            "goal_progress": round(progress, 3),
            "escort_pct": int(round(100 * progress)),
        }


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
