"""Discrete projectiles for swarm battle — travel time, range falloff, misses."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Projectile:
    """In-flight round fired from a snapshot aim point."""

    x: float
    y: float
    vx: float
    vy: float
    damage: float
    team: int
    target_bot_idx: int
    cover: float = 1.0
    age: float = 0.0
    ttl: float = 0.5
    hit_radius: float = 0.42


@dataclass
class ProjectileState:
    """Active rounds and per-shooter reload timers (keyed by ``bots`` index)."""

    projectiles: list = field(default_factory=list)
    cooldowns: dict = field(default_factory=dict)


def shot_accuracy(dist: float, max_range: float, *, acc_min: float, acc_max: float) -> float:
    """Hit probability falls off linearly from point-blank to max range."""
    if max_range <= 1e-9:
        return acc_max
    t = min(1.0, max(0.0, dist / max_range))
    return acc_max - (acc_max - acc_min) * t


def _aim_point(ax, ay, tx, ty, dist, accuracy, rng, miss_spread):
    """Deterministic aim with lateral miss offset when the roll fails."""
    dx, dy = tx - ax, ty - ay
    d = max(dist, 1e-9)
    ux, uy = dx / d, dy / d
    if rng.random() <= accuracy:
        return tx, ty
    sign = -1.0 if rng.random() < 0.5 else 1.0
    spread = miss_spread * (1.0 - accuracy) * (0.55 + 0.45 * rng.random())
    return tx + (-uy) * spread * sign, ty + ux * spread * sign


def spawn_projectile_from_bot(ax, ay, target, dist, b_range, cfg, *, team,
                              target_bot_idx, dps, cover, tick,
                              shooter_bot_idx):
    """Fire one round using per-bot range for accuracy falloff."""
    rng = random.Random(tick * 7919 + shooter_bot_idx * 104729)
    acc = shot_accuracy(dist, b_range, acc_min=cfg.accuracy_min,
                        acc_max=cfg.accuracy_max)
    aim_x, aim_y = _aim_point(ax, ay, target[0], target[1], dist, acc, rng,
                              cfg.miss_spread)
    ddx, ddy = aim_x - ax, aim_y - ay
    dd = math.hypot(ddx, ddy)
    if dd <= 1e-9:
        ddx, ddy, dd = 1.0, 0.0, 1.0
    speed = cfg.projectile_speed
    damage = dps * cfg.fire_interval
    return Projectile(
        x=ax, y=ay,
        vx=ddx / dd * speed, vy=ddy / dd * speed,
        damage=damage, team=team, target_bot_idx=target_bot_idx,
        cover=cover, ttl=cfg.projectile_ttl,
        hit_radius=cfg.projectile_hit_radius,
    ), (ax, ay, aim_x, aim_y, team)


def advance_projectiles(state: ProjectileState, bots, cfg, *, dt):
    """Move active rounds, apply hits, return damage per living-bot index."""
    live = [b for b in bots if b.alive]
    live_idx = {id(b): i for i, b in enumerate(live)}
    damage = [0.0] * len(live)
    remaining = []
    snapshots = []

    for p in state.projectiles:
        p.age += dt
        if p.age > p.ttl:
            continue
        target = bots[p.target_bot_idx] if p.target_bot_idx < len(bots) else None
        if target is not None and target.alive:
            speed = math.hypot(p.vx, p.vy) or cfg.projectile_speed
            dx, dy = target.x - p.x, target.y - p.y
            dd = math.hypot(dx, dy)
            if dd > 1e-9:
                p.vx = dx / dd * speed
                p.vy = dy / dd * speed
        p.x += p.vx * dt
        p.y += p.vy * dt
        snapshots.append((p.x, p.y, p.team))

        hit = False
        if target is not None and target.alive:
            d = math.hypot(p.x - target.x, p.y - target.y)
            if d <= p.hit_radius:
                li = live_idx.get(id(target))
                if li is not None:
                    damage[li] += p.damage * p.cover
                hit = True
        if not hit and p.age < p.ttl:
            remaining.append(p)

    state.projectiles = remaining
    return damage, snapshots


def tick_cooldowns(state: ProjectileState, bots, cfg):
    """Decay reload timers for every bot slot."""
    dt = cfg.dt
    for i in range(len(bots)):
        cd = state.cooldowns.get(i, 0.0) - dt
        state.cooldowns[i] = max(0.0, cd)


def can_fire(state: ProjectileState, bot_idx):
    return state.cooldowns.get(bot_idx, 0.0) <= 0.0


def mark_fired(state: ProjectileState, bot_idx, cfg):
    state.cooldowns[bot_idx] = cfg.fire_interval
