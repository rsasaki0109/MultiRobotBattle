"""Fog of war — limited enemy sensing for decentralized swarm battle.

Each robot only knows about enemies within ``sense_range`` (optionally requiring
clear line of sight through terrain). Reuses the same segment geometry as fire
cover checks. Scout / sniper classes see farther than soldiers.
"""

from __future__ import annotations

import math

from .battle_teams import teams_are_enemies
from .battle_terrain import cover_along_segment_rect


def sense_range_for(bot, cfg) -> float:
    """Effective sensing radius for one bot."""
    if getattr(bot, "sense_range", None) is not None:
        return bot.sense_range
    base = cfg.sense_range if cfg.sense_range is not None else cfg.perception
    if bot.kind == "scout":
        return base * 1.55
    if bot.kind == "sniper":
        return base * 1.25
    return base


def _segment_point_distance(a0, a1, c):
    dx, dy = a1[0] - a0[0], a1[1] - a0[1]
    dd = dx * dx + dy * dy
    if dd <= 1e-15:
        return math.hypot(a0[0] - c[0], a0[1] - c[1])
    t = ((c[0] - a0[0]) * dx + (c[1] - a0[1]) * dy) / dd
    t = max(0.0, min(1.0, t))
    px, py = a0[0] + t * dx, a0[1] + t * dy
    return math.hypot(px - c[0], py - c[1])


def _circle_blocks_vision(ax, ay, tx, ty, ox, oy, radius):
    d = _segment_point_distance((ax, ay), (tx, ty), (ox, oy))
    return d <= radius


def vision_clear(ax, ay, tx, ty, cfg) -> bool:
    """Hard line-of-sight — any terrain disc or wall along the ray blocks vision."""
    if not cfg.fog_requires_los:
        return True
    for ox, oy, r in cfg.obstacles:
        if _circle_blocks_vision(ax, ay, tx, ty, ox, oy, r):
            return False
    for cx, cy, hw, hh in cfg.walls:
        if cover_along_segment_rect(ax, ay, tx, ty, cx, cy, hw, hh, 0.0) < 1.0:
            return False
    return True


def can_see_enemy(live, observer_idx, enemy_idx, cfg) -> bool:
    """Whether ``live[observer_idx]`` can sense ``live[enemy_idx]``."""
    if not cfg.fog_of_war:
        return True
    me = live[observer_idx]
    other = live[enemy_idx]
    if not teams_are_enemies(cfg.alliances, me.team, other.team):
        return False
    dist = math.hypot(me.x - other.x, me.y - other.y)
    if dist > sense_range_for(me, cfg):
        return False
    return vision_clear(me.x, me.y, other.x, other.y, cfg)


def visible_enemy_indices(live, observer_idx, cfg) -> list:
    """Live indices of enemies visible to ``live[observer_idx]``."""
    if not cfg.fog_of_war:
        me = live[observer_idx]
        return [j for j, b in enumerate(live)
                if j != observer_idx
                and teams_are_enemies(cfg.alliances, me.team, b.team)]
    out = []
    for j in range(len(live)):
        if j == observer_idx:
            continue
        if can_see_enemy(live, observer_idx, j, cfg):
            out.append(j)
    return out


def filter_decision(decision, live, observer_idx, cfg):
    """Drop a policy target that is not currently visible under fog."""
    if decision is None or not cfg.fog_of_war:
        return decision
    t = decision.target_index
    if t < 0 or t >= len(live) or not can_see_enemy(live, observer_idx, t, cfg):
        return None
    return decision


def team_visible_enemy_bot_indices(bots, cfg, team) -> list:
    """Full ``bots`` indices of enemies seen by any living ally on ``team``."""
    if not cfg.fog_of_war:
        return [i for i, b in enumerate(bots)
                if b.alive and teams_are_enemies(cfg.alliances, team, b.team)]
    live = [b for b in bots if b.alive]
    visible = set()
    for i, me in enumerate(live):
        if me.team != team:
            continue
        for j in visible_enemy_indices(live, i, cfg):
            visible.add(bots.index(live[j]))
    return sorted(visible)


def blind_advance_vector(bot, cfg) -> tuple:
    """Patrol toward expected enemy contact when no targets are visible."""
    if bot.team == 0:  # RED — press right
        tx = cfg.width * 0.78
    elif bot.team == 1:  # BLUE — press left
        tx = cfg.width * 0.22
    else:
        tx = cfg.width * 0.5
    ty = cfg.height * 0.5
    dx, dy = tx - bot.x, ty - bot.y
    d = max(math.hypot(dx, dy), 1e-9)
    boost = 1.55 if bot.kind == "scout" else 1.0
    return (dx / d * boost, dy / d * boost)
