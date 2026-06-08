"""Morale and rout — collapsing teams flee off the field instead of stalling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class MoraleState:
    """Per-team starting strength and rout tracking."""

    start_counts: dict = field(default_factory=dict)
    routed_total: dict = field(default_factory=dict)


def init_morale_state(bots, teams):
    counts = {t: sum(1 for b in bots if b.team == t) for t in teams}
    return MoraleState(start_counts=counts, routed_total={t: 0 for t in teams})


def team_strength_frac(bots, team, state: MoraleState) -> float:
    start = state.start_counts.get(team, 0)
    if start <= 0:
        return 0.0
    alive = sum(1 for b in bots if b.alive and b.team == team)
    return alive / start


def team_is_routing(bots, team, cfg, state: MoraleState) -> bool:
    if not cfg.morale:
        return False
    return team_strength_frac(bots, team, state) <= cfg.morale_rout_frac


def rout_direction(bot, cfg) -> tuple:
    """Flee toward the nearest spawn flank."""
    if bot.team == 0:
        tx = -2.0
    elif bot.team == 1:
        tx = cfg.width + 2.0
    else:
        tx = cfg.width * 0.5
    ty = cfg.height * 0.5
    dx, dy = tx - bot.x, ty - bot.y
    d = max(math.hypot(dx, dy), 1e-9)
    return (dx / d, dy / d)


def apply_rout_steering(bots, live, desired, cfg, state: MoraleState):
    """Override desired velocity for bots on routing teams."""
    if not cfg.morale:
        return
    routing = {t for t in state.start_counts if team_is_routing(bots, t, cfg, state)}
    if not routing:
        return
    for i, b in enumerate(live):
        if b.team not in routing:
            continue
        ux, uy = rout_direction(b, cfg)
        scale = cfg.w_retreat * cfg.morale_rout_speed
        desired[i][0] = scale * ux
        desired[i][1] = scale * uy


def remove_routed_off_field(bots, cfg, state: MoraleState):
    """Eliminate bots that have left the arena while routing."""
    if not cfg.morale:
        return
    margin = cfg.morale_exit_margin
    for b in bots:
        if not b.alive:
            continue
        if team_is_routing(bots, b.team, cfg, state):
            if (b.x < -margin or b.x > cfg.width + margin
                    or b.y < -margin or b.y > cfg.height + margin):
                b.alive = False
                b.hp = 0.0
                state.routed_total[b.team] = state.routed_total.get(b.team, 0) + 1


def morale_snapshot(bots, teams, state: MoraleState):
    """Per-team strength fraction for animation / metrics."""
    return {t: round(team_strength_frac(bots, t, state), 3) for t in teams}
