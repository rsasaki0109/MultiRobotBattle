"""Local battle observations as TeamHOI-style agent tokens.

Each living bot encodes nearby teammates and enemies as fixed-layout token lists
(relative position and velocity in the observer's heading frame, plus health).
Global alive counts are included so a policy can react to *team size* without a
central commander.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentToken:
    """One neighbour expressed in the observer's local frame."""

    rel_x: float
    rel_y: float
    rel_vx: float
    rel_vy: float
    hp_frac: float
    kind: str
    dist: float
    live_index: int


@dataclass
class BattleObservation:
    """What one bot can see when choosing tactics (decentralized, same layout for all)."""

    live_index: int
    team: int
    hp_frac: float
    kind: str
    heading: float
    n_allies: int
    n_enemies: int
    n_local_allies: int
    n_local_enemies: int
    teammate_tokens: list = field(default_factory=list)
    enemy_tokens: list = field(default_factory=list)


def _heading_from_bot(bot, enemies):
    """Observer heading: velocity, else toward the nearest enemy, else +x."""
    speed = math.hypot(bot.vx, bot.vy)
    if speed > 0.05:
        return math.atan2(bot.vy, bot.vx)
    if enemies:
        ex, ey = enemies[0][1].x, enemies[0][1].y
        return math.atan2(ey - bot.y, ex - bot.x)
    return 0.0


def _to_local(dx, dy, heading):
    c, s = math.cos(heading), math.sin(heading)
    return dx * c + dy * s, -dx * s + dy * c


def _make_token(observer, other, heading, live_index):
    dx, dy = other.x - observer.x, other.y - observer.y
    rx, ry = _to_local(dx, dy, heading)
    rvx, rvy = _to_local(other.vx - observer.vx, other.vy - observer.vy, heading)
    hp_frac = other.hp / other.max_hp if other.max_hp else 0.0
    return AgentToken(rx, ry, rvx, rvy, hp_frac, other.kind or "",
                      math.hypot(dx, dy), live_index)


def build_observation(live, index, *, perception=6.0, max_tokens=8, spatial=None):
    """Build a :class:`BattleObservation` for ``live[index]``."""
    me = live[index]
    positions = [(b.x, b.y) for b in live]
    enemies = []
    allies = []
    if spatial is not None:
        cand = spatial.query_disk(me.x, me.y, perception * 2.5, positions)
        for j in cand:
            if j == index:
                continue
            other = live[j]
            d = math.hypot(me.x - other.x, me.y - other.y)
            if other.team == me.team:
                allies.append((d, other, j))
            else:
                enemies.append((d, other, j))
    else:
        for j, other in enumerate(live):
            if j == index:
                continue
            d = math.hypot(me.x - other.x, me.y - other.y)
            if other.team == me.team:
                allies.append((d, other, j))
            else:
                enemies.append((d, other, j))

    allies.sort(key=lambda t: t[0])
    enemies.sort(key=lambda t: t[0])
    heading = _heading_from_bot(me, enemies)

    n_allies = sum(1 for b in live if b.team == me.team)
    n_enemies = len(live) - n_allies
    n_local_allies = sum(1 for d, _, _ in allies if d <= perception)
    n_local_enemies = sum(1 for d, _, _ in enemies if d <= perception)

    teammate_tokens = [_make_token(me, o, heading, j)
                       for d, o, j in allies[:max_tokens]]
    enemy_tokens = [_make_token(me, o, heading, j)
                    for d, o, j in enemies[:max_tokens]]

    return BattleObservation(
        live_index=index,
        team=me.team,
        hp_frac=me.hp / me.max_hp if me.max_hp else 0.0,
        kind=me.kind or "",
        heading=heading,
        n_allies=n_allies,
        n_enemies=n_enemies,
        n_local_allies=n_local_allies,
        n_local_enemies=n_local_enemies,
        teammate_tokens=teammate_tokens,
        enemy_tokens=enemy_tokens,
    )
