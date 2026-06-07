"""Swarm battle — two flocking armies fight to the last robot.

This is the swarm counterpart's natural endgame: take the decentralized Boids
flocking of :mod:`mrn_coord.flocking` and split the swarm into **two teams** that
seek and fight each other. Every robot steers from only local information — there
is no central commander — yet coherent battlefield behaviour emerges:

- **flock with living teammates** (separation / alignment / cohesion) — the army
  stays together instead of scattering;
- **advance to contact** — each robot pursues its nearest living enemy;
- **keep spacing** — mutual repulsion so robots do not pile up;
- **fire when in range** — a robot deals continuous damage to its nearest enemy
  within ``attack_range``; health falls, and at zero the robot is eliminated;
- **fall back when wounded** (optional) — with ``retreat_frac > 0`` a robot below
  that fraction of max health flees its nearest enemy instead of pressing in. It
  is **off by default** (``retreat_frac = 0``): a default battle is fought to the
  death so it always reaches a decisive result; enabling it trades decisiveness
  for skirmishing, since the last wounded survivors may flee indefinitely.

The one tactically interesting consequence is **focus fire for free**: damage is
per-attacker, so a robot caught by three enemies at once takes triple damage.
The team that keeps formation and concentrates locally therefore wears the other
down — an emergent property of the local rules, not anything scripted.

Everything is pure Python (no numpy), deterministic given the seed, and built on
the existing flocking primitives. :func:`run_battle` plays a whole engagement and
returns the per-tick history for animation plus the winner.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .flocking import _clamp_speed, flock_velocities, mutual_avoidance

RED, BLUE = 0, 1
TEAM_NAMES = {RED: "red", BLUE: "blue"}


@dataclass
class Bot:
    """One combat robot: position, velocity, team, and health."""

    x: float
    y: float
    vx: float
    vy: float
    team: int
    hp: float
    max_hp: float
    alive: bool = True


@dataclass
class BattleConfig:
    """Arena and behaviour parameters for a swarm battle."""

    width: float = 40.0
    height: float = 24.0
    # flocking / movement
    perception: float = 6.0
    separation: float = 1.6
    sep_radius: float = 2.0
    sep_strength: float = 1.4
    max_speed: float = 2.2
    dt: float = 0.1
    # combat
    attack_range: float = 3.5
    dps: float = 22.0
    max_hp: float = 100.0
    retreat_frac: float = 0.0    # 0 = fight to the death (decisive); >0 = skirmish
    # steering weights
    w_flock: float = 1.0
    w_pursue: float = 1.8
    w_sep: float = 1.4
    w_retreat: float = 2.6


@dataclass
class BattleResult:
    """The full history of an engagement, ready to animate."""

    frames: list = field(default_factory=list)   # per tick: list of bot snapshots
    shots: list = field(default_factory=list)     # per tick: list of firing lines
    counts: list = field(default_factory=list)    # per tick: (red_alive, blue_alive)
    winner: object = None                         # RED / BLUE / None (draw)
    ticks: int = 0
    survivors: dict = field(default_factory=dict)


def make_armies(n_per_team, cfg, *, seed=0):
    """Two clustered armies facing off: red on the left, blue on the right."""
    rng = random.Random(seed)
    bots = []
    for team, cx in ((RED, cfg.width * 0.17), (BLUE, cfg.width * 0.83)):
        for _ in range(n_per_team):
            x = cx + rng.uniform(-3.0, 3.0)
            y = cfg.height * 0.5 + rng.uniform(-7.0, 7.0)
            bots.append(Bot(x, y, 0.0, 0.0, team, cfg.max_hp, cfg.max_hp))
    return bots


def _wall_turn(x, y, vx, vy, width, height, margin=2.0, push=2.0):
    if x < margin:
        vx += push
    elif x > width - margin:
        vx -= push
    if y < margin:
        vy += push
    elif y > height - margin:
        vy -= push
    return vx, vy


def battle_step(bots, cfg):
    """Advance the battle one tick (in place); return this tick's firing lines.

    Steering and damage are both computed from the *same* snapshot of living
    bots, so the update is order-independent and deterministic.
    """
    live = [b for b in bots if b.alive]
    n = len(live)
    desired = [[0.0, 0.0] for _ in range(n)]
    damage = [0.0] * n
    shots = []

    # 1) flock with living teammates (reuse the Boids step per team)
    for team in (RED, BLUE):
        idx = [i for i in range(n) if live[i].team == team]
        if not idx:
            continue
        pos = [(live[i].x, live[i].y) for i in idx]
        vel = [(live[i].vx, live[i].vy) for i in idx]
        fv = flock_velocities(pos, vel, perception=cfg.perception,
                              separation=cfg.separation, max_speed=cfg.max_speed)
        for k, i in enumerate(idx):
            desired[i][0] += cfg.w_flock * fv[k][0]
            desired[i][1] += cfg.w_flock * fv[k][1]

    # 2) pursue / retreat from the nearest enemy, and 3) fire if in range
    for i in range(n):
        b = live[i]
        best, bd = -1, float("inf")
        for j in range(n):
            if live[j].team == b.team:
                continue
            d = math.hypot(b.x - live[j].x, b.y - live[j].y)
            if d < bd:
                bd, best = d, j
        if best < 0:
            continue
        e = live[best]
        d = max(bd, 1e-9)
        ux, uy = (e.x - b.x) / d, (e.y - b.y) / d
        if b.hp < cfg.retreat_frac * b.max_hp:
            desired[i][0] -= cfg.w_retreat * ux
            desired[i][1] -= cfg.w_retreat * uy
        else:
            desired[i][0] += cfg.w_pursue * ux
            desired[i][1] += cfg.w_pursue * uy
        if bd <= cfg.attack_range:
            damage[best] += cfg.dps * cfg.dt
            shots.append((b.x, b.y, e.x, e.y, b.team))

    # 4) keep spacing — mutual repulsion across everyone (reuse flocking)
    sep = mutual_avoidance([(b.x, b.y) for b in live], radius=cfg.sep_radius,
                           strength=cfg.sep_strength)
    for i in range(n):
        desired[i][0] += cfg.w_sep * sep[i][0]
        desired[i][1] += cfg.w_sep * sep[i][1]

    # integrate motion (snapshot-consistent)
    for i in range(n):
        b = live[i]
        vx, vy = _wall_turn(b.x, b.y, desired[i][0], desired[i][1],
                            cfg.width, cfg.height)
        vx, vy = _clamp_speed(vx, vy, cfg.max_speed)
        b.vx, b.vy = vx, vy
        b.x += vx * cfg.dt
        b.y += vy * cfg.dt

    # apply damage and resolve eliminations (simultaneous)
    for i in range(n):
        if damage[i] <= 0.0:
            continue
        b = live[i]
        b.hp -= damage[i]
        if b.hp <= 0.0:
            b.hp = 0.0
            b.alive = False
    return shots


def _alive_counts(bots):
    red = sum(1 for b in bots if b.alive and b.team == RED)
    blue = sum(1 for b in bots if b.alive and b.team == BLUE)
    return red, blue


def _snapshot(bots):
    return [(b.x, b.y, b.team, b.hp / b.max_hp if b.max_hp else 0.0, b.alive)
            for b in bots]


def run_battle(n_per_team=14, cfg=None, *, seed=0, max_ticks=800):
    """Play a whole engagement; return a :class:`BattleResult` for animation.

    Records the per-tick bot snapshots and firing lines, stops as soon as one
    team is wiped out (or ``max_ticks``), and reports the winner. Deterministic
    given ``(n_per_team, cfg, seed)``.
    """
    cfg = cfg or BattleConfig()
    bots = make_armies(n_per_team, cfg, seed=seed)
    result = BattleResult()
    for _ in range(max_ticks):
        result.frames.append(_snapshot(bots))
        red, blue = _alive_counts(bots)
        result.counts.append((red, blue))
        if red == 0 or blue == 0:
            result.shots.append([])
            break
        result.shots.append(battle_step(bots, cfg))
    else:
        # ran out of ticks without a wipeout: record the final state
        result.frames.append(_snapshot(bots))
        result.counts.append(_alive_counts(bots))
        result.shots.append([])

    red, blue = _alive_counts(bots)
    result.survivors = {RED: red, BLUE: blue}
    result.ticks = len(result.frames) - 1
    if blue == 0 and red > 0:
        result.winner = RED
    elif red == 0 and blue > 0:
        result.winner = BLUE
    else:
        result.winner = None
    return result
