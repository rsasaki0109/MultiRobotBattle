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

The same engine drives several **kinds** of fight (see :func:`battle_scenario`):
a two-army duel, a **free-for-all** of three or more armies
(:func:`make_free_for_all`), **unit classes** with per-bot stats — scout / soldier
/ tank / sniper (:data:`CLASSES`, :func:`make_company`), so quality-vs-quantity
and combined-arms matchups fall out — and **terrain**, a battlefield with circular
obstacles the robots flow around (``BattleConfig.obstacles``, reusing
:func:`~mrn_coord.flocking.obstacle_avoidance`).

Everything is pure Python (no numpy), deterministic given the seed, and built on
the existing flocking primitives. :func:`run_battle` plays a two-army (or N-army)
engagement; :func:`simulate` runs any prepared list of bots; both return the
per-tick history for animation plus the winning team.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .flocking import (
    _clamp_speed,
    flock_velocities,
    mutual_avoidance,
    obstacle_avoidance,
)

RED, BLUE, GREEN, YELLOW = 0, 1, 2, 3
TEAM_NAMES = {RED: "red", BLUE: "blue", GREEN: "green", YELLOW: "yellow"}

# Unit classes — per-bot stats that override the config defaults. Mixing them
# gives a rough rock-paper-scissors: tanks soak, snipers out-range, scouts swarm.
CLASSES = {
    "scout":   dict(hp=55.0,  dps=15.0, attack_range=3.0, max_speed=3.2),
    "soldier": dict(hp=100.0, dps=22.0, attack_range=3.5, max_speed=2.2),
    "tank":    dict(hp=240.0, dps=26.0, attack_range=2.6, max_speed=1.25),
    "sniper":  dict(hp=60.0,  dps=30.0, attack_range=6.5, max_speed=1.7),
}


@dataclass
class Bot:
    """One combat robot: position, velocity, team, and health.

    ``dps`` / ``attack_range`` / ``max_speed`` default to ``None``, meaning *use
    the config value*; a unit class sets them per-bot. ``kind`` is the class label
    (for colouring / sizing in an animation).
    """

    x: float
    y: float
    vx: float
    vy: float
    team: int
    hp: float
    max_hp: float
    alive: bool = True
    dps: float = None
    attack_range: float = None
    max_speed: float = None
    kind: str = ""


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
    w_obstacle: float = 1.4
    # optional terrain: circular obstacles (x, y, radius) the robots flow around
    obstacles: tuple = ()


@dataclass
class BattleResult:
    """The full history of an engagement, ready to animate."""

    frames: list = field(default_factory=list)   # per tick: list of bot snapshots
    shots: list = field(default_factory=list)     # per tick: list of firing lines
    counts: list = field(default_factory=list)    # per tick: alive count per team (in `teams` order)
    teams: list = field(default_factory=list)     # team ids present, sorted
    winner: object = None                         # winning team id / None (draw)
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


def make_unit(x, y, team, kind):
    """A single bot of unit class ``kind`` (see :data:`CLASSES`)."""
    st = CLASSES[kind]
    return Bot(x, y, 0.0, 0.0, team, st["hp"], st["hp"], dps=st["dps"],
               attack_range=st["attack_range"], max_speed=st["max_speed"],
               kind=kind)


def make_company(cfg, team, center, roster, rng, *, jitter=2.6):
    """A clustered company around ``center``; ``roster`` is ``[(kind, count), ...]``."""
    bots = []
    for kind, count in roster:
        for _ in range(count):
            x = center[0] + rng.uniform(-jitter, jitter)
            y = center[1] + rng.uniform(-jitter, jitter)
            bots.append(make_unit(x, y, team, kind))
    return bots


def make_free_for_all(n_per_team, cfg, *, seed=0, num_teams=3, kind="soldier"):
    """``num_teams`` clustered armies placed around the arena for a free-for-all."""
    rng = random.Random(seed)
    cx, cy = cfg.width / 2.0, cfg.height / 2.0
    rx, ry = cfg.width * 0.33, cfg.height * 0.36
    bots = []
    for t in range(num_teams):
        ang = -math.pi / 2.0 + 2.0 * math.pi * t / num_teams
        center = (cx + rx * math.cos(ang), cy + ry * math.sin(ang))
        for _ in range(n_per_team):
            x = center[0] + rng.uniform(-2.4, 2.4)
            y = center[1] + rng.uniform(-2.4, 2.4)
            bots.append(make_unit(x, y, t, kind))
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
    for team in sorted({b.team for b in live}):
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

    # 2) pursue / retreat from the nearest enemy (any other team), 3) fire if in
    #    range — using each bot's own class stats where set
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
        b_range = b.attack_range if b.attack_range is not None else cfg.attack_range
        if bd <= b_range:
            damage[best] += (b.dps if b.dps is not None else cfg.dps) * cfg.dt
            shots.append((b.x, b.y, e.x, e.y, b.team))

    # 4) keep spacing — mutual repulsion across everyone (reuse flocking)
    sep = mutual_avoidance([(b.x, b.y) for b in live], radius=cfg.sep_radius,
                           strength=cfg.sep_strength)
    for i in range(n):
        desired[i][0] += cfg.w_sep * sep[i][0]
        desired[i][1] += cfg.w_sep * sep[i][1]

    # 5) optional terrain: flow around circular obstacles (reuse flocking)
    if cfg.obstacles:
        obs = obstacle_avoidance([(b.x, b.y) for b in live], list(cfg.obstacles))
        for i in range(n):
            desired[i][0] += cfg.w_obstacle * obs[i][0]
            desired[i][1] += cfg.w_obstacle * obs[i][1]

    # integrate motion (snapshot-consistent), each bot at its own top speed
    for i in range(n):
        b = live[i]
        mspeed = b.max_speed if b.max_speed is not None else cfg.max_speed
        vx, vy = _wall_turn(b.x, b.y, desired[i][0], desired[i][1],
                            cfg.width, cfg.height)
        vx, vy = _clamp_speed(vx, vy, mspeed)
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


def _counts(bots, teams):
    return tuple(sum(1 for b in bots if b.alive and b.team == t) for t in teams)


def _snapshot(bots):
    return [(b.x, b.y, b.team, b.hp / b.max_hp if b.max_hp else 0.0, b.alive,
             b.kind) for b in bots]


def simulate(bots, cfg=None, *, max_ticks=800):
    """Play a whole engagement from a prepared list of bots (any teams/classes).

    Records the per-tick snapshots and firing lines, stops as soon as at most one
    team has living robots (or ``max_ticks``), and reports the winning team id
    (or ``None`` for a draw). Deterministic given the inputs.
    """
    cfg = cfg or BattleConfig()
    teams = sorted({b.team for b in bots})
    result = BattleResult(teams=teams)
    for _ in range(max_ticks):
        result.frames.append(_snapshot(bots))
        c = _counts(bots, teams)
        result.counts.append(c)
        if sum(1 for n in c if n > 0) <= 1:   # one (or zero) team left standing
            result.shots.append([])
            break
        result.shots.append(battle_step(bots, cfg))
    else:
        # ran out of ticks without a wipeout: record the final state
        result.frames.append(_snapshot(bots))
        result.counts.append(_counts(bots, teams))
        result.shots.append([])

    c = _counts(bots, teams)
    result.survivors = {t: n for t, n in zip(teams, c)}
    result.ticks = len(result.frames) - 1
    standing = [t for t, n in zip(teams, c) if n > 0]
    result.winner = standing[0] if len(standing) == 1 else None
    return result


def run_battle(n_per_team=14, cfg=None, *, seed=0, max_ticks=800, num_teams=2):
    """Play a whole engagement; return a :class:`BattleResult` for animation.

    With ``num_teams == 2`` (default) it is the classic two-army duel (red left,
    blue right); with more teams it sets up a free-for-all placed around the
    arena. Deterministic given ``(n_per_team, cfg, seed, num_teams)``.
    """
    cfg = cfg or BattleConfig()
    if num_teams == 2:
        bots = make_armies(n_per_team, cfg, seed=seed)
    else:
        bots = make_free_for_all(n_per_team, cfg, seed=seed, num_teams=num_teams)
    return simulate(bots, cfg, max_ticks=max_ticks)


# A handful of distinct showcase battles, each with a baked seed so the result is
# reproducible. Each returns ``(bots, cfg, title)``; render them with
# ``scripts/make_battle_gallery_gif.py``.
SCENARIO_NAMES = ("duel", "free_for_all", "quality_vs_quantity", "chokepoint")


def battle_scenario(name):
    """Return ``(bots, cfg, title)`` for a named showcase battle (deterministic)."""
    if name == "duel":
        cfg = BattleConfig()
        return (make_armies(14, cfg, seed=14), cfg, "Duel — 14 vs 14")
    if name == "free_for_all":
        cfg = BattleConfig()
        bots = make_free_for_all(10, cfg, seed=6, num_teams=3)
        return (bots, cfg, "Free-for-all — three armies")
    if name == "quality_vs_quantity":
        cfg = BattleConfig()
        rng = random.Random(0)
        red = make_company(cfg, RED, (cfg.width * 0.18, cfg.height * 0.5),
                           [("tank", 5)], rng)
        blue = make_company(cfg, BLUE, (cfg.width * 0.82, cfg.height * 0.5),
                            [("scout", 16)], rng)
        return (red + blue, cfg, "Quality vs quantity — 5 tanks vs 16 scouts")
    if name == "chokepoint":
        obstacles = ((20.0, 4.5, 2.6), (20.0, 12.0, 2.6), (20.0, 19.5, 2.6))
        cfg = BattleConfig(obstacles=obstacles)
        rng = random.Random(7)
        red = make_company(cfg, RED, (cfg.width * 0.13, cfg.height * 0.5),
                           [("soldier", 13)], rng, jitter=3.2)
        blue = make_company(cfg, BLUE, (cfg.width * 0.87, cfg.height * 0.5),
                            [("soldier", 13)], rng, jitter=3.2)
        return (red + blue, cfg, "Chokepoint — terrain splits the field")
    raise KeyError(name)
