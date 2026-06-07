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
- **line of sight & cover** (optional) — circular obstacles (and optionally other
  robot bodies) block or attenuate fire along the segment to the target; partial
  cover scales damage down instead of a hard on/off hitscan;
- **fall back when wounded** (optional) — with ``retreat_frac > 0`` a robot below
  that fraction of max health flees its nearest enemy instead of pressing in. It
  is **off by default** (``retreat_frac = 0``): a default battle is fought to the
  death so it always reaches a decisive result; enabling it trades decisiveness
  for skirmishing, since the last wounded survivors may flee indefinitely.
- **count-aware tactics** (optional) — :mod:`mrn_coord.battle_policy` builds
  TeamHOI-style teammate/enemy tokens and a decentralized policy that adapts
  pursue / flock / retreat to ally-vs-enemy counts, focus-fires wounded targets,
  and kites snipers (``BattleConfig.tactics = "count_aware"``).
- **formations** (optional) — :mod:`mrn_coord.battle_formations` drives
  line / wedge / screen / square shapes via displacement consensus
  (``BattleConfig.formation``); ``auto`` picks from force ratio.
- **planned maneuver** (optional) — :mod:`mrn_coord.battle_maneuver` swaps the
  movement layer for grid A* / prioritized / CBS / LaCAM-PIBT
  (``BattleConfig.maneuver``) while tactics still pick targets.

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

from .battle_assignment import apply_assignments, AssignmentState
from .spatial_hash import SpatialHash
from .battle_formations import formation_commands
from .battle_maneuver import ManeuverState, maneuver_direction, maneuver_for_team, replan_paths
from .battle_policy.count_aware import policy_for_name
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
    # line of sight / cover (terrain + optional body blocking)
    require_los: bool = True
    body_blocks_fire: bool = False   # other robots attenuate fire along the ray
    body_radius: float = 0.55        # effective disc radius for body LoS
    cover_margin: float = 0.75       # partial-cover band beyond obstacle/body surface
    # tactics — ``nearest`` (default), ``count_aware``, ``transformer``, or per-team
    tactics: str = "nearest"
    tactics_by_team: dict = None   # e.g. {RED: "count_aware", BLUE: "nearest"}
    # formations — ``none`` (default), ``auto``, ``line``, ``wedge``, ``screen``, ``square``
    formation: str = "none"
    formation_by_team: dict = None
    formation_spacing: float = 2.0
    formation_gain: float = 0.45
    w_formation: float = 0.9
    # planned maneuver — ``greedy`` (default) or ``astar`` / ``prioritized`` / ``cbs`` / ``pibt``
    maneuver: str = "greedy"
    maneuver_by_team: dict = None
    maneuver_cell_size: float = 1.0
    maneuver_replan_ticks: int = 20
    maneuver_lookahead: float = 1.4
    maneuver_goal_tol: float = 1.2
    # target assignment — ``none``, ``hungarian``, or ``cbs_ta`` (grid CBS-TA)
    assignment: str = "none"
    assignment_by_team: dict = None
    assignment_replan_ticks: int = 15
    assignment_cell_size: float = 2.0
    # steering weights
    w_flock: float = 1.0
    w_pursue: float = 1.8
    w_sep: float = 1.4
    w_retreat: float = 2.6
    w_obstacle: float = 1.4
    # scale-out — spatial hash when alive count >= spatial_min_bots
    spatial_min_bots: int = 36
    spatial_cell: float = 4.0
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


def make_grand_army(cfg, team, front_center, *, rows, cols, kind="soldier",
                    spacing=2.1, rng=None, face_right=True):
    """Rectangular block deployment — Total War / Kingdom-style battle lines.

    ``front_center`` is the midpoint of the front rank; ``face_right`` means +x
    is forward (red attacking right, blue attacking left when placed on flanks).
    """
    rng = rng or random.Random(0)
    sign = 1.0 if face_right else -1.0
    bots = []
    for r in range(rows):
        for c in range(cols):
            lateral = (c - (cols - 1) / 2.0) * spacing
            back = -r * spacing * sign
            x = front_center[0] + back + rng.uniform(-0.15, 0.15)
            y = front_center[1] + lateral + rng.uniform(-0.15, 0.15)
            bots.append(make_unit(x, y, team, kind))
    return bots


def kingdom_config(**overrides):
    """Wide arena defaults for hundred-bot clashes."""
    base = dict(
        width=100.0,
        height=56.0,
        perception=5.0,
        separation=1.4,
        sep_radius=1.8,
        formation="line",
        tactics="count_aware:aggressive",
        assignment="none",
        maneuver="greedy",
        dps=28.0,
        spatial_min_bots=30,
        spatial_cell=4.5,
    )
    base.update(overrides)
    return BattleConfig(**base)


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


def _segment_point_distance(a0, a1, c):
    """Minimum distance from the segment ``a0->a1`` to the point ``c``."""
    dx, dy = a1[0] - a0[0], a1[1] - a0[1]
    dd = dx * dx + dy * dy
    if dd <= 1e-15:
        return math.hypot(a0[0] - c[0], a0[1] - c[1])
    t = ((c[0] - a0[0]) * dx + (c[1] - a0[1]) * dy) / dd
    t = max(0.0, min(1.0, t))
    px, py = a0[0] + t * dx, a0[1] + t * dy
    return math.hypot(px - c[0], py - c[1])


def _cover_along_segment(ax, ay, tx, ty, ox, oy, radius, cover_margin):
    """Cover factor for one circular blocker (0 = blocked, 1 = clear).

    A segment that passes through the disc is fully blocked; one that grazes
    within ``cover_margin`` of the surface scales damage linearly.
    """
    d = _segment_point_distance((ax, ay), (tx, ty), (ox, oy))
    if d >= radius + cover_margin:
        return 1.0
    if d <= radius:
        return 0.0
    return (d - radius) / cover_margin


def _fire_cover(ax, ay, tx, ty, cfg, live, skip):
    """Combined cover factor along the shot segment (1.0 = clear line of fire)."""
    if not cfg.require_los:
        return 1.0
    factor = 1.0
    for (ox, oy, r) in cfg.obstacles:
        factor = min(factor, _cover_along_segment(ax, ay, tx, ty, ox, oy, r,
                                                  cfg.cover_margin))
    if cfg.body_blocks_fire:
        for j, other in enumerate(live):
            if j in skip:
                continue
            factor = min(factor, _cover_along_segment(ax, ay, tx, ty,
                                                      other.x, other.y,
                                                      cfg.body_radius,
                                                      cfg.cover_margin))
    return factor


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


def battle_step(bots, cfg, *, maneuver_state=None, assignment_state=None, tick=0):
    """Advance the battle one tick (in place); return this tick's firing lines.

    Steering and damage are both computed from the *same* snapshot of living
    bots, so the update is order-independent and deterministic.
    """
    if maneuver_state is None:
        maneuver_state = ManeuverState()
    if assignment_state is None:
        assignment_state = AssignmentState()
    live = [b for b in bots if b.alive]
    n = len(live)
    spatial = None
    if n >= cfg.spatial_min_bots:
        positions = [(b.x, b.y) for b in live]
        spatial = SpatialHash(cfg.spatial_cell)
        spatial.build(positions)
    desired = [[0.0, 0.0] for _ in range(n)]
    damage = [0.0] * n
    shots = []
    policy = policy_for_name(cfg.tactics)
    policy_cache = {cfg.tactics: policy}

    def _policy_for(team):
        name = (cfg.tactics_by_team or {}).get(team, cfg.tactics)
        if name not in policy_cache:
            policy_cache[name] = policy_for_name(name)
        return policy_cache[name]

    decisions = [_policy_for(live[i].team).decide(live, i, cfg, spatial=spatial)
                 for i in range(n)]
    decisions = apply_assignments(decisions, live, cfg,
                                  assignment_state=assignment_state, tick=tick)
    replan_paths(bots, live, decisions, cfg, maneuver_state, tick)

    def _formation_for(team):
        return (cfg.formation_by_team or {}).get(team, cfg.formation)

    # 1) flock with living teammates (reuse the Boids step per team)
    for team in sorted({b.team for b in live}):
        idx = [i for i in range(n) if live[i].team == team]
        if not idx:
            continue
        pos = [(live[i].x, live[i].y) for i in idx]
        vel = [(live[i].vx, live[i].vy) for i in idx]
        team_spatial = None
        if spatial is not None and len(idx) >= cfg.spatial_min_bots:
            team_spatial = SpatialHash(cfg.spatial_cell)
            team_spatial.build(pos)
        fv = flock_velocities(pos, vel, perception=cfg.perception,
                              separation=cfg.separation, max_speed=cfg.max_speed,
                              spatial=team_spatial)
        for k, i in enumerate(idx):
            flock_scale = decisions[i].flock_scale if decisions[i] else 1.0
            desired[i][0] += cfg.w_flock * flock_scale * fv[k][0]
            desired[i][1] += cfg.w_flock * flock_scale * fv[k][1]

    # 1b) optional team formations (displacement consensus toward shape)
    for team in sorted({b.team for b in live}):
        mode = _formation_for(team)
        if mode in (None, "", "none"):
            continue
        idx = [i for i in range(n) if live[i].team == team]
        fcmds = formation_commands(bots, live, idx, mode,
                                   spacing=cfg.formation_spacing,
                                   gain=cfg.formation_gain)
        for i, (ux, uy) in fcmds.items():
            desired[i][0] += cfg.w_formation * ux
            desired[i][1] += cfg.w_formation * uy

    # 2) pursue / retreat toward the policy target, 3) fire if in range
    for i in range(n):
        b = live[i]
        decision = decisions[i]
        if decision is None:
            continue
        best = decision.target_index
        if best < 0 or best >= n:
            continue
        e = live[best]
        if e.team == b.team:
            continue
        bd = math.hypot(b.x - e.x, b.y - e.y)
        d = max(bd, 1e-9)
        ux, uy = (e.x - b.x) / d, (e.y - b.y) / d
        mode = maneuver_for_team(b.team, cfg)
        if mode not in (None, "", "greedy"):
            path = maneuver_state.paths.get(bots.index(b))
            directed = maneuver_direction(b, path, e, cfg)
            if directed is not None:
                ux, uy = directed
        wounded = b.hp < cfg.retreat_frac * b.max_hp
        if wounded or decision.kite:
            scale = cfg.w_retreat * decision.retreat_scale
            desired[i][0] -= scale * ux
            desired[i][1] -= scale * uy
        else:
            scale = cfg.w_pursue * decision.pursue_scale
            desired[i][0] += scale * ux
            desired[i][1] += scale * uy
        b_range = b.attack_range if b.attack_range is not None else cfg.attack_range
        if bd <= b_range:
            cover = _fire_cover(b.x, b.y, e.x, e.y, cfg, live, {i, best})
            if cover > 0.0:
                dps = b.dps if b.dps is not None else cfg.dps
                damage[best] += dps * cfg.dt * cover
                shots.append((b.x, b.y, e.x, e.y, b.team))

    # 4) keep spacing — mutual repulsion across everyone (reuse flocking)
    sep = mutual_avoidance([(b.x, b.y) for b in live], radius=cfg.sep_radius,
                           strength=cfg.sep_strength, spatial=spatial)
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


def simulate(bots, cfg=None, *, max_ticks=800, frame_stride=1):
    """Play a whole engagement from a prepared list of bots (any teams/classes).

    Records the per-tick snapshots and firing lines, stops as soon as at most one
    team has living robots (or ``max_ticks``), and reports the winning team id
    (or ``None`` for a draw). Deterministic given the inputs.

    ``frame_stride`` subsamples recorded frames (and shots) for large battles.
    """
    cfg = cfg or BattleConfig()
    frame_stride = max(1, frame_stride)
    teams = sorted({b.team for b in bots})
    result = BattleResult(teams=teams)
    maneuver_state = ManeuverState()
    assignment_state = AssignmentState()
    for tick in range(max_ticks):
        record = (tick % frame_stride == 0)
        if record:
            result.frames.append(_snapshot(bots))
            result.counts.append(_counts(bots, teams))
        c = _counts(bots, teams)
        if sum(1 for n in c if n > 0) <= 1:   # one (or zero) team left standing
            if record:
                result.shots.append([])
            break
        shots = battle_step(bots, cfg, maneuver_state=maneuver_state,
                            assignment_state=assignment_state, tick=tick)
        if record:
            result.shots.append(shots)
    else:
        if tick % frame_stride == 0:
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
SCENARIO_NAMES = ("duel", "free_for_all", "quality_vs_quantity", "chokepoint",
                  "maneuver_duel", "mapf_stack_duel", "kingdom")


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
        cfg = BattleConfig(obstacles=obstacles, formation="wedge",
                           tactics="count_aware")
        rng = random.Random(7)
        red = make_company(cfg, RED, (cfg.width * 0.13, cfg.height * 0.5),
                           [("soldier", 13)], rng, jitter=3.2)
        blue = make_company(cfg, BLUE, (cfg.width * 0.87, cfg.height * 0.5),
                            [("soldier", 13)], rng, jitter=3.2)
        return (red + blue, cfg, "Chokepoint — wedge through terrain")
    if name == "maneuver_duel":
        obstacles = ((20.0, 4.5, 2.6), (20.0, 12.0, 2.6), (20.0, 19.5, 2.6))
        cfg = BattleConfig(
            obstacles=obstacles,
            tactics="count_aware",
            assignment="hungarian",
            formation="wedge",
            maneuver="greedy",
            maneuver_by_team={RED: "prioritized", BLUE: "greedy"},
            maneuver_replan_ticks=15,
        )
        rng = random.Random(11)
        red = make_company(cfg, RED, (cfg.width * 0.13, cfg.height * 0.5),
                           [("soldier", 10)], rng, jitter=2.8)
        blue = make_company(cfg, BLUE, (cfg.width * 0.87, cfg.height * 0.5),
                            [("soldier", 10)], rng, jitter=2.8)
        return (red + blue, cfg,
                "Maneuver duel — planned red vs greedy blue")
    if name == "mapf_stack_duel":
        obstacles = ((20.0, 4.5, 2.6), (20.0, 12.0, 2.6), (20.0, 19.5, 2.6))
        cfg = BattleConfig(
            obstacles=obstacles,
            tactics="count_aware",
            formation="wedge",
            assignment="none",
            assignment_by_team={RED: "cbs_ta"},
            maneuver="greedy",
            maneuver_by_team={RED: "prioritized", BLUE: "greedy"},
            maneuver_replan_ticks=15,
            assignment_replan_ticks=15,
        )
        rng = random.Random(11)
        red = make_company(cfg, RED, (cfg.width * 0.13, cfg.height * 0.5),
                           [("soldier", 10)], rng, jitter=2.8)
        blue = make_company(cfg, BLUE, (cfg.width * 0.87, cfg.height * 0.5),
                            [("soldier", 10)], rng, jitter=2.8)
        return (red + blue, cfg,
                "MAPF stack — CBS-TA assignment + planned maneuver")
    if name == "kingdom":
        cfg = kingdom_config(formation_by_team={RED: "line", BLUE: "line"})
        rng = random.Random(42)
        red = make_grand_army(cfg, RED, (cfg.width * 0.28, cfg.height * 0.5),
                              rows=8, cols=10, rng=rng, face_right=True)
        blue = make_grand_army(cfg, BLUE, (cfg.width * 0.72, cfg.height * 0.5),
                               rows=8, cols=10, rng=random.Random(43),
                               face_right=False)
        return (red + blue, cfg, "Kingdom clash — 80 vs 80 battle lines")
    raise KeyError(name)
