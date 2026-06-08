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
from .battle_objectives import (
    CtfTracker, ObjectiveTracker, ctf_render_zones, objective_zone,
    winner_from_objective, zone_leader,
)
from .battle_teams import alliance_of, teams_are_enemies
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
    # alliances — map team id -> alliance id; allied teams do not fire on each other
    alliances: dict = None
    # objectives — ``annihilation`` (default), ``hill``, ``domination``, ``ctf``
    objective: str = "annihilation"
    objective_center: tuple = None
    objective_radius: float = 5.0
    objective_hold_ticks: int = 200
    base_radius: float = None   # CTF home-base radius (defaults to objective_radius)


@dataclass
class BattleResult:
    """The full history of an engagement, ready to animate."""

    frames: list = field(default_factory=list)   # per tick: list of bot snapshots
    shots: list = field(default_factory=list)     # per tick: list of firing lines
    counts: list = field(default_factory=list)    # per tick: alive count per team (in `teams` order)
    teams: list = field(default_factory=list)     # team ids present, sorted
    winner: object = None                         # winning team id / None (draw)
    winning_alliance: object = None             # winning alliance id when ``alliances`` set
    alliances: dict = field(default_factory=dict)
    objective: str = "annihilation"
    objective_zone: tuple = field(default_factory=tuple)
    objective_progress: list = field(default_factory=list)
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


def make_allied_armies(cfg, deployments):
    """Deploy several battle lines for a multi-army allied campaign.

    Each entry in ``deployments`` is a dict with ``team``, ``front_center``, and
    optional ``rows``, ``cols``, ``kind``, ``spacing``, ``face_right``, ``seed`` /
    ``rng`` (passed to :func:`make_grand_army`).
    """
    bots = []
    for dep in deployments:
        rng = dep.get("rng")
        if rng is None:
            rng = random.Random(dep.get("seed", 0))
        bots.extend(make_grand_army(
            cfg, dep["team"], dep["front_center"],
            rows=dep.get("rows", 6), cols=dep.get("cols", 8),
            kind=dep.get("kind", "soldier"),
            spacing=dep.get("spacing", 2.0),
            rng=rng, face_right=dep.get("face_right", True)))
    return bots


def make_contest_armies(n_per_team, cfg, *, seed=0):
    """Two armies spawned closer to centre — for hill / domination fights."""
    rng = random.Random(seed)
    bots = []
    for team, cx in ((RED, cfg.width * 0.28), (BLUE, cfg.width * 0.72)):
        for _ in range(n_per_team):
            x = cx + rng.uniform(-2.8, 2.8)
            y = cfg.height * 0.5 + rng.uniform(-6.5, 6.5)
            bots.append(Bot(x, y, 0.0, 0.0, team, cfg.max_hp, cfg.max_hp))
    return bots


def clone_bots(bots):
    """Deep-copy bot state for a fair rematch on the same spawn."""
    return [Bot(b.x, b.y, b.vx, b.vy, b.team, b.hp, b.max_hp, alive=b.alive,
                dps=b.dps, attack_range=b.attack_range, max_speed=b.max_speed,
                kind=b.kind)
            for b in bots]


HILL_CONTEST_KW = dict(
    objective="hill",
    objective_radius=4.8,
    objective_hold_ticks=120,
    tactics="count_aware",
    formation="wedge",
    maneuver_replan_ticks=12,
    assignment_replan_ticks=12,
)


def mapf_total_war_pair(*, n=18, seed=8):
    """Same spawn, two configs — Hungarian+greedy vs CBS-TA+prioritized MAPF."""
    cfg_base = BattleConfig(**HILL_CONTEST_KW)
    spawn = make_contest_armies(n, cfg_base, seed=seed)
    cfg_local = BattleConfig(
        **HILL_CONTEST_KW,
        assignment="none",
        assignment_by_team={RED: "hungarian"},
        maneuver="greedy",
        maneuver_by_team={BLUE: "greedy"},
    )
    cfg_mapf = BattleConfig(
        **HILL_CONTEST_KW,
        assignment="none",
        assignment_by_team={RED: "cbs_ta"},
        maneuver="greedy",
        maneuver_by_team={RED: "prioritized", BLUE: "greedy"},
    )
    titles = (
        "Hungarian + greedy",
        "CBS-TA + prioritized MAPF",
    )
    return spawn, cfg_local, cfg_mapf, titles


CTF_CONTEST_KW = dict(
    objective="ctf",
    objective_radius=3.2,
    base_radius=4.2,
    tactics="count_aware",
    formation="wedge",
    maneuver_replan_ticks=12,
    assignment_replan_ticks=12,
)


def ctf_mapf_pair(*, n=10, seed=10):
    """Same CTF spawn, two configs — Hungarian+greedy vs CBS-TA+prioritized MAPF."""
    cfg_base = BattleConfig(**CTF_CONTEST_KW)
    spawn = make_armies(n, cfg_base, seed=seed)
    cfg_local = BattleConfig(
        **CTF_CONTEST_KW,
        assignment="none",
        assignment_by_team={RED: "hungarian"},
        maneuver="greedy",
        maneuver_by_team={BLUE: "greedy"},
    )
    cfg_mapf = BattleConfig(
        **CTF_CONTEST_KW,
        assignment="none",
        assignment_by_team={RED: "cbs_ta"},
        maneuver="greedy",
        maneuver_by_team={RED: "prioritized", BLUE: "greedy"},
    )
    titles = (
        "Hungarian + greedy",
        "CBS-TA + prioritized MAPF",
    )
    return spawn, cfg_local, cfg_mapf, titles


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


def battle_step(bots, cfg, *, maneuver_state=None, assignment_state=None,
                tick=0, ctf_tracker=None):
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
    ctf_mode = cfg.objective == "ctf" and ctf_tracker is not None
    for i in range(n):
        b = live[i]
        bot_idx = bots.index(b)
        decision = decisions[i]
        if ctf_mode:
            carrier_idx = ctf_tracker.carrier_idx
            if carrier_idx == bot_idx:
                base = ctf_tracker.bases.get(b.team)
                if base is not None:
                    bx, by, _ = base
                    dx, dy = bx - b.x, by - b.y
                    d = max(math.hypot(dx, dy), 1e-9)
                    scale = cfg.w_pursue * 2.4
                    desired[i][0] += scale * dx / d
                    desired[i][1] += scale * dy / d
            elif carrier_idx is None:
                dx = ctf_tracker.flag_x - b.x
                dy = ctf_tracker.flag_y - b.y
                d = math.hypot(dx, dy)
                if d < cfg.perception * 1.3:
                    scale = cfg.w_pursue * 1.1
                    desired[i][0] += scale * dx / max(d, 1e-9)
                    desired[i][1] += scale * dy / max(d, 1e-9)
            elif carrier_idx is not None:
                carrier = bots[carrier_idx]
                if carrier.alive and teams_are_enemies(cfg.alliances, b.team,
                                                       carrier.team):
                    dx, dy = carrier.x - b.x, carrier.y - b.y
                    d = max(math.hypot(dx, dy), 1e-9)
                    scale = cfg.w_pursue * 1.6
                    desired[i][0] += scale * dx / d
                    desired[i][1] += scale * dy / d
        if decision is None:
            continue
        best = decision.target_index
        if best < 0 or best >= n:
            continue
        e = live[best]
        if not teams_are_enemies(cfg.alliances, b.team, e.team):
            continue
        if ctf_mode and ctf_tracker.carrier_idx == bot_idx:
            bd = math.hypot(b.x - e.x, b.y - e.y)
            b_range = b.attack_range if b.attack_range is not None else cfg.attack_range
            if bd <= b_range:
                cover = _fire_cover(b.x, b.y, e.x, e.y, cfg, live, {i, best})
                if cover > 0.0:
                    dps = b.dps if b.dps is not None else cfg.dps
                    damage[best] += dps * cfg.dt * cover
                    shots.append((b.x, b.y, e.x, e.y, b.team))
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


def _standing_alliances(bots, teams, alliances):
    """Alliance ids (or team ids in FFA) that still have living robots."""
    alive = set()
    for t in teams:
        if any(b.alive and b.team == t for b in bots):
            if alliances:
                alive.add(alliance_of(alliances, t))
            else:
                alive.add(t)
    return alive


def _pick_winner(teams, survivors, alliances):
    standing = _standing_alliances_from_counts(teams, survivors, alliances)
    if len(standing) != 1:
        return None, None
    win_key = next(iter(standing))
    if alliances:
        best_team, best_n = None, -1
        for t, n in survivors.items():
            if alliance_of(alliances, t) == win_key and n > best_n:
                best_n, best_team = n, t
        return best_team, win_key
    return win_key, None


def _standing_alliances_from_counts(teams, survivors, alliances):
    alive = set()
    for t in teams:
        if survivors.get(t, 0) > 0:
            if alliances:
                alive.add(alliance_of(alliances, t))
            else:
                alive.add(t)
    return alive


def _alliance_scores(teams, survivors, alliances):
    scores = {}
    for t in teams:
        a = alliance_of(alliances, t)
        scores[a] = scores.get(a, 0) + survivors.get(t, 0)
    return scores


def _pick_winner_by_survivors(teams, survivors, alliances):
    """Break a timed-out alliance battle by total survivors."""
    scores = _alliance_scores(teams, survivors, alliances)
    if not scores or max(scores.values()) <= 0:
        return None, None
    win_key = max(scores, key=scores.get)
    best_team, best_n = None, -1
    for t in teams:
        if alliance_of(alliances, t) == win_key:
            n = survivors.get(t, 0)
            if n > best_n:
                best_n, best_team = n, t
    return best_team, win_key


def _snapshot(bots):
    return [(b.x, b.y, b.team, b.hp / b.max_hp if b.max_hp else 0.0, b.alive,
             b.kind) for b in bots]


def simulate(bots, cfg=None, *, max_ticks=800, frame_stride=1):
    """Play a whole engagement from a prepared list of bots (any teams/classes).

    Records the per-tick snapshots and firing lines. Stops when at most one team
    or alliance remains (annihilation), when an objective is secured (hill /
    domination), or at ``max_ticks``. Deterministic given the inputs.

    ``frame_stride`` subsamples recorded frames (and shots) for large battles.
    """
    cfg = cfg or BattleConfig()
    frame_stride = max(1, frame_stride)
    teams = sorted({b.team for b in bots})
    obj_mode = cfg.objective or "annihilation"
    zone = ()
    if obj_mode in ("hill", "domination"):
        zone = objective_zone(cfg)
    elif obj_mode == "ctf":
        zone = tuple(ctf_render_zones(cfg, teams))
    result = BattleResult(
        teams=teams,
        alliances=dict(cfg.alliances or {}),
        objective=obj_mode,
        objective_zone=zone,
    )
    maneuver_state = ManeuverState()
    assignment_state = AssignmentState()
    obj_tracker = ObjectiveTracker(cfg) if obj_mode in ("hill", "domination") else None
    ctf_tracker = CtfTracker(cfg, teams) if obj_mode == "ctf" else None
    objective_win = None
    for tick in range(max_ticks):
        record = (tick % frame_stride == 0)
        if record:
            result.frames.append(_snapshot(bots))
            result.counts.append(_counts(bots, teams))
            if ctf_tracker is not None:
                result.objective_progress.append(ctf_tracker.snapshot(bots, cfg))
            elif obj_tracker is not None:
                result.objective_progress.append(obj_tracker.snapshot())
        if obj_mode != "ctf" and len(_standing_alliances(bots, teams, cfg.alliances)) <= 1:
            if record:
                result.shots.append([])
            break
        shots = battle_step(bots, cfg, maneuver_state=maneuver_state,
                            assignment_state=assignment_state, tick=tick,
                            ctf_tracker=ctf_tracker)
        if record:
            result.shots.append(shots)
        if obj_tracker is not None:
            leader = zone_leader(bots, cfg, teams)
            objective_win = obj_tracker.tick(leader)
            if objective_win is not None:
                break
        if ctf_tracker is not None:
            objective_win = ctf_tracker.tick(bots, cfg)
            if objective_win is not None:
                break
    else:
        if tick % frame_stride == 0:
            result.frames.append(_snapshot(bots))
            result.counts.append(_counts(bots, teams))
            if ctf_tracker is not None:
                result.objective_progress.append(ctf_tracker.snapshot(bots, cfg))
            elif obj_tracker is not None:
                result.objective_progress.append(obj_tracker.snapshot())
            result.shots.append([])

    c = _counts(bots, teams)
    result.survivors = {t: n for t, n in zip(teams, c)}
    result.ticks = len(result.frames) - 1
    if objective_win is not None:
        result.winner, result.winning_alliance = winner_from_objective(
            objective_win, cfg, teams)
    else:
        result.winner, result.winning_alliance = _pick_winner(
            teams, result.survivors, cfg.alliances)
        if (result.winning_alliance is None and cfg.alliances
                and len(_standing_alliances_from_counts(
                    teams, result.survivors, cfg.alliances)) > 1):
            result.winner, result.winning_alliance = _pick_winner_by_survivors(
                teams, result.survivors, cfg.alliances)
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
                  "maneuver_duel", "mapf_stack_duel", "mapf_total_war_local",
                  "mapf_total_war_mapf", "ctf_mapf_local", "ctf_mapf_mapf",
                  "kingdom", "grand_alliance",
                  "hill", "domination", "ctf")

ALLIANCE_NAMES = {0: "western", 1: "eastern"}


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
    if name == "grand_alliance":
        alliances = {RED: 0, GREEN: 0, BLUE: 1, YELLOW: 1}
        cfg = kingdom_config(
            width=140.0,
            height=72.0,
            alliances=alliances,
            dps=56.0,
            tactics="count_aware:aggressive",
            spatial_min_bots=24,
            formation_by_team={
                RED: "line", BLUE: "line",
                GREEN: "wedge", YELLOW: "wedge",
            },
        )
        w, h = cfg.width, cfg.height
        dense = 1.78
        # Total war — four full-strength infantry battle lines (upper/lower per
        # alliance) plus armoured and fire-support echelons tucked behind each wing.
        bots = make_allied_armies(cfg, [
            dict(team=RED, front_center=(w * 0.36, h * 0.36),
                 rows=9, cols=14, face_right=True, spacing=dense, seed=19),
            dict(team=RED, front_center=(w * 0.29, h * 0.36),
                 rows=2, cols=10, kind="tank", face_right=True, spacing=2.6,
                 seed=23),
            dict(team=GREEN, front_center=(w * 0.36, h * 0.64),
                 rows=9, cols=14, face_right=True, spacing=dense, seed=20),
            dict(team=GREEN, front_center=(w * 0.27, h * 0.64),
                 rows=2, cols=8, kind="sniper", face_right=True, spacing=2.5,
                 seed=24),
            dict(team=BLUE, front_center=(w * 0.64, h * 0.36),
                 rows=9, cols=14, face_right=False, spacing=dense, seed=21),
            dict(team=BLUE, front_center=(w * 0.71, h * 0.36),
                 rows=2, cols=10, kind="tank", face_right=False, spacing=2.6,
                 seed=25),
            dict(team=YELLOW, front_center=(w * 0.64, h * 0.64),
                 rows=9, cols=14, face_right=False, spacing=dense, seed=22),
            dict(team=YELLOW, front_center=(w * 0.73, h * 0.64),
                 rows=2, cols=8, kind="sniper", face_right=False, spacing=2.5,
                 seed=26),
        ])
        n = len(bots)
        return (bots, cfg,
                f"Total war — {n} robots, four armies, two allied fronts")
    if name == "grand_alliance_lite":
        alliances = {RED: 0, GREEN: 0, BLUE: 1, YELLOW: 1}
        cfg = kingdom_config(
            width=88.0,
            height=50.0,
            alliances=alliances,
            dps=50.0,
            tactics="count_aware:aggressive",
            spatial_min_bots=48,
            formation_by_team={
                RED: "line", BLUE: "line", GREEN: "wedge", YELLOW: "wedge",
            },
        )
        w, h = cfg.width, cfg.height
        dense = 2.0
        bots = make_allied_armies(cfg, [
            dict(team=RED, front_center=(w * 0.34, h * 0.38),
                 rows=4, cols=8, face_right=True, spacing=dense, seed=19),
            dict(team=GREEN, front_center=(w * 0.34, h * 0.62),
                 rows=4, cols=8, face_right=True, spacing=dense, seed=20),
            dict(team=BLUE, front_center=(w * 0.66, h * 0.38),
                 rows=4, cols=8, face_right=False, spacing=dense, seed=21),
            dict(team=YELLOW, front_center=(w * 0.66, h * 0.62),
                 rows=4, cols=8, face_right=False, spacing=dense, seed=22),
        ])
        n = len(bots)
        return (bots, cfg,
                f"Allied fronts — {n} robots (browser scale)")
    if name == "kingdom_lite":
        cfg = kingdom_config(formation_by_team={RED: "line", BLUE: "line"})
        w, h = cfg.width, cfg.height
        red = make_grand_army(cfg, RED, (w * 0.30, h * 0.5),
                              rows=5, cols=8, rng=random.Random(42),
                              face_right=True)
        blue = make_grand_army(cfg, BLUE, (w * 0.70, h * 0.5),
                               rows=5, cols=8, rng=random.Random(43),
                               face_right=False)
        return (red + blue, cfg, "Kingdom clash — 40 vs 40 battle lines")
    if name == "hill":
        cfg = BattleConfig(
            objective="hill",
            objective_radius=4.2,
            objective_hold_ticks=140,
            tactics="count_aware",
            formation="wedge",
        )
        return (make_contest_armies(12, cfg, seed=8), cfg,
                "King of the hill — hold the centre")
    if name == "domination":
        cfg = BattleConfig(
            objective="domination",
            objective_radius=6.0,
            objective_hold_ticks=220,
            tactics="count_aware",
            formation="line",
        )
        return (make_contest_armies(14, cfg, seed=9), cfg,
                "Domination — control the centre")
    if name == "ctf":
        cfg = BattleConfig(
            objective="ctf",
            objective_radius=3.2,
            base_radius=4.2,
            tactics="count_aware",
            formation="wedge",
        )
        return (make_armies(10, cfg, seed=10), cfg,
                "Capture the flag — return home")
    if name in ("mapf_total_war_local", "mapf_total_war_mapf"):
        spawn, cfg_local, cfg_mapf, titles = mapf_total_war_pair()
        cfg = cfg_local if name == "mapf_total_war_local" else cfg_mapf
        title = f"MAPF total war — {titles[0 if name == 'mapf_total_war_local' else 1]}"
        return (clone_bots(spawn), cfg, title)
    if name in ("ctf_mapf_local", "ctf_mapf_mapf"):
        spawn, cfg_local, cfg_mapf, titles = ctf_mapf_pair()
        cfg = cfg_local if name == "ctf_mapf_local" else cfg_mapf
        title = f"CTF MAPF — {titles[0 if name == 'ctf_mapf_local' else 1]}"
        return (clone_bots(spawn), cfg, title)
    raise KeyError(name)
