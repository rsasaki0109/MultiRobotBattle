"""Optimal shooter–target assignment for swarm battle.

- ``hungarian`` — min-cost matching on combat utility (who engages whom).
- ``cbs_ta`` — Murty K-best assignment on grid BFS distances (the CBS-TA root
  forest from :mod:`mrn_coord.mapf.cbs_ta`): path-aware target matching around
  terrain, without running the full joint CBS search each tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .battle_fog import can_see_enemy
from .battle_maneuver import _nearest_free, grid_from_battle, world_to_cell
from .battle_teams import teams_are_enemies
from .lifelong.allocation import INF, hungarian
from .mapf.cbs_ta import _bfs_dist, _murty

_ASSIGNMENT_MODES = ("none", "hungarian", "cbs_ta")
_MAX_CBS_TA_TEAM = 12
_CBS_TA_TRY = 4


@dataclass
class AssignmentState:
    """Cached live-index -> enemy live-index targets between replans."""

    targets: dict = field(default_factory=dict)
    last_plan_tick: int = -9999


def _needs_assignment(cfg):
    if cfg.assignment not in (None, "", "none"):
        return True
    by = cfg.assignment_by_team or {}
    return any(m in ("hungarian", "cbs_ta") for m in by.values())


def assignment_mode_for_team(team, cfg):
    return (cfg.assignment_by_team or {}).get(team, cfg.assignment)


def assignment_cost(shooter, enemy, dist, cfg):
    """Lower is better: prefer in-range, wounded, and nearer targets."""
    b_range = (shooter.attack_range if shooter.attack_range is not None
               else cfg.attack_range)
    cost = dist
    if dist > b_range:
        cost += 8.0 + 2.0 * (dist - b_range)
    hp_frac = enemy.hp / enemy.max_hp if enemy.max_hp else 1.0
    cost -= 2.5 * (1.0 - hp_frac)
    if shooter.kind == "sniper":
        ideal = 0.85 * b_range
        cost += abs(dist - ideal)
        if dist < 0.35 * b_range:
            cost += 4.0
    return cost


def _hungarian_for_team(live, allies, enemies, cfg):
    """Hungarian matching for one team; maps live ally index -> live enemy index."""
    targets = {}
    if not enemies:
        return targets
    if len(allies) == 1:
        best = min(enemies, key=lambda j: math.hypot(
            live[allies[0]].x - live[j].x, live[allies[0]].y - live[j].y))
        targets[allies[0]] = best
        return targets
    cost = []
    for ai in allies:
        row = []
        sh = live[ai]
        for ej in enemies:
            en = live[ej]
            d = math.hypot(sh.x - en.x, sh.y - en.y)
            row.append(assignment_cost(sh, en, d, cfg))
        cost.append(row)
    matching = hungarian(cost)
    for r, c in matching.items():
        targets[allies[r]] = enemies[c]
    for ai in allies:
        if ai not in targets:
            targets[ai] = min(enemies, key=lambda j, ai=ai: math.hypot(
                live[ai].x - live[j].x, live[ai].y - live[j].y))
    return targets


def _enemy_goal_cells(grid, live, enemies, cell_size):
    """Map each enemy to a free grid cell; dedupe cells that collide."""
    cell_to_enemy = {}
    goal_cells = []
    for ej in enemies:
        en = live[ej]
        cell = _nearest_free(grid, world_to_cell((en.x, en.y), cell_size))
        if cell not in cell_to_enemy:
            cell_to_enemy[cell] = ej
            goal_cells.append(cell)
    return goal_cells, cell_to_enemy


def _cbs_ta_for_team(live, allies, enemies, cfg):
    """Murty assignment on grid BFS distances; fall back to Hungarian."""
    if not enemies:
        return {}
    if (len(allies) > _MAX_CBS_TA_TEAM or len(enemies) > _MAX_CBS_TA_TEAM
            or len(allies) > len(enemies)):
        return _hungarian_for_team(live, allies, enemies, cfg)

    cell_size = cfg.assignment_cell_size or max(cfg.maneuver_cell_size, 1.5)
    grid = grid_from_battle(cfg, cell_size=cell_size)
    goal_cells, cell_to_enemy = _enemy_goal_cells(grid, live, enemies, cell_size)
    if len(allies) > len(goal_cells):
        return _hungarian_for_team(live, allies, enemies, cfg)

    targets_list = sorted(goal_cells)
    tindex = {t: j for j, t in enumerate(targets_list)}
    starts = []
    for ai in allies:
        sh = live[ai]
        start = _nearest_free(grid, world_to_cell((sh.x, sh.y), cell_size))
        if not grid.is_free(start):
            return _hungarian_for_team(live, allies, enemies, cfg)
        starts.append(start)

    dists = [_bfs_dist(grid, s) for s in starts]
    cost = []
    for di in dists:
        row = [INF] * len(targets_list)
        for t in goal_cells:
            if grid.is_free(t) and t in di:
                row[tindex[t]] = float(di[t])
        cost.append(row)

    gen = _murty(cost)
    for _ in range(_CBS_TA_TRY):
        nxt = next(gen, None)
        if nxt is None:
            break
        assign, _ = nxt
        targets = {}
        for r, ai in enumerate(allies):
            goal = targets_list[assign[r]]
            targets[ai] = cell_to_enemy[goal]
        if len(targets) == len(allies):
            return targets
    return _hungarian_for_team(live, allies, enemies, cfg)


def team_assignments(live, cfg):
    """Per live-index target for bots on teams using assignment modes."""
    by_team = {}
    for i, b in enumerate(live):
        by_team.setdefault(b.team, []).append(i)

    targets = {}
    for team, allies in by_team.items():
        mode = assignment_mode_for_team(team, cfg)
        if mode not in ("hungarian", "cbs_ta"):
            continue
        enemies = [j for j, b in enumerate(live)
                   if teams_are_enemies(cfg.alliances, team, b.team)]
        if mode == "hungarian":
            targets.update(_hungarian_for_team(live, allies, enemies, cfg))
        else:
            targets.update(_cbs_ta_for_team(live, allies, enemies, cfg))
    return targets


def hungarian_assignments(live, cfg):
    """Per live-index target for bots on teams using Hungarian assignment."""
    by_team = {}
    for i, b in enumerate(live):
        by_team.setdefault(b.team, []).append(i)

    targets = {}
    for team, allies in by_team.items():
        if assignment_mode_for_team(team, cfg) != "hungarian":
            continue
        enemies = [j for j, b in enumerate(live)
                   if teams_are_enemies(cfg.alliances, team, b.team)]
        targets.update(_hungarian_for_team(live, allies, enemies, cfg))
    return targets


def apply_assignments(decisions, live, cfg, *, assignment_state=None, tick=0):
    """Override policy targets for teams with ``assignment='hungarian'`` or ``cbs_ta``."""
    if not _needs_assignment(cfg):
        return decisions
    replan = cfg.assignment_replan_ticks
    if assignment_state is None:
        assigns = team_assignments(live, cfg)
    elif (tick - assignment_state.last_plan_tick >= replan
          or not assignment_state.targets):
        assignment_state.targets = team_assignments(live, cfg)
        assignment_state.last_plan_tick = tick
        assigns = assignment_state.targets
    else:
        assigns = assignment_state.targets
    if not assigns:
        return decisions
    out = list(decisions)
    for i, decision in enumerate(out):
        if decision is None or i not in assigns:
            continue
        t = assigns[i]
        if cfg.fog_of_war and not can_see_enemy(live, i, t, cfg):
            continue
        out[i] = type(decision)(
            target_index=t,
            pursue_scale=decision.pursue_scale,
            retreat_scale=decision.retreat_scale,
            flock_scale=decision.flock_scale,
            kite=decision.kite,
        )
    return out
