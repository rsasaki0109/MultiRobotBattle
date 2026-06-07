"""Planned maneuver for swarm battle — MAPF zoo as the movement layer.

Replaces greedy straight-line pursuit with grid paths planned by the same solvers
the repo benchmarks: independent space-time A*, prioritized planning, CBS, and
LaCAM/PIBT. Each bot still picks its combat target via the tactics policy; the
planner routes it there collision-free around terrain (and, for joint solvers,
around teammates).
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field

from .mapf.cbs import cbs
from .mapf.grid import GridWorld, manhattan
from .mapf.lacam import lacam
from .mapf.path_follower import carrot_point

MANEUVER_MODES = ("greedy", "astar", "prioritized", "cbs", "pibt")


@dataclass
class ManeuverState:
    """Cached world-space paths keyed by stable bot id (index in the full bot list)."""

    paths: dict = field(default_factory=dict)
    last_plan_tick: int = -9999


def world_to_cell(xy, cell_size: float):
    return (int(math.floor(xy[0] / cell_size)), int(math.floor(xy[1] / cell_size)))


def cell_to_world(cell, cell_size: float):
    return ((cell[0] + 0.5) * cell_size, (cell[1] + 0.5) * cell_size)


def grid_from_battle(cfg, *, cell_size=None, inflation=0.35):
    """Discretize the battle arena and circular obstacles into a grid."""
    cell_size = cell_size or cfg.maneuver_cell_size
    nx = max(1, int(math.ceil(cfg.width / cell_size)))
    ny = max(1, int(math.ceil(cfg.height / cell_size)))
    blocked = set()
    for cx in range(nx):
        for cy in range(ny):
            wx, wy = cell_to_world((cx, cy), cell_size)
            for (ox, oy, r) in cfg.obstacles:
                if math.hypot(wx - ox, wy - oy) <= r + inflation:
                    blocked.add((cx, cy))
                    break
    return GridWorld(nx, ny, blocked=frozenset(blocked))


def _nearest_free(grid: GridWorld, cell):
    if grid.is_free(cell):
        return cell
    q = deque([cell])
    seen = {cell}
    while q:
        c = q.popleft()
        for nb in grid.neighbors(c):
            if nb in seen:
                continue
            seen.add(nb)
            if grid.is_free(nb):
                return nb
            q.append(nb)
    return cell


def _spatial_dedupe(cells):
    out = []
    for c in cells:
        if not out or c != out[-1]:
            out.append(c)
    return out


def _cells_to_world(cells, cell_size):
    return [cell_to_world(c, cell_size) for c in _spatial_dedupe(cells)]


def build_agent_goals(bots, live, decisions, cfg, grid):
    """Map stable bot id -> ``(start_cell, goal_cell)`` for planned teams only."""
    cell_size = cfg.maneuver_cell_size
    agents = {}
    for i, decision in enumerate(decisions):
        b = live[i]
        if maneuver_for_team(b.team, cfg) in (None, "", "greedy"):
            continue
        bid = bots.index(b)
        start = _nearest_free(grid, world_to_cell((b.x, b.y), cell_size))
        if decision is None:
            goal = start
        else:
            target = live[decision.target_index]
            goal = _nearest_free(grid, world_to_cell((target.x, target.y), cell_size))
        if grid.is_free(start):
            agents[bid] = (start, goal)
    return agents


def _blockers_for_greedy_teams(bots, live, cfg):
    """Cells occupied by greedy-team bots — treated as static for one replan."""
    cell_size = cfg.maneuver_cell_size
    blocked = set()
    for b in live:
        if maneuver_for_team(b.team, cfg) not in (None, "", "greedy"):
            continue
        blocked.add(world_to_cell((b.x, b.y), cell_size))
    return blocked


def spatial_astar(grid: GridWorld, start, goal, *, blocked=frozenset()):
    """Static grid A* — one step per cell, no time dimension (fast for battle)."""
    if start == goal:
        return [start]
    if not grid.is_free(start) or not grid.is_free(goal):
        return None
    open_heap = [(manhattan(start, goal), 0, start)]
    came = {}
    gscore = {start: 0}
    while open_heap:
        _, g, cell = heapq.heappop(open_heap)
        if cell == goal:
            path = [cell]
            while cell in came:
                cell = came[cell]
                path.append(cell)
            return list(reversed(path))
        x, y = cell
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (x + dx, y + dy)
            if not grid.is_free(nb) or nb in blocked:
                continue
            ng = g + 1
            if ng < gscore.get(nb, float("inf")):
                gscore[nb] = ng
                came[nb] = cell
                heapq.heappush(open_heap, (ng + manhattan(nb, goal), ng, nb))
    return None


def prioritized_spatial(grid, agents, order=None):
    """Priority-ordered static paths; later agents avoid earlier paths' cells."""
    order = list(order) if order is not None else sorted(agents)
    reserved = set()
    paths = {}
    for aid in order:
        start, goal = agents[aid]
        path = spatial_astar(grid, start, goal, blocked=frozenset(reserved))
        if path is None:
            return None
        paths[aid] = path
        reserved.update(path)
    return paths


def _plan_independent_astar(grid, agents):
    paths = {}
    for aid, (start, goal) in agents.items():
        path = spatial_astar(grid, start, goal)
        if path is not None:
            paths[aid] = path
    return paths


def _plan_joint(grid, agents, mode):
    if not agents:
        return None
    if mode == "prioritized":
        paths = prioritized_spatial(grid, agents)
        if paths is None:
            return None
        from .mapf.solution import Solution, sum_of_costs
        return Solution(paths=paths, cost=sum_of_costs(paths))
    if mode == "cbs":
        if len(agents) <= 8:
            return cbs(grid, agents, max_expansions=8_000)
        paths = prioritized_spatial(grid, agents)
        if paths is None:
            return None
        from .mapf.solution import Solution, sum_of_costs
        return Solution(paths=paths, cost=sum_of_costs(paths))
    if mode == "pibt":
        n = len(agents)
        if n <= 8:
            budget = min(200_000, 20_000 + 5_000 * n)
            return lacam(grid, agents, max_iterations=budget)
        paths = prioritized_spatial(grid, agents)
        if paths is None:
            return None
        from .mapf.solution import Solution, sum_of_costs
        return Solution(paths=paths, cost=sum_of_costs(paths))
    return None


def plan_maneuver(grid, agents, mode):
    """Return ``bot_id -> grid cell path`` for the requested maneuver layer."""
    if mode == "astar":
        return _plan_independent_astar(grid, agents)
    sol = _plan_joint(grid, agents, mode)
    if sol is None:
        return _plan_independent_astar(grid, agents)
    return dict(sol.paths)


def maneuver_for_team(team, cfg):
    return (cfg.maneuver_by_team or {}).get(team, cfg.maneuver)


def replan_paths(bots, live, decisions, cfg, state, tick):
    """Refresh cached world paths when the replan interval elapses."""
    if all(maneuver_for_team(live[i].team, cfg) in (None, "", "greedy")
           for i in range(len(live))):
        state.paths = {}
        return
    if tick - state.last_plan_tick < cfg.maneuver_replan_ticks and state.paths:
        return
    grid = grid_from_battle(cfg)
    extra = _blockers_for_greedy_teams(bots, live, cfg)
    if extra:
        grid = GridWorld(grid.width, grid.height,
                         frozenset(set(grid.blocked) | extra))
    agents = build_agent_goals(bots, live, decisions, cfg, grid)
    if not agents:
        state.paths = {}
        return
    mode = cfg.maneuver
    for team in {b.team for b in live}:
        tm = maneuver_for_team(team, cfg)
        if tm not in (None, "", "greedy"):
            mode = tm
            break
    cell_paths = plan_maneuver(grid, agents, mode)
    state.paths = {
        aid: _cells_to_world(path, cfg.maneuver_cell_size)
        for aid, path in cell_paths.items()
    }
    state.last_plan_tick = tick


def _bot_heading(b, target):
    if math.hypot(b.vx, b.vy) > 0.05:
        return math.atan2(b.vy, b.vx)
    if target is not None:
        return math.atan2(target.y - b.y, target.x - b.x)
    return 0.0


def maneuver_direction(b, path, target, cfg):
    """Unit velocity toward the path carrot, or ``None`` to fall back to greedy."""
    if not path:
        return None
    heading = _bot_heading(b, target)
    pose = (b.x, b.y, heading)
    gx, gy = path[-1]
    if math.hypot(gx - b.x, gy - b.y) <= cfg.maneuver_goal_tol:
        return None
    cx, cy = carrot_point(pose, path, cfg.maneuver_lookahead)
    dx, dy = cx - b.x, cy - b.y
    d = math.hypot(dx, dy)
    if d <= 1e-9:
        return None
    return (dx / d, dy / d)
