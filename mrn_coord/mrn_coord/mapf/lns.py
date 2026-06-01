"""MAPF-LNS: anytime MAPF by Large Neighborhood Search (Li et al., 2021).

CBS/ECBS *search* for a good solution from scratch; LaCAM finds *a* solution
fast. LNS takes a different, anytime route: start from any feasible solution
(here prioritized planning, falling back to complete LaCAM), then repeatedly
**destroy** a small neighborhood — rip out the paths of a handful of agents —
and **repair** it by replanning just those agents around everyone else's fixed
paths. Keep the repair if it does not raise the sum-of-costs. Each round is
cheap (it replans a few agents, not all), the cost decreases monotonically, and
you can stop whenever the budget runs out — so a rough initial solution is
polished toward the optimum over time, on team sizes far beyond CBS's reach.

Two destroy heuristics, chosen at random each round (the adaptive ensemble that
makes LNS robust):

- **random** — a random set of agents.
- **worst** — the most *delayed* agent (largest gap between its current path
  cost and its obstacle-aware shortest path) plus the agents whose paths cross
  it; replanning this cluster together is what unsticks a bad detour.

Pure and deterministic given the seed. Repair is collision-free by construction
(prioritized replanning over the frozen paths), so every accepted solution stays
valid.
"""

from __future__ import annotations

import random
from collections import deque

from .grid import Cell, GridWorld
from .lacam import lacam
from .prioritized import prioritized_planning
from .solution import Solution, sum_of_costs
from .space_time_astar import plan_path


def _bfs_dist_from(grid: GridWorld, goal: Cell) -> dict:
    dist = {goal: 0}
    q = deque([goal])
    while q:
        cell = q.popleft()
        d = dist[cell]
        for nb in grid.neighbors(cell):
            if nb not in dist:
                dist[nb] = d + 1
                q.append(nb)
    return dist


def _reservations(paths, subset, horizon):
    """Vertex/edge reservations from every agent *not* in ``subset``."""
    vertex: set = set()
    edge: set = set()
    for agent, path in paths.items():
        if agent in subset:
            continue
        for t, cell in enumerate(path):
            vertex.add((cell, t))
        goal_cell = path[-1]
        for t in range(len(path) - 1, horizon + 1):
            vertex.add((goal_cell, t))
        for t in range(len(path) - 1):
            edge.add((path[t + 1], path[t], t + 1))
    return vertex, edge


def _repair(grid, agents, paths, subset, horizon):
    """Replan ``subset`` around the frozen paths; return new paths or ``None``.

    Reservations (including each frozen agent's goal hold) extend to ``horizon``
    and every replan is capped at ``max_time=horizon``, so a repaired path can
    never run past the reserved window and slip through a held goal cell.
    """
    vertex, edge = _reservations(paths, subset, horizon)
    # replan the longest-detoured agents first (id breaks ties so the order is
    # independent of the set's iteration order — keeps the search deterministic)
    order = sorted(subset, key=lambda a: (-(len(paths[a]) - 1), a))
    new_paths = {}
    for agent in order:
        start, goal = agents[agent]
        path = plan_path(grid, start, goal, frozenset(vertex), frozenset(edge),
                         max_time=horizon)
        if path is None:
            return None
        new_paths[agent] = path
        for t, cell in enumerate(path):
            vertex.add((cell, t))
        for t in range(len(path) - 1, horizon + 1):
            vertex.add((path[-1], t))
        for t in range(len(path) - 1):
            edge.add((path[t + 1], path[t], t + 1))
    return new_paths


def _worst_neighborhood(paths, shortest, rng, k):
    """Most-delayed agent plus the agents whose paths cross it (padded random)."""
    delay = {a: (len(paths[a]) - 1) - shortest[a] for a in paths}
    seed_agent = max(delay, key=lambda a: (delay[a], a))
    seed_cells = set(paths[seed_agent])
    crossing = [a for a in paths
                if a != seed_agent and any(c in seed_cells for c in paths[a])]
    rng.shuffle(crossing)
    chosen = [seed_agent] + crossing
    if len(chosen) < k:
        rest = [a for a in paths if a not in chosen]
        rng.shuffle(rest)
        chosen += rest
    return set(chosen[:k])


def mapf_lns(
    grid: GridWorld,
    agents: dict,
    *,
    neighborhood_size: int | None = None,
    iterations: int = 100,
    seed: int = 0,
    init: Solution | None = None,
    stats: dict | None = None,
):
    """Improve a MAPF solution by Large Neighborhood Search (anytime).

    ``agents`` maps an agent id to a ``(start, goal)`` tuple. Returns the best
    :class:`Solution` found (collision-free), or ``None`` if no initial solution
    exists. ``init`` seeds the search (default: prioritized planning, then LaCAM
    if that fails). If ``stats`` is given, it records ``initial_cost``,
    ``final_cost``, ``iterations``, and ``accepted`` (rounds that improved).
    """
    ids = sorted(agents)
    if init is None:
        init = prioritized_planning(grid, agents) or lacam(grid, agents)
    if init is None:
        return None

    paths = {a: list(init.paths[a]) for a in ids}
    cur_cost = sum_of_costs(paths)
    initial_cost = cur_cost

    k = neighborhood_size or max(2, min(len(ids), 8))
    k = min(k, len(ids))
    rng = random.Random(seed)
    shortest = {a: _bfs_dist_from(grid, agents[a][1]).get(agents[a][0], 0)
                for a in ids}
    # Generous, fixed reservation/replan horizon: at least the prioritized
    # feasibility bound, and never shorter than the longest initial path, so
    # every frozen agent's goal hold and every capped repair fit inside the
    # reserved window (a repaired path can never run past a held goal cell).
    formula = (2 * grid.width * grid.height + len(ids)
               + grid.width + grid.height + 5)
    horizon = max(formula, max(len(p) for p in paths.values())) + len(ids) + 5
    accepted = 0

    for _ in range(iterations):
        if k >= len(ids):
            subset = set(ids)
        elif rng.random() < 0.5:
            subset = set(rng.sample(ids, k))                  # random destroy
        else:
            subset = _worst_neighborhood(paths, shortest, rng, k)  # worst destroy

        repaired = _repair(grid, agents, paths, subset, horizon)
        if repaired is None:
            continue
        candidate = dict(paths)
        candidate.update(repaired)
        new_cost = sum_of_costs(candidate)
        if new_cost <= cur_cost:
            if new_cost < cur_cost:
                accepted += 1
            paths = candidate
            cur_cost = new_cost

    if stats is not None:
        stats["initial_cost"] = initial_cost
        stats["final_cost"] = cur_cost
        stats["iterations"] = iterations
        stats["accepted"] = accepted
    return Solution(paths=paths, cost=cur_cost)
