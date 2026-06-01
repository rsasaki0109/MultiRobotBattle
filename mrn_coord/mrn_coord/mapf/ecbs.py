"""Enhanced Conflict-Based Search (ECBS): bounded-suboptimal MAPF.

CBS (:mod:`cbs`) is optimal, but its constraint tree explodes as agents and
conflicts grow — every conflict can fork the search, and it insists on the very
best sum-of-costs. ECBS (Barer et al., 2014) keeps CBS's two-level structure but
trades a *provable* suboptimality factor ``w >= 1`` for speed: the returned
solution costs at most ``w`` times the optimum. With a little slack it can
expand orders of magnitude fewer nodes and solve instances CBS cannot within a
budget.

The trick is **focal search** at both levels. A focal search keeps the usual
best-first OPEN list (ordered by a lower bound ``f``) but expands from a FOCAL
sublist — the nodes whose ``f`` is within ``w`` of the current minimum — chosen
by a *secondary* heuristic. Here that secondary heuristic is "fewest conflicts",
so among all the cost-bounded options the search greedily steers toward
collision-free solutions:

- **Low level** (:func:`_focal_low_level`) — a single-agent A* whose FOCAL is
  ranked by the number of conflicts the path has with the *other* agents'
  current paths. It returns the path together with ``f_min``, a lower bound on
  that agent's optimal cost under its constraints.
- **High level** (:func:`ecbs`) — the constraint tree. OPEN is ordered by
  ``LB(N) = sum of the agents' low-level lower bounds``; FOCAL holds the nodes
  whose actual cost is ``<= w * min LB`` and is ranked by total conflict count.
  Popping a conflict-free node from FOCAL yields a solution with
  ``cost <= w * min LB <= w * optimal``.
"""

from __future__ import annotations

import heapq
import itertools

from .conflicts import VertexConflict, cell_at, detect_first_conflict
from .grid import GridWorld, manhattan
from .solution import Solution, sum_of_costs
from .space_time_astar import _default_max_time, _reconstruct


def _count_conflicts(paths: dict) -> int:
    """Total number of vertex + edge (swap) conflicts across all path pairs."""
    agents = list(paths)
    if len(agents) < 2:
        return 0
    horizon = max(len(p) for p in paths.values())
    total = 0
    for t in range(horizon):
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                if cell_at(paths[a], t) == cell_at(paths[b], t):
                    total += 1
                a_t, a_t1 = cell_at(paths[a], t), cell_at(paths[a], t + 1)
                b_t, b_t1 = cell_at(paths[b], t), cell_at(paths[b], t + 1)
                if a_t != a_t1 and a_t == b_t1 and a_t1 == b_t:
                    total += 1
    return total


def _focal_low_level(grid, start, goal, vertex_constraints, edge_constraints,
                     reserved, w):
    """Bounded-suboptimal single-agent plan minimizing conflicts with ``reserved``.

    ``reserved`` is the list of the other agents' current paths. Returns
    ``(path, lb)`` where ``path`` costs at most ``w`` times the constrained
    optimum and ``lb`` is a lower bound on that optimum (the minimum ``f`` left
    in OPEN), or ``(None, None)`` if infeasible.
    """
    if not grid.is_free(start) or not grid.is_free(goal):
        return None, None
    if (start, 0) in vertex_constraints:
        return None, None
    max_time = _default_max_time(grid, vertex_constraints, edge_constraints)
    goal_constraint_times = [t for (c, t) in vertex_constraints if c == goal]
    last_goal_time = max(goal_constraint_times) if goal_constraint_times else 0

    counter = itertools.count()
    came_from: dict = {}
    conflicts_to: dict = {}            # (cell, t) -> conflict count along its path
    open_heap: list = []               # (f, cnt, cell, t)        — orders by f
    focal_cand: list = []              # (f, cnt, cell, t)        — to migrate
    focal: list = []                   # (d, f, cnt, cell, t)     — orders by conflicts
    closed: set = set()

    def add(cell, t, d, parent):
        key = (cell, t)
        if key in conflicts_to:
            return
        conflicts_to[key] = d
        came_from[key] = parent
        f = t + manhattan(cell, goal)
        c = next(counter)
        heapq.heappush(open_heap, (f, c, cell, t))
        heapq.heappush(focal_cand, (f, c, cell, t))

    conflicts_to[(start, 0)] = 0
    c0 = next(counter)
    heapq.heappush(open_heap, (manhattan(start, goal), c0, start, 0))
    heapq.heappush(focal_cand, (manhattan(start, goal), c0, start, 0))

    while open_heap:
        while open_heap and (open_heap[0][2], open_heap[0][3]) in closed:
            heapq.heappop(open_heap)
        if not open_heap:
            break
        f_min = open_heap[0][0]
        threshold = w * f_min
        while focal_cand and focal_cand[0][0] <= threshold:
            f, c, cell, t = heapq.heappop(focal_cand)
            if (cell, t) not in closed:
                heapq.heappush(focal, (conflicts_to[(cell, t)], f, c, cell, t))
        while focal and (focal[0][3], focal[0][4]) in closed:
            heapq.heappop(focal)

        _, _, _, cell, t = heapq.heappop(focal)
        if (cell, t) in closed:
            continue
        closed.add((cell, t))

        if cell == goal and t >= last_goal_time:
            return _reconstruct(came_from, (cell, t)), f_min
        if t >= max_time:
            continue

        nt = t + 1
        base = conflicts_to[(cell, t)]
        for ncell in grid.neighbors(cell):
            if (ncell, nt) in vertex_constraints:
                continue
            if (cell, ncell, nt) in edge_constraints:
                continue
            if (ncell, nt) in conflicts_to:
                continue
            added = 0
            for op in reserved:
                if cell_at(op, nt) == ncell:                       # vertex
                    added += 1
                elif cell_at(op, nt) == cell and cell_at(op, t) == ncell:
                    added += 1                                     # swap
            add(ncell, nt, base + added, (cell, t))

    return None, None


def _root(grid, agents, w):
    """Plan every agent independently; return the root node dict or ``None``."""
    paths: dict = {}
    costs: dict = {}
    lbs: dict = {}
    for agent in agents:
        start, goal = agents[agent]
        reserved = list(paths.values())
        path, lb = _focal_low_level(
            grid, start, goal, frozenset(), frozenset(), reserved, w)
        if path is None:
            return None
        paths[agent] = path
        costs[agent] = len(path) - 1
        lbs[agent] = lb
    return {
        "vertex": {a: frozenset() for a in agents},
        "edge": {a: frozenset() for a in agents},
        "paths": paths,
        "costs": costs,
        "lbs": lbs,
        "cost": sum(costs.values()),
        "lb": sum(lbs.values()),
        "conf": _count_conflicts(paths),
    }


def ecbs(grid: GridWorld, agents: dict, *, w: float = 1.5,
         max_expansions: int = 100_000, stats: dict | None = None):
    """Solve a MAPF instance bounded-suboptimally (cost ``<= w * optimal``).

    ``agents`` maps an agent id to ``(start, goal)``; ``w >= 1`` is the
    suboptimality factor (``w = 1`` reduces to optimal CBS, just via focal
    search). Returns a :class:`Solution`, or ``None`` if infeasible or the
    expansion budget is exhausted. If ``stats`` is given,
    ``stats["expansions"]`` is set to the number of high-level nodes expanded.
    """
    root = _root(grid, agents, w)
    if root is None:
        if stats is not None:
            stats["expansions"] = 0
        return None

    order = itertools.count()
    root["id"] = next(order)
    live = [root]
    expansions = 0

    while live:
        lb_min = min(n["lb"] for n in live)
        threshold = w * lb_min
        focal = [n for n in live if n["cost"] <= threshold]
        node = min(focal, key=lambda n: (n["conf"], n["cost"], n["id"]))
        live.remove(node)

        expansions += 1
        if expansions > max_expansions:
            if stats is not None:
                stats["expansions"] = expansions
            return None

        conflict = detect_first_conflict(node["paths"])
        if conflict is None:
            if stats is not None:
                stats["expansions"] = expansions
            return Solution(paths=dict(node["paths"]), cost=node["cost"])

        if isinstance(conflict, VertexConflict):
            branches = [
                (conflict.agent_a, ("v", conflict.cell, conflict.time)),
                (conflict.agent_b, ("v", conflict.cell, conflict.time)),
            ]
        else:
            branches = [
                (conflict.agent_a,
                 ("e", conflict.cell_a, conflict.cell_b, conflict.time)),
                (conflict.agent_b,
                 ("e", conflict.cell_b, conflict.cell_a, conflict.time)),
            ]

        for agent, constraint in branches:
            child_vertex = dict(node["vertex"])
            child_edge = dict(node["edge"])
            if constraint[0] == "v":
                _, cell, time = constraint
                child_vertex[agent] = child_vertex[agent] | {(cell, time)}
            else:
                _, frm, to, time = constraint
                child_edge[agent] = child_edge[agent] | {(frm, to, time)}

            start, goal = agents[agent]
            reserved = [node["paths"][o] for o in agents if o != agent]
            path, lb = _focal_low_level(
                grid, start, goal,
                child_vertex[agent], child_edge[agent], reserved, w)
            if path is None:
                continue

            child_paths = dict(node["paths"])
            child_paths[agent] = path
            child_costs = dict(node["costs"])
            child_costs[agent] = len(path) - 1
            child_lbs = dict(node["lbs"])
            child_lbs[agent] = lb
            live.append({
                "vertex": child_vertex,
                "edge": child_edge,
                "paths": child_paths,
                "costs": child_costs,
                "lbs": child_lbs,
                "cost": sum(child_costs.values()),
                "lb": sum(child_lbs.values()),
                "conf": _count_conflicts(child_paths),
                "id": next(order),
            })

    if stats is not None:
        stats["expansions"] = expansions
    return None
