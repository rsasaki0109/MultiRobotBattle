"""BCBS: Bounded-suboptimal Conflict-Based Search (Barer, Sharon, Stern &
Felner, "Suboptimal Variants of the Conflict-Based Search Algorithm", SoCS 2014).

The same paper introduced **two** suboptimal CBS variants, and the package
already has the better one: :func:`ecbs <mrn_coord.mapf.ecbs.ecbs>`. This is the
*other* one — BCBS — kept as a faithful, gated contrast that exposes exactly what
ECBS improved.

Both run focal search at both levels (OPEN by a primary key, expand from a FOCAL
sublist ranked by fewest conflicts). The difference is the **high-level focal
bound**:

- **BCBS(w_H, w_L)** orders the high-level OPEN by the node's actual sum-of-costs
  ``cost`` and takes FOCAL = nodes with ``cost <= w_H * min(cost in OPEN)``. The
  low level is ``w_L``-suboptimal. Because the high-level bound is taken against
  the best *cost* (not a lower bound on the optimum), the two factors **multiply**:
  the returned solution is at most ``w_H * w_L`` times optimal.
- **ECBS(w)** instead tracks ``LB(N) = sum of the low-level lower bounds`` and
  bounds FOCAL by ``w * min(LB)``. Since ``LB`` is a true lower bound on the
  optimum, the suboptimality is just ``w`` — no squaring. That tighter accounting
  is ECBS's whole contribution.

So at a matched *effective* bound, BCBS's per-factor bounds are looser: BCBS(w, w)
only guarantees ``w^2``, and its returned cost can exceed ``w * optimal`` where
ECBS(w) cannot. BCBS(1, 1) is optimal CBS. This module reuses ECBS's focal low
level and conflict counter unchanged; only the high-level bound differs.
"""

from __future__ import annotations

import itertools

from .conflicts import VertexConflict, detect_first_conflict
from .ecbs import _count_conflicts, _focal_low_level
from .grid import GridWorld
from .solution import Solution


def bcbs(grid: GridWorld, agents: dict, *, w_high: float = 1.5,
         w_low: float = 1.5, max_expansions: int = 100_000,
         stats: dict | None = None):
    """Solve a MAPF instance with cost ``<= w_high * w_low * optimal``.

    ``agents`` maps an agent id to ``(start, goal)``. ``w_high >= 1`` bounds the
    high-level focal list (against the best *cost* in OPEN); ``w_low >= 1`` bounds
    the low-level focal search. Returns a :class:`Solution`, or ``None`` if
    infeasible or the budget is exhausted. With ``stats``, ``stats["expansions"]``
    is the number of high-level nodes expanded.
    """
    paths: dict = {}
    costs: dict = {}
    for agent in agents:
        start, goal = agents[agent]
        reserved = list(paths.values())
        path, _lb = _focal_low_level(
            grid, start, goal, frozenset(), frozenset(), reserved, w_low)
        if path is None:
            if stats is not None:
                stats["expansions"] = 0
            return None
        paths[agent] = path
        costs[agent] = len(path) - 1

    order = itertools.count()
    root = {
        "vertex": {a: frozenset() for a in agents},
        "edge": {a: frozenset() for a in agents},
        "paths": paths,
        "costs": costs,
        "cost": sum(costs.values()),
        "conf": _count_conflicts(paths),
        "id": next(order),
    }
    live = [root]
    expansions = 0

    while live:
        # High-level focal bound is taken against the best COST in OPEN (this is
        # what makes BCBS's factor multiply with the low level's, vs ECBS's LB).
        cost_min = min(n["cost"] for n in live)
        threshold = w_high * cost_min
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
            path, _lb = _focal_low_level(
                grid, start, goal,
                child_vertex[agent], child_edge[agent], reserved, w_low)
            if path is None:
                continue

            child_paths = dict(node["paths"])
            child_paths[agent] = path
            child_costs = dict(node["costs"])
            child_costs[agent] = len(path) - 1
            live.append({
                "vertex": child_vertex,
                "edge": child_edge,
                "paths": child_paths,
                "costs": child_costs,
                "cost": sum(child_costs.values()),
                "conf": _count_conflicts(child_paths),
                "id": next(order),
            })

    if stats is not None:
        stats["expansions"] = expansions
    return None
