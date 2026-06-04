"""FECBS — ECBS with Flex Distribution (Chan, Li, Harabor, Koenig, SoCS 2021).

ECBS keeps every agent's path within ``w`` of *its own* constrained optimum: the
low-level focal search bounds the path cost by ``w * f_min`` (see
:func:`mrn_coord.mapf.ecbs._focal_low_level`). That per-agent bound is stricter
than the only guarantee the user asked for — that the *total* sum-of-costs is
within ``w`` of optimal. Bounding each path individually wastes leeway: an agent
sitting exactly at its optimum has spare budget that a *different*, conflict-prone
agent could use to route around a collision.

**Flex distribution** hands that spare budget over. When ECBS replans agent
``i``, FECBS widens its focal threshold from ``w * f_min,i`` to

    tau_i = w * f_min,i + flex_i,   flex_i = sum_{j != i} ( w * lb_j - c_j )

where ``lb_j`` / ``c_j`` are the other agents' lower bound and current path cost
in this node. ``flex_i`` is exactly the suboptimality budget the *other* agents
have not spent, so the agent may overshoot ``w * lb_i`` to dodge conflicts — while
the **global** bound still holds by construction:

    c_i <= tau_i  =>  sum_k c_k = c_i + sum_{j!=i} c_j
                              <= (w*lb_i + sum_{j!=i}(w*lb_j - c_j)) + sum_{j!=i} c_j
                              =  w * sum_k lb_k  =  w * LB(N).

So FECBS returns a solution of cost ``<= w * optimal``, the same guarantee as
ECBS, but the looser low level resolves fewer conflicts on the high level — fewer
constraint-tree nodes. This is the greedy variant (GFD): the full flex goes to
the one agent being replanned. With ``w = 1`` the flex is non-positive (a path
never beats its lower bound) so FECBS collapses to optimal CBS, exactly like ECBS.

This module reuses ECBS's focal low level, conflict counter, and root unchanged;
:mod:`mrn_coord.mapf.ecbs` stays byte-for-byte the same (its low level merely
gained a ``flex=0`` keyword that defaults to plain ECBS).
"""

from __future__ import annotations

import itertools

from .conflicts import EdgeConflict, VertexConflict, detect_first_conflict
from .ecbs import _count_conflicts, _focal_low_level, _root
from .grid import GridWorld
from .solution import Solution


def fecbs(grid: GridWorld, agents: dict, *, w: float = 1.5,
          max_expansions: int = 100_000, stats: dict | None = None,
          highways=frozenset()):
    """Solve a MAPF instance bounded-suboptimally (cost ``<= w * optimal``) with
    flex distribution.

    Same interface and guarantee as :func:`mrn_coord.mapf.ecbs.ecbs`, but the
    low level spends the other agents' unused suboptimality budget to avoid
    conflicts, so the high level expands fewer constraint-tree nodes. If ``stats``
    is given, ``stats["expansions"]`` is the number of high-level nodes expanded
    and ``stats["flex_replans"]`` the number of low-level replans that were handed
    a strictly positive flex budget.
    """
    root = _root(grid, agents, w, highways)
    if root is None:
        if stats is not None:
            stats["expansions"] = 0
            stats["flex_replans"] = 0
        return None

    order = itertools.count()
    root["id"] = next(order)
    live = [root]
    expansions = 0
    flex_replans = 0

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
                stats["flex_replans"] = flex_replans
            return None

        conflict = detect_first_conflict(node["paths"])
        if conflict is None:
            if stats is not None:
                stats["expansions"] = expansions
                stats["flex_replans"] = flex_replans
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

            # Flex: the unused budget of every *other* agent in this node. Used
            # raw (it can be slightly negative, never below what keeps the focal
            # non-empty) so the global w*LB bound holds exactly by construction.
            flex = sum(w * node["lbs"][o] - node["costs"][o]
                       for o in agents if o != agent)
            if flex > 0:
                flex_replans += 1

            start, goal = agents[agent]
            reserved = [node["paths"][o] for o in agents if o != agent]
            path, lb = _focal_low_level(
                grid, start, goal, child_vertex[agent], child_edge[agent],
                reserved, w, highways, flex=flex)
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
        stats["flex_replans"] = flex_replans
    return None
