"""Conflict-Based Search (Sharon et al., 2015), with disjoint splitting.

A two-level optimal MAPF solver. The low level (:func:`plan_path`) plans each
agent independently under a set of constraints. The high level searches a
binary *constraint tree* best-first by sum-of-costs: at each node it looks for
the first conflict between the current paths and, if any, branches into two
children — each adds one constraint resolving the conflict to one of the two
agents — and replans only that agent. The first conflict-free node popped is an
optimal solution.

**Standard splitting** (the default) resolves a vertex conflict ``(a1, a2, v,
t)`` by forbidding ``v`` at ``t`` to ``a1`` in one child and to ``a2`` in the
other. The two children *overlap*: any solution in which *neither* agent is at
``v`` at ``t`` satisfies both, so it is re-searched in both subtrees — wasted
work.

**Disjoint splitting** (``disjoint=True``, Li et al., ICAPS 2019) picks *one*
agent ``ai`` from the conflict and branches into ``ai`` is at ``v`` at ``t``
(a *positive* constraint) versus ``ai`` is *not* at ``v`` at ``t`` (a negative
constraint). Those two children *partition* the solution space — every solution
satisfies exactly one — so nothing is searched twice. The positive constraint is
sound because vertex occupancy is exclusive: if ``ai`` is at ``v`` at ``t``, no
other agent can be, so the positive child also pins every *other* agent off
``v`` at ``t`` without dropping a single valid solution. Same optimal
sum-of-costs as standard CBS, fewer high-level expansions.
"""

from __future__ import annotations

import heapq
import itertools

from .conflicts import VertexConflict, cell_at, detect_first_conflict
from .grid import GridWorld
from .solution import Solution, sum_of_costs
from .space_time_astar import plan_path


def cbs(grid: GridWorld, agents: dict, *, disjoint: bool = False,
        max_expansions: int = 100_000, stats: dict | None = None):
    """Solve a MAPF instance optimally (sum-of-costs).

    ``agents`` maps an agent id to a ``(start, goal)`` tuple. Returns a
    :class:`Solution` whose paths are collision-free, or ``None`` if the
    instance is infeasible (or the expansion budget is exhausted). With
    ``disjoint=True`` the high level uses disjoint splitting on vertex conflicts
    (positive/negative on one agent) instead of the standard two-negative split;
    it finds the same optimum with fewer expansions. If ``stats`` is given,
    ``stats["expansions"]`` is set to the number of high-level constraint-tree
    nodes expanded — for comparison against standard CBS or bounded-suboptimal
    ECBS (:func:`mrn_coord.mapf.ecbs.ecbs`).
    """
    # Root: plan each agent with no constraints.
    vertex: dict = {a: frozenset() for a in agents}
    edge: dict = {a: frozenset() for a in agents}
    pos_v: dict = {a: frozenset() for a in agents}
    pos_e: dict = {a: frozenset() for a in agents}
    paths: dict = {}
    for agent, (start, goal) in agents.items():
        path = plan_path(grid, start, goal, vertex[agent], edge[agent])
        if path is None:
            return None
        paths[agent] = path

    counter = itertools.count()
    open_heap = [(sum_of_costs(paths), next(counter),
                  vertex, edge, pos_v, pos_e, paths)]

    expansions = 0
    while open_heap:
        cost, _, vertex, edge, pos_v, pos_e, paths = heapq.heappop(open_heap)
        expansions += 1
        if expansions > max_expansions:
            if stats is not None:
                stats["expansions"] = expansions
            return None

        conflict = detect_first_conflict(paths)
        if conflict is None:
            if stats is not None:
                stats["expansions"] = expansions
            return Solution(paths=dict(paths), cost=cost)

        if disjoint and isinstance(conflict, VertexConflict):
            children = _disjoint_vertex_children(
                agents, conflict, vertex, edge, pos_v, pos_e, paths
            )
        else:
            children = _standard_children(
                agents, conflict, vertex, edge, pos_v, pos_e
            )

        for child in children:
            c_vertex, c_edge, c_pos_v, c_pos_e, replan = child
            child_paths = dict(paths)
            ok = True
            for ag in replan:
                start, goal = agents[ag]
                new_path = plan_path(
                    grid, start, goal, c_vertex[ag], c_edge[ag],
                    positive_vertex=c_pos_v[ag], positive_edge=c_pos_e[ag],
                )
                if new_path is None:
                    ok = False
                    break
                child_paths[ag] = new_path
            if not ok:
                continue
            heapq.heappush(
                open_heap,
                (sum_of_costs(child_paths), next(counter),
                 c_vertex, c_edge, c_pos_v, c_pos_e, child_paths),
            )

    if stats is not None:
        stats["expansions"] = expansions
    return None


def _standard_children(agents, conflict, vertex, edge, pos_v, pos_e):
    """The classic two-negative split: each agent is forbidden its half of the
    conflict. Returns a list of ``(vertex, edge, pos_v, pos_e, replan)`` child
    specs, where ``replan`` is the set of agents whose path must be replanned."""
    if isinstance(conflict, VertexConflict):
        branches = [
            (conflict.agent_a, ("v", conflict.cell, conflict.time)),
            (conflict.agent_b, ("v", conflict.cell, conflict.time)),
        ]
    else:  # EdgeConflict — each agent is forbidden its half of the swap.
        branches = [
            (conflict.agent_a,
             ("e", conflict.cell_a, conflict.cell_b, conflict.time)),
            (conflict.agent_b,
             ("e", conflict.cell_b, conflict.cell_a, conflict.time)),
        ]

    children = []
    for agent, constraint in branches:
        c_vertex = dict(vertex)
        c_edge = dict(edge)
        if constraint[0] == "v":
            _, cell, time = constraint
            c_vertex[agent] = c_vertex[agent] | {(cell, time)}
        else:
            _, frm, to, time = constraint
            c_edge[agent] = c_edge[agent] | {(frm, to, time)}
        children.append((c_vertex, c_edge, dict(pos_v), dict(pos_e), {agent}))
    return children


def _disjoint_vertex_children(agents, conflict, vertex, edge, pos_v, pos_e,
                              paths):
    """Disjoint split on a vertex conflict: pick one agent ``ai`` and partition
    the solution space into ``ai`` is / is not at ``(cell, time)``.

    - *positive* child: ``ai`` must occupy ``(cell, time)``; every other agent
      is forbidden it (vertex exclusivity makes this lossless). ``ai`` and every
      other agent whose current path passes through ``(cell, time)`` are
      replanned.
    - *negative* child: ``ai`` is forbidden ``(cell, time)`` — only ``ai`` is
      replanned.
    """
    cell, time = conflict.cell, conflict.time
    ai = conflict.agent_a  # deterministic choice; either agent is sound

    # Positive child.
    pos_vertex = dict(vertex)
    pos_pos_v = dict(pos_v)
    pos_pos_v[ai] = pos_pos_v[ai] | {(cell, time)}
    replan_pos = {ai}
    for ag in agents:
        if ag == ai:
            continue
        pos_vertex[ag] = pos_vertex[ag] | {(cell, time)}
        if cell_at(paths[ag], time) == cell:
            replan_pos.add(ag)
    positive = (pos_vertex, dict(edge), pos_pos_v, dict(pos_e), replan_pos)

    # Negative child.
    neg_vertex = dict(vertex)
    neg_vertex[ai] = neg_vertex[ai] | {(cell, time)}
    negative = (neg_vertex, dict(edge), dict(pos_v), dict(pos_e), {ai})

    return [positive, negative]
