"""Conflict-Based Search (Sharon et al., 2015).

A two-level optimal MAPF solver. The low level (:func:`plan_path`) plans each
agent independently under a set of constraints. The high level searches a
binary *constraint tree* best-first by sum-of-costs: at each node it looks for
the first conflict between the current paths and, if any, branches into two
children — each adds one constraint resolving the conflict to one of the two
agents — and replans only that agent. The first conflict-free node popped is an
optimal solution.
"""

from __future__ import annotations

import heapq
import itertools

from .conflicts import VertexConflict, detect_first_conflict
from .grid import GridWorld
from .solution import Solution, sum_of_costs
from .space_time_astar import plan_path


def cbs(grid: GridWorld, agents: dict, *, max_expansions: int = 100_000):
    """Solve a MAPF instance optimally (sum-of-costs).

    ``agents`` maps an agent id to a ``(start, goal)`` tuple. Returns a
    :class:`Solution` whose paths are collision-free, or ``None`` if the
    instance is infeasible (or the expansion budget is exhausted).
    """
    # Root: plan each agent with no constraints.
    vertex: dict = {a: frozenset() for a in agents}
    edge: dict = {a: frozenset() for a in agents}
    paths: dict = {}
    for agent, (start, goal) in agents.items():
        path = plan_path(grid, start, goal, vertex[agent], edge[agent])
        if path is None:
            return None
        paths[agent] = path

    counter = itertools.count()
    open_heap = [(sum_of_costs(paths), next(counter), vertex, edge, paths)]

    expansions = 0
    while open_heap:
        cost, _, vertex, edge, paths = heapq.heappop(open_heap)
        expansions += 1
        if expansions > max_expansions:
            return None

        conflict = detect_first_conflict(paths)
        if conflict is None:
            return Solution(paths=dict(paths), cost=cost)

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

        for agent, constraint in branches:
            child_vertex = dict(vertex)
            child_edge = dict(edge)
            if constraint[0] == "v":
                _, cell, time = constraint
                child_vertex[agent] = child_vertex[agent] | {(cell, time)}
            else:
                _, frm, to, time = constraint
                child_edge[agent] = child_edge[agent] | {(frm, to, time)}

            start, goal = agents[agent]
            new_path = plan_path(
                grid, start, goal, child_vertex[agent], child_edge[agent]
            )
            if new_path is None:
                continue
            child_paths = dict(paths)
            child_paths[agent] = new_path
            heapq.heappush(
                open_heap,
                (sum_of_costs(child_paths), next(counter),
                 child_vertex, child_edge, child_paths),
            )

    return None
