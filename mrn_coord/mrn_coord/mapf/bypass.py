"""CBS with bypassing conflicts (BP).

Boyarski, Felner, Sharon, Stern, Tolpin & Shimony, *"Don't Split, Try to Work
It Out: Bypassing Conflicts in Multi-Agent Pathfinding"* (ICAPS 2015), the BP
component of ICBS. Standard Conflict-Based Search, when it picks a conflict to
resolve, always **splits**: it adds a node to the constraint tree for each of the
two agents. BP observes that a split is often unnecessary. When it generates the
two children, it checks whether either child is a **valid bypass** of the current
node ``N``:

- the child has the **same cost** as ``N`` (no agent paid for resolving the
  conflict), and
- the child has **strictly fewer conflicts** than ``N``.

If so, instead of growing the tree, BP simply **adopts the child's new path** for
the replanned agent into ``N`` — *without* recording the constraint that produced
it — and re-examines ``N``. The bypass path is valid under ``N``'s (weaker)
constraints because it was found under *more* constraints, and its cost is
unchanged, so ``g(N)`` and the optimal bound are preserved: BP returns the **same
optimal sum-of-costs as CBS**. Each adoption strictly drops the conflict count,
so a node can only be bypassed finitely often before it is solved or genuinely
split. A cardinal conflict can never be bypassed (both children must gain cost,
so the same-cost test fails) — BP shrinks the tree precisely on the non-cardinal
conflicts that plain CBS wastefully splits.

This reuses the conflict machinery (MDD classification, cardinal-first conflict
choice) from :mod:`mrn_coord.mapf.cbsh`; the plain :func:`mrn_coord.mapf.cbs.cbs`
is left untouched as the baseline.
"""

from __future__ import annotations

import heapq
import itertools

from .cbsh import _all_conflicts, _as_sets, _branches, _choose_conflict
from .grid import GridWorld
from .solution import Solution, sum_of_costs
from .space_time_astar import plan_path


def cbs_bypass(
    grid: GridWorld,
    agents: dict,
    *,
    bypass: bool = True,
    max_expansions: int = 100_000,
    stats: dict | None = None,
):
    """Solve a MAPF instance optimally with CBS + conflict bypassing.

    ``agents`` maps an agent id to ``(start, goal)``. Returns an optimal
    :class:`Solution` (same sum-of-costs as :func:`cbs`) or ``None`` if
    infeasible / the expansion budget is exhausted. With ``bypass=False`` it is
    plain cardinal-prioritized CBS (no adoption) — the ablation baseline these
    numbers are measured against. ``stats`` records ``expansions`` (high-level
    pops), ``bypasses`` (adopted paths) and ``splits``.
    """
    agent_ids = list(agents)
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

    expansions = bypasses = splits = generated = 0
    while open_heap:
        g, _, vertex, edge, paths = heapq.heappop(open_heap)
        expansions += 1
        if expansions > max_expansions:
            return _finish(stats, expansions, bypasses, splits, generated, None)

        # Re-examine this node in place: keep adopting valid bypasses (same cost,
        # fewer conflicts) until none remains, then split. Each adoption avoids
        # generating the two children a standard split would, *without* spending
        # another high-level expansion.
        while True:
            conflict = _choose_conflict(grid, agents, vertex, edge, paths)
            if conflict is None:
                return _finish(stats, expansions, bypasses, splits, generated,
                               Solution(paths=dict(paths), cost=g))

            # Build the two children (replan one agent under one new constraint).
            parent_conflicts = len(_all_conflicts(paths, agent_ids))
            children = []
            for agent, constraint in _branches(conflict):
                add_v, add_e = _as_sets(constraint)
                c_vertex = dict(vertex)
                c_edge = dict(edge)
                if add_v:
                    c_vertex[agent] = c_vertex[agent] | add_v
                if add_e:
                    c_edge[agent] = c_edge[agent] | add_e
                start, goal = agents[agent]
                new_path = plan_path(grid, start, goal,
                                     c_vertex[agent], c_edge[agent])
                if new_path is None:
                    continue
                c_paths = dict(paths)
                c_paths[agent] = new_path
                children.append((agent, new_path, c_vertex, c_edge, c_paths))

            # Bypass: adopt a same-cost, fewer-conflicts child's path into THIS
            # node (constraints unchanged) and re-examine -- no new tree nodes.
            adopted = None
            if bypass:
                for agent, new_path, _, _, c_paths in children:
                    if (sum_of_costs(c_paths) == g
                            and len(_all_conflicts(c_paths, agent_ids))
                            < parent_conflicts):
                        adopted = (agent, new_path)
                        break
            if adopted is not None:
                agent, new_path = adopted
                bypasses += 1
                paths = dict(paths)
                paths[agent] = new_path
                continue

            # No bypass: grow the tree as standard CBS does.
            splits += 1
            for agent, new_path, c_vertex, c_edge, c_paths in children:
                generated += 1
                heapq.heappush(
                    open_heap,
                    (sum_of_costs(c_paths), next(counter),
                     c_vertex, c_edge, c_paths))
            break

    return _finish(stats, expansions, bypasses, splits, generated, None)


def _finish(stats, expansions, bypasses, splits, generated, result):
    if stats is not None:
        stats["expansions"] = expansions
        stats["bypasses"] = bypasses
        stats["splits"] = splits
        stats["generated"] = generated
    return result
