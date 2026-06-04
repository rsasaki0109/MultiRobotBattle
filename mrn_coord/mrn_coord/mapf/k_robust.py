"""k-robust CBS: plans that survive delays (Atzmon et al., 2018).

A reproduction of Atzmon, Stern, Felner, Wagner, Barták & Zhou, *"Robust
Multi-Agent Path Finding"* (SoCS 2018; extended in JAIR 2020). A plan executed by
real robots rarely runs on the planned clock — a robot stalls, a wheel slips, a
sensor blocks for a tick. A **k-robust** plan is one that stays collision-free as
long as no agent is delayed more than ``k`` timesteps in total: it leaves a
``k``-step *buffer* at every shared cell.

Formally, a solution is k-robust iff for every pair of agents ``i != j`` and
every cell ``v``, if ``i`` is at ``v`` at time ``t`` and ``j`` is at ``v`` at
time ``t'`` then ``|t - t'| > k``. With ``k = 0`` that is ordinary
collision-freedom; with ``k = 1`` no two agents may use the same cell on
consecutive steps, so a one-step delay can never make them collide.

This is built as the smallest change to :func:`mrn_coord.mapf.cbs.cbs`:

- **Conflict detection** looks for a *k-delay* vertex conflict — two agents at the
  same cell within ``k`` steps of each other — in addition to the ordinary swap
  (edge) conflict. The k-delay test also catches a head-on swap *that would
  become a vertex collision under a delay*: the two agents occupy each other's
  cell one step apart, which is within ``k`` for ``k >= 1``.
- **Resolution** is the standard CBS split. A k-delay conflict ``(i@v@t_i,
  j@v@t_j)`` branches into "``i`` not at ``v`` at ``t_i``" versus "``j`` not at
  ``v`` at ``t_j``" — single negative vertex constraints that partition every
  k-robust solution (one of the two occupancies must go). Swaps split on edges as
  usual.

Because the low level and cost model are unchanged, the result is the optimal
(minimum sum-of-costs) k-robust plan. At ``k = 0`` the detection and the split
are byte-for-byte the standard ones, so ``k_robust_cbs(grid, agents, k=0)``
expands the same tree and returns the same solution as :func:`cbs`. Raising ``k``
buys robustness at a (monotone non-decreasing) cost in path length.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

from .conflicts import EdgeConflict, cell_at
from .grid import GridWorld
from .solution import Solution, sum_of_costs
from .space_time_astar import plan_path


@dataclass(frozen=True)
class KVertexConflict:
    """Agents ``agent_a`` and ``agent_b`` use ``cell`` within ``k`` steps:
    ``agent_a`` at ``time_a`` and ``agent_b`` at ``time_b``."""

    agent_a: object
    agent_b: object
    cell: object
    time_a: int
    time_b: int


def detect_first_k_conflict(paths: dict, k: int):
    """Earliest k-delay vertex conflict or swap between any pair, or ``None``.

    Scans time forward; within a timestep checks every agent pair. A k-delay
    vertex conflict is reported when two agents occupy the same cell at times
    within ``k`` of each other (the earlier occupancy time orders the scan). Swap
    (edge) conflicts are reported exactly as in ordinary CBS. With ``k = 0`` this
    is identical to :func:`mrn_coord.mapf.conflicts.detect_first_conflict`.
    """
    agents = list(paths)
    if len(agents) < 2:
        return None
    horizon = max(len(p) for p in paths.values()) + k

    for t in range(horizon):
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                ca = cell_at(paths[a], t)
                # k-delay vertex: a at t, b at some t' in [t-k, t+k]. We anchor on
                # a's time t and look at b within the window, taking the earliest.
                for dt in range(-k, k + 1):
                    tb = t + dt
                    if tb < 0:
                        continue
                    if ca == cell_at(paths[b], tb):
                        # order so the conflict is deterministic and minimal
                        return KVertexConflict(a, b, ca, t, tb)
        # swap conflicts between t and t+1 (k=0 collision, always checked)
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                a_t, a_t1 = cell_at(paths[a], t), cell_at(paths[a], t + 1)
                b_t, b_t1 = cell_at(paths[b], t), cell_at(paths[b], t + 1)
                if a_t != a_t1 and a_t == b_t1 and a_t1 == b_t:
                    return EdgeConflict(a, b, a_t, a_t1, t + 1)

    return None


def k_robust_cbs(grid: GridWorld, agents: dict, *, k: int = 1,
                 max_expansions: int = 100_000, stats: dict | None = None):
    """Optimal (sum-of-costs) k-robust MAPF solution, or ``None`` if infeasible.

    ``agents`` maps an agent id to ``(start, goal)``. ``k`` is the delay the plan
    must tolerate: the returned paths stay collision-free as long as no agent is
    delayed by more than ``k`` steps. ``k = 0`` reproduces plain
    :func:`mrn_coord.mapf.cbs.cbs`. ``stats["expansions"]`` records the
    high-level nodes expanded.
    """
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
            if stats is not None:
                stats["expansions"] = expansions
            return None

        conflict = detect_first_k_conflict(paths, k)
        if conflict is None:
            if stats is not None:
                stats["expansions"] = expansions
            return Solution(paths=dict(paths), cost=cost)

        if isinstance(conflict, KVertexConflict):
            branches = [
                (conflict.agent_a, ("v", conflict.cell, conflict.time_a)),
                (conflict.agent_b, ("v", conflict.cell, conflict.time_b)),
            ]
        else:  # EdgeConflict — swap, split on directed edges
            branches = [
                (conflict.agent_a,
                 ("e", conflict.cell_a, conflict.cell_b, conflict.time)),
                (conflict.agent_b,
                 ("e", conflict.cell_b, conflict.cell_a, conflict.time)),
            ]

        for agent, constraint in branches:
            c_vertex = dict(vertex)
            c_edge = dict(edge)
            if constraint[0] == "v":
                _, cell, time = constraint
                c_vertex[agent] = c_vertex[agent] | {(cell, time)}
            else:
                _, frm, to, time = constraint
                c_edge[agent] = c_edge[agent] | {(frm, to, time)}
            start, goal = agents[agent]
            new_path = plan_path(grid, start, goal, c_vertex[agent], c_edge[agent])
            if new_path is None:
                continue
            child_paths = dict(paths)
            child_paths[agent] = new_path
            heapq.heappush(
                open_heap,
                (sum_of_costs(child_paths), next(counter),
                 c_vertex, c_edge, child_paths),
            )

    if stats is not None:
        stats["expansions"] = expansions
    return None
