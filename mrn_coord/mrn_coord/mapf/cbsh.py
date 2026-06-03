"""CBS with improved heuristics and conflict prioritization (CBSH).

Li et al., *"Improved Heuristics for Multi-Agent Path Finding with Conflict-
Based Search"* (IJCAI 2019), plus the conflict prioritization of Boyarski et
al.'s ICBS (2015). This is the same optimal constraint-tree search as
:func:`mrn_coord.mapf.cbs.cbs` — it returns the identical optimal sum-of-costs
— but it expands far fewer high-level nodes by:

1. **An admissible high-level heuristic** ``h(N)`` added to ``g(N)``. Plain CBS
   orders OPEN by ``g`` alone (``h = 0``); the conflicts in a node carry
   information about how much more cost is unavoidable, and ``h`` extracts it.
   Three strengths, selectable via ``heuristic``:

   - ``"cg"``  — minimum vertex cover of the *cardinal* conflict graph.
   - ``"dg"``  — minimum vertex cover of the *dependency* graph (a superset).
   - ``"wdg"`` — weighted minimum vertex cover of the dependency graph, edges
     weighted by the true pairwise cost increase. The tightest, and the
     default.

2. **Conflict prioritization** — when it must branch, it splits on a *cardinal*
   conflict if one exists (both children then provably gain cost), else a
   semi-cardinal, else any. ``heuristic=None`` gives this prioritization with
   no ``h`` — a clean ablation isolating the two ideas.

The machinery (MDDs, conflict classification, dependency, MVC/WMVC) lives in
:mod:`mrn_coord.mapf.mdd`. The plain :func:`cbs` is left untouched as the
baseline these numbers are measured against.
"""

from __future__ import annotations

import heapq
import itertools

from .conflicts import EdgeConflict, VertexConflict, cell_at, detect_first_conflict
from .grid import GridWorld
from .mdd import (
    CARDINAL,
    are_dependent,
    build_mdd,
    classify_edge,
    classify_vertex,
    min_vertex_cover,
    weighted_min_vertex_cover,
)
from .rectangle import find_rectangle_barriers
from .solution import Solution, sum_of_costs
from .space_time_astar import plan_path


def cbsh(
    grid: GridWorld,
    agents: dict,
    *,
    heuristic: str | None = "wdg",
    rectangle: bool = False,
    max_expansions: int = 100_000,
    stats: dict | None = None,
):
    """Solve a MAPF instance optimally with improved heuristics.

    ``agents`` maps an agent id to ``(start, goal)``. ``heuristic`` is one of
    ``"cg"``, ``"dg"``, ``"wdg"`` (default), or ``None`` for prioritization
    only. Returns an optimal :class:`Solution` or ``None`` if infeasible / the
    expansion budget is exhausted. ``stats["expansions"]`` is set to the number
    of high-level nodes expanded — directly comparable to
    :func:`mrn_coord.mapf.cbs.cbs`'s own ``expansions``.

    With ``rectangle=True`` (off by default), the high level resolves a
    **rectangle symmetry** — two agents crossing the same open region whose every
    pair of optimal paths collides — with a single *barrier* split instead of the
    standard one-cell vertex split, collapsing the symmetric permutations CBS
    would otherwise enumerate (Li et al. 2019; :mod:`mrn_coord.mapf.rectangle`).
    The optimum is unchanged; ``stats["rectangles"]`` counts the barrier splits.
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

    pairwise_memo: dict = {}
    counter = itertools.count()
    root_g = sum_of_costs(paths)
    root_h = _heuristic(grid, agents, agent_ids, vertex, edge, paths,
                        heuristic, pairwise_memo)
    open_heap = [(root_g + root_h, root_g, next(counter), vertex, edge, paths)]

    expansions = 0
    rectangles = 0
    while open_heap:
        _, g, _, vertex, edge, paths = heapq.heappop(open_heap)
        expansions += 1
        if expansions > max_expansions:
            return _finish(stats, expansions, rectangles, None)

        # A rectangle symmetry, when present and enabled, is split by barriers;
        # otherwise fall back to the standard cardinal-first single-cell split.
        rect = None
        if rectangle:
            rect = _choose_rectangle(grid, agents, agent_ids, vertex, edge, paths)
        if rect is None:
            conflict = _choose_conflict(grid, agents, vertex, edge, paths)
            if conflict is None:
                return _finish(stats, expansions, rectangles,
                               Solution(paths=dict(paths), cost=g))
            branches = [(agent, *_as_sets(constraint))
                        for agent, constraint in _branches(conflict)]
        else:
            agent_a, barrier_a, agent_b, barrier_b = rect
            rectangles += 1
            branches = [(agent_a, barrier_a, frozenset()),
                        (agent_b, barrier_b, frozenset())]

        for agent, add_vertex, add_edge in branches:
            child_vertex = dict(vertex)
            child_edge = dict(edge)
            if add_vertex:
                child_vertex[agent] = child_vertex[agent] | add_vertex
            if add_edge:
                child_edge[agent] = child_edge[agent] | add_edge

            start, goal = agents[agent]
            new_path = plan_path(
                grid, start, goal, child_vertex[agent], child_edge[agent]
            )
            if new_path is None:
                continue
            child_paths = dict(paths)
            child_paths[agent] = new_path
            child_g = sum_of_costs(child_paths)
            child_h = _heuristic(grid, agents, agent_ids, child_vertex,
                                child_edge, child_paths, heuristic, pairwise_memo)
            heapq.heappush(
                open_heap,
                (child_g + child_h, child_g, next(counter),
                 child_vertex, child_edge, child_paths),
            )

    return _finish(stats, expansions, rectangles, None)


def _finish(stats, expansions, rectangles, result):
    if stats is not None:
        stats["expansions"] = expansions
        stats["rectangles"] = rectangles
    return result


def _as_sets(constraint):
    """A standard single constraint as ``(vertex_set, edge_set)`` to add."""
    if constraint[0] == "v":
        _, cell, time = constraint
        return frozenset({(cell, time)}), frozenset()
    _, frm, to, time = constraint
    return frozenset(), frozenset({(frm, to, time)})


def _choose_rectangle(grid, agents, agent_ids, vertex, edge, paths):
    """The best rectangle symmetry among the node's vertex conflicts, as a
    barrier split ``(agent_a, barrier_a, agent_b, barrier_b)``, or ``None``.

    Only semi/cardinal rectangles (``klass >= 1``) are taken — a non-cardinal one
    need not reduce the search. Ties break toward higher type, then more barrier
    cells (a larger swept rectangle)."""
    conflicts = _all_conflicts(paths, agent_ids)
    if not conflicts:
        return None
    get_mdd = _mdd_provider(grid, agents, vertex, edge, paths)
    best = None
    best_key = None
    for _, _, conflict in conflicts:
        if not isinstance(conflict, VertexConflict):
            continue
        mdd_a = get_mdd(conflict.agent_a)
        mdd_b = get_mdd(conflict.agent_b)
        if mdd_a is None or mdd_b is None:
            continue
        found = find_rectangle_barriers(mdd_a, mdd_b, conflict.time)
        if found is None:
            continue
        barrier_a, barrier_b, klass = found
        if klass < 1:
            continue
        key = (klass, len(barrier_a) + len(barrier_b))
        if best_key is None or key > best_key:
            best_key = key
            best = (conflict.agent_a, barrier_a, conflict.agent_b, barrier_b)
    return best


def _branches(conflict):
    """The two constraints a conflict splits into (same vocabulary as CBS)."""
    if isinstance(conflict, VertexConflict):
        return [
            (conflict.agent_a, ("v", conflict.cell, conflict.time)),
            (conflict.agent_b, ("v", conflict.cell, conflict.time)),
        ]
    return [
        (conflict.agent_a, ("e", conflict.cell_a, conflict.cell_b, conflict.time)),
        (conflict.agent_b, ("e", conflict.cell_b, conflict.cell_a, conflict.time)),
    ]


def _all_conflicts(paths, agent_ids):
    """Every pairwise conflict in ``paths``, in (time, pair-index) order.

    Yields ``(klass_inputs, conflict)`` where ``klass_inputs`` carries what the
    classifier needs. Scans like :func:`detect_first_conflict` but does not stop
    at the first hit — CBSH needs to see them all to build its conflict graph.
    """
    n = len(agent_ids)
    horizon = max((len(paths[a]) for a in agent_ids), default=0)
    out = []
    for t in range(horizon):
        for i in range(n):
            for j in range(i + 1, n):
                a, b = agent_ids[i], agent_ids[j]
                pa, pb = paths[a], paths[b]
                if cell_at(pa, t) == cell_at(pb, t):
                    out.append((i, j, VertexConflict(a, b, cell_at(pa, t), t)))
                    continue
                a_t, a_t1 = cell_at(pa, t), cell_at(pa, t + 1)
                b_t, b_t1 = cell_at(pb, t), cell_at(pb, t + 1)
                if a_t != a_t1 and a_t == b_t1 and a_t1 == b_t:
                    out.append(
                        (i, j, EdgeConflict(a, b, a_t, a_t1, t + 1)))
    return out


def _mdd_provider(grid, agents, vertex, edge, paths):
    """Lazily build and cache the MDD of each agent at the current node."""
    cache: dict = {}

    def get(agent):
        if agent not in cache:
            start, goal = agents[agent]
            cost = len(paths[agent]) - 1
            cache[agent] = build_mdd(
                grid, start, goal, cost, vertex[agent], edge[agent])
        return cache[agent]

    return get


def _conflict_class(conflict, get_mdd):
    mdd_a = get_mdd(conflict.agent_a)
    mdd_b = get_mdd(conflict.agent_b)
    if mdd_a is None or mdd_b is None:
        return 0  # cannot prove anyone is pinned -> treat as non-cardinal
    if isinstance(conflict, VertexConflict):
        return classify_vertex(mdd_a, mdd_b, conflict.time)
    return classify_edge(mdd_a, mdd_b, conflict.time)


def _choose_conflict(grid, agents, vertex, edge, paths):
    """Pick the conflict to split on: the earliest of the highest class.

    Cardinal conflicts first (both children must gain cost), then
    semi-cardinal, then non-cardinal. Returns ``None`` if conflict-free.
    """
    agent_ids = list(agents)
    conflicts = _all_conflicts(paths, agent_ids)
    if not conflicts:
        return None
    get_mdd = _mdd_provider(grid, agents, vertex, edge, paths)
    best = None
    best_key = None
    for order, (i, j, conflict) in enumerate(conflicts):
        klass = _conflict_class(conflict, get_mdd)
        t = conflict.time
        key = (-klass, t, i, j)
        if best_key is None or key < best_key:
            best_key = key
            best = conflict
            if klass == CARDINAL and t == 0:
                break  # cannot do better than an earliest cardinal
    return best


def _heuristic(grid, agents, agent_ids, vertex, edge, paths, mode, pairwise_memo):
    """Admissible high-level heuristic ``h(N)`` for the given node."""
    if mode is None:
        return 0
    conflicts = _all_conflicts(paths, agent_ids)
    if not conflicts:
        return 0
    get_mdd = _mdd_provider(grid, agents, vertex, edge, paths)

    if mode == "cg":
        # Edge per pair that has at least one cardinal conflict.
        cardinal_pairs = set()
        for i, j, conflict in conflicts:
            if _conflict_class(conflict, get_mdd) == CARDINAL:
                cardinal_pairs.add((agent_ids[i], agent_ids[j]))
        if not cardinal_pairs:
            return 0
        verts = {a for pair in cardinal_pairs for a in pair}
        return min_vertex_cover(list(verts), list(cardinal_pairs))

    # DG / WDG both need the dependency relation over conflicting pairs.
    conflicting_pairs = {(i, j) for i, j, _ in conflicts}
    dep_pairs = []
    for i, j in sorted(conflicting_pairs):
        a, b = agent_ids[i], agent_ids[j]
        mdd_a, mdd_b = get_mdd(a), get_mdd(b)
        if mdd_a is None or mdd_b is None:
            continue
        if are_dependent(grid, mdd_a, mdd_b, agents[a][0], agents[b][0],
                         edge[a], edge[b]):
            dep_pairs.append((a, b))
    if not dep_pairs:
        return 0

    if mode == "dg":
        verts = {a for pair in dep_pairs for a in pair}
        return min_vertex_cover(list(verts), dep_pairs)

    if mode == "wdg":
        weighted = []
        for a, b in dep_pairs:
            w = _pairwise_delta(grid, agents, a, b, vertex, edge, paths,
                               pairwise_memo)
            if w > 0:
                weighted.append((a, b, w))
        if not weighted:
            return 0
        verts = {x for a, b, _ in weighted for x in (a, b)}
        return weighted_min_vertex_cover(list(verts), weighted)

    raise ValueError(f"unknown heuristic {mode!r}")


def _pairwise_delta(grid, agents, a, b, vertex, edge, paths, memo):
    """Extra cost to *jointly* resolve agents ``a`` and ``b`` beyond their
    current individual costs — the WDG edge weight.

    Solves the 2-agent MAPF for ``a`` and ``b`` under their current node
    constraints (a small CBS), and subtracts their current path costs. ``>= 0``;
    a positive value is a lower bound on the cost-to-go those two contribute.
    """
    key = (a, b, vertex[a], edge[a], vertex[b], edge[b])
    if key in memo:
        return memo[key]
    cur = (len(paths[a]) - 1) + (len(paths[b]) - 1)
    opt = _two_agent_optimal(
        grid, agents[a], agents[b],
        vertex[a], edge[a], vertex[b], edge[b])
    delta = 0 if opt is None else max(0, opt - cur)
    memo[key] = delta
    return delta


def _two_agent_optimal(grid, sg_a, sg_b, va, ea, vb, eb, *, max_expansions=4000):
    """Optimal sum-of-costs for two agents under initial constraints (small CBS)."""
    pa = plan_path(grid, sg_a[0], sg_a[1], va, ea)
    pb = plan_path(grid, sg_b[0], sg_b[1], vb, eb)
    if pa is None or pb is None:
        return None
    counter = itertools.count()
    paths = {0: pa, 1: pb}
    cons_v = {0: va, 1: vb}
    cons_e = {0: ea, 1: eb}
    sg = {0: sg_a, 1: sg_b}
    heap = [(sum_of_costs(paths), next(counter), cons_v, cons_e, paths)]
    exp = 0
    while heap:
        cost, _, cons_v, cons_e, paths = heapq.heappop(heap)
        exp += 1
        if exp > max_expansions:
            return None
        conflict = detect_first_conflict(paths)
        if conflict is None:
            return cost
        for agent, constraint in _branches(conflict):
            cv = dict(cons_v)
            ce = dict(cons_e)
            if constraint[0] == "v":
                _, cell, time = constraint
                cv[agent] = cv[agent] | {(cell, time)}
            else:
                _, frm, to, time = constraint
                ce[agent] = ce[agent] | {(frm, to, time)}
            np_ = plan_path(grid, sg[agent][0], sg[agent][1], cv[agent], ce[agent])
            if np_ is None:
                continue
            cp = dict(paths)
            cp[agent] = np_
            heapq.heappush(
                heap, (sum_of_costs(cp), next(counter), cv, ce, cp))
    return None
