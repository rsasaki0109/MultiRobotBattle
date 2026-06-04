"""CBS-TA: Conflict-Based Search with optimal Target Assignment (Hönig et al.,
ICAPS 2018, "Conflict-Based Search with Optimal Task Assignment").

Plain :func:`cbs <mrn_coord.mapf.cbs.cbs>` is handed *one* goal per agent and
finds the optimal collision-free paths to it. But many problems leave the
target assignment *open*: each agent may serve any goal from a set (a pool of
delivery stations, a team of interchangeable parking bays), and we want the
*jointly* optimal choice — the assignment **and** the paths whose combined
sum-of-costs is least. Solving the assignment first (cheapest matching by
free-space distance) then routing is *not* optimal: the cheapest matching can
force an expensive collision that a slightly-costlier matching avoids.

CBS-TA searches assignments and paths *together*. It keeps CBS's two-level
structure but replaces the single root with a **forest of roots**, one per target
assignment, unfolded lazily in increasing assignment-cost order by **Murty's
K-best algorithm** (:func:`_murty`) over the agent×target distance matrix
(min-cost matching by :func:`hungarian <mrn_coord.lifelong.allocation.hungarian>`).
The first root is the cheapest assignment; only when a root is *expanded* is the
*next*-cheapest assignment materialized as a new root and pushed. Each root plans
every agent to its assigned target with no constraints — so a root's
sum-of-costs equals its assignment cost exactly — and from there ordinary CBS
constraint nodes resolve conflicts. The whole forest is searched best-first by
sum-of-costs, so the first conflict-free node popped is optimal over *both*
assignment and paths.

Why lazy unfolding stays optimal: Murty yields assignments non-decreasing in
cost, a root's node cost equals its assignment cost, and constraints only raise
cost — so any not-yet-materialized assignment costs at least as much as the last
materialized root. A conflict-free root popped at cost ``f`` therefore beats
every assignment still to come. **It interpolates two solvers we already have:**
give each agent a single distinct goal and the forest has one root — byte-for-byte
:func:`cbs`; let all agents share one target pool and CBS-TA is the labeled
sum-of-costs anonymous MAPF optimum, the search cousin of the max-flow
:func:`anonymous_makespan <mrn_coord.mapf.flow.anonymous_makespan>` (which
optimizes makespan) and the min-cost-flow :func:`cbm <mrn_coord.mapf.cbm.cbm>`
(which handles *teams*). Optimal sum-of-costs; pure and deterministic.
"""

from __future__ import annotations

import heapq
import itertools
from collections import deque

from .conflicts import VertexConflict, detect_first_conflict
from .grid import Cell, GridWorld
from .solution import Solution, sum_of_costs
from .space_time_astar import plan_path

INF = float("inf")


def _bfs_dist(grid: GridWorld, source: Cell) -> dict:
    """4-connected BFS distance from ``source`` to every reachable free cell."""
    dist = {source: 0}
    q = deque([source])
    while q:
        cell = q.popleft()
        d = dist[cell]
        for nb in grid.neighbors(cell):
            if nb not in dist:
                dist[nb] = d + 1
                q.append(nb)
    return dist


def _murty(cost):
    """Yield ``(assignment, total_cost)`` pairs in non-decreasing cost order —
    Murty's algorithm for the K-best assignments of a cost matrix.

    ``cost`` is an ``R x C`` matrix (``R <= C``; ``INF`` = forbidden);
    ``assignment`` maps every row to a distinct column (a perfect matching of the
    rows) at finite total cost. After yielding a solution the remaining space is
    partitioned by, for each free edge of that solution in turn, forcing the
    earlier free edges *in* and the current one *out* — so every other assignment
    lands in exactly one child node. Each child's best assignment is recomputed by
    :func:`hungarian` on a matrix with the forced edges baked in. Lazy: the
    generator computes one child only when ``next`` is called.
    """
    # Imported lazily: :mod:`lifelong` imports back from :mod:`mapf`, so a
    # top-level import here would be a circular import while :mod:`mapf` loads.
    from ..lifelong.allocation import hungarian

    R = len(cost)
    C = len(cost[0]) if R else 0

    # `hungarian` assumes a feasible matching exists and loops forever on an
    # all-forbidden row, so never hand it INF. Replace every forbidden cell with
    # a big finite cost (so a real edge is always preferred), solve, then reject
    # the result if it was forced to use a forbidden cell — the big-M trick.
    finite = [v for row in cost for v in row if v < INF]
    BIG = (sum(finite) + 1.0) if finite else 1.0

    def solve(forced_in, forced_out):
        # Bake the node's constraints into the matrix as forbidden cells.
        forbidden = set(forced_out)
        for r, c in forced_in.items():
            for cc in range(C):
                if cc != c:
                    forbidden.add((r, cc))  # row r may only take column c
            for rr in range(R):
                if rr != r:
                    forbidden.add((rr, c))  # column c is reserved for row r
        m = [[BIG if (r, c) in forbidden or cost[r][c] >= INF else cost[r][c]
              for c in range(C)] for r in range(R)]
        a = hungarian(m)
        if len(a) < R:
            return None                     # rows outnumber columns: no matching
        total = 0.0
        for r in range(R):
            if (r, a[r]) in forbidden or cost[r][a[r]] >= INF:
                return None                 # forced onto a forbidden cell
            total += cost[r][a[r]]
        return (a, total)

    first = solve({}, set())
    if first is None:
        return
    counter = itertools.count()
    pq = [(first[1], next(counter), first[0], {}, frozenset())]
    while pq:
        total, _, assign, fin, fout = heapq.heappop(pq)
        yield (assign, total)
        # Partition: walk this assignment's free edges, forcing each successive
        # one out while pinning the earlier ones in.
        cur_in = dict(fin)
        for r in sorted(assign):
            if r in fin:
                continue
            c = assign[r]
            child_out = fout | {(r, c)}
            child = solve(cur_in, child_out)
            if child is not None:
                heapq.heappush(
                    pq, (child[1], next(counter), child[0], dict(cur_in), child_out)
                )
            cur_in[r] = c                   # later children keep this edge


def cbs_ta(grid: GridWorld, agents: dict, *,
           max_expansions: int = 100_000, stats: dict | None = None):
    """Solve a MAPF instance with open target assignment, optimally.

    ``agents`` maps an agent id to ``(start, goals)`` where ``goals`` is an
    iterable of candidate target cells the agent may be assigned (give a
    single-element iterable to pin a goal — then this degenerates to :func:`cbs`).
    Returns a :class:`Solution` whose paths are collision-free and whose combined
    assignment+path sum-of-costs is minimal, or ``None`` if infeasible (no perfect
    assignment, or the expansion budget is exhausted).

    If ``stats`` is given, ``stats["expansions"]`` counts high-level nodes
    expanded and ``stats["roots"]`` counts assignments materialized (how far into
    Murty's K-best sequence the search had to reach).
    """
    ids = sorted(agents)
    starts = {a: agents[a][0] for a in ids}
    cand = {a: list(dict.fromkeys(agents[a][1])) for a in ids}   # de-dup, ordered
    for a in ids:
        if not grid.is_free(starts[a]):
            return None

    # Agent×target distance matrix; a target an agent can't serve / reach is INF.
    targets = sorted({t for a in ids for t in cand[a]})
    tindex = {t: j for j, t in enumerate(targets)}
    dist = {a: _bfs_dist(grid, starts[a]) for a in ids}
    cost = [[INF] * len(targets) for _ in ids]
    for i, a in enumerate(ids):
        di = dist[a]
        for t in cand[a]:
            if grid.is_free(t) and t in di:
                cost[i][tindex[t]] = float(di[t])

    gen = _murty(cost)

    counter = itertools.count()
    open_heap: list = []
    roots = 0
    expansions = 0

    def push_root():
        """Materialize the next-cheapest assignment as a CBS root, or do nothing
        when Murty is exhausted. Returns ``True`` if a root was pushed."""
        nonlocal roots
        nxt = next(gen, None)
        if nxt is None:
            return False
        assign, _ = nxt
        goals = {ids[i]: targets[assign[i]] for i in range(len(ids))}
        vertex = {a: frozenset() for a in ids}
        edge = {a: frozenset() for a in ids}
        pos_v = {a: frozenset() for a in ids}
        pos_e = {a: frozenset() for a in ids}
        paths = {}
        for a in ids:
            path = plan_path(grid, starts[a], goals[a], vertex[a], edge[a])
            if path is None:
                return push_root()          # this assignment is unroutable; skip
        for a in ids:                        # re-plan kept simple & explicit
            paths[a] = plan_path(grid, starts[a], goals[a], vertex[a], edge[a])
        roots += 1
        heapq.heappush(open_heap, (
            sum_of_costs(paths), next(counter), True,
            goals, vertex, edge, pos_v, pos_e, paths,
        ))
        return True

    if not push_root():
        return None                          # no feasible assignment at all

    while open_heap:
        cost_n, _, is_root, goals, vertex, edge, pos_v, pos_e, paths = \
            heapq.heappop(open_heap)
        expansions += 1
        if expansions > max_expansions:
            if stats is not None:
                stats["expansions"], stats["roots"] = expansions, roots
            return None

        if is_root:
            push_root()                      # unfold the next assignment lazily

        conflict = detect_first_conflict(paths)
        if conflict is None:
            if stats is not None:
                stats["expansions"], stats["roots"] = expansions, roots
            return Solution(paths=dict(paths), cost=cost_n)

        for child in _children(conflict, vertex, edge):
            c_vertex, c_edge, ag = child
            new_path = plan_path(grid, starts[ag], goals[ag],
                                 c_vertex[ag], c_edge[ag])
            if new_path is None:
                continue
            child_paths = dict(paths)
            child_paths[ag] = new_path
            heapq.heappush(open_heap, (
                sum_of_costs(child_paths), next(counter), False,
                goals, c_vertex, c_edge, pos_v, pos_e, child_paths,
            ))

    if stats is not None:
        stats["expansions"], stats["roots"] = expansions, roots
    return None


def _children(conflict, vertex, edge):
    """Standard two-negative CBS split (assignment is fixed within a node, so no
    target reassignment here — that lives entirely in the root forest). Returns
    ``(vertex, edge, replanned_agent)`` specs."""
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
    out = []
    for agent, c in branches:
        c_vertex = dict(vertex)
        c_edge = dict(edge)
        if c[0] == "v":
            _, cell, time = c
            c_vertex[agent] = c_vertex[agent] | {(cell, time)}
        else:
            _, frm, to, time = c
            c_edge[agent] = c_edge[agent] | {(frm, to, time)}
        out.append((c_vertex, c_edge, agent))
    return out
