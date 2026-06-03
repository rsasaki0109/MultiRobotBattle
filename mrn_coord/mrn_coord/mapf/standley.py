"""Standley's optimal MAPF: operator decomposition and independence detection.

David Silver's neighbours aside, the two ideas that made *optimal* multi-agent
A* practical come from Trevor Standley, *Finding Optimal Solutions to
Cooperative Pathfinding Problems* (AAAI 2010). Both attack the same enemy — the
joint search whose branching factor is ``b**n`` (``b`` moves each for ``n``
agents) — from opposite ends:

- **Operator decomposition (OD)** — :func:`od_astar`. Instead of moving all
  agents at once (``b**n`` children), assign a move to **one** agent at a time.
  Between two full ("standard") configurations the search passes through
  ``n - 1`` *intermediate* states, each branching only ``b`` ways, so the
  effective branching factor drops from ``b**n`` to ``b``. A move is checked for
  collisions against the agents already assigned in this round, so a hopeless
  partial assignment is pruned long before all ``n`` agents have committed. The
  result is the same optimal sum-of-costs as CBS, reached while *generating* a
  small fraction of the successors a fully coupled joint A* would.

- **Independence detection (ID)** — :func:`independence_detection`. Don't solve
  ``n`` agents together at all if you don't have to. Plan each agent (group)
  on its own; whenever two groups' paths collide, merge them into one group and
  replan that group jointly; repeat until no two groups collide. Independent
  agents are never searched together, so an ``n``-agent instance is solved as a
  handful of small joint searches instead of one huge one. At convergence the
  groups are mutually conflict-free and each is individually optimal, so their
  union is an optimal joint solution — the same optimum CBS returns.

OD gives ID a fast optimal group solver; ID keeps the groups OD must solve
small. Together they are the classic optimal-MAPF workhorse that predates and
motivates CBS.
"""

from __future__ import annotations

import heapq
import itertools

from .cbs import cbs
from .conflicts import detect_first_conflict
from .grid import Cell, GridWorld
from .mstar import _dist_to_goal
from .solution import Solution


def od_astar(grid: GridWorld, agents: dict, *, max_expansions: int = 200_000,
             stats: dict | None = None):
    """Optimal (sum-of-costs) MAPF by A* with operator decomposition.

    ``agents`` maps an id to ``(start, goal)``. Returns a :class:`Solution` with
    the optimal sum-of-costs — the same value :func:`mrn_coord.mapf.cbs.cbs`
    finds — or ``None`` if infeasible or the budget is exhausted. If ``stats`` is
    given: ``stats["expansions"]`` counts standard (full-configuration) nodes
    expanded, ``stats["generated"]`` the successors generated (the branching
    work OD shrinks), and ``stats["max_collision_set"]`` is unused (kept absent).
    """
    ids = list(agents)
    n = len(ids)
    starts = tuple(agents[a][0] for a in ids)
    goals = tuple(agents[a][1] for a in ids)
    dist = [_dist_to_goal(grid, g) for g in goals]
    for i in range(n):
        if dist[i].get(starts[i]) is None:
            return None

    def h_full(config):
        return sum(dist[i][config[i]] for i in range(n))

    # A node is either standard ("S", config, settled) or intermediate
    # ("I", base, settled, assigned, k): agents 0..k-1 have committed their next
    # cell (assigned[j] = (to_cell, settled_flag)), agents k..n-1 still at base.
    INF = float("inf")
    start_node = ("S", starts, frozenset())
    gscore = {start_node: 0}
    parent = {}
    counter = itertools.count()
    open_heap = [(h_full(starts), next(counter), start_node)]

    def h_partial(base, settled, assigned, k):
        tot = 0
        for j in range(k):
            tot += dist[j][assigned[j][0]]
        for j in range(k, n):
            tot += dist[j][base[j]]
        return tot

    expansions = 0
    generated = 0
    while open_heap:
        _, _, node = heapq.heappop(open_heap)
        if node[0] == "S":
            _, config, settled = node
            if config == goals:
                if stats is not None:
                    stats["expansions"] = expansions
                    stats["generated"] = generated
                return _reconstruct_od(parent, node, ids, goals, gscore[node])
            expansions += 1
            if expansions > max_expansions:
                if stats is not None:
                    stats["expansions"] = expansions
                    stats["generated"] = generated
                return None
            base, k, assigned = config, 0, ()
            g_here = gscore[node]
        else:
            _, base, settled, assigned, k = node
            g_here = gscore[node]

        # Assign agent k every legal move; emit k+1 nodes (or a standard node).
        gi = goals[k]
        frm = base[k]
        if k in settled:
            moves = [(gi, True, 0)]
        else:
            on_goal = frm == gi
            moves = []
            for m in grid.neighbors(frm):
                if on_goal and m == gi:
                    moves.append((m, True, 0))
                    moves.append((m, False, 1))
                else:
                    moves.append((m, False, 1))

        for to, flag, cost in moves:
            ok = True
            for j in range(k):
                tj = assigned[j][0]
                if to == tj or (to == base[j] and tj == frm):  # vertex / swap
                    ok = False
                    break
            if not ok:
                continue
            generated += 1
            nassigned = assigned + ((to, flag),)
            ng = g_here + cost
            if k + 1 == n:
                nconfig = tuple(a[0] for a in nassigned)
                nsettled = settled | frozenset(
                    i for i in range(n) if nassigned[i][1])
                child = ("S", nconfig, nsettled)
                if ng < gscore.get(child, INF):
                    gscore[child] = ng
                    parent[child] = node
                    heapq.heappush(open_heap,
                                   (ng + h_full(nconfig), next(counter), child))
            else:
                child = ("I", base, settled, nassigned, k + 1)
                if ng < gscore.get(child, INF):
                    gscore[child] = ng
                    parent[child] = node
                    f = ng + h_partial(base, settled, nassigned, k + 1)
                    heapq.heappush(open_heap, (f, next(counter), child))

    if stats is not None:
        stats["expansions"] = expansions
        stats["generated"] = generated
    return None


def _reconstruct_od(parent, goal_node, ids, goals, cost) -> Solution:
    # Walk parents, keep only standard nodes (full configurations).
    configs = []
    node = goal_node
    while node is not None:
        if node[0] == "S":
            configs.append(node[1])
        node = parent.get(node)
    configs.reverse()
    paths = {}
    for idx, a in enumerate(ids):
        seq = [cfg[idx] for cfg in configs]
        gc = goals[idx]
        while len(seq) > 1 and seq[-1] == gc and seq[-2] == gc:
            seq.pop()
        paths[a] = seq
    return Solution(paths=paths, cost=cost)


def independence_detection(grid: GridWorld, agents: dict, *, solver=None,
                           max_expansions: int = 200_000,
                           stats: dict | None = None):
    """Optimal MAPF by independence detection over a group solver.

    Each agent starts in its own group; groups are solved in isolation by
    ``solver`` (default :func:`od_astar`); whenever two groups' paths collide
    they merge and the merged group is replanned, until no collision remains.
    Returns a :class:`Solution` with the optimal sum-of-costs (== CBS) or
    ``None``. If ``stats`` is given, ``stats["num_groups"]`` is the final group
    count and ``stats["max_group"]`` the largest group ever solved — the coupling
    ID actually paid for (1 ⇒ fully independent).
    """
    if solver is None:
        solver = od_astar
    ids = list(agents)

    # group id -> set of agent ids; agent -> its current path
    groups = {a: {a} for a in ids}
    paths: dict = {}

    def solve_group(members) -> bool:
        sub = {a: agents[a] for a in members}
        sol = solver(grid, sub, max_expansions=max_expansions)
        if sol is None:
            return False
        for a in members:
            paths[a] = sol.paths[a]
        return True

    for a in ids:
        if not solve_group({a}):
            return None

    max_group = 1
    while True:
        conflict = detect_first_conflict(paths)
        if conflict is None:
            break
        ga = groups[conflict.agent_a]
        gb = groups[conflict.agent_b]
        if ga is gb:
            # Two agents of one group collide — its solver should have prevented
            # this; bail rather than loop.
            return None
        merged = ga | gb
        for a in merged:
            groups[a] = merged
        max_group = max(max_group, len(merged))
        if not solve_group(merged):
            return None

    if stats is not None:
        stats["num_groups"] = len({id(groups[a]) for a in ids})
        stats["max_group"] = max_group
    cost = sum(max(0, len(paths[a]) - 1) for a in ids)
    return Solution(paths={a: paths[a] for a in ids}, cost=cost)
