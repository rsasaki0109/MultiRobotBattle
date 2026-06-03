"""EPEA* — Enhanced Partial Expansion A* (Goldenberg et al., JAIR 2014).

Goldenberg, Felner, Stern, Sharon, Sturtevant, Holte & Schaeffer, *"Enhanced
Partial Expansion A\\*"* (JAIR 2014; AAAI 2012), with multi-agent pathfinding as
the showcase domain. EPEA* attacks a specific waste in ordinary A* over the joint
configuration space (:func:`mrn_coord.mapf.mstar.joint_astar`): when a node is
expanded, plain A* generates **all** of its successors and pushes them onto OPEN,
including the many whose ``f`` exceeds the parent's — they just sit there
consuming memory until (if ever) the search reaches their ``f`` level.

**Partial Expansion A\\*** generates only the successors whose ``f`` equals the
node's current ``f``, then re-inserts the node with its ``f`` bumped to the next
larger child ``f`` (its *stored value*), so its higher-``f`` children are
generated later, lazily — and not at all if the search finishes first.

**EPEA\\*** removes the generate-then-discard cost with a domain **Operator
Selection Function (OSF)**: it computes, *without* enumerating them, exactly which
operators yield a child of a given ``f``. For sum-of-costs MAPF the OSF is cheap.
Each agent's move changes ``f`` by a small per-agent amount

    δ_i(move) = (1 if the move is not "wait on goal" else 0)
                + (dist_i(next) - dist_i(cur))

which is ``0`` for a step along a shortest path (or staying on the goal), ``1``
for waiting off-goal, ``2`` for a step away. A joint move's ``Δf`` is the sum of
the per-agent ``δ_i``; so to expand a node at offset ``Δf`` the OSF enumerates
only the operator tuples whose ``δ_i`` sum to ``Δf``, and the next stored value is
the smallest achievable sum greater than ``Δf`` (a sumset of the per-agent ``δ``
sets). The heuristic is the sum of individual true distances (SIC), reusing
:func:`mrn_coord.mapf.mstar._dist_to_goal`; the cost model is identical to
``joint_astar``, so EPEA* returns the **same optimal sum-of-costs as CBS** while
generating far fewer nodes than the fully-expanding joint A* it is built to beat.
"""

from __future__ import annotations

import heapq
import itertools

from .grid import GridWorld
from .mstar import _dist_to_goal
from .solution import Solution


def epea_star(grid: GridWorld, agents: dict, *, max_expansions: int = 200_000,
              stats: dict | None = None):
    """Solve a MAPF instance optimally (sum-of-costs) with EPEA*.

    ``agents`` maps an agent id to ``(start, goal)``. Returns an optimal
    :class:`Solution` (the same cost :func:`mrn_coord.mapf.cbs.cbs` finds), or
    ``None`` if infeasible or the expansion budget is exhausted. If ``stats`` is
    given it records ``expansions`` (node pops, including partial re-expansions),
    ``generated`` (children pushed) and ``partial_reinsertions`` — compare
    ``generated`` against :func:`mrn_coord.mapf.mstar.joint_astar` to see partial
    expansion pay off.
    """
    ids = list(agents)
    n = len(ids)
    starts = tuple(agents[a][0] for a in ids)
    goals = tuple(agents[a][1] for a in ids)
    dist = [_dist_to_goal(grid, g) for g in goals]
    for i in range(n):
        if dist[i].get(starts[i]) is None:
            return None

    def h(cfg):
        return sum(dist[i][cfg[i]] for i in range(n))

    def legal(u, v):
        if len(set(v)) != n:                         # vertex collision
            return False
        for i in range(n):
            for j in range(i + 1, n):
                if u[i] == v[j] and u[j] == v[i] and u[i] != u[j]:
                    return False                     # swap collision
        return True

    def osf(u):
        """Per-agent ``{δ: [next cells]}`` and the set of achievable joint Δf."""
        per_agent = []
        for i in range(n):
            ci = u[i]
            di = dist[i]
            here = di[ci]
            ops: dict = {}
            for c in grid.neighbors(ci):
                dc = di.get(c)
                if dc is None:
                    continue
                cost = 0 if (ci == c == goals[i]) else 1
                delta = cost + dc - here
                ops.setdefault(delta, []).append(c)
            per_agent.append(ops)
        # All achievable joint Δf (sumset of the per-agent δ key sets).
        sums = {0}
        for ops in per_agent:
            sums = {s + d for s in sums for d in ops}
        return per_agent, sums

    def children_at(per_agent, target):
        """Yield child configs whose per-agent δ sum to exactly ``target``."""
        # suffix max δ for pruning
        suff_max = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suff_max[i] = suff_max[i + 1] + max(per_agent[i])

        def rec(i, remaining, acc):
            if i == n:
                if remaining == 0:
                    yield tuple(acc)
                return
            if remaining < 0 or remaining > suff_max[i]:
                return
            for d, cells in per_agent[i].items():
                if d > remaining:
                    continue
                for c in cells:
                    acc.append(c)
                    yield from rec(i + 1, remaining - d, acc)
                    acc.pop()

        yield from rec(0, target, [])

    INF = float("inf")
    g_best = {starts: 0}
    parent: dict = {}
    closed: set = set()
    counter = itertools.count()
    open_heap = [(h(starts), next(counter), starts, 0, 0)]  # f, cnt, cfg, g, dv

    expansions = generated = reinsertions = 0
    while open_heap:
        f, _, u, gu, dv = heapq.heappop(open_heap)
        if u in closed or gu > g_best.get(u, INF):
            continue                                 # fully expanded or stale
        expansions += 1
        if expansions > max_expansions:
            return _finish(stats, expansions, generated, reinsertions, None)

        if u == goals:
            return _finish(stats, expansions, generated, reinsertions,
                           _reconstruct(ids, goals, parent, u, gu))

        per_agent, sums = osf(u)
        for v in children_at(per_agent, dv):
            if not legal(u, v):
                continue
            gv = gu + sum(0 if (u[i] == v[i] == goals[i]) else 1
                          for i in range(n))
            if gv < g_best.get(v, INF):
                g_best[v] = gv
                parent[v] = u
                generated += 1
                heapq.heappush(open_heap,
                               (gv + h(v), next(counter), v, gv, 0))

        nxt = min((s for s in sums if s > dv), default=None)
        if nxt is None:
            closed.add(u)
        else:
            reinsertions += 1
            heapq.heappush(open_heap, (gu + h(u) + nxt, next(counter), u, gu, nxt))

    return _finish(stats, expansions, generated, reinsertions, None)


def _finish(stats, expansions, generated, reinsertions, result):
    if stats is not None:
        stats["expansions"] = expansions
        stats["generated"] = generated
        stats["partial_reinsertions"] = reinsertions
    return result


def _reconstruct(ids, goals, parent, goal_cfg, cost) -> Solution:
    configs = [goal_cfg]
    cur = goal_cfg
    while cur in parent:
        cur = parent[cur]
        configs.append(cur)
    configs.reverse()
    paths: dict = {}
    for idx, a in enumerate(ids):
        seq = [cfg[idx] for cfg in configs]
        gc = goals[idx]
        while len(seq) > 1 and seq[-1] == gc and seq[-2] == gc:
            seq.pop()
        paths[a] = seq
    return Solution(paths=paths, cost=cost)
