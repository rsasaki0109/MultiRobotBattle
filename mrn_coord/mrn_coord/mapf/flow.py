"""Makespan-optimal anonymous MAPF by network flow (Yu & LaValle, AAAI 2013).

A reproduction of the celebrated result from Jingjin Yu & Steven LaValle,
*"Multi-agent Path Planning and Network Flow"* / *"Optimal Multi-Robot Path
Planning on Graphs"* (AAAI 2013; WAFR 2012): when the targets are
*interchangeable* (the **anonymous** / unlabeled problem — any agent may fill any
goal), minimum-**makespan** collision-free routing of ``n`` agents is solvable in
**polynomial time** by reduction to integer **maximum flow** on a time-expanded
network. No constraint-tree search, no priorities — a completely different
paradigm from the CBS family.

The reduction, for a fixed horizon ``T``:

- **Time-expand** the grid: every free cell ``v`` at every step ``t`` becomes a
  pair ``v_in(t) -> v_out(t)`` joined by a **capacity-1** edge. That single edge
  is the whole vertex-collision argument: at most one agent passes through ``v``
  at time ``t``.
- A **wait** is ``v_out(t) -> v_in(t+1)``.
- A **move** along grid edge ``{u, v}`` uses a shared **capacity-1 gadget**
  (``u_out(t), v_out(t) -> mid -> u_in(t+1), v_in(t+1)``). Because both directions
  funnel through the one cap-1 ``mid``, the head-on **swap** ``u->v`` and
  ``v->u`` cannot both happen.
- A super-source feeds every start at ``t=0``; every goal at ``t=T`` drains to a
  super-sink.

A feasible integer flow of value ``n`` then corresponds exactly to ``n``
collision-free trajectories reaching the goal set by time ``T``. Feasibility is
**monotone** in ``T`` (park at the goal to extend), so a binary search finds the
minimum makespan; the optimum is **self-certified** — flow ``= n`` at ``T`` and
flow ``< n`` at ``T-1``.

Honest scope (see ``docs/coordination.md``): this is the *anonymous* (target-
interchangeable) problem Yu & LaValle solve in polynomial time — a relaxation of
labeled MAPF, so its makespan lower-bounds any labeled solution's. The labeled
makespan-optimal problem is NP-hard and is *not* reproduced here.
"""

from __future__ import annotations

from collections import deque

from .grid import GridWorld


# --------------------------------------------------------------------------- #
# A small unit-capacity max-flow (Edmonds-Karp; the value is at most n)        #
# --------------------------------------------------------------------------- #
class _MaxFlow:
    def __init__(self):
        self.adj: dict = {}          # node -> list of edge indices
        self.to: list = []           # edge -> head node
        self.cap: list = []          # edge -> residual capacity
        self.orig: list = []         # edge -> original capacity (0 for back-edges)

    def _node(self, u):
        if u not in self.adj:
            self.adj[u] = []

    def add(self, u, v, c):
        self._node(u)
        self._node(v)
        self.adj[u].append(len(self.to))
        self.to.append(v)
        self.cap.append(c)
        self.orig.append(c)
        self.adj[v].append(len(self.to))
        self.to.append(u)
        self.cap.append(0)
        self.orig.append(0)

    def max_flow(self, s, t):
        flow = 0
        while True:
            parent_edge = {s: -1}
            q = deque([s])
            while q:
                u = q.popleft()
                if u == t:
                    break
                for ei in self.adj[u]:
                    v = self.to[ei]
                    if self.cap[ei] > 0 and v not in parent_edge:
                        parent_edge[v] = ei
                        q.append(v)
            if t not in parent_edge:
                return flow
            # augment by 1 (unit capacities)
            v = t
            while v != s:
                ei = parent_edge[v]
                self.cap[ei] -= 1
                self.cap[ei ^ 1] += 1
                v = self.to[ei ^ 1]
            flow += 1

    def used(self, u, v) -> bool:
        """Was the original edge ``u -> v`` saturated (carries a unit of flow)?"""
        for ei in self.adj.get(u, ()):
            if self.to[ei] == v and self.orig[ei] == 1 and self.cap[ei] == 0:
                return True
        return False


# --------------------------------------------------------------------------- #
# Time-expanded network for a fixed horizon T                                  #
# --------------------------------------------------------------------------- #
def _neighbors4(grid: GridWorld, cell):
    x, y = cell
    return [c for c in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
            if grid.is_free(c)]


def _build(grid: GridWorld, starts, goals, T):
    g = _MaxFlow()
    S, K = ("S",), ("K",)

    free = [(x, y) for x in range(grid.width) for y in range(grid.height)
            if grid.is_free((x, y))]

    for t in range(T + 1):
        for v in free:
            g.add(("in", v, t), ("out", v, t), 1)           # vertex capacity
    for t in range(T):
        for v in free:
            g.add(("out", v, t), ("in", v, t + 1), 1)        # wait
        seen = set()
        for u in free:
            for v in _neighbors4(grid, u):
                key = (min(u, v), max(u, v))
                if key in seen:
                    continue
                seen.add(key)
                a, b = key
                mid = ("mid", a, b, t)
                g.add(("out", a, t), ("midin", a, b, t), 1)
                g.add(("out", b, t), ("midin", a, b, t), 1)
                g.add(("midin", a, b, t), mid, 1)            # shared cap-1
                g.add(mid, ("in", a, t + 1), 1)
                g.add(mid, ("in", b, t + 1), 1)

    for v in starts:
        g.add(S, ("in", v, 0), 1)
    for v in goals:
        g.add(("out", v, T), K, 1)
    return g, S, K


def _extract_paths(g, S, K, starts, goals, T):
    """Decompose the unit flow into one cell-per-timestep path per agent."""
    paths = []
    for _ in starts:
        # find a start whose source edge still carries flow
        start_node = None
        for v in starts:
            if g.used(S, ("in", v, 0)):
                start_node = ("in", v, 0)
                # consume the source edge so we don't reuse this unit
                _consume(g, S, start_node)
                break
        if start_node is None:
            break
        path = [None] * (T + 1)
        node = start_node
        while node != K:
            kind = node[0]
            if kind == "in":
                _, cell, t = node
                path[t] = cell
                nxt = ("out", cell, t)
            elif kind == "out":
                _, cell, t = node
                nxt = _follow(g, node)
            else:  # midin / mid pass-through
                nxt = _follow(g, node)
            _consume(g, node, nxt)
            node = nxt
        paths.append(path)
    return paths


def _follow(g, node):
    for ei in g.adj.get(node, ()):
        if g.orig[ei] == 1 and g.cap[ei] == 0:  # saturated original edge
            return g.to[ei]
    return ("K",)


def _consume(g, u, v):
    for ei in g.adj.get(u, ()):
        if g.to[ei] == v and g.orig[ei] == 1 and g.cap[ei] == 0:
            g.cap[ei] = 1          # release so this unit is not traced twice
            return


def _feasible(grid, starts, goals, T):
    g, S, K = _build(grid, starts, goals, T)
    return g.max_flow(S, K) == len(starts), g, S, K


def anonymous_makespan(grid: GridWorld, starts, goals, *,
                       max_makespan: int | None = None, stats: dict | None = None):
    """Minimum-makespan collision-free routing of interchangeable agents.

    ``starts`` and ``goals`` are equal-length lists of distinct free cells; any
    agent may fill any goal. Returns ``(paths, makespan)`` — ``paths`` a list of
    per-timestep cell lists (length ``makespan + 1``), each starting at some start
    and ending at some goal, collectively a perfect start->goal matching and
    pairwise collision-free — or ``None`` if infeasible within the horizon. If
    ``stats`` is given it records ``stats["makespan"]`` and
    ``stats["certified"]`` (True iff the horizon one below is provably
    infeasible, i.e. the makespan is optimal)."""
    starts = list(starts)
    goals = list(goals)
    if len(starts) != len(goals):
        raise ValueError("starts and goals must have equal length")
    n = len(starts)
    if n == 0:
        return [], 0
    for v in starts + goals:
        if not grid.is_free(v):
            return None

    if max_makespan is None:
        free = sum(1 for x in range(grid.width) for y in range(grid.height)
                   if grid.is_free((x, y)))
        max_makespan = free + n + 2

    # Binary search the minimum feasible horizon (feasibility is monotone in T).
    lo, hi, best = 0, max_makespan, None
    if not _feasible(grid, starts, goals, hi)[0]:
        return None
    while lo <= hi:
        mid = (lo + hi) // 2
        if _feasible(grid, starts, goals, mid)[0]:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1

    T = best
    ok, g, S, K = _feasible(grid, starts, goals, T)
    paths = _extract_paths(g, S, K, starts, goals, T)
    if stats is not None:
        stats["makespan"] = T
        stats["certified"] = (T == 0) or (
            not _feasible(grid, starts, goals, T - 1)[0])
    return paths, T
