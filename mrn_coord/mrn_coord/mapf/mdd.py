"""Multi-valued decision diagrams and the admissible CBS heuristics.

This module is the machinery behind :func:`mrn_coord.mapf.cbsh.cbsh` — the
improved-heuristic Conflict-Based Search of Li et al., *"Improved Heuristics
for Multi-Agent Path Finding with Conflict-Based Search"* (IJCAI 2019). Plain
CBS (:func:`mrn_coord.mapf.cbs.cbs`) orders its constraint tree by ``g`` (the
sum-of-costs) alone; CBSH adds an admissible ``h`` derived from the structure
of the conflicts, and prioritizes which conflict to split on.

The unit of structure is the **MDD** (multi-valued decision diagram): for an
agent whose shortest path under its current constraints costs ``c``, the MDD is
the union of *all* cost-``c`` paths, laid out by timestep. Its width at level
``t`` — the number of cells the agent could occupy at time ``t`` without
leaving an optimal path — is what tells us how much freedom the agent has, and
hence how expensive a conflict is to resolve:

- **cardinal** conflict — both agents are pinned (width 1) at the conflict, so
  *both* children of the split must increase cost. A guaranteed ``+1`` (at
  least) on every resolution.
- **semi-cardinal** — exactly one agent is pinned; one child must pay.
- **non-cardinal** — neither is pinned; the conflict may resolve for free.

From these we build three increasingly tight heuristics, all admissible:

- **CG** (conflict graph) — an edge per pair with a cardinal conflict; ``h`` is
  the minimum vertex cover (each edge forces ``>= 1``).
- **DG** (dependency graph) — an edge per *dependent* pair (their joint MDD has
  no conflict-free pair of optimal paths); ``h`` is the minimum vertex cover.
- **WDG** (weighted dependency graph) — edges weighted by the true pairwise
  cost increase; ``h`` is the weighted minimum vertex cover.

The MVC/WMVC solvers here are exact (branch and bound); the graphs that arise
are tiny (only agents currently in conflict), with a matching lower bound as a
safe admissible fallback if one is ever large.
"""

from __future__ import annotations

from .grid import GridWorld


class Mdd:
    """The cost-``cost`` MDD of one agent: ``levels[t]`` is the set of cells it
    could occupy at time ``t`` on some optimal (length-``cost``) path."""

    __slots__ = ("levels", "goal", "cost")

    def __init__(self, levels: list[set], goal, cost: int) -> None:
        self.levels = levels
        self.goal = goal
        self.cost = cost

    def width(self, t: int) -> int:
        """Number of cells reachable at time ``t``. Past ``cost`` the agent
        waits at its goal, so the width is 1."""
        if t <= self.cost:
            return len(self.levels[t])
        return 1

    def cells(self, t: int) -> set:
        """The cells reachable at time ``t`` (``{goal}`` past ``cost``)."""
        if t <= self.cost:
            return self.levels[t]
        return {self.goal}


def build_mdd(
    grid: GridWorld,
    start,
    goal,
    cost: int,
    vertex_constraints=frozenset(),
    edge_constraints=frozenset(),
) -> Mdd | None:
    """Build the MDD of all cost-``cost`` paths from ``start`` to ``goal``.

    Respects the same ``(cell, time)`` / ``(frm, to, time)`` constraint
    vocabulary as :func:`mrn_coord.mapf.space_time_astar.plan_path`. Returns
    ``None`` if no length-``cost`` path exists (which should not happen when
    ``cost`` is taken from a real path the planner already found).
    """
    # Forward reachability: forward[t] = cells reachable at t from the start.
    forward = [set() for _ in range(cost + 1)]
    if (start, 0) in vertex_constraints:
        return None
    forward[0].add(start)
    for t in range(cost):
        nxt = forward[t + 1]
        for cell in forward[t]:
            for ncell in grid.neighbors(cell):
                if (ncell, t + 1) in vertex_constraints:
                    continue
                if (cell, ncell, t + 1) in edge_constraints:
                    continue
                nxt.add(ncell)
    if goal not in forward[cost]:
        return None

    # Backward reachability: cells from which the goal is reachable exactly at
    # time ``cost``. A cell v is in backward[t] iff some neighbour w it can
    # legally step to lives in backward[t+1].
    backward = [set() for _ in range(cost + 1)]
    backward[cost].add(goal)
    for t in range(cost - 1, -1, -1):
        here = backward[t]
        for w in backward[t + 1]:
            if (w, t + 1) in vertex_constraints:
                continue
            for v in grid.neighbors(w):  # predecessors of w (grid is undirected)
                if (v, t) in vertex_constraints:
                    continue
                if (v, w, t + 1) in edge_constraints:
                    continue
                here.add(v)

    levels = [forward[t] & backward[t] for t in range(cost + 1)]
    if any(not lvl for lvl in levels):
        return None
    return Mdd(levels, goal, cost)


def are_dependent(
    grid: GridWorld,
    mdd_a: Mdd,
    mdd_b: Mdd,
    start_a,
    start_b,
    edge_a=frozenset(),
    edge_b=frozenset(),
) -> bool:
    """Are two agents *dependent* — is there NO pair of optimal paths (one in
    each MDD) that avoids every vertex and swap conflict?

    Searched as a reachability problem over the joint MDD: a state is the pair
    of cells ``(va, vb)`` at a timestep, transitions step each agent along its
    own MDD, and we forbid the two from sharing a cell or swapping. If the
    horizon ``max(cost_a, cost_b)`` is reachable with both agents parked at
    their goals, they are independent; otherwise dependent.
    """
    horizon = max(mdd_a.cost, mdd_b.cost)

    def succ(mdd: Mdd, edge, v, t):
        # Cells the agent can move to for time t+1 within its MDD.
        if t >= mdd.cost:
            return (mdd.goal,)  # parked at goal
        out = []
        nxt = mdd.levels[t + 1]
        for w in grid.neighbors(v):
            if w not in nxt:
                continue
            if (v, w, t + 1) in edge:
                continue
            out.append(w)
        return out

    # BFS over joint (va, vb) states by timestep, deduping within each level.
    frontier = {(start_a, start_b)}
    for t in range(horizon):
        nxt_frontier = set()
        for va, vb in frontier:
            for na in succ(mdd_a, edge_a, va, t):
                for nb in succ(mdd_b, edge_b, vb, t):
                    if na == nb:
                        continue  # vertex conflict
                    if na == vb and nb == va:
                        continue  # swap conflict
                    nxt_frontier.add((na, nb))
        if not nxt_frontier:
            return True  # no conflict-free joint path survives -> dependent
        frontier = nxt_frontier
    return False


# --- conflict classification -------------------------------------------------

CARDINAL = 2
SEMI = 1
NON = 0


def classify_vertex(mdd_a: Mdd, mdd_b: Mdd, t: int) -> int:
    """Classify a vertex conflict at time ``t``: an agent is *pinned* there iff
    its MDD width is 1, so it cannot dodge without lengthening its path."""
    pinned = (mdd_a.width(t) == 1) + (mdd_b.width(t) == 1)
    return (CARDINAL, SEMI, NON)[2 - pinned]


def classify_edge(mdd_a: Mdd, mdd_b: Mdd, t: int) -> int:
    """Classify an edge (swap) conflict arriving at time ``t``. An agent is
    pinned to the swapped edge iff it has width 1 at both endpoints."""
    a_pinned = mdd_a.width(t - 1) == 1 and mdd_a.width(t) == 1
    b_pinned = mdd_b.width(t - 1) == 1 and mdd_b.width(t) == 1
    pinned = a_pinned + b_pinned
    return (CARDINAL, SEMI, NON)[2 - pinned]


# --- (weighted) minimum vertex cover ----------------------------------------

def min_vertex_cover(vertices: list, edges: list) -> int:
    """Exact minimum vertex cover size of an unweighted graph (branch & bound).

    ``edges`` is a list of ``(u, v)`` pairs over ``vertices``. The graphs are
    tiny (only agents in conflict), so the exponential branch is cheap; a
    maximum-matching lower bound prunes and also caps a pathological case.
    """
    if not edges:
        return 0
    adj = {v: set() for v in vertices}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    edge_set = {frozenset(e) for e in edges}
    edge_list = [tuple(e) for e in edge_set]

    best = [len(vertices)]

    def lower_bound(remaining_edges, chosen):
        # A matching among the uncovered edges: each matched edge needs >= 1
        # more vertex in the cover, so chosen + |matching| <= optimum.
        used = set()
        m = 0
        for u, v in remaining_edges:
            if u not in used and v not in used:
                used.add(u)
                used.add(v)
                m += 1
        return chosen + m

    def branch(cover: set, idx: int):
        # All edges covered?
        uncovered = [e for e in edge_list if e[0] not in cover and e[1] not in cover]
        if not uncovered:
            best[0] = min(best[0], len(cover))
            return
        if lower_bound(uncovered, len(cover)) >= best[0]:
            return
        # Pick an uncovered edge; one of its endpoints must be in the cover.
        u, v = uncovered[0]
        for pick in (u, v):
            branch(cover | {pick}, idx + 1)

    branch(set(), 0)
    return best[0]


def weighted_min_vertex_cover(vertices: list, weighted_edges: list) -> int:
    """Exact weighted minimum vertex cover (Li et al.'s WDG ``h``).

    Assign each vertex an integer potential ``x_v >= 0`` such that
    ``x_u + x_v >= w(u, v)`` for every edge; minimize ``sum x_v``. Solved by
    branch and bound over vertex potentials. A matching lower bound (disjoint
    edges contribute their full weight) keeps it admissible and fast.
    """
    if not weighted_edges:
        return 0
    verts = [v for v in vertices]
    incident = {v: [] for v in verts}
    for u, v, w in weighted_edges:
        incident[u].append((v, w))
        incident[v].append((u, w))
    max_w = max(w for _, _, w in weighted_edges)

    # Matching lower bound: greedily pick heavy disjoint edges.
    used = set()
    lb = 0
    for u, v, w in sorted(weighted_edges, key=lambda e: -e[2]):
        if u not in used and v not in used:
            used.add(u)
            used.add(v)
            lb += w

    best = [sum(w for _, _, w in weighted_edges)]  # trivial upper bound
    x = {v: 0 for v in verts}

    def branch(i: int, total: int):
        if total >= best[0]:
            return
        if i == len(verts):
            # All potentials assigned; check feasibility of every edge.
            for u, v, w in weighted_edges:
                if x[u] + x[v] < w:
                    return
            best[0] = total
            return
        v = verts[i]
        # The potential never needs to exceed the heaviest incident edge.
        cap = max((w for _, w in incident[v]), default=0)
        for val in range(cap + 1):
            x[v] = val
            branch(i + 1, total + val)
        x[v] = 0

    branch(0, 0)
    return max(best[0], lb)
