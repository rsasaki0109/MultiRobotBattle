"""DDM — database-driven multi-robot path planning (Han & Yu, RA-L 2020).

Shuai D. Han & Jingjin Yu, *"DDM: Fast Near-Optimal Multi-Robot Path Planning
using Diversified-Path and Optimal Sub-Problem Solution Database Heuristics"*
(IEEE RA-L 2020). DDM is a centralized, *decoupled* planner aimed at dense,
warehouse-like grids. It plans each robot's path almost independently and leans
on two heuristics where the robots actually interfere:

1. **Optimal sub-problem solution database.** Conflicts are resolved *locally*.
   When robots collide, DDM carves out a small region (the paper uses 2×3 and
   3×3 windows) and looks up the **optimal** collision-free joint motion that
   moves the involved robots to their desired cells inside that window. Those
   sub-problems are tiny, so their optimal solutions are precomputed once into a
   database and reused in O(1) — turning the expensive part of coupling into a
   table lookup. The database here is an exhaustive joint breadth-first search
   over the window's labeled configuration space (min-makespan optimal),
   memoized on a translation-invariant key so a pattern is solved once and reused
   wherever it recurs on the big grid.

2. **Path diversification.** Rather than send every robot down the *same*
   shortest path (which manufactures congestion), DDM keeps several shortest
   paths per robot and picks, greedily, the one that overlaps the
   already-committed paths least — spreading the load so fewer conflicts ever
   reach the database.

The online loop steps the robots forward along their diversified paths; at each
timestep the conflicting robots are gathered into a window and advanced by the
database's optimal local motion, while everyone else moves freely. The result is
collision-free by construction (every committed step is a database-certified
collision-free joint move or an unconflicted advance) and near-optimal, solving
dense instances where decoupled planners (prioritized planning) deadlock.

This module reproduces the two defining heuristics and a database-driven online
loop; it is centralized and deterministic. Like the paper it is incomplete (it
can fail on a genuinely sealed instance, returning ``None``).
"""

from __future__ import annotations

from collections import deque

from .grid import Cell, GridWorld
from .solution import Solution, sum_of_costs


# --------------------------------------------------------------------------
# (1) Optimal sub-problem solution database
# --------------------------------------------------------------------------
class LocalDatabase:
    """Optimal joint motions for small windows, computed once and memoized.

    A query ``solve(cells, starts, goals)`` returns the min-makespan
    collision-free joint plan that moves a labeled set of robots from ``starts``
    to ``goals`` using only the cells of ``cells`` (a small connected region), or
    ``None`` if no such plan exists. Results are cached on a translation-invariant
    key, so the same local pattern — wherever it occurs on the big grid — is
    solved only once.
    """

    def __init__(self) -> None:
        self._cache: dict = {}
        self.lookups = 0
        self.solves = 0

    @staticmethod
    def _adj(cells):
        cellset = set(cells)
        adj = {}
        for (x, y) in cells:
            nbrs = [(x, y)]  # waiting is always allowed
            for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nb in cellset:
                    nbrs.append(nb)
            adj[(x, y)] = tuple(nbrs)
        return adj

    def solve(self, cells, starts, goals):
        """Optimal collision-free joint plan within ``cells`` or ``None``.

        ``starts`` / ``goals`` map robot id -> cell; the returned plan is a list
        of ``{robot: cell}`` configurations (index 0 == ``starts``), of minimal
        makespan.
        """
        self.lookups += 1
        minx = min(x for x, _ in cells)
        miny = min(y for _, y in cells)
        order = sorted(starts, key=lambda r: starts[r])  # by absolute start

        def sh(c):
            return (c[0] - minx, c[1] - miny)

        rcells = tuple(sorted(sh(c) for c in cells))
        rs = tuple(sh(starts[r]) for r in order)
        rg = tuple(sh(goals[r]) for r in order)
        key = (rcells, rs, rg)

        cached = self._cache.get(key, "miss")
        if cached == "miss":
            self.solves += 1
            shifted_cells = {sh(c) for c in cells}
            shifted_starts = {i: rs[i] for i in range(len(order))}
            shifted_goals = {i: rg[i] for i in range(len(order))}
            plan = self._search(shifted_cells, shifted_starts, shifted_goals)
            cached = (None if plan is None
                      else [tuple(cfg[i] for i in range(len(order)))
                            for cfg in plan])
            self._cache[key] = cached

        if cached is None:
            return None
        # Rehydrate to this query's absolute coordinates and robot ids.
        out = []
        for cfg in cached:
            out.append({order[i]: (cfg[i][0] + minx, cfg[i][1] + miny)
                        for i in range(len(order))})
        return out

    def _search(self, cellset, starts, goals):
        order = sorted(starts)
        adj = self._adj(cellset)
        start_cfg = tuple(starts[r] for r in order)
        goal_cfg = tuple(goals[r] for r in order)
        n = len(order)
        if start_cfg == goal_cfg:
            return [dict(zip(order, start_cfg))]

        def legal(u, v):
            if len(set(v)) != n:               # vertex collision
                return False
            for i in range(n):
                for j in range(i + 1, n):
                    if u[i] == v[j] and u[j] == v[i] and u[i] != u[j]:
                        return False           # swap
            return True

        seen = {start_cfg: None}
        q = deque([start_cfg])
        while q:
            u = q.popleft()
            if u == goal_cfg:
                return self._reconstruct(order, seen, u)
            for v in self._joint_moves(u, adj, n):
                if v in seen or not legal(u, v):
                    continue
                seen[v] = u
                q.append(v)
        return None

    @staticmethod
    def _joint_moves(u, adj, n):
        partial = [((), frozenset())]
        for i in range(n):
            nxt = []
            for cells_so_far, used in partial:
                for c in adj[u[i]]:
                    if c in used:
                        continue
                    nxt.append((cells_so_far + (c,), used | {c}))
            partial = nxt
        return [p[0] for p in partial]

    @staticmethod
    def _reconstruct(order, seen, goal_cfg):
        chain = [goal_cfg]
        cur = goal_cfg
        while seen[cur] is not None:
            cur = seen[cur]
            chain.append(cur)
        chain.reverse()
        return [dict(zip(order, cfg)) for cfg in chain]


# --------------------------------------------------------------------------
# (2) Path diversification
# --------------------------------------------------------------------------
def _dist_field(grid: GridWorld, goal: Cell) -> dict:
    """Backward BFS cost-to-go from ``goal`` over free cells."""
    if not grid.is_free(goal):
        return {}
    dist = {goal: 0}
    q = deque([goal])
    while q:
        c = q.popleft()
        x, y = c
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if grid.is_free(nb) and nb not in dist:
                dist[nb] = dist[c] + 1
                q.append(nb)
    return dist


def _shortest_paths(grid, start, goal, field, limit):
    """Up to ``limit`` distinct shortest paths from ``start`` to ``goal``.

    Walks the cost-to-go field downhill; at each cell the optimal next steps are
    the neighbors one closer to the goal. A bounded DFS enumerates distinct
    downhill paths (every one is a shortest path), capped at ``limit``.
    """
    if field.get(start) is None:
        return []
    out = []

    def dfs(cell, acc):
        if len(out) >= limit:
            return
        if cell == goal:
            out.append(list(acc))
            return
        here = field[cell]
        nxts = sorted((nb for nb in grid.neighbors(cell)
                       if nb != cell and field.get(nb, here) == here - 1))
        for nb in nxts:
            acc.append(nb)
            dfs(nb, acc)
            acc.pop()
            if len(out) >= limit:
                return

    dfs(start, [start])
    return out


def _diversified_paths(grid, agents, fields, *, candidates=4):
    """Pick each robot a shortest path overlapping the others' the least.

    Greedy: process robots in id order; for each, enumerate up to ``candidates``
    shortest paths and keep the one whose ``(cell, time)`` footprint adds the
    fewest hits to the cells already claimed by earlier robots. This is the
    path-diversification heuristic — it spreads robots over alternative shortest
    routes so fewer conflicts ever reach the database.
    """
    claimed: dict = {}
    chosen: dict = {}
    for r in agents:
        start, goal = agents[r]
        cands = _shortest_paths(grid, start, goal, fields[r], candidates)
        if not cands:
            return None
        best, best_overlap = None, None
        for p in cands:
            overlap = sum(claimed.get((c, t), 0) for t, c in enumerate(p))
            if best_overlap is None or overlap < best_overlap:
                best_overlap = overlap
                best = p
        chosen[r] = best
        for t, c in enumerate(best):
            claimed[(c, t)] = claimed.get((c, t), 0) + 1
    return chosen


# --------------------------------------------------------------------------
# (3) Database-driven online loop
# --------------------------------------------------------------------------
def _window_cells(grid, focus, blocked, max_side):
    """A small window (≤ ``max_side`` per side) of free cells around ``focus``.

    ``focus`` are the cells the conflicting group sits on / heads to; ``blocked``
    are cells held by *other* robots, treated as static obstacles the local
    solver routes around. The paper resolves conflicts strictly inside small 2×3
    / 3×3 windows, so the box is the focus's bounding box padded by one and then
    *clamped* to ``max_side`` per side (centered on the focus). Returns ``None``
    if the focus itself does not fit in a ``max_side`` square — that conflict is
    larger than a local window and out of DDM's scope."""
    xs = [c[0] for c in focus]
    ys = [c[1] for c in focus]
    x0, x1 = min(xs) - 1, max(xs) + 1
    y0, y1 = min(ys) - 1, max(ys) + 1
    if (max(xs) - min(xs) + 1 > max_side) or (max(ys) - min(ys) + 1 > max_side):
        return None
    # Clamp each axis to a window of at most max_side cells, biased to cover the
    # focus span.
    if x1 - x0 + 1 > max_side:
        x1 = x0 + max_side - 1
    if y1 - y0 + 1 > max_side:
        y1 = y0 + max_side - 1
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)
            if grid.is_free((x, y)) and (x, y) not in blocked]


def _fail(stats, t, database, resolutions):
    if stats is not None:
        stats.update(makespan=t, database_solves=database.solves,
                     database_lookups=database.lookups, resolutions=resolutions)
    return None


def _groups(conflict_pairs, members):
    """Union-find the conflicting robots into connected groups."""
    parent = {r: r for r in members}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in conflict_pairs:
        parent[find(a)] = find(b)
    groups: dict = {}
    for r in members:
        groups.setdefault(find(r), []).append(r)
    return list(groups.values())


def ddm(grid: GridWorld, agents: dict, *, candidates: int = 4,
        diversify: bool = True, max_side: int = 3, max_group: int = 3,
        max_steps: int | None = None,
        database: LocalDatabase | None = None, stats: dict | None = None):
    """Solve a MAPF instance with the database-driven method (DDM).

    ``agents`` maps robot id -> ``(start, goal)``. Returns a :class:`Solution`
    whose paths are collision-free, or ``None`` if a robot cannot reach its goal
    or the step budget is exhausted (DDM is incomplete). ``diversify`` toggles
    the path-diversification heuristic (when off, every robot takes the first
    shortest path). ``database`` lets a precomputed :class:`LocalDatabase` be
    shared/reused; one is created if omitted. ``stats`` records ``makespan``,
    ``database_solves`` (distinct sub-problems computed), ``database_lookups``
    and ``resolutions`` (timesteps a database resolution fired).
    """
    ids = list(agents)
    goals = {r: agents[r][1] for r in ids}
    fields = {r: _dist_field(grid, goals[r]) for r in ids}
    for r in ids:
        if fields[r].get(agents[r][0]) is None:
            return None
    if database is None:
        database = LocalDatabase()
    if max_steps is None:
        max_steps = 8 * (grid.width * grid.height) + 4 * len(ids) + 40

    if diversify:
        chosen = _diversified_paths(grid, agents, fields, candidates=candidates)
    else:
        chosen = {r: _shortest_paths(grid, agents[r][0], goals[r], fields[r], 1)[0]
                  for r in ids}
    if chosen is None:
        return None

    pos = {r: agents[r][0] for r in ids}
    paths = {r: chosen[r] for r in ids}
    prog = {r: 0 for r in ids}
    trace = {r: [pos[r]] for r in ids}
    resolutions = 0

    def want(r):
        p = paths[r]
        if prog[r] + 1 < len(p):
            return p[prog[r] + 1]
        return pos[r]

    t = 0
    while t < max_steps:
        if all(pos[r] == goals[r] for r in ids):
            break

        # Conflicts in the naive "advance to want" assignment.
        w = {r: want(r) for r in ids}
        by_cell: dict = {}
        for r in ids:
            by_cell.setdefault(w[r], []).append(r)
        conflict_pairs = []
        conflicted = set()
        for cell, rs in by_cell.items():            # vertex
            if len(rs) > 1:
                for i in range(len(rs)):
                    for j in range(i + 1, len(rs)):
                        conflict_pairs.append((rs[i], rs[j]))
                        conflicted.update((rs[i], rs[j]))
        for a in ids:                               # swap / move-into-stay
            for b in ids:
                if a < b and w[a] == pos[b] and w[b] == pos[a] and a != b:
                    conflict_pairs.append((a, b))
                    conflicted.update((a, b))
        # A mover heading into a cell occupied by a robot that stays put there.
        occ = {pos[r]: r for r in ids}
        for r in ids:
            if w[r] != pos[r] and w[r] in occ:
                s = occ[w[r]]
                if w[s] == pos[s]:                  # s is staying on that cell
                    conflict_pairs.append((r, s))
                    conflicted.update((r, s))

        nxt = {r: pos[r] for r in ids}
        committed: set = set()
        window_cells_all: set = set()

        if conflicted:
            resolutions += 1
            for group in _groups(conflict_pairs, conflicted):
                # A coupling larger than a local window is M*/CBS territory.
                if len(group) > max_group:
                    return _fail(stats, t, database, resolutions)
                gset = set(group)
                focus = [pos[r] for r in group] + [w[r] for r in group]
                # Other robots are static obstacles the local solver routes
                # around (they wait this step); cells already used by an earlier
                # group's window this step are off-limits too, keeping windows
                # disjoint so resolutions can never collide across groups.
                blocked = {pos[r] for r in ids if r not in gset} | window_cells_all
                cells = _window_cells(grid, focus, blocked, max_side)
                if cells is None:                   # conflict too spread out
                    return _fail(stats, t, database, resolutions)
                cellset = set(cells)
                if not all(pos[r] in cellset for r in group):  # pinned / overlapping
                    for r in group:
                        committed.add(r)
                    continue
                # Assign each group robot a distinct window cell that most
                # reduces its cost-to-go (greedy, hardest robot first).
                local_starts = {r: pos[r] for r in group}
                targets: dict = {}
                taken: set = set()
                for r in sorted(group, key=lambda r: -fields[r].get(pos[r], 0)):
                    best, bestd = None, None
                    for c in cells:
                        if c in taken:
                            continue
                        d = fields[r].get(c)
                        if d is None:
                            continue
                        if bestd is None or d < bestd or (
                                d == bestd and c == w[r]):
                            bestd = d
                            best = c
                    if best is None:
                        best = pos[r]
                    targets[r] = best
                    taken.add(best)
                plan = database.solve(cells, local_starts, targets)
                if plan is None or len(plan) < 2:
                    for r in group:                 # cannot progress -> wait
                        committed.add(r)
                else:
                    step1 = plan[1]
                    for r in group:
                        nxt[r] = step1[r]
                        committed.add(r)
                window_cells_all |= cellset

        # Free (uncommitted) robots advance to their want; window cells are
        # off-limits. A move is committed only if it survives a fixpoint that
        # demotes any move whose target is contested or held by a robot that does
        # not vacate -- so a robot never follows into a cell that stays occupied.
        free = [r for r in ids if r not in committed]
        move = {r: (w[r] if (w[r] != pos[r] and w[r] not in window_cells_all)
                    else pos[r]) for r in free}

        def end_cell(r):
            return nxt[r] if r in committed else move[r]

        changed = True
        while changed:
            changed = False
            for r in free:
                if move[r] == pos[r]:
                    continue
                tgt = move[r]
                contested = any(x != r and end_cell(x) == tgt for x in ids)
                swap = any(x != r and pos[x] == tgt and end_cell(x) == pos[r]
                           for x in ids)
                if contested or swap:
                    move[r] = pos[r]
                    changed = True
        for r in free:
            nxt[r] = move[r]

        # Commit the step; advance progress, replan robots pushed off their path.
        for r in ids:
            new = nxt[r]
            if new == pos[r]:
                pass
            elif prog[r] + 1 < len(paths[r]) and new == paths[r][prog[r] + 1]:
                prog[r] += 1
            else:
                np_ = _shortest_paths(grid, new, goals[r], fields[r], 1)
                paths[r] = np_[0] if np_ else [new]
                prog[r] = 0
            pos[r] = new
            trace[r].append(new)
        t += 1

    if not all(pos[r] == goals[r] for r in ids):
        if stats is not None:
            stats.update(makespan=t, database_solves=database.solves,
                         database_lookups=database.lookups,
                         resolutions=resolutions)
        return None

    for r in ids:                                   # trim synchronized goal waits
        p = trace[r]
        while len(p) > 1 and p[-1] == goals[r] and p[-2] == goals[r]:
            p.pop()
    if stats is not None:
        stats.update(makespan=max(len(p) - 1 for p in trace.values()),
                     database_solves=database.solves,
                     database_lookups=database.lookups,
                     resolutions=resolutions)
    return Solution(paths=trace, cost=sum_of_costs(trace))
