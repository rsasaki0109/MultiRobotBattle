"""MDD-SAT: makespan-optimal MAPF by reduction to satisfiability.

Pavel Surynek's line of work — here the MDD-SAT encoding of *Time-Expanded
Decision Diagrams for movement planning* / *Efficient SAT approach to MAPF*
(Surynek et al., ECAI 2016 / IJCAI 2016) — solves MAPF without any search over
configurations at all. It is the **declarative** paradigm: write the question
"is there a collision-free plan of makespan ``mu``?" as a Boolean formula in CNF,
hand it to a SAT solver, and read a plan off any satisfying assignment.

The encoding is kept small by the **MDD** (multi-valued decision diagram): for a
target makespan ``mu`` an agent can only ever sit, at time ``t``, on a cell it
can both *reach* from its start in ``t`` steps and still get to its goal in the
remaining ``mu - t`` — so a variable ``x[a, v, t]`` is created only for those
cells. The clauses say: each agent is on exactly one of its MDD cells at each
time; it starts at its start and ends at its goal; consecutive cells are
adjacent (a step or a wait); no two agents share a cell (vertex) or swap across
an edge.

Optimality is **self-certified** by an incremental sweep: start ``mu`` at the
trivial lower bound (the largest single-agent shortest path — no plan can finish
sooner) and increase it until the formula is satisfiable. The first satisfiable
``mu`` is the optimal makespan, and *because every smaller ``mu`` was proved
UNSAT*, that optimum carries its own proof.

This module ships a small stock DPLL solver (unit propagation + chronological
backtracking, no external dependency) standing in for the off-the-shelf SAT
solver the paper uses; the reproduction is the *encoding*, not the solver.

MDD-SAT optimizes **labeled makespan** — distinct from the sum-of-costs CBS
minimizes and from the *anonymous* makespan
:func:`mrn_coord.mapf.flow.anonymous_makespan` computes (which it lower-bounds:
labels can only cost more).
"""

from __future__ import annotations

from collections import deque

from .grid import Cell, GridWorld
from .solution import Solution


# --------------------------------------------------------------------------
# A minimal DPLL SAT solver. Variables are 1..num_vars; a clause is a list of
# nonzero ints (``+v`` / ``-v``). Returns a truth list indexed by variable, or
# ``None`` if unsatisfiable.
# --------------------------------------------------------------------------
def _dpll(num_vars: int, clauses: list[list[int]]):
    val = [0] * (num_vars + 1)  # 0 unassigned, 1 true, -1 false

    # Branch on the most-constrained variables first (static occurrence count):
    # the collision-prone cells get decided early, which collapses the naive
    # search dramatically on MAPF formulas.
    freq = [0] * (num_vars + 1)
    for cl in clauses:
        for lit in cl:
            freq[abs(lit)] += 1
    order = sorted(range(1, num_vars + 1), key=lambda v: -freq[v])

    def propagate(trail: list[int]) -> bool:
        changed = True
        while changed:
            changed = False
            for cl in clauses:
                sat = False
                unassigned = 0
                last = 0
                for lit in cl:
                    v = val[abs(lit)]
                    if v == 0:
                        unassigned += 1
                        last = lit
                    elif (v == 1) == (lit > 0):
                        sat = True
                        break
                if sat:
                    continue
                if unassigned == 0:
                    return False
                if unassigned == 1:
                    val[abs(last)] = 1 if last > 0 else -1
                    trail.append(abs(last))
                    changed = True
        return True

    def undo(trail: list[int]) -> None:
        for v in trail:
            val[v] = 0

    def pick() -> int:
        for v in order:
            if val[v] == 0:
                return v
        return 0

    def search() -> bool:
        trail: list[int] = []
        if not propagate(trail):
            undo(trail)
            return False
        var = pick()
        if var == 0:
            return True
        for value in (1, -1):
            val[var] = value
            decided = [var]
            if search():
                return True
            undo(decided)
        undo(trail)
        return False

    if search():
        return [b == 1 for b in val]
    return None


# --------------------------------------------------------------------------
# Time-expanded MDD and the CNF encoding for a target makespan.
# --------------------------------------------------------------------------
def _bfs(grid: GridWorld, source: Cell) -> dict:
    if not grid.is_free(source):
        return {}
    dist = {source: 0}
    q = deque([source])
    while q:
        c = q.popleft()
        for nb in grid.neighbors(c):
            if nb not in dist:
                dist[nb] = dist[c] + 1
                q.append(nb)
    return dist


def _mdd_levels(grid: GridWorld, start: Cell, goal: Cell, mu: int):
    """Cells the agent may occupy at each time ``0..mu``: reachable from start by
    ``t`` and able to reach goal within ``mu - t``. Returns ``None`` if the goal
    is not reachable within ``mu``."""
    d_start = _bfs(grid, start)
    d_goal = _bfs(grid, goal)
    if d_goal.get(start, 10 ** 9) > mu:
        return None
    levels = []
    for t in range(mu + 1):
        lvl = [v for v, ds in d_start.items()
               if ds <= t and d_goal.get(v, 10 ** 9) <= mu - t]
        levels.append(set(lvl))
    return levels


def _encode(grid: GridWorld, agents: dict, mu: int):
    """Build ``(num_vars, clauses, var_index, ids)`` for makespan ``mu`` or
    return ``None`` if some agent cannot reach its goal in time."""
    ids = list(agents)
    levels = {}
    for a in ids:
        start, goal = agents[a]
        lv = _mdd_levels(grid, start, goal, mu)
        if lv is None:
            return None
        levels[a] = lv

    var_index: dict = {}
    nv = 0

    def var(a, v, t) -> int:
        nonlocal nv
        key = (a, v, t)
        idx = var_index.get(key)
        if idx is None:
            nv += 1
            var_index[key] = nv
            idx = nv
        return idx

    clauses: list[list[int]] = []
    for a in ids:
        start, goal = agents[a]
        lv = levels[a]
        # exactly one cell per time step
        for t in range(mu + 1):
            cells = list(lv[t])
            clauses.append([var(a, v, t) for v in cells])           # at least one
            for i in range(len(cells)):
                for j in range(i + 1, len(cells)):
                    clauses.append([-var(a, cells[i], t),
                                    -var(a, cells[j], t)])           # at most one
        # endpoints
        clauses.append([var(a, start, 0)])
        clauses.append([var(a, goal, mu)])
        # transitions: occupying v at t forces some adjacent (or same) cell next
        for t in range(mu):
            for v in lv[t]:
                succ = [w for w in ([v] + grid.neighbors(v)) if w in lv[t + 1]]
                clauses.append([-var(a, v, t)] + [var(a, w, t + 1)
                                                  for w in succ])

    # vertex collisions: at most one agent per cell per time
    for t in range(mu + 1):
        cell_users: dict = {}
        for a in ids:
            for v in levels[a][t]:
                cell_users.setdefault(v, []).append(a)
        for v, users in cell_users.items():
            for i in range(len(users)):
                for j in range(i + 1, len(users)):
                    clauses.append([-var(users[i], v, t),
                                    -var(users[j], v, t)])
    # edge (swap) collisions
    for t in range(mu):
        for ia in range(len(ids)):
            for ib in range(ia + 1, len(ids)):
                a, b = ids[ia], ids[ib]
                la, lb = levels[a], levels[b]
                for v in la[t] & lb[t + 1]:
                    for w in (la[t + 1] & lb[t]):
                        if w == v:
                            continue
                        if w in la[t + 1] and v in lb[t + 1]:
                            clauses.append([-var(a, v, t), -var(a, w, t + 1),
                                            -var(b, w, t), -var(b, v, t + 1)])
    return nv, clauses, var_index, ids, levels


def _decode(model, var_index, ids, levels, agents, mu) -> dict:
    paths = {}
    for a in ids:
        seq = []
        for t in range(mu + 1):
            here = None
            for v in levels[a][t]:
                if model[var_index[(a, v, t)]]:
                    here = v
                    break
            seq.append(here)
        goal = agents[a][1]
        while len(seq) > 1 and seq[-1] == goal and seq[-2] == goal:
            seq.pop()
        paths[a] = seq
    return paths


def satmdd(grid: GridWorld, agents: dict, *, max_makespan: int | None = None,
           stats: dict | None = None):
    """Solve a MAPF instance for the optimal **makespan** via MDD-SAT.

    Returns a :class:`Solution` whose paths are collision-free and finish in the
    minimum makespan, or ``None`` if infeasible within ``max_makespan``. The
    sweep starts at the trivial lower bound and stops at the first satisfiable
    makespan, so the result is makespan-optimal by construction. If ``stats`` is
    given: ``stats["makespan"]`` is that optimum, ``stats["lower_bound"]`` the
    starting bound, ``stats["certified"]`` is ``True`` when optimality is proved
    (either the optimum equals the lower bound, or the makespan one below it was
    shown UNSAT), and ``stats["unsat_below"]`` is the count of smaller makespans
    proved infeasible.
    """
    ids = list(agents)
    lb = 0
    for a in ids:
        start, goal = agents[a]
        d = _bfs(grid, goal)
        if start not in d:
            return None
        lb = max(lb, d[start])
    if max_makespan is None:
        max_makespan = lb + 2 * grid.width * grid.height + 4

    unsat_below = 0
    mu = lb
    while mu <= max_makespan:
        enc = _encode(grid, agents, mu)
        if enc is not None:
            nv, clauses, var_index, eids, levels = enc
            model = _dpll(nv, clauses)
            if model is not None:
                paths = _decode(model, var_index, eids, levels, agents, mu)
                cost = sum(max(0, len(p) - 1) for p in paths.values())
                if stats is not None:
                    stats["makespan"] = mu
                    stats["lower_bound"] = lb
                    stats["unsat_below"] = unsat_below
                    stats["certified"] = (mu == lb) or (unsat_below > 0)
                return Solution(paths=paths, cost=cost)
        unsat_below += 1
        mu += 1

    if stats is not None:
        stats["makespan"] = None
        stats["lower_bound"] = lb
        stats["unsat_below"] = unsat_below
        stats["certified"] = False
    return None
