"""Branch-and-cut-and-price for MAPF (Lam et al., IJCAI 2019).

A fifth optimal (sum-of-costs) paradigm in this package, and the first that is
*not* a search over states. CBS / M* / Standley search; MDD-SAT decides
satisfiability; :mod:`flow` routes a network. BCP **optimizes a linear
program** and certifies optimality by *LP duality*.

The model is the path-based (Dantzig-Wolfe / set-partitioning) formulation.
For each agent ``a`` let ``Omega_a`` be its set of candidate paths; a binary
variable ``lambda_{a,p}`` selects path ``p``. Minimize the total
sum-of-costs ``sum cost(p) * lambda_{a,p}`` subject to

* **convexity** — each agent picks exactly one path: ``sum_p lambda_{a,p} = 1``;
* **vertex** — at most one agent per cell per time;
* **edge** — at most one agent per swap (``u->v`` and ``v->u`` at the same step).

Two things make this tractable without writing down the (astronomically many)
paths and conflict rows up front, and they are exactly the "price" and the
"cut" of the name:

* **Pricing (column generation).** We start with one path per agent and only
  *generate* a new path-column when it has negative reduced cost. The pricing
  subproblem — minimize ``cost(p) - sigma_a - sum_r pi_r [p covers r]`` over
  paths — is a shortest path in the time-expanded graph where occupying a
  congested cell carries the LP dual price ``-pi_r >= 0`` as a penalty. When no
  agent has an improving column, the LP is optimal: its objective is a valid
  *lower bound*, certified by the reduced-cost optimality condition.
* **Cutting (lazy separation).** Vertex/edge conflict rows are added only when
  the LP solution violates one. We never materialize a row for a cell-time no
  agent contends.

Branching closes the integrality gap: when the aggregate usage
``y_{a,v,t} = sum_{p uses (v,t)} lambda_{a,p}`` is fractional we branch on it,
forcing the agent onto ``(v,t)`` in one child and off it in the other (a
disjoint, CBS-style split, but imposed inside the pricing subproblem). The
incumbent whose cost equals the LP bound (gap zero) is optimal — the same
sum-of-costs as :func:`mrn_coord.mapf.cbs.cbs`.

The LP master is solved with SciPy's HiGHS. This reproduces the BCP *skeleton*
— path LP + pricing + lazy conflict cuts + branch-and-price; the published
solver layers further specialized cut families (rectangle, corridor, target)
on this same exact frame.
"""

from __future__ import annotations

import heapq
import itertools
import math

import numpy as np
from scipy.optimize import linprog

from .grid import Cell, GridWorld
from .mstar import _dist_to_goal
from .solution import Solution

_EPS = 1e-6
_RECT_CAP = 4   # rectangle cuts separated per agent per node before branching


def _canon_edge(u: Cell, v: Cell, t: int):
    """A swap resource is undirected in (u, v) but pinned to step t->t+1."""
    return (u, v, t) if u <= v else (v, u, t)


def _horizon(grid: GridWorld, agents: dict, dist: dict) -> int:
    """A time bound that provably contains a sum-of-costs optimum here.

    Tiny gate instances only; a generous-but-bounded horizon (longest single
    -agent distance plus slack for vacate-and-return detours) suffices and is
    validated against CBS in the test/gate.
    """
    maxd = max((dist[a].get(agents[a][0], 0) for a in agents), default=0)
    free = sum(1 for x in range(grid.width) for y in range(grid.height)
               if grid.is_free((x, y)))
    return min(maxd + 2 * len(agents) + 4, maxd + free + 2)


def _price(grid, start, goal, horizon, dist, *, vpen, epen, force, block,
           rect=()):
    """Min reduced-cost path for one agent (the pricing subproblem).

    Returns ``(positions, soc_cost, reduced_value)`` or ``None`` if no path to
    the goal honors the branching constraints. ``positions`` has length
    ``horizon + 1`` (settled agents hold the goal). ``soc_cost`` is the genuine
    sum-of-costs contribution (steps before the agent settles at its goal);
    ``reduced_value`` adds the dual penalties used for the improvement test.

    State is ``(cell, settled, crossed)``: an unsettled agent pays 1 per step
    (even a wait), a settled agent is frozen at goal at cost 0 and may settle
    only while on its goal. This is the done-bit cost model shared with
    :mod:`mstar` / :mod:`standley`; it prices vacate-and-return exactly.

    ``vpen[(v, t)] >= 0`` is the penalty for occupying ``(v, t)``;
    ``epen[(u, v, t)] >= 0`` the penalty for the swap edge at ``t``. ``force[t]``
    pins the cell at time ``t``; ``block`` is a set of forbidden ``(cell, t)``.

    ``rect`` is a tuple of ``(barrier_frozenset, penalty)`` for *rectangle* cuts
    this agent is in: the penalty is charged **once**, the first time the path
    enters any ``(cell, time)`` of the barrier (its cut coefficient is binary, so
    a second touch is free). ``crossed`` is the per-rect tuple of "already paid"
    bits that makes this exact — without it the additive penalty would over-count
    a path that grazes a barrier twice and the LP lower bound would be wrong. With
    ``rect == ()`` the bit tuple is empty and the DP is identical to plain BCP.
    """
    # Forward DP over a DAG (time strictly increases).
    def allowed(cell, t):
        if t in force and cell != force[t]:
            return False
        if (cell, t) in block:
            return False
        return True

    nrect = len(rect)
    base = (False,) * nrect

    def cross(crossed, cell, t):
        """Charge each not-yet-paid rectangle barrier this arrival enters once."""
        if nrect == 0:
            return crossed, 0.0
        nc = None
        extra = 0.0
        for i in range(nrect):
            barrier, pen = rect[i]
            if not crossed[i] and (cell, t) in barrier:
                if nc is None:
                    nc = list(crossed)
                nc[i] = True
                extra += pen
        return (tuple(nc) if nc is not None else crossed), extra

    # cost = accumulated SOC; rc = accumulated reduced value. Minimize rc, break
    # ties by cost then path for determinism.
    cur = {}
    if allowed(start, 0):
        c0, e0 = cross(base, start, 0)
        v0 = vpen.get((start, 0), 0.0) + e0
        cur[(start, False, c0)] = (v0, 0.0, (start,))
        if start == goal:
            cur[(start, True, c0)] = (v0, 0.0, (start,))
    for t in range(horizon):
        nxt = {}
        for (cell, settled, crossed), (rc, cost, path) in cur.items():
            if settled:
                # frozen at goal: only stay, cost 0, still occupies (goal, t+1)
                if not allowed(goal, t + 1):
                    continue
                nc, extra = cross(crossed, goal, t + 1)
                nrc = rc + vpen.get((goal, t + 1), 0.0) + extra
                key = (goal, True, nc)
                cand = (nrc, cost, path + (goal,))
                if key not in nxt or cand < nxt[key]:
                    nxt[key] = cand
                continue
            x, y = cell
            for nb in ((x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if not grid.is_free(nb) or not allowed(nb, t + 1):
                    continue
                step = vpen.get((nb, t + 1), 0.0)
                if nb != cell:
                    step += epen.get(_canon_edge(cell, nb, t), 0.0)
                nc, extra = cross(crossed, nb, t + 1)
                nrc = rc + 1.0 + step + extra
                ncost = cost + 1
                npath = path + (nb,)
                # unsettled arrival
                key = (nb, False, nc)
                cand = (nrc, ncost, npath)
                if key not in nxt or cand < nxt[key]:
                    nxt[key] = cand
                # option to settle, only on goal (cost of this step still paid;
                # future steps free)
                if nb == goal:
                    key = (nb, True, nc)
                    if key not in nxt or cand < nxt[key]:
                        nxt[key] = cand
        cur = nxt
        if not cur:
            break
    # accept only fully-parked agents: settled at goal at the horizon (the best
    # over all crossing states -- a path that detours *around* a barrier pays no
    # penalty but more SOC; pricing compares them by reduced cost).
    best = None
    for (cell, settled, _crossed), val in cur.items():
        if cell == goal and settled and (best is None or val < best):
            best = val
    if best is None:
        return None
    rc, cost, path = best
    return (list(path), cost, rc)


def _shortest(grid, start, goal, horizon, dist, *, force, block):
    """A feasible (zero-penalty) starting column honoring the branch."""
    return _price(grid, start, goal, horizon, dist,
                  vpen={}, epen={}, force=force, block=block)


class _Column:
    __slots__ = ("agent", "pos", "cost", "dummy")

    def __init__(self, agent, pos, cost, dummy=False):
        self.agent = agent
        self.pos = tuple(pos) if pos is not None else None
        self.cost = cost
        self.dummy = dummy

    def at(self, t):
        return self.pos[t] if t < len(self.pos) else self.pos[-1]

    def occupies(self, v, t):
        # A dummy (artificial) column keeps the restricted master feasible but
        # contends for no resource, so it covers no conflict cut.
        return not self.dummy and self.at(t) == v

    def traverses(self, key):
        if self.dummy:
            return False
        u, v, t = key
        a, b = self.at(t), self.at(t + 1)
        return (a, b) == (u, v) or (a, b) == (v, u)

    def crosses(self, barrier):
        # Whether this path touches any (cell, time) of a rectangle barrier --
        # the membership a rectangle cut sums over. A monotone optimal path
        # crosses an exit barrier exactly once.
        if self.dummy:
            return False
        return any(self.at(t) == v for (v, t) in barrier)


def _solve_master(columns, agents, vcuts, ecuts, rcuts=()):
    """Solve the path LP for the current columns and active cuts.

    Returns ``(obj, lam, sigma, vpi, epi, rpi)`` or ``None`` if infeasible.
    ``sigma[a]`` is the convexity dual; ``vpi/epi/rpi`` the (<=0) cut duals
    (``rpi`` is a list aligned with ``rcuts``, each ``(a1, B1, a2, B2)``).
    """
    n = len(columns)
    aidx = {a: i for i, a in enumerate(agents)}
    c = [col.cost for col in columns]

    # equality: one path per agent (convexity)
    A_eq = [[0.0] * n for _ in agents]
    for j, col in enumerate(columns):
        A_eq[aidx[col.agent]][j] = 1.0
    b_eq = [1.0] * len(agents)

    A_ub = []
    b_ub = []
    for (v, t) in vcuts:
        A_ub.append([1.0 if col.occupies(v, t) else 0.0 for col in columns])
        b_ub.append(1.0)
    for key in ecuts:
        A_ub.append([1.0 if col.traverses(key) else 0.0 for col in columns])
        b_ub.append(1.0)
    for (a1, b1, a2, b2) in rcuts:
        # rectangle cut: [a1 crosses its barrier] + [a2 crosses its barrier] <= 1
        row = []
        for col in columns:
            if col.agent == a1 and col.crosses(b1):
                row.append(1.0)
            elif col.agent == a2 and col.crosses(b2):
                row.append(1.0)
            else:
                row.append(0.0)
        A_ub.append(row)
        b_ub.append(1.0)

    res = linprog(
        c,
        A_ub=A_ub or None, b_ub=b_ub or None,
        A_eq=A_eq, b_eq=b_eq,
        bounds=(0.0, 1.0), method="highs",
    )
    if not res.success:
        return None
    sigma = {a: res.eqlin.marginals[aidx[a]] for a in agents}
    marg = list(res.ineqlin.marginals) if A_ub else []
    vpi = {}
    epi = {}
    r = 0
    for (v, t) in vcuts:
        vpi[(v, t)] = marg[r]
        r += 1
    for key in ecuts:
        epi[key] = marg[r]
        r += 1
    rpi = []
    for _ in rcuts:
        rpi.append(marg[r])
        r += 1
    return res.fun, list(res.x), sigma, vpi, epi, rpi


def _separate(columns, lam, vcuts, ecuts, horizon):
    """Find one most-violated vertex or edge conflict not yet cut."""
    vuse = {}
    euse = {}
    for col, x in zip(columns, lam):
        if x <= _EPS or col.dummy:
            continue
        for t in range(horizon + 1):
            v = col.at(t)
            vuse[(v, t)] = vuse.get((v, t), 0.0) + x
        for t in range(horizon):
            a, b = col.at(t), col.at(t + 1)
            if a != b:
                key = _canon_edge(a, b, t)
                euse[key] = euse.get(key, 0.0) + x
    best = None
    for (v, t), u in vuse.items():
        if u > 1.0 + _EPS and (v, t) not in vcuts:
            if best is None or u > best[2]:
                best = ("v", (v, t), u)
    for key, u in euse.items():
        if u > 1.0 + _EPS and key not in ecuts:
            if best is None or u > best[2]:
                best = ("e", key, u)
    return best


def _fractional_branch(columns, lam, agents, horizon):
    """Pick the agent-vertex-time usage closest to 0.5 to branch on."""
    yuse = {}
    for col, x in zip(columns, lam):
        if x <= _EPS or col.dummy:
            continue
        for t in range(horizon + 1):
            key = (col.agent, col.at(t), t)
            yuse[key] = yuse.get(key, 0.0) + x
    best = None
    bestdist = 1.0
    for key, y in yuse.items():
        frac = abs(y - round(y))
        if frac > _EPS:
            d = abs(y - 0.5)
            if d < bestdist:
                bestdist = d
                best = key
    return best


def _separate_rectangle(grid, columns, lam, ids, agents, dist, horizon, rcuts):
    """Find one violated, not-yet-added rectangle cut, or ``None``.

    A rectangle symmetry (Li et al. AAAI'19) is two agents crossing an open
    rectangle in the same direction so *every* pair of their Manhattan-optimal
    paths collides inside. We look at the LP's dominant path per agent, find a
    pair whose paths share a cell, build both agents' optimal-cost MDDs, and ask
    :func:`mrn_coord.mapf.rectangle.find_rectangle_barriers` for the two exit
    barriers ``B1``, ``B2`` (each an anti-diagonal of ``(cell, time)`` an optimal
    crossing path hits exactly once). The cut
    ``sum_{B1} y_{a1} + sum_{B2} y_{a2} <= 1`` is valid — both agents crossing
    means a collision — so it is returned only when the current LP violates it
    (the two agents' barrier usage exceeds 1).
    """
    from .mdd import build_mdd
    from .rectangle import find_rectangle_barriers

    dom = {}
    for a in ids:
        best = None
        for j, col in enumerate(columns):
            if col.agent == a and not col.dummy and (
                    best is None or lam[j] > best[0]):
                best = (lam[j], col)
        if best is not None and best[0] > _EPS:
            dom[a] = best[1]

    existing = set(rcuts)
    for ia in range(len(ids)):
        for ib in range(ia + 1, len(ids)):
            a1, a2 = ids[ia], ids[ib]
            if a1 not in dom or a2 not in dom:
                continue
            c1 = dist[a1].get(agents[a1][0])
            c2 = dist[a2].get(agents[a2][0])
            if c1 is None or c2 is None:
                continue
            # a shared cell between the two dominant paths, within both MDDs
            ctime = None
            for t in range(min(c1, c2) + 1):
                if dom[a1].at(t) == dom[a2].at(t):
                    ctime = t
                    break
            if ctime is None:
                continue
            m1 = build_mdd(grid, agents[a1][0], agents[a1][1], c1)
            m2 = build_mdd(grid, agents[a2][0], agents[a2][1], c2)
            if m1 is None or m2 is None:
                continue
            found = find_rectangle_barriers(m1, m2, ctime)
            if found is None:
                continue
            b1, b2, klass = found
            if klass < 1:                      # need at least a semi-cardinal cut
                continue
            rcut = (a1, b1, a2, b2)
            if rcut in existing:
                continue
            lhs = 0.0
            for j, col in enumerate(columns):
                if lam[j] <= _EPS or col.dummy:
                    continue
                if col.agent == a1 and col.crosses(b1):
                    lhs += lam[j]
                elif col.agent == a2 and col.crosses(b2):
                    lhs += lam[j]
            if lhs > 1.0 + _EPS:
                return rcut
    return None


def bcp(grid: GridWorld, agents: dict, *, max_nodes: int = 5000,
        rectangle: bool = False, stats: dict | None = None):
    """Solve a MAPF instance optimally (sum-of-costs) by branch-and-price.

    ``agents`` maps an agent id to a ``(start, goal)`` tuple. Returns a
    :class:`Solution` (collision-free, optimal sum-of-costs — the same optimum
    as :func:`mrn_coord.mapf.cbs.cbs`) or ``None`` if infeasible or the node
    budget is exhausted.

    With ``rectangle=True`` the lazy separation also adds **rectangle cuts** (Lam
    et al.'s specialized family, on top of the vertex/edge cuts): when two agents
    cross an open rectangle in the same direction — a symmetry that makes plain
    branch-and-price enumerate exponentially many equivalent crossings — a single
    cut ``sum_{B1} y_{a1} + sum_{B2} y_{a2} <= 1`` forbids both crossing it. Same
    optimum as plain BCP / CBS; fewer nodes. ``rectangle=False`` (default) is the
    plain BCP, byte-for-byte unchanged.

    If ``stats`` is given it records the price-and-cut mechanism:
    ``nodes`` (branch-and-bound nodes solved), ``columns`` (path-columns priced
    over the whole run), ``cuts`` (vertex/edge conflict rows separated lazily),
    ``rcuts`` (rectangle cuts separated), ``root_integral`` (True iff the root LP
    already solved the IP — branch-and-price found the optimum with no
    branching), ``lp_bound`` (the root LP objective, a valid lower bound), and
    ``cost`` (the certified optimum).
    """
    ids = list(agents)
    dist = {a: _dist_to_goal(grid, agents[a][1]) for a in ids}
    for a in ids:
        if agents[a][0] not in dist[a]:
            return None  # goal unreachable from start
    horizon = _horizon(grid, agents, dist)

    st = {"nodes": 0, "columns": 0, "cuts": 0, "rcuts": 0,
          "root_integral": False, "lp_bound": None, "cost": None}

    def node_columns(force, block):
        """Column-generate to LP optimality + separate cuts at one node.

        Returns ``(obj, lam, columns, vcuts, ecuts)`` or ``None`` (infeasible).
        """
        # A big-M artificial column per agent keeps the restricted master
        # feasible at all times (so the LP always has duals to price against);
        # its cost exceeds any real plan, so the LP drives it to zero whenever a
        # real conflict-free assignment exists. A leaf that still leans on a
        # dummy is genuinely infeasible.
        big = float(len(ids) * (horizon + 2) + 100)
        columns = [_Column(a, None, big, dummy=True) for a in ids]
        for a in ids:
            seed = _shortest(grid, agents[a][0], agents[a][1], horizon, dist,
                             force=force.get(a, {}), block=block.get(a, set()))
            if seed is not None:
                columns.append(_Column(a, seed[0], seed[1]))
                st["columns"] += 1
        vcuts, ecuts, rcuts = [], [], []
        while True:
            master = _solve_master(columns, ids, vcuts, ecuts, rcuts)
            if master is None:
                return None
            obj, lam, sigma, vpi, epi, rpi = master
            # pricing: an improving column per agent
            vpen = {k: -p for k, p in vpi.items()}
            epen = {k: -p for k, p in epi.items()}
            improved = False
            for a in ids:
                # rectangle barriers (and their dual penalties) this agent is in
                rect = []
                for i, (a1, b1, a2, b2) in enumerate(rcuts):
                    if a == a1:
                        rect.append((b1, -rpi[i]))
                    elif a == a2:
                        rect.append((b2, -rpi[i]))
                col = _price(grid, agents[a][0], agents[a][1], horizon, dist,
                             vpen=vpen, epen=epen,
                             force=force.get(a, {}), block=block.get(a, set()),
                             rect=tuple(rect))
                if col is None:
                    continue
                _pos, cost, rc = col
                if rc - sigma[a] < -1e-6:
                    columns.append(_Column(a, _pos, cost))
                    st["columns"] += 1
                    improved = True
            if improved:
                continue
            # LP optimal for this cut set: separate a violated conflict
            cut = _separate(columns, lam, vcuts, ecuts, horizon)
            if cut is not None:
                kind, key, _u = cut
                if kind == "v":
                    vcuts.append(key)
                else:
                    ecuts.append(key)
                st["cuts"] += 1
                continue
            # then a violated rectangle symmetry, if enabled. Cap the rectangle
            # cuts per node: a shifting fractional solution can keep offering
            # slightly different barriers, so after a bounded round we stop
            # separating and let branching finish the job (optimality is the
            # branch tree's guarantee, not the cut's).
            if rectangle and len(rcuts) < _RECT_CAP * len(ids):
                rcut = _separate_rectangle(grid, columns, lam, ids, agents,
                                           dist, horizon, rcuts)
                if rcut is not None:
                    rcuts.append(rcut)
                    st["rcuts"] += 1
                    continue
            return obj, lam, columns, vcuts, ecuts

    incumbent = None
    best_cost = math.inf
    # B&B stack of (force, block) constraint sets; force[a]={t:cell}, block[a]=set
    root = ({a: {} for a in ids}, {a: set() for a in ids})
    stack = [root]
    is_root = True
    while stack and st["nodes"] < max_nodes:
        force, block = stack.pop()
        st["nodes"] += 1
        res = node_columns(force, block)
        root_now, is_root = is_root, False
        if res is None:
            continue
        obj, lam, columns, vcuts, ecuts = res
        if root_now:
            st["lp_bound"] = obj
        lb = math.ceil(obj - _EPS)
        if lb >= best_cost:
            continue
        br = _fractional_branch(columns, lam, ids, horizon)
        if br is None:
            # integral & (cuts ensure) conflict-free: decode the picked paths.
            # If any agent still rides its artificial column the node has no
            # real assignment under its branch -- prune it.
            picks = {a: max((j for j in range(len(columns))
                             if columns[j].agent == a), key=lambda j: lam[j])
                     for a in ids}
            if any(columns[picks[a]].dummy for a in ids):
                continue
            cost = int(round(obj))
            paths = {}
            for a in ids:
                pick = picks[a]
                pos = list(columns[pick].pos)
                while len(pos) > 1 and pos[-1] == pos[-2] == agents[a][1]:
                    pos.pop()
                paths[a] = pos
            if cost < best_cost:
                best_cost = cost
                incumbent = paths
                if root_now:
                    st["root_integral"] = True
            continue
        a, v, t = br
        # left: force a at (v, t); right: forbid a at (v, t)
        lf = {k: dict(force[k]) for k in ids}
        lf[a] = dict(lf[a]); lf[a][t] = v
        lb_ = {k: set(block[k]) for k in ids}
        rf = {k: dict(force[k]) for k in ids}
        rb = {k: set(block[k]) for k in ids}
        rb[a] = set(rb[a]); rb[a].add((v, t))
        stack.append((lf, lb_))
        stack.append((rf, rb))

    if stats is not None:
        st["cost"] = best_cost if incumbent is not None else None
        stats.update(st)
    if incumbent is None:
        return None
    return Solution(paths=incumbent, cost=best_cost)
