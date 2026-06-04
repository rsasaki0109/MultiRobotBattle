"""CBM — Conflict-Based Min-cost-flow for Target Assignment and Path Finding.

A reproduction of Hang Ma & Sven Koenig, *"Optimal Target Assignment and Path
Finding for Teams of Agents"* (AAMAS 2016). TAPF generalizes both labeled and
anonymous MAPF. Agents are partitioned into **teams**; each team owns a set of
**targets** equal in number to its agents. Targets *within* a team are
interchangeable (any team member may fill any of its team's targets — the
**anonymous** problem); targets *across* teams are not. The task is to assign
each agent a target of its own team **and** route everyone collision-free,
minimizing makespan.

The two extremes recover the problems we already have:

- **one team** containing every agent ⇒ fully anonymous ⇒ pure network flow
  (:func:`mrn_coord.mapf.flow.anonymous_makespan`);
- **singleton teams** (one agent, one target each) ⇒ fully labeled ⇒ ordinary
  makespan-optimal MAPF.

CBM interpolates between them with a two-level search that marries the two
paradigms:

- **Low level — per-team min-cost flow.** Each team is solved *independently* as
  an anonymous makespan problem by integer max-flow on the time-expanded grid
  (the Yu & LaValle reduction, reused here), but with the high-level constraints
  baked in: a forbidden ``(cell, t)`` drops that time-expanded vertex; a
  forbidden directed move drops that gadget entry. The flow simultaneously
  *assigns* the team's targets and routes its agents, collision-free *within* the
  team by construction.
- **High level — CBS over teams.** The teams are planned together; the first
  conflict between agents of *different* teams (a shared cell, or a head-on swap)
  is resolved CBS-style by branching: forbid that cell/move to one team or the
  other and re-solve only that team's flow. Best-first on makespan, the first
  conflict-free node is makespan-optimal.

So a within-team interaction is dissolved for free by the polynomial flow, and
only the *inter-team* interactions pay for tree search — exactly the structure
that makes TAPF tractable when teams are few and large.
"""

from __future__ import annotations

import heapq
import itertools

from .flow import _MaxFlow, _neighbors4
from .grid import GridWorld


# --------------------------------------------------------------------------- #
# Constrained team flow: anonymous makespan with forbidden vertices/edges      #
# --------------------------------------------------------------------------- #
def _build(grid, starts, goals, T, forbidden_v, forbidden_e):
    """Time-expanded flow network for one team at horizon ``T``.

    ``forbidden_v`` is a set of ``(cell, t)`` the team may not occupy;
    ``forbidden_e`` a set of ``(frm, to, t)`` directed moves arriving at ``t``
    the team may not make. Both are realized by *omitting* the corresponding
    edges, so no flow (no agent of this team) can use them."""
    g = _MaxFlow()
    S, K = ("S",), ("K",)
    free = [(x, y) for x in range(grid.width) for y in range(grid.height)
            if grid.is_free((x, y))]

    for t in range(T + 1):
        for v in free:
            if (v, t) in forbidden_v:
                continue                       # team may not be at v at t
            g.add(("in", v, t), ("out", v, t), 1)
    for t in range(T):
        for v in free:
            if (v, t) in forbidden_v or (v, t + 1) in forbidden_v:
                continue
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
                midin = ("midin", a, b, t)
                # u_out -> midin entry; omit a direction if it is forbidden or its
                # endpoint cell is blocked at the relevant time.
                if ((a, t) not in forbidden_v and (b, t + 1) not in forbidden_v
                        and (a, b, t + 1) not in forbidden_e):
                    g.add(("out", a, t), midin, 1)
                if ((b, t) not in forbidden_v and (a, t + 1) not in forbidden_v
                        and (b, a, t + 1) not in forbidden_e):
                    g.add(("out", b, t), midin, 1)
                g.add(midin, mid, 1)                          # shared cap-1 (swap)
                g.add(mid, ("in", a, t + 1), 1)
                g.add(mid, ("in", b, t + 1), 1)

    for v in starts:
        g.add(S, ("in", v, 0), 1)
    for v in goals:
        g.add(("out", v, T), K, 1)
    return g, S, K


def _extract(g, S, K, starts, goals, T):
    """Decompose the unit flow into one cell-per-timestep path per agent."""
    paths = []
    for _ in starts:
        start_node = None
        for v in starts:
            if g.used(S, ("in", v, 0)):
                start_node = ("in", v, 0)
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
            else:
                nxt = _follow(g, node)
            _consume(g, node, nxt)
            node = nxt
        paths.append(path)
    return paths


def _follow(g, node):
    for ei in g.adj.get(node, ()):
        if g.orig[ei] == 1 and g.cap[ei] == 0:
            return g.to[ei]
    return ("K",)


def _consume(g, u, v):
    for ei in g.adj.get(u, ()):
        if g.to[ei] == v and g.orig[ei] == 1 and g.cap[ei] == 0:
            g.cap[ei] = 1
            return


def _team_feasible(grid, starts, goals, T, fv, fe):
    g, S, K = _build(grid, starts, goals, T, fv, fe)
    return g.max_flow(S, K) == len(starts), g, S, K


def _team_plan(grid, starts, goals, fv, fe, max_makespan):
    """Minimum-makespan anonymous flow for one team under constraints.

    Returns ``(paths, makespan)`` with ``paths`` a list of per-timestep cell
    lists (length ``makespan + 1``), or ``None`` if infeasible within
    ``max_makespan``."""
    if not _team_feasible(grid, starts, goals, max_makespan, fv, fe)[0]:
        return None
    lo, hi, best = 0, max_makespan, max_makespan
    while lo <= hi:
        mid = (lo + hi) // 2
        if _team_feasible(grid, starts, goals, mid, fv, fe)[0]:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    T = best
    _, g, S, K = _team_feasible(grid, starts, goals, T, fv, fe)
    return _extract(g, S, K, starts, goals, T), T


# --------------------------------------------------------------------------- #
# Inter-team conflict detection                                               #
# --------------------------------------------------------------------------- #
def _cell(path, t):
    return path[t] if t < len(path) else path[-1]


def _first_inter_team_conflict(team_paths):
    """Earliest conflict between agents of *different* teams, or ``None``.

    ``team_paths`` maps a team id to a list of per-agent paths. Returns
    ``("v", ti, tj, cell, t)`` for a shared cell or
    ``("e", ti, tj, u, v, t)`` for a head-on swap (team ``ti`` moves ``u->v``,
    team ``tj`` moves ``v->u``, arriving at ``t``)."""
    teams = list(team_paths)
    horizon = max((len(p) for paths in team_paths.values() for p in paths),
                  default=0)
    for t in range(horizon):
        # vertex: an agent of team ti and an agent of team tj sharing a cell
        occ: dict = {}
        for ti in teams:
            for p in team_paths[ti]:
                occ.setdefault(_cell(p, t), []).append(ti)
        for cell, owners in occ.items():
            for a in range(len(owners)):
                for b in range(a + 1, len(owners)):
                    if owners[a] != owners[b]:
                        return ("v", owners[a], owners[b], cell, t)
        # swap between teams
        for ia in range(len(teams)):
            for ib in range(ia + 1, len(teams)):
                ti, tj = teams[ia], teams[ib]
                for pa in team_paths[ti]:
                    for pb in team_paths[tj]:
                        au, av = _cell(pa, t), _cell(pa, t + 1)
                        bu, bv = _cell(pb, t), _cell(pb, t + 1)
                        if au != av and au == bv and av == bu:
                            return ("e", ti, tj, au, av, t + 1)
    return None


# --------------------------------------------------------------------------- #
# High level: CBS over teams                                                  #
# --------------------------------------------------------------------------- #
def cbm(grid: GridWorld, teams, *, max_makespan: int | None = None,
        max_expansions: int = 100_000, stats: dict | None = None):
    """Solve a TAPF instance (makespan-optimal), or return ``None``.

    ``teams`` is a list of ``(starts, goals)`` pairs — each a list of distinct
    free cells, equal in length (the team's agents and its interchangeable
    targets). Returns ``(paths, makespan)`` where ``paths`` maps an agent id
    ``(team_index, agent_index)`` to a per-timestep cell list (padded to the
    makespan), collision-free across all teams, each agent ending on one of its
    own team's targets — or ``None`` if infeasible. ``stats["expansions"]``
    records the high-level nodes expanded."""
    teams = [(list(s), list(go)) for (s, go) in teams]
    if max_makespan is None:
        free = sum(1 for x in range(grid.width) for y in range(grid.height)
                   if grid.is_free((x, y)))
        total = sum(len(s) for (s, _g) in teams)
        max_makespan = free + total + 2

    # Root: solve each team independently, no constraints.
    fv = {i: frozenset() for i in range(len(teams))}
    fe = {i: frozenset() for i in range(len(teams))}
    team_paths: dict = {}
    team_ms: dict = {}
    for i, (s, go) in enumerate(teams):
        res = _team_plan(grid, s, go, fv[i], fe[i], max_makespan)
        if res is None:
            return None
        team_paths[i], team_ms[i] = res

    counter = itertools.count()
    root_ms = max(team_ms.values()) if team_ms else 0
    open_heap = [(root_ms, next(counter), fv, fe, team_paths, team_ms)]

    expansions = 0
    while open_heap:
        ms, _, fv, fe, team_paths, team_ms = heapq.heappop(open_heap)
        expansions += 1
        if expansions > max_expansions:
            if stats is not None:
                stats["expansions"] = expansions
            return None

        conflict = _first_inter_team_conflict(team_paths)
        if conflict is None:
            if stats is not None:
                stats["expansions"] = expansions
            return _assemble(team_paths, ms), ms

        if conflict[0] == "v":
            _, ti, tj, cell, t = conflict
            branches = [(ti, "v", (cell, t)), (tj, "v", (cell, t))]
        else:
            _, ti, tj, u, v, t = conflict
            branches = [(ti, "e", (u, v, t)), (tj, "e", (v, u, t))]

        for team, kind, c in branches:
            c_fv = dict(fv)
            c_fe = dict(fe)
            if kind == "v":
                c_fv[team] = c_fv[team] | {c}
            else:
                c_fe[team] = c_fe[team] | {c}
            s, go = teams[team]
            res = _team_plan(grid, s, go, c_fv[team], c_fe[team], max_makespan)
            if res is None:
                continue
            c_paths = dict(team_paths)
            c_ms = dict(team_ms)
            c_paths[team], c_ms[team] = res
            heapq.heappush(
                open_heap,
                (max(c_ms.values()), next(counter), c_fv, c_fe, c_paths, c_ms),
            )

    if stats is not None:
        stats["expansions"] = expansions
    return None


def _assemble(team_paths, makespan):
    """Flatten per-team flow paths into an agent-id -> padded-path dict."""
    out: dict = {}
    for ti, paths in team_paths.items():
        for ai, p in enumerate(paths):
            padded = list(p) + [p[-1]] * (makespan + 1 - len(p))
            out[(ti, ai)] = padded
    return out
