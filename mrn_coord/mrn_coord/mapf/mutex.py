"""Mutex propagation for Conflict-Based Search.

Zhang, Li, Surynek, Koenig & Kumar, *"Multi-Agent Path Finding with Mutex
Propagation"* (ICAPS 2020; AIJ 2022). Mutex propagation is a constraint-
propagation technique from classical planning, adapted here to a pair of MDDs.
It generalizes the hand-designed rectangle reasoning of
:mod:`mrn_coord.mapf.rectangle`: rather than recognising one geometric pattern,
it *derives* which pairs of MDD nodes can never be reached conflict-free, and
from that both classifies cardinal conflicts and synthesizes symmetry-breaking
constraints automatically.

The unit is a **mutex** between two MDD nodes (or edges) at the same level:

- *Initial* node mutex — same cell at the same level (a vertex conflict).
- *Initial* edge mutex — the two edges swap endpoints (a swap conflict).
- *Propagated* node mutex — two nodes whose every pair of incoming edges is
  mutex.
- *Propagated* edge mutex — two edges whose source nodes are mutex.

The central guarantee (the paper's Theorem 1) is: two MDD nodes at level ``t``
are mutex **iff** there is no pair of conflict-free sub-paths reaching them. So a
node mutex between the two agents' *sinks* means every pair of cost-optimal paths
collides — a cardinal conflict — and the cells that are mutex with the whole
opposite MDD become the disjunctive symmetry-breaking constraints.

This module exposes:

- :func:`generate_mutexes` — all node/edge mutexes between two MDDs (Algorithm 1).
- :func:`classify_conflict` — ``"PC"`` / ``"AC"`` / ``"NC"`` (Algorithm 2): a
  pre-goal cardinal, after-goal cardinal, or not-cardinal conflict. By Theorem 2,
  ``"NC"`` iff a conflict-free pair of paths of those costs exists — the same
  fact :func:`mrn_coord.mapf.mdd.are_dependent` computes, here from mutexes.
- :func:`pc_constraints` — the disjunctive constraint sets for a pre-goal
  cardinal (Algorithm 3); these *generalize* the rectangle barrier (on a
  rectangle conflict they reduce to exactly the barrier of
  :mod:`mrn_coord.mapf.rectangle`).

**Honest scope.** This module reproduces mutex propagation as a *detector* — the
verified core. It is deliberately not wired into the CBS high level as a
brancher: the paper's full constraint-generation loop (Algorithm 5) grows the
MDD levels to the cardinal boundary and, for after-goal cardinals, adds *cost*
constraints, regenerating all mutexes at each grown level. In pure Python that is
prohibitively slow on corridor-style conflicts (the paper itself notes mutex
propagation is "computationally expensive"), so a gated solver built on it would
not be practical. What *is* fast, correct, and verifiable — and what the gate
pins — is the detector: mutex classification agrees with the direct dependency
test (Theorem 2) and catches cardinal conflicts the width-based test misses.
"""

from __future__ import annotations

from .grid import GridWorld
from .mdd import Mdd


def _nodes(mdd: Mdd, t: int) -> set:
    """Cells of the MDD at level ``t`` (``{goal}`` once the agent has parked)."""
    return mdd.cells(t)


def _out_edges(grid: GridWorld, mdd: Mdd, t: int):
    """MDD edges from level ``t`` to ``t+1`` as ``(u, v)`` cell pairs."""
    here = _nodes(mdd, t)
    nxt = _nodes(mdd, t + 1)
    out = []
    for u in here:
        for v in grid.neighbors(u):
            if v in nxt:
                out.append((u, v))
    return out


def generate_mutexes(grid: GridWorld, mdd_i: Mdd, mdd_j: Mdd):
    """All mutexes between two MDDs (Algorithm 1, level-ordered propagation).

    Returns ``(node_mutex, horizon)`` where ``node_mutex`` is a set of
    ``(level, loc_i, loc_j)`` triples — the node mutexes, which are what conflict
    classification and constraint generation need.
    """
    horizon = max(mdd_i.cost, mdd_j.cost)

    node_mutex: set = set()
    edge_mutex: set = set()

    # Initial node mutexes: same cell, same level (vertex conflict).
    for t in range(horizon + 1):
        common = _nodes(mdd_i, t) & _nodes(mdd_j, t)
        for loc in common:
            node_mutex.add((t, loc, loc))

    # Initial edge mutexes: the two edges swap endpoints (swap conflict).
    out_i = [_out_edges(grid, mdd_i, t) for t in range(horizon)]
    out_j = [_out_edges(grid, mdd_j, t) for t in range(horizon)]
    for t in range(horizon):
        for ui, vi in out_i[t]:
            for uj, vj in out_j[t]:
                if ui == vj and uj == vi:
                    edge_mutex.add((t, ui, vi, uj, vj))

    # Forward propagation, level by level. Node mutexes at level t spawn edge
    # mutexes at level t (out-edges); edge mutexes at level t are gathered to
    # decide node mutexes at level t+1 (a node pair is mutex iff *all* of its
    # incoming edge pairs are mutex).
    for t in range(horizon):
        # node mutex (t) -> edge mutex (t)
        for (tt, ai, bj) in [m for m in node_mutex if m[0] == t]:
            for ui, vi in out_i[t]:
                if ui != ai:
                    continue
                for uj, vj in out_j[t]:
                    if uj != bj:
                        continue
                    edge_mutex.add((t, ui, vi, uj, vj))

        # edge mutex (t) -> candidate node mutex (t+1)
        em_t = {(ui, vi, uj, vj) for (tt, ui, vi, uj, vj) in edge_mutex
                if tt == t}
        # group incoming edges by their target cell pair (vi, vj)
        nodes_i1 = _nodes(mdd_i, t + 1)
        nodes_j1 = _nodes(mdd_j, t + 1)
        in_i = {v: [] for v in nodes_i1}
        for ui, vi in out_i[t]:
            in_i[vi].append(ui)
        in_j = {v: [] for v in nodes_j1}
        for uj, vj in out_j[t]:
            in_j[vj].append(uj)
        for vi in nodes_i1:
            preds_i = in_i[vi]
            if not preds_i:
                continue
            for vj in nodes_j1:
                preds_j = in_j[vj]
                if not preds_j:
                    continue
                if all((ui, vi, uj, vj) in em_t
                       for ui in preds_i for uj in preds_j):
                    node_mutex.add((t + 1, vi, vj))

    return node_mutex, horizon


def classify_conflict(grid: GridWorld, mdd_i: Mdd, mdd_j: Mdd) -> str:
    """Classify the conflict between two agents at the given MDD costs.

    Requires ``mdd_i.cost <= mdd_j.cost``. Returns ``"PC"`` (pre-goal cardinal),
    ``"AC"`` (after-goal cardinal), or ``"NC"`` (not cardinal). By the paper's
    Theorem 2, a conflict-free pair of paths of these costs exists iff the result
    is ``"NC"`` (Algorithm 2).
    """
    li = mdd_i.cost
    gi = mdd_i.goal
    node_mutex, _ = generate_mutexes(grid, mdd_i, mdd_j)

    # MDD nodes of MDD_j at level li that are NOT mutex with MDD_i's sink (gi).
    nj = [v for v in _nodes(mdd_j, li) if (li, gi, v) not in node_mutex]
    if not nj:
        return "PC"
    for v in nj:
        if _reaches_sink_avoiding(grid, mdd_j, v, li, gi):
            return "NC"
    return "AC"


def _reaches_sink_avoiding(grid, mdd_j, start_loc, start_level, avoid_loc):
    """Is there a sub-path in ``mdd_j`` from ``(start_loc, start_level)`` to its
    sink that never occupies ``avoid_loc`` (the other agent's goal)?"""
    if start_loc == avoid_loc:
        return False
    reach = {start_loc}
    for t in range(start_level, mdd_j.cost):
        nxt_level = _nodes(mdd_j, t + 1)
        nxt = set()
        for u in reach:
            for v in grid.neighbors(u):
                if v in nxt_level and v != avoid_loc:
                    nxt.add(v)
        reach = nxt
        if not reach:
            return False
    return mdd_j.goal in reach


def pc_constraints(grid: GridWorld, mdd_i: Mdd, mdd_j: Mdd):
    """Disjunctive constraint sets for a pre-goal cardinal conflict (Algorithm 3).

    Requires ``mdd_i.cost <= mdd_j.cost``. Returns ``(ci, cj)``: ``ci`` is the
    set of ``(cell, time)`` vertex constraints for agent *i* (every MDD-``i`` node
    mutex with *all* of MDD-``j`` at its level), ``cj`` likewise for agent *j*.
    Adding ``ci`` to one child and ``cj`` to the other is a complete disjunctive
    split (the paper's Property 3).
    """
    node_mutex, horizon = generate_mutexes(grid, mdd_i, mdd_j)
    ci = set()
    cj = set()
    for t in range(horizon + 1):
        nodes_i = _nodes(mdd_i, t)
        nodes_j = _nodes(mdd_j, t)
        for u in nodes_i:
            if all((t, u, w) in node_mutex for w in nodes_j):
                ci.add((u, t))
        for w in nodes_j:
            if all((t, u, w) in node_mutex for u in nodes_i):
                cj.add((w, t))
    return frozenset(ci), frozenset(cj)
