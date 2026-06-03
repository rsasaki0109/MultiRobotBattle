"""Corridor symmetry reasoning for Conflict-Based Search.

Li, Harabor, Stuckey, Felner & Koenig, *"New Techniques for Pairwise Symmetry
Breaking in Multi-Agent Path Finding"* (ICAPS 2020) and its journal extension
*"Pairwise symmetry reasoning for multi-agent path finding search"* (AIJ 2021).
This completes the symmetry-breaking trilogy alongside the rectangle
(:mod:`mrn_coord.mapf.rectangle`) and mutex (:mod:`mrn_coord.mapf.mutex`)
reasoning already in the repo.

A **corridor symmetry** arises when two agents traverse the same one-wide
passage in *opposite* directions. They must conflict somewhere inside, and plain
CBS/CBSH resolves it one cell at a time: forbidding the meeting cell to one
agent just shifts the head-on collision by one cell, so the search branches
through a number of symmetric resolutions that grows with the corridor length
before one agent is finally forced to wait the whole corridor out.

The fix is a **range (length-of-time) constraint** — a new constraint shape
beyond the single ``(cell, time)`` vertex constraint — that forbids an agent
from a corridor *endpoint* across a whole *range* of timesteps, forcing the
other agent through first in a single split. Because the two agents cross in
opposite directions, they *share* the two openings: one agent's **exit** opening
``P`` is the other's **entry**, and vice versa for ``Q``. To let ``a1`` go first
we forbid ``a2`` from its entry ``P`` until ``a1`` has cleared it — and since the
corridor is one-wide and ``P`` is the only way in, that *holds ``a2`` outside*
rather than merely delaying where it surfaces. With ``d_i`` the earliest time
agent ``a_i`` reaches its own exit opening, the two children are:

- block ``a2`` from ``P`` (its entry, ``a1``'s exit) for all ``t in [0, d1]``
  (so ``a1`` goes first), or
- block ``a1`` from ``Q`` (its entry, ``a2``'s exit) for all ``t in [0, d2]``
  (``a2`` first).

These two are *mutually disjunctive* **when the corridor is the only route
between its two sides**: every collision-free solution then has one agent fully
traverse before the other enters, so it satisfies at least one child and the
split preserves CBS's optimality and completeness. The reasoning therefore fires
only when neither agent has a **bypass** (a route to its exit opening that
avoids the corridor); if a detour exists, a bypassing solution could satisfy
neither child, so the high level falls back to the plain single-cell split. This
is the conservative, provably-sound core of the technique — the meeting cell can
no longer be shifted one step at a time, collapsing the whole chain to one
split.

A range constraint needs no new low-level machinery: it expands to the set of
``(endpoint, t)`` vertex constraints over the range, which
:func:`mrn_coord.mapf.space_time_astar.plan_path` already honors. The reasoning
is wired into :func:`mrn_coord.mapf.cbsh.cbsh` behind ``corridor=True`` (off by
default).
"""

from __future__ import annotations

from collections import deque

from .conflicts import EdgeConflict, VertexConflict
from .grid import GridWorld


def _free_neighbors(grid: GridWorld, cell):
    x, y = cell
    return [n for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
            if grid.is_free(n)]


def _degree(grid: GridWorld, cell) -> int:
    """Number of free orthogonal neighbours — a *corridor cell* has degree 2."""
    return len(_free_neighbors(grid, cell))


def _extend(grid, frm, to):
    """Walk the maximal degree-2 chain from ``to`` (entered from ``frm``).

    Returns ``(corridor_cells, opening)``: the degree-2 cells in order leading
    away from ``frm``, and the first cell with degree != 2 at the end (the
    *opening* into the rest of the map), or ``None`` if the chain has no opening.
    """
    cells = []
    prev, cur = frm, to
    while _degree(grid, cur) == 2:
        cells.append(cur)
        nxt = [n for n in _free_neighbors(grid, cur) if n != prev]
        if not nxt:
            return cells, None
        prev, cur = cur, nxt[0]
    return cells, cur


def _trace_corridor(grid, seed):
    """The maximal degree-2 chain through ``seed`` with its two openings.

    Returns ``(cells, opening_a, opening_b)`` — ``cells`` ordered from the
    ``opening_a`` end to the ``opening_b`` end — or ``None`` if ``seed`` is not a
    corridor cell, an opening is missing, or the chain loops onto itself.
    """
    if _degree(grid, seed) != 2:
        return None
    n0, n1 = _free_neighbors(grid, seed)
    left_cells, opening_a = _extend(grid, seed, n0)
    right_cells, opening_b = _extend(grid, seed, n1)
    if opening_a is None or opening_b is None:
        return None
    if opening_a == opening_b:
        return None  # a ring, not a through-corridor
    cells = list(reversed(left_cells)) + [seed] + right_cells
    return cells, opening_a, opening_b


def _bfs_dist(grid, src, dst, blocked=frozenset()):
    """Shortest 4-connected distance ``src -> dst``, avoiding ``blocked`` cells,
    or ``None`` if unreachable."""
    if src == dst:
        return 0
    seen = {src}
    queue = deque([(src, 0)])
    while queue:
        cell, dist = queue.popleft()
        for nb in _free_neighbors(grid, cell):
            if nb in blocked or nb in seen:
                continue
            if nb == dst:
                return dist + 1
            seen.add(nb)
            queue.append((nb, dist + 1))
    return None


def _first_at(path, cell):
    """First timestep ``path`` occupies ``cell``, or ``None``."""
    for t, c in enumerate(path):
        if c == cell:
            return t
    return None


def try_corridor(grid: GridWorld, agents: dict, paths: dict, conflict):
    """A corridor symmetry bracketing ``conflict``, as a disjunctive split.

    Returns ``(agent_a, barrier_a, agent_b, barrier_b, klass)`` where each
    ``barrier`` is a set of ``(endpoint, time)`` vertex constraints to add to its
    agent (one of the two range constraints), and ``klass`` is 2 when both
    barriers cut their agent's current path (the case the reasoning fires on).
    Returns ``None`` if the conflict is not an opposite-direction corridor
    crossing, or the split would make no progress.
    """
    a1, a2 = conflict.agent_a, conflict.agent_b
    if isinstance(conflict, VertexConflict):
        seeds = [conflict.cell]
    elif isinstance(conflict, EdgeConflict):
        seeds = [conflict.cell_a, conflict.cell_b]
    else:
        return None

    for seed in seeds:
        traced = _trace_corridor(grid, seed)
        if traced is None:
            continue
        cells, opening_a, opening_b = traced
        split = _build_split(grid, agents, paths, a1, a2, cells,
                             opening_a, opening_b)
        if split is not None:
            return split
    return None


def _build_split(grid, agents, paths, a1, a2, cells, oa, ob):
    corridor_set = frozenset(cells)
    p1, p2 = paths[a1], paths[a2]

    # Both agents must traverse the corridor end to end (pass both openings).
    t1a, t1b = _first_at(p1, oa), _first_at(p1, ob)
    t2a, t2b = _first_at(p2, oa), _first_at(p2, ob)
    if None in (t1a, t1b, t2a, t2b):
        return None

    # Entry is the earlier opening, exit the later one.
    a1_entry, a1_exit = (oa, ob) if t1a < t1b else (ob, oa)
    a2_entry, a2_exit = (oa, ob) if t2a < t2b else (ob, oa)
    # The corridor symmetry is the *opposite*-direction crossing: the agents
    # share the openings, so a1's exit is a2's entry and vice versa.
    if a1_exit == a2_exit:
        return None
    if a1_exit != a2_entry or a2_exit != a1_entry:
        return None  # not a clean shared-opening crossing

    s1, s2 = agents[a1][0], agents[a2][0]
    # P = a1's exit = a2's entry; Q = a2's exit = a1's entry.
    P, Q = a1_exit, a2_exit
    if s2 == P or s1 == Q:
        return None  # degenerate: an agent already sits on the cell to forbid

    # Earliest each agent reaches its OWN exit opening — a lower bound, so the
    # range stays sound however long the current paths happen to be.
    d1 = _bfs_dist(grid, s1, a1_exit)
    d2 = _bfs_dist(grid, s2, a2_exit)
    if d1 is None or d2 is None:
        return None
    # The disjunction is only exhaustive when the corridor is the SOLE route
    # between its two sides. If either agent can reach its exit opening without
    # the corridor (a bypass), a go-around solution might satisfy neither child,
    # so we decline and let the caller fall back to a plain split.
    if (_bfs_dist(grid, s1, a1_exit, blocked=corridor_set) is not None or
            _bfs_dist(grid, s2, a2_exit, blocked=corridor_set) is not None):
        return None

    # Child A ("a1 first"): hold a2 outside its entry P until a1 has cleared it.
    # Child B ("a2 first"): hold a1 outside its entry Q until a2 has cleared it.
    barrier_a = frozenset((P, t) for t in range(d1 + 1))
    barrier_b = frozenset((Q, t) for t in range(d2 + 1))
    # Child A constrains a2; child B constrains a1. Both are cardinal here (each
    # forces its agent to wait out the whole corridor), so klass = 2.
    return (a2, barrier_a, a1, barrier_b, 2)
