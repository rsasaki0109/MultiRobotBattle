"""Multi-Label A* (Grenouilleau, van Hoeve & Hooker, ICAPS 2019).

The single-agent low level for **pickup-and-delivery** MAPF (MAPD): instead of
one start->goal path, an agent must visit an *ordered sequence* of goals — drive
to the pickup, collect, then drive to the delivery. The natural baseline plans
this with **two sequential** :func:`mrn_coord.mapf.space_time_astar.plan_path`
calls (start->pickup, then pickup->delivery). That over-constrains: the first
call treats the pickup as a *terminal* goal and so assumes the agent **rests at
the pickup until the end of the horizon**. Two failure modes follow (the paper's
Case 1 / Case 2):

- **Case 1** — the pickup is another agent's parked endpoint (reserved
  indefinitely). The two-step plan cannot settle there, so it reports *no path* —
  even though the agent could pass *through* the pickup and reach its delivery.
- **Case 2** — another agent is scheduled to pass through the pickup later. The
  two-step plan delays the agent at the pickup until after that reservation,
  where it could instead pass through and carry on.

Multi-Label A* (MLA*) plans the whole start->pickup->delivery route in **one** A*
search over augmented states ``(cell, time, label)``. The *label* records the
current leg: ``1`` = heading to the pickup, ``2`` = heading to the delivery.
Reaching the pickup at label ``1`` spawns a label-``2`` node at the **same cell
and time** (collect the package, no rest); the search ends when a label-``2``
node reaches the delivery. The heuristic stays admissible across the legs:

    h(cell, 1) = dist(cell, pickup) + dist(pickup, delivery)
    h(cell, 2) = dist(cell, delivery)

Because the agent never has to settle at the pickup, MLA* finds paths the
two-step search misses (Case 1), shorter paths where it would wait (Case 2), and
— in the contended regime it is built for — expands fewer states than the two
separate searches combined. All search respects a ``(vertex, edge)`` reservation
table (the shared *token* of the lifelong engines), so the returned path is
collision-free against the already-committed agents.
"""

from __future__ import annotations

import heapq

from .grid import Cell, GridWorld, manhattan
from .space_time_astar import plan_path


def mla_star(
    grid: GridWorld,
    start: Cell,
    pickup: Cell,
    delivery: Cell,
    vertex_constraints=frozenset(),
    edge_constraints=frozenset(),
    *,
    max_time: int | None = None,
    stats: dict | None = None,
) -> list[Cell] | None:
    """Plan one agent ``start -> pickup -> delivery`` in a single multi-label A*.

    ``vertex_constraints`` is a set of ``(cell, time)`` and ``edge_constraints``
    a set of ``(frm, to, time)`` the agent must avoid — the reservation table of
    the other agents, in :func:`mrn_coord.mapf.space_time_astar.plan_path`'s
    format. Returns the path as a list of cells indexed by timestep (it visits
    ``pickup`` before ending at ``delivery``), or ``None`` if no such path exists
    within the horizon. If ``stats`` is given, ``stats["expanded"]`` is the number
    of ``(cell, time, label)`` states expanded and ``stats["created"]`` the number
    generated — compare against the two sequential searches of
    :func:`two_step_plan`.

    The agent is *not* required to rest at the delivery (the paper's Algorithm 1
    returns on reaching it); holding the endpoint afterward is the assignment
    layer's concern, not the low level's.
    """
    if not (grid.is_free(start) and grid.is_free(pickup)
            and grid.is_free(delivery)):
        return None
    if (start, 0) in vertex_constraints:
        return None
    if max_time is None:
        max_time = (4 * grid.width * grid.height
                    + len(vertex_constraints) + len(edge_constraints) + 10)

    def h(cell: Cell, label: int) -> int:
        if label == 1:
            return manhattan(cell, pickup) + manhattan(pickup, delivery)
        return manhattan(cell, delivery)

    # node: (f, g, tiebreak, cell, t, label)
    open_heap = [(h(start, 1), 0, 0, start, 0, 1)]
    came_from: dict = {}
    visited: set = set()
    counter = 0
    created = 1

    while open_heap:
        _, g, _, cell, t, label = heapq.heappop(open_heap)
        state = (cell, t, label)
        if state in visited:
            continue
        visited.add(state)

        if label == 2 and cell == delivery:
            if stats is not None:
                stats["expanded"] = len(visited)
                stats["created"] = created
            return _reconstruct(came_from, state)

        # collect at the pickup: a label-2 node at the same cell and time (the
        # leg switch costs no timestep), so the agent passes through rather than
        # resting there.
        if label == 1 and cell == pickup:
            nxt = (pickup, t, 2)
            if nxt not in visited:
                came_from.setdefault(nxt, state)
                counter += 1
                created += 1
                heapq.heappush(open_heap,
                               (g + h(pickup, 2), g, counter, pickup, t, 2))

        if t >= max_time:
            continue
        nt = t + 1
        for ncell in grid.neighbors(cell):
            if (ncell, nt) in vertex_constraints:
                continue
            if (cell, ncell, nt) in edge_constraints:
                continue
            nstate = (ncell, nt, label)
            if nstate in visited:
                continue
            came_from.setdefault(nstate, state)
            counter += 1
            created += 1
            f = nt + h(ncell, label)
            heapq.heappush(open_heap, (f, nt, counter, ncell, nt, label))

    if stats is not None:
        stats["expanded"] = len(visited)
        stats["created"] = created
    return None


def _reconstruct(came_from: dict, state: tuple) -> list[Cell]:
    seq = [state]
    while state in came_from:
        state = came_from[state]
        seq.append(state)
    seq.reverse()
    # The label flip at the pickup shares a timestep with its predecessor; emit
    # one cell per timestep so the result is a plain space-time path.
    cells: list[Cell] = []
    last_t = None
    for (cell, t, _label) in seq:
        if t == last_t:
            continue
        cells.append(cell)
        last_t = t
    return cells


def two_step_plan(
    grid: GridWorld,
    start: Cell,
    pickup: Cell,
    delivery: Cell,
    vertex_constraints=frozenset(),
    edge_constraints=frozenset(),
    *,
    max_time: int | None = None,
    stats: dict | None = None,
) -> list[Cell] | None:
    """The two-sequential-A* baseline MLA* improves on.

    Plans ``start -> pickup`` and then ``pickup -> delivery`` with two
    :func:`plan_path` calls, composing them at the pickup arrival time. Because
    ``plan_path`` settles the agent at its goal, the first leg assumes the agent
    rests at the pickup — the over-constraint that makes this fail (Case 1) or
    wait (Case 2) where :func:`mla_star` does not. Reservations on the second leg
    are shifted into its local clock (``t -> t - arrival``). ``stats["expanded"]``
    is the combined number of states the two searches expand.
    """
    s1: dict = {}
    leg1 = plan_path(grid, start, pickup, vertex_constraints, edge_constraints,
                     max_time=max_time, stats=s1)
    if leg1 is None:
        if stats is not None:
            stats["expanded"] = s1.get("expansions", 0)
        return None
    arrival = len(leg1) - 1
    v2 = frozenset((c, t - arrival) for (c, t) in vertex_constraints
                   if t >= arrival)
    e2 = frozenset((a, b, t - arrival) for (a, b, t) in edge_constraints
                   if t >= arrival)
    mt2 = None if max_time is None else max_time - arrival
    s2: dict = {}
    leg2 = plan_path(grid, pickup, delivery, v2, e2, max_time=mt2, stats=s2)
    if stats is not None:
        stats["expanded"] = s1.get("expansions", 0) + s2.get("expansions", 0)
    if leg2 is None:
        return None
    return leg1 + leg2[1:]
