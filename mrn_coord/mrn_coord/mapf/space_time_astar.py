"""Low-level single-agent planner: space-time A* with constraints.

Plans one agent over ``(cell, time)`` states, one timestep per move (a wait
counts as a step). It honors two kinds of constraints, the vocabulary
Conflict-Based Search and prioritized planning branch on:

- **vertex** ``(cell, time)`` — the agent may not occupy ``cell`` at ``time``.
- **edge** ``(frm, to, time)`` — the agent may not move ``frm -> to`` arriving
  at ``time`` (i.e. be at ``frm`` at ``time-1`` and ``to`` at ``time``). This
  is how swap conflicts are forbidden.

It also honors the *positive* (must-occupy) constraints that disjoint splitting
(:func:`mrn_coord.mapf.cbs.cbs` with ``disjoint=True``) branches on:

- **positive vertex** ``(cell, time)`` — the agent *must* be at ``cell`` at
  ``time``.
- **positive edge** ``(frm, to, time)`` — the agent *must* move ``frm -> to``
  arriving at ``time``.

These prune every successor that would violate them, and they hold the agent
past the last positive timestep before it may settle at its goal. They default
empty, so a call that passes none behaves exactly as before.

The goal test requires the agent to be at the goal *and* past the last time the
goal is vertex-constrained, so a returned path can be safely held at the goal
forever (the planner assumes the agent waits at its goal after arrival).
"""

from __future__ import annotations

import heapq

from .grid import Cell, GridWorld, manhattan


def _default_max_time(grid: GridWorld, vertex_constraints, edge_constraints,
                      positive_vertex=(), positive_edge=()) -> int:
    # Generous upper bound on the time horizon so the (cell, time) state space
    # is finite. Big enough to let an agent wait out the longest reservation.
    return (
        2 * grid.width * grid.height
        + len(vertex_constraints)
        + len(edge_constraints)
        + len(positive_vertex)
        + len(positive_edge)
        + grid.width
        + grid.height
        + 5
    )


def plan_path(
    grid: GridWorld,
    start: Cell,
    goal: Cell,
    vertex_constraints=frozenset(),
    edge_constraints=frozenset(),
    *,
    positive_vertex=frozenset(),
    positive_edge=frozenset(),
    max_time: int | None = None,
    stats: dict | None = None,
) -> list[Cell] | None:
    """Find a minimal-time path from ``start`` to ``goal``.

    ``vertex_constraints`` is a set of ``(cell, time)``; ``edge_constraints`` is
    a set of ``(frm, to, time)``. ``positive_vertex`` is a set of ``(cell,
    time)`` the agent *must* occupy and ``positive_edge`` a set of ``(frm, to,
    time)`` moves it *must* make — the must-occupy half of disjoint splitting;
    both default empty (a call passing none behaves exactly as plain CBS). Returns
    the path as a list of cells indexed by timestep (``path[t]`` is the cell at
    time ``t``), or ``None`` if no path exists within the time horizon (including
    when the positive constraints are mutually contradictory). If ``stats`` is
    given, ``stats["expansions"]`` is set to the number of ``(cell, time)`` states
    expanded — for comparison against the safe-interval planner
    (:func:`mrn_coord.mapf.sipp.plan_sipp`).
    """
    if not grid.is_free(start) or not grid.is_free(goal):
        return None
    if (start, 0) in vertex_constraints:
        return None

    # Index the positive constraints by their (arrival) time. Two positive
    # vertices at the same time, or a start that contradicts a positive vertex
    # at t=0, make the instance infeasible.
    pos_v: dict[int, Cell] = {}
    for cell, t in positive_vertex:
        if pos_v.get(t, cell) != cell:
            return None
        pos_v[t] = cell
    if pos_v.get(0, start) != start:
        return None
    pos_e: dict[int, tuple[Cell, Cell]] = {}
    for frm, to, t in positive_edge:
        if pos_e.get(t, (frm, to)) != (frm, to):
            return None
        pos_e[t] = (frm, to)

    if max_time is None:
        max_time = _default_max_time(
            grid, vertex_constraints, edge_constraints,
            positive_vertex, positive_edge,
        )

    # The agent must settle at the goal after the last constraint that touches it
    # — its goal vertex constraints and any positive constraint that pins it to a
    # *non-goal* cell (which it cannot honor once it has parked at the goal, so it
    # must not terminate before then). A positive constraint *on* the goal is free
    # under stay-at-goal semantics — the must-occupy check below already forces an
    # arrival by that time — so it does not extend the settle time.
    goal_constraint_times = [t for (cell, t) in vertex_constraints if cell == goal]
    last_goal_time = max(goal_constraint_times) if goal_constraint_times else 0
    for t, cell in pos_v.items():
        if cell != goal:
            last_goal_time = max(last_goal_time, t)
    for t, (frm, to) in pos_e.items():
        if to != goal:
            last_goal_time = max(last_goal_time, t)

    open_heap: list[tuple[int, int, int, Cell, int]] = []
    counter = 0
    heapq.heappush(open_heap, (manhattan(start, goal), 0, counter, start, 0))
    came_from: dict[tuple[Cell, int], tuple[Cell, int]] = {}
    visited: set[tuple[Cell, int]] = set()

    while open_heap:
        _, g, _, cell, t = heapq.heappop(open_heap)
        state = (cell, t)
        if state in visited:
            continue
        visited.add(state)

        if cell == goal and t >= last_goal_time:
            if stats is not None:
                stats["expansions"] = len(visited)
            return _reconstruct(came_from, state)

        if t >= max_time:
            continue

        nt = t + 1
        must_cell = pos_v.get(nt)
        must_edge = pos_e.get(nt)
        for ncell in grid.neighbors(cell):
            if (ncell, nt) in vertex_constraints:
                continue
            if (cell, ncell, nt) in edge_constraints:
                continue
            if must_cell is not None and ncell != must_cell:
                continue
            if must_edge is not None and (cell, ncell) != must_edge:
                continue
            nstate = (ncell, nt)
            if nstate in visited:
                continue
            came_from.setdefault(nstate, state)
            counter += 1
            f = nt + manhattan(ncell, goal)
            heapq.heappush(open_heap, (f, nt, counter, ncell, nt))

    if stats is not None:
        stats["expansions"] = len(visited)
    return None


def _reconstruct(came_from, state) -> list[Cell]:
    path = [state]
    while state in came_from:
        state = came_from[state]
        path.append(state)
    path.reverse()
    return [cell for (cell, _) in path]
