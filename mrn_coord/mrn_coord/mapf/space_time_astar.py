"""Low-level single-agent planner: space-time A* with constraints.

Plans one agent over ``(cell, time)`` states, one timestep per move (a wait
counts as a step). It honors two kinds of constraints, the vocabulary
Conflict-Based Search and prioritized planning branch on:

- **vertex** ``(cell, time)`` — the agent may not occupy ``cell`` at ``time``.
- **edge** ``(frm, to, time)`` — the agent may not move ``frm -> to`` arriving
  at ``time`` (i.e. be at ``frm`` at ``time-1`` and ``to`` at ``time``). This
  is how swap conflicts are forbidden.

The goal test requires the agent to be at the goal *and* past the last time the
goal is vertex-constrained, so a returned path can be safely held at the goal
forever (the planner assumes the agent waits at its goal after arrival).
"""

from __future__ import annotations

import heapq

from .grid import Cell, GridWorld, manhattan


def _default_max_time(grid: GridWorld, vertex_constraints, edge_constraints) -> int:
    # Generous upper bound on the time horizon so the (cell, time) state space
    # is finite. Big enough to let an agent wait out the longest reservation.
    return (
        2 * grid.width * grid.height
        + len(vertex_constraints)
        + len(edge_constraints)
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
    max_time: int | None = None,
) -> list[Cell] | None:
    """Find a minimal-time path from ``start`` to ``goal``.

    ``vertex_constraints`` is a set of ``(cell, time)``; ``edge_constraints`` is
    a set of ``(frm, to, time)``. Returns the path as a list of cells indexed by
    timestep (``path[t]`` is the cell at time ``t``), or ``None`` if no path
    exists within the time horizon.
    """
    if not grid.is_free(start) or not grid.is_free(goal):
        return None
    if (start, 0) in vertex_constraints:
        return None
    if max_time is None:
        max_time = _default_max_time(grid, vertex_constraints, edge_constraints)

    # The agent must settle at the goal after the last constraint that touches it.
    goal_constraint_times = [t for (cell, t) in vertex_constraints if cell == goal]
    last_goal_time = max(goal_constraint_times) if goal_constraint_times else 0

    start_state = (start, 0)
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
            return _reconstruct(came_from, state)

        if t >= max_time:
            continue

        nt = t + 1
        for ncell in grid.neighbors(cell):
            if (ncell, nt) in vertex_constraints:
                continue
            if (cell, ncell, nt) in edge_constraints:
                continue
            nstate = (ncell, nt)
            if nstate in visited:
                continue
            came_from.setdefault(nstate, state)
            counter += 1
            f = nt + manhattan(ncell, goal)
            heapq.heappush(open_heap, (f, nt, counter, ncell, nt))

    return None


def _reconstruct(came_from, state) -> list[Cell]:
    path = [state]
    while state in came_from:
        state = came_from[state]
        path.append(state)
    path.reverse()
    return [cell for (cell, _) in path]
