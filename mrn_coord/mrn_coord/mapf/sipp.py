"""Safe Interval Path Planning (SIPP, Phillips & Likhachev 2011).

A drop-in alternative to :func:`mrn_coord.mapf.space_time_astar.plan_path` that
plans the *same* minimal-time single-agent path, but searches a far smaller
state space. Time-expanded A* expands one state per ``(cell, time)``; when an
agent must wait out a long reservation, that is many near-identical states.
SIPP instead expands one state per ``(cell, safe interval)`` — a maximal run of
contiguous timesteps during which the cell is collision-free — and lets the
agent wait *anywhere* inside an interval for free. A corridor blocked for 200
ticks costs SIPP one state, not 200.

Same constraint vocabulary as the time-expanded planner, so it is a drop-in low
level for prioritized planning (and, in principle, CBS):

- **vertex** ``(cell, time)`` — partitions each cell's timeline into safe
  intervals.
- **edge** ``(frm, to, time)`` — forbids the transition ``frm -> to`` arriving
  at ``time``; handled by skipping that single arrival time into the successor.

Returns a path as a list of cells indexed by timestep (waits expanded back out),
identical in format and in arrival time to ``plan_path``, so the two are
interchangeable and their costs/makespans match. Pass a ``stats`` dict to read
back the number of expansions — the headline SIPP win.
"""

from __future__ import annotations

import heapq

from .grid import Cell, GridWorld, manhattan
from .space_time_astar import _default_max_time

INF = float("inf")


def _safe_intervals(blocked) -> list:
    """Maximal runs of timesteps a cell is free, given its blocked instants.

    ``blocked`` is an iterable of integer times the cell is occupied. Returns a
    sorted list of ``(lo, hi)`` closed intervals covering ``[0, INF)``; the last
    interval always extends to ``INF`` so an agent can settle forever.
    """
    times = sorted(set(blocked))
    if not times:
        return [(0, INF)]
    intervals = []
    lo = 0
    for bt in times:
        if bt > lo:
            intervals.append((lo, bt - 1))
        lo = bt + 1
    intervals.append((lo, INF))
    return intervals


def plan_sipp(
    grid: GridWorld,
    start: Cell,
    goal: Cell,
    vertex_constraints=frozenset(),
    edge_constraints=frozenset(),
    *,
    max_time: int | None = None,
    stats: dict | None = None,
) -> list | None:
    """Minimal-time ``start -> goal`` path via safe-interval search.

    Signature matches :func:`plan_path` so this is a drop-in low-level planner.
    Returns the path as a list of cells indexed by timestep, or ``None`` if no
    path exists within the horizon. If ``stats`` is given, ``stats["expansions"]``
    is set to the number of states expanded.
    """
    if not grid.is_free(start) or not grid.is_free(goal):
        return None
    if (start, 0) in vertex_constraints:
        return None
    if max_time is None:
        max_time = _default_max_time(grid, vertex_constraints, edge_constraints)

    blocked_by_cell: dict = {}
    for (cell, t) in vertex_constraints:
        blocked_by_cell.setdefault(cell, set()).add(t)
    edge_by: dict = {}
    for (frm, to, t) in edge_constraints:
        edge_by.setdefault((frm, to), set()).add(t)

    interval_cache: dict = {}

    def intervals(cell):
        iv = interval_cache.get(cell)
        if iv is None:
            iv = _safe_intervals(blocked_by_cell.get(cell, ()))
            interval_cache[cell] = iv
        return iv

    # The interval of ``start`` containing time 0 exists (we checked (start, 0)).
    start_lo, start_hi = next(
        (lo, hi) for (lo, hi) in intervals(start) if lo <= 0 <= hi)

    goal_constraint_times = [t for (cell, t) in vertex_constraints if cell == goal]
    last_goal_time = max(goal_constraint_times) if goal_constraint_times else 0

    open_heap: list = []
    counter = 0
    heapq.heappush(
        open_heap, (manhattan(start, goal), 0, counter, start, start_lo, start_hi))
    best_g: dict = {(start, start_lo): 0}
    came_from: dict = {}
    expansions = 0

    while open_heap:
        _, g, _, cell, lo, hi = heapq.heappop(open_heap)
        state = (cell, lo)
        if g > best_g.get(state, g):
            continue  # stale heap entry, superseded by a better arrival
        expansions += 1

        if cell == goal and hi == INF and g >= last_goal_time:
            if stats is not None:
                stats["expansions"] = expansions
            return _expand_path(came_from, best_g, state, start)

        if g >= max_time:
            continue

        for ncell in grid.neighbors(cell):
            if ncell == cell:
                continue  # waiting is captured by the interval, not a move
            forbidden = edge_by.get((cell, ncell), ())
            for (nlo, nhi) in intervals(ncell):
                # Depart any time in [g, hi]; arrive one tick later, landing in
                # this successor interval.
                earliest = max(g + 1, nlo)
                latest = min(hi + 1, nhi)
                if earliest > latest:
                    continue
                arrival = earliest
                while arrival in forbidden and arrival <= latest:
                    arrival += 1
                if arrival > latest or arrival > max_time:
                    continue
                nstate = (ncell, nlo)
                if arrival < best_g.get(nstate, arrival + 1):
                    best_g[nstate] = arrival
                    came_from[nstate] = state
                    counter += 1
                    heapq.heappush(
                        open_heap,
                        (arrival + manhattan(ncell, goal), arrival, counter,
                         ncell, nlo, nhi))

    if stats is not None:
        stats["expansions"] = expansions
    return None


def _expand_path(came_from, best_g, end_state, start) -> list:
    """Turn the interval state chain into a per-timestep cell list.

    Each state carries its arrival time in ``best_g``; between two consecutive
    states the agent waited in place, so we fill those timesteps with the
    earlier cell before stepping to the next.
    """
    chain = [end_state]
    while chain[-1] in came_from:
        chain.append(came_from[chain[-1]])
    chain.reverse()

    path = []
    prev_cell = start
    prev_t = 0
    for cell, _lo in chain:
        g = best_g[(cell, _lo)]
        while prev_t < g:
            path.append(prev_cell)
            prev_t += 1
        path.append(cell)
        prev_t = g + 1
        prev_cell = cell
    return path
