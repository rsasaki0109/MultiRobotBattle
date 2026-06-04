"""SIPPS — Safe Interval Path Planning with Soft constraints (Li et al. 2022).

The low-level planner behind MAPF-LNS2 (Li, Chen, Harabor, Stuckey & Koenig,
*"MAPF-LNS2: Fast Repairing for Multi-Agent Path Finding via Large Neighborhood
Search"*, AAAI 2022). It is to :func:`mrn_coord.mapf.lns2._plan_min_collision`
what plain :func:`mrn_coord.mapf.sipp.plan_sipp` is to the time-expanded
:func:`mrn_coord.mapf.space_time_astar.plan_path`: the *same* answer over a far
smaller state space.

The repair low level of MAPF-LNS2 must replan one agent through a grid where the
other agents' paths are still present. It distinguishes two kinds of occupancy:

- **hard** constraints — static obstacles and any reservation the agent must
  never violate. These partition each cell's timeline into **safe intervals**,
  exactly as in SIPP, and the agent may never be in a cell during a hard-blocked
  instant.
- **soft** constraints — the other agents' current paths. The agent *may* pass
  through (or wait inside) a soft-occupied cell, but every such overlap counts
  one **collision**. SIPPS finds the path with the **fewest collisions**, and the
  shortest among those — the lexicographic ``(collisions, length)`` objective
  that lets LNS2's repair make progress on a tangle with no collision-free
  completion *yet*.

The win over the time-expanded collision-minimizer is the same SIPP win: it
expands one state per ``(cell, safe interval)`` rather than one per ``(cell,
time)``, so a cell free for a long stretch is a single state, not hundreds.
Collisions are still priced exactly — a wait through a soft-occupied instant
costs a collision, and settling on a goal that a soft agent later crosses costs
one per crossing — so the ``(collisions, length)`` cost SIPPS returns matches the
time-expanded optimum.

A subtlety SIPPS gets right: the cheapest path sometimes *vacates* the goal while
a soft agent crosses it and returns afterwards, rather than parking early and
eating the crossing. Each arrival at the goal is therefore offered as its own
terminal (its collisions already including the forever-park cost from that
arrival), independent of the safe-interval dominance that prunes ordinary nodes —
so a later, collision-free settle is never lost.

Pass a ``stats`` dict to read back ``stats["expansions"]`` (the SIPP win) and
``stats["collisions"]`` (the minimized objective). The returned value is a
per-timestep cell list, the same format as ``plan_path`` / ``plan_sipp``.
"""

from __future__ import annotations

import heapq

from .grid import Cell, GridWorld, manhattan
from .space_time_astar import _default_max_time

INF = float("inf")


def _safe_intervals(blocked) -> list:
    """Maximal runs of timesteps a cell is free, given its hard-blocked instants.

    Same construction as :func:`mrn_coord.mapf.sipp._safe_intervals`: a sorted
    list of ``(lo, hi)`` closed intervals covering ``[0, INF)`` with the last
    extending to ``INF`` so an agent can settle forever.
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


def plan_sipps(
    grid: GridWorld,
    start: Cell,
    goal: Cell,
    *,
    hard_vertex=frozenset(),
    hard_edge=frozenset(),
    soft_vertex=None,
    soft_edge=None,
    max_time: int | None = None,
    stats: dict | None = None,
) -> list | None:
    """Fewest-collision (then shortest) ``start -> goal`` path by safe intervals.

    ``hard_vertex`` is a set of ``(cell, time)`` the agent must never occupy and
    ``hard_edge`` a set of ``(frm, to, time)`` arrivals it must never make — they
    define the safe intervals. ``soft_vertex`` maps ``(cell, time)`` to how many
    other agents sit there (each a collision the path incurs by being there at
    that time, including while waiting) and ``soft_edge`` maps ``(frm, to,
    time)`` to swap collisions. Returns the path as a per-timestep cell list with
    the minimum number of soft collisions, shortest among ties, or ``None`` if
    the goal is unreachable under the hard constraints. ``stats["expansions"]``
    and ``stats["collisions"]`` are filled when ``stats`` is given.
    """
    if not grid.is_free(start) or not grid.is_free(goal):
        return None
    if (start, 0) in hard_vertex:
        return None
    soft_v = dict(soft_vertex or {})
    soft_e = dict(soft_edge or {})
    # Beyond the last soft constraint, occupancy is free and constant, so the
    # search never needs to enumerate arrivals past this horizon.
    soft_horizon = 0
    for (_c, t) in soft_v:
        soft_horizon = max(soft_horizon, t)
    for (_f, _t2, t) in soft_e:
        soft_horizon = max(soft_horizon, t)
    if max_time is None:
        # The minimum-collision path may wait out *every* soft reservation before
        # traversing, so the horizon must clear the last soft constraint plus a
        # full obstacle-aware crossing — not just the constraint-free bound.
        max_time = soft_horizon + _default_max_time(grid, hard_vertex, hard_edge)

    hard_v_by_cell: dict = {}
    for (cell, t) in hard_vertex:
        hard_v_by_cell.setdefault(cell, set()).add(t)
    hard_e_by: dict = {}
    for (frm, to, t) in hard_edge:
        hard_e_by.setdefault((frm, to), set()).add(t)

    interval_cache: dict = {}

    def intervals(cell):
        iv = interval_cache.get(cell)
        if iv is None:
            iv = _safe_intervals(hard_v_by_cell.get(cell, ()))
            interval_cache[cell] = iv
        return iv

    soft_times_by_cell: dict = {}
    for (cell, t), cnt in soft_v.items():
        soft_times_by_cell.setdefault(cell, []).append((t, cnt))

    def wait_cost(cell, frm_t, to_t):
        """Soft collisions accrued waiting at ``cell`` over ``(frm_t, to_t]``."""
        if to_t <= frm_t:
            return 0
        total = 0
        for (t, cnt) in soft_times_by_cell.get(cell, ()):
            if frm_t < t <= to_t:
                total += cnt
        return total

    start_lo, start_hi = next(
        (lo, hi) for (lo, hi) in intervals(start) if lo <= 0 <= hi)
    start_col = soft_v.get((start, 0), 0)

    goal_hard_times = [t for (cell, t) in hard_vertex if cell == goal]
    last_goal_hard = max(goal_hard_times) if goal_hard_times else 0

    # Node bookkeeping for reconstruction, keyed by a unique id (so two arrivals
    # at the same cell/interval with the same collision count never clash).
    nid = 0
    parent: dict = {0: None}
    node_cell: dict = {0: (start, 0)}      # id -> (cell, arrival time)

    # Per (cell, interval-lo): the non-dominated (collisions, g) labels reached,
    # to prune ordinary expansion (the goal terminal is exempt — see below).
    labels: dict = {(start, start_lo): [(start_col, 0)]}

    # Dominance must account for waiting being *costly* when the cell is
    # soft-occupied: an earlier label (c2, g2) dominates (col, g) only if it can
    # wait inside the interval to time g for no more than ``col`` collisions,
    # i.e. ``c2 + wait_cost(cell, g2, g) <= col``. Plain SIPP's (c, g) dominance
    # assumes free waiting (true for hard intervals) and is unsound here.
    def dominated(cell, lo, col, g):
        for (c2, g2) in labels.get((cell, lo), ()):
            if g2 <= g and c2 + wait_cost(cell, g2, g) <= col:
                return True
        return False

    def record(cell, lo, col, g):
        bucket = labels.setdefault((cell, lo), [])
        kept = [(c2, g2) for (c2, g2) in bucket
                if not (g <= g2 and col + wait_cost(cell, g, g2) <= c2)]
        kept.append((col, g))
        labels[(cell, lo)] = kept

    open_heap: list = []
    counter = 0
    expansions = 0

    def push_terminal(col_at_arrival, g, arrival_id):
        """Offer to park at the goal forever from this arrival (charging every
        later soft crossing). Exempt from label dominance so a later,
        collision-free settle is never pruned by an earlier, costlier one."""
        nonlocal counter
        total = col_at_arrival + wait_cost(goal, g, soft_horizon)
        counter += 1
        heapq.heappush(
            open_heap,
            (total, total, g, counter, _GOAL, 0, 0, arrival_id))

    heapq.heappush(
        open_heap,
        (start_col, manhattan(start, goal), 0, counter, start, start_lo,
         start_hi, 0))
    if start == goal and start_hi == INF and 0 >= last_goal_hard:
        push_terminal(start_col, 0, 0)

    while open_heap:
        col, _f, g, _tie, cell, lo, hi, node_id = heapq.heappop(open_heap)

        if cell is _GOAL:
            if stats is not None:
                stats["expansions"] = expansions
                stats["collisions"] = col
            return _reconstruct(parent, node_cell, node_id, start)

        # Skip a stale heap entry whose label was later superseded (a better one
        # at the same or earlier time that can wait here at no greater cost).
        superseded = False
        for (c2, g2) in labels.get((cell, lo), ()):
            if (c2, g2) != (col, g) and g2 <= g \
                    and c2 + wait_cost(cell, g2, g) <= col:
                superseded = True
                break
        if superseded:
            continue
        expansions += 1

        if g >= max_time:
            continue

        for ncell in grid.neighbors(cell):
            if ncell == cell:
                continue  # waiting is captured inside the interval
            hard_arr = hard_e_by.get((cell, ncell), ())
            for (nlo, nhi) in intervals(ncell):
                arrive_lo = max(g + 1, nlo)
                arrive_hi = min(hi + 1, nhi)
                if arrive_lo > arrive_hi:
                    continue
                # Candidate arrivals: the earliest, every soft-event boundary in
                # range and the instant after it, and one beyond the soft horizon.
                # All finite — they capture the Pareto trade between leaving early
                # (cheaper length) and waiting for a collision-free arrival.
                cands = {arrive_lo}
                for (t, _cnt) in soft_times_by_cell.get(ncell, ()):
                    if arrive_lo <= t <= arrive_hi:
                        cands.add(t)
                    if arrive_lo <= t + 1 <= arrive_hi:
                        cands.add(t + 1)
                for (frm, to, t) in soft_e:
                    if frm == cell and to == ncell:
                        if arrive_lo <= t <= arrive_hi:
                            cands.add(t)
                        if arrive_lo <= t + 1 <= arrive_hi:
                            cands.add(t + 1)
                beyond = max(arrive_lo, min(arrive_hi, soft_horizon + 1))
                if arrive_lo <= beyond <= arrive_hi:
                    cands.add(beyond)
                for arrival in sorted(cands):
                    if arrival > arrive_hi or arrival > max_time:
                        continue
                    if arrival in hard_arr:
                        continue
                    depart = arrival - 1
                    waited = wait_cost(cell, g, depart)
                    move_col = soft_e.get((cell, ncell, arrival), 0)
                    arr_col = soft_v.get((ncell, arrival), 0)
                    ncol = col + waited + move_col + arr_col
                    nid += 1
                    parent[nid] = node_id
                    node_cell[nid] = (ncell, arrival)
                    # The goal's final interval is always offered as a terminal,
                    # exempt from dominance (the park cost shrinks as arrival
                    # grows, so a later arrival can settle more cheaply).
                    if (ncell == goal and nhi == INF
                            and arrival >= last_goal_hard):
                        push_terminal(ncol, arrival, nid)
                    if dominated(ncell, nlo, ncol, arrival):
                        continue
                    record(ncell, nlo, ncol, arrival)
                    counter += 1
                    heapq.heappush(
                        open_heap,
                        (ncol, arrival + manhattan(ncell, goal), arrival,
                         counter, ncell, nlo, nhi, nid))

    if stats is not None:
        stats["expansions"] = expansions
        stats["collisions"] = INF
    return None


_GOAL = ("__settled__",)  # sentinel cell marking a terminal (parked-on-goal) node


def _reconstruct(parent, node_cell, end_id, start) -> list:
    """Walk the id chain back to the root and expand waits into a cell list."""
    chain = []
    nid = end_id
    while nid is not None:
        chain.append(node_cell[nid])
        nid = parent[nid]
    chain.reverse()

    path = []
    prev_cell = start
    prev_t = 0
    for (cell, g) in chain:
        while prev_t < g:
            path.append(prev_cell)
            prev_t += 1
        path.append(cell)
        prev_t = g + 1
        prev_cell = cell
    return path
