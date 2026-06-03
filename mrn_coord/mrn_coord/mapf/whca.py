"""Windowed Hierarchical Cooperative A* (Silver, AIIDE 2005).

David Silver, *Cooperative Pathfinding* (AIIDE 2005). A scalable, online,
incomplete MAPF solver built from three ideas layered on prioritized planning:

- **Cooperative A* (CA*).** Plan agents one at a time in priority order. Each
  agent reserves its whole space-time path in a *reservation table*; later
  agents treat those reservations (vertex + swap) as moving obstacles. This is
  exactly :func:`mrn_coord.mapf.prioritized.prioritized_planning`.

- **Hierarchical (HCA*).** Replace the Manhattan heuristic with the *true*
  shortest-path distance to the goal on the static map (ignoring other agents),
  computed on demand by **Reverse Resumable A\\* (RRA\\*)** — a backward A* from
  the goal that resumes expanding only as far as each queried cell. The true
  distance is perfect on the obstacle map, so the cooperative space-time A*
  stops exploring dead ends a wall would create and expands far fewer states.

- **Windowed (WHCA*).** Do the cooperation only within a fixed lookahead
  *window* of ``w`` timesteps: an agent searches (and reserves) just ``w`` steps
  ahead toward the goal, then everyone advances and the window rolls. Beyond the
  window agents ignore each other. This bounds the per-step search depth (so it
  scales) and, by re-deciding every window with a *rotating* priority order,
  lets a blocked agent take precedence next window — breaking the transient
  deadlocks a single fixed priority order livelocks on.

WHCA* is neither optimal nor complete (it can fail on a genuinely sealed map),
but it is **collision-free by construction** — every committed segment is laid
into the reservation table in priority order, so no two agents ever share a cell
or swap. It scales to large, dense teams where full-horizon optimal search blows
up, at the cost of optimality. Pure and deterministic.
"""

from __future__ import annotations

import heapq

from .grid import Cell, GridWorld, manhattan
from .solution import Solution, sum_of_costs


class RRAStar:
    """Reverse Resumable A* — an on-demand oracle for the true distance to a goal.

    A backward A* rooted at ``goal`` (moves are reversible on the 4-connected
    grid, so backward distance == forward distance). A query ``distance(cell)``
    resumes the search only until ``cell`` is settled, caching every distance
    found along the way; later queries pick up where the last left off. The
    heuristic for the backward search aims at ``origin`` (the agent's start), so
    expansion concentrates along the corridor between start and goal.
    """

    def __init__(self, grid: GridWorld, goal: Cell, origin: Cell) -> None:
        self.grid = grid
        self.goal = goal
        self.origin = origin
        self.g: dict[Cell, int] = {goal: 0}
        self.closed: set[Cell] = set()
        self._open: list[tuple[int, int, Cell]] = []
        self._counter = 0
        heapq.heappush(self._open, (manhattan(goal, origin), 0, goal))

    def _push(self, cell: Cell) -> None:
        self._counter += 1
        heapq.heappush(
            self._open, (self.g[cell] + manhattan(cell, self.origin),
                         self._counter, cell)
        )

    def distance(self, cell: Cell) -> int | None:
        """True shortest-path distance from ``cell`` to the goal, or ``None``."""
        if cell in self.closed:
            return self.g[cell]
        while self._open:
            _, _, node = heapq.heappop(self._open)
            if node in self.closed:
                continue
            self.closed.add(node)
            ng = self.g[node] + 1
            for nb in self.grid.neighbors(node):
                if nb == node:
                    continue  # waiting is not a move on the static map
                if nb not in self.g or ng < self.g[nb]:
                    self.g[nb] = ng
                    self._push(nb)
            if node == cell:
                return self.g[cell]
        return None


def _segment_search(
    grid: GridWorld, start: Cell, goal: Cell, t0: int, window: int,
    vres: dict, eres: set, h, *, agent=None, stats: dict | None = None,
):
    """Windowed cooperative space-time A* for one agent.

    Searches from ``(start, t0)`` toward ``goal``. Reservations in ``vres`` (a
    ``{(cell, time): owner}`` map) and ``eres`` (a set of ``(frm, to, time)``
    forbidden swaps) are honored only for arrival times ``<= t0 + window`` — the
    cooperation window. ``h(cell)`` is the true-distance heuristic.

    Returns the cell sequence ``[start, ...]`` indexed from ``t0``: it settles on
    the goal the moment it can *hold* the goal through the window edge (no
    higher-priority agent reserves it later in the window), otherwise it runs to
    the window edge ``t0 + window`` and returns the path to the frontier cell
    closest to the goal (minimum ``h``) — letting a lower-priority agent vacate
    and return when a higher one passes through its goal. Returns ``None`` only
    if the agent cannot even stay safe for one step. ``stats['expansions']``
    counts states expanded.
    """
    limit = t0 + window

    def _can_hold(t: int) -> bool:
        # The goal must be free (or already this agent's) for every step from
        # arrival ``t`` through the window edge, so the settle is genuinely safe.
        for tt in range(t, limit + 1):
            owner = vres.get((goal, tt))
            if owner is not None and owner != agent:
                return False
        return True

    start_h = h(start)
    if start_h is None:
        return None
    open_heap: list[tuple[int, int, int, Cell, int]] = []
    counter = 0
    heapq.heappush(open_heap, (start_h, 0, counter, start, t0))
    came: dict[tuple[Cell, int], tuple[Cell, int]] = {}
    best_g: dict[tuple[Cell, int], int] = {(start, t0): 0}
    expansions = 0

    while open_heap:
        f, g, _, cell, t = heapq.heappop(open_heap)
        if best_g.get((cell, t), g) < g:
            continue
        expansions += 1

        # Settle on the goal as soon as we can hold it through the window edge.
        if cell == goal and h(cell) == 0 and _can_hold(t):
            if stats is not None:
                stats["expansions"] = expansions
            return _reconstruct(came, (cell, t), start, t0)

        if t >= limit:
            # Frontier node; min-f frontier == min-h == closest to goal.
            if stats is not None:
                stats["expansions"] = expansions
            return _reconstruct(came, (cell, t), start, t0)

        nt = t + 1
        for ncell in grid.neighbors(cell):
            if vres.get((ncell, nt)) is not None and nt <= limit:
                continue
            if ncell != cell and (ncell, cell, nt) in eres and nt <= limit:
                continue
            ng = g + 1
            key = (ncell, nt)
            if best_g.get(key, ng + 1) <= ng:
                continue
            best_g[key] = ng
            came[key] = (cell, t)
            counter += 1
            hh = h(ncell)
            if hh is None:
                continue
            heapq.heappush(open_heap, (ng + hh, ng, counter, ncell, nt))

    if stats is not None:
        stats["expansions"] = expansions
    return None


def _reconstruct(came, state, start, t0) -> list[Cell]:
    seq = [state]
    while state in came:
        state = came[state]
        seq.append(state)
    seq.reverse()
    return [cell for (cell, _) in seq]


def _default_max_steps(grid: GridWorld, agents: dict) -> int:
    return 4 * (grid.width * grid.height) + 4 * len(agents) + 20


def whca_star(
    grid: GridWorld, agents: dict, *, window: int = 8, replan_period: int | None = None,
    rotate_priority: bool = True, order=None, max_steps: int | None = None,
    stats: dict | None = None,
):
    """Solve a MAPF instance with Windowed Hierarchical Cooperative A*.

    ``agents`` maps an agent id to ``(start, goal)``. Returns a :class:`Solution`
    whose paths are collision-free (each agent waits at its goal after arrival),
    or ``None`` if some agent cannot be routed within ``max_steps`` (WHCA* is
    incomplete). ``window`` is the cooperation lookahead in timesteps; a window
    ``>= max_steps`` recovers full-horizon HCA* (one reservation of every whole
    path). ``replan_period`` is how many steps the team advances before the
    window rolls and everyone replans (default ``max(1, window // 2)``); with
    ``rotate_priority`` the priority order rotates by one each round so a blocked
    agent eventually leads. ``order`` fixes the base priority (default insertion
    order). If ``stats`` is given it records ``replans``, ``rounds``,
    ``low_level_expansions`` (summed) and ``makespan``.
    """
    ids = list(order) if order is not None else list(agents)
    if not ids:
        return Solution(paths={}, cost=0)
    if replan_period is None:
        replan_period = max(1, window // 2)
    if max_steps is None:
        max_steps = _default_max_steps(grid, agents)

    goals = {a: agents[a][1] for a in ids}
    rra = {a: RRAStar(grid, goals[a], agents[a][0]) for a in ids}
    pos = {a: agents[a][0] for a in ids}
    paths = {a: [pos[a]] for a in ids}
    committed: dict = {a: [] for a in ids}

    replans = rounds = 0
    total_exp = 0
    t = 0
    while t < max_steps:
        if all(pos[a] == goals[a] for a in ids):
            break

        if t % replan_period == 0:
            # Roll the window: rebuild the reservation table and replan everyone
            # in priority order (rotated so a previously-blocked agent can lead).
            shift = rounds if rotate_priority else 0
            order_now = ids[shift % len(ids):] + ids[:shift % len(ids)]
            vres: dict = {(pos[a], t): a for a in ids}  # seed current positions
            eres: set = set()
            new_committed: dict = {}
            failed = False
            for a in order_now:
                seg_stats: dict = {}
                seg = _segment_search(
                    grid, pos[a], goals[a], t, window, vres, eres,
                    rra[a].distance, agent=a, stats=seg_stats,
                )
                total_exp += seg_stats.get("expansions", 0)
                if seg is None:
                    failed = True
                    break
                # Reserve this agent's committed segment (and hold its tail cell
                # through the window edge so lower-priority agents route around).
                for k, cell in enumerate(seg):
                    vres[(cell, t + k)] = a
                tail = seg[-1]
                for tt in range(t + len(seg), t + window + 1):
                    vres[(tail, tt)] = a
                for k in range(len(seg) - 1):
                    frm, to = seg[k], seg[k + 1]
                    if frm != to:
                        eres.add((frm, to, t + k + 1))  # forbid the reverse swap
                new_committed[a] = seg[1:]
            if failed:
                if stats is not None:
                    stats.update(replans=replans, rounds=rounds,
                                 low_level_expansions=total_exp,
                                 makespan=max(len(p) - 1 for p in paths.values()))
                return None
            committed = new_committed
            replans += len(ids)
            rounds += 1

        # Advance every agent one step along its committed segment.
        for a in ids:
            if committed[a]:
                pos[a] = committed[a].pop(0)
            paths[a].append(pos[a])
        t += 1

    if not all(pos[a] == goals[a] for a in ids):
        if stats is not None:
            stats.update(replans=replans, rounds=rounds,
                         low_level_expansions=total_exp,
                         makespan=max(len(p) - 1 for p in paths.values()))
        return None

    # Trim the synchronized trailing waits (everyone already at goal).
    for a in ids:
        p = paths[a]
        while len(p) > 1 and p[-1] == goals[a] and p[-2] == goals[a]:
            p.pop()
    if stats is not None:
        stats.update(replans=replans, rounds=rounds,
                     low_level_expansions=total_exp,
                     makespan=max(len(p) - 1 for p in paths.values()))
    return Solution(paths=paths, cost=sum_of_costs(paths))
