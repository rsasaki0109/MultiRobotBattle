"""Priority-Based Search (PBS): MAPF by searching over priority orderings.

Ma, Harabor, Stuckey, Li & Koenig, *Searching with Consistent Prioritization for
Multi-Agent Path Finding* (AAAI 2019). Like Conflict-Based Search, PBS is a
two-level search — but it branches on **priorities** instead of constraints.

- **High level.** A node holds a *partial* priority order (a DAG over agents).
  The root has no priorities. On the first conflict between agents ``a`` and
  ``b`` the node branches into two children that add the ordering ``a < b`` or
  ``b < a`` (one must yield to the other); a child that would close a priority
  cycle is pruned. Depth-first, returning the first conflict-free plan.
- **Low level.** Under a fixed partial order, each agent is planned with
  prioritized planning: it treats every *strictly-higher-priority* agent's path
  as a moving obstacle (vertex + swap reservations) and is replanned, together
  with everything below it, in topological order whenever a new ordering is
  added.

PBS is neither optimal nor complete, but it scales far past CBS and resolves the
priority deadlocks that plain fixed-order prioritized planning cannot (it can
reorder a head-on pair so the right agent yields). That makes it the windowed
solver of choice for lifelong MAPF (RHCR, :mod:`mrn_coord.lifelong.rhcr`): pass
``window=w`` to resolve only the conflicts within the next ``w`` timesteps and
ignore everything beyond, which is what the rolling horizon replans away anyway.

Pure and deterministic: ties (which agent to branch first, topological order)
break by an optional ``order_hint`` then by agent id.
"""

from __future__ import annotations

from .conflicts import EdgeConflict, VertexConflict, detect_first_conflict
from .grid import GridWorld
from .solution import Solution, sum_of_costs
from .space_time_astar import plan_path


def _reachable(pairs, start, *, forward: bool) -> set:
    """Agents reachable from ``start`` in the priority DAG.

    ``forward`` follows ``hi -> lo`` edges (the descendants / lower-priority set);
    otherwise it follows ``lo -> hi`` (the ancestors / strictly-higher set).
    """
    adj: dict = {}
    for hi, lo in pairs:
        a, b = (hi, lo) if forward else (lo, hi)
        adj.setdefault(a, set()).add(b)
    seen: set = set()
    stack = [start]
    while stack:
        n = stack.pop()
        for m in adj.get(n, ()):
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return seen


def _topo_order(agents, pairs, rank) -> list:
    """Topological order of ``agents`` under ``pairs`` (higher priority first).

    ``rank`` maps agent -> sort key; ready agents are taken lowest-rank first so
    the result is deterministic. ``pairs`` may reference agents outside the set;
    only edges within ``agents`` constrain the order.
    """
    members = set(agents)
    indeg = {a: 0 for a in agents}
    succ: dict = {a: [] for a in agents}
    for hi, lo in pairs:
        if hi in members and lo in members:
            succ[hi].append(lo)
            indeg[lo] += 1
    ready = sorted((a for a in agents if indeg[a] == 0), key=rank)
    out = []
    while ready:
        a = ready.pop(0)
        out.append(a)
        for lo in succ[a]:
            indeg[lo] -= 1
            if indeg[lo] == 0:
                # insert keeping the ready list rank-sorted
                ready.append(lo)
                ready.sort(key=rank)
    return out


def _reserve(path, cap):
    """Vertex + edge reservations imposed by a higher-priority ``path`` up to ``cap``.

    Every occupied ``(cell, time)`` for ``time <= cap`` is a vertex reservation;
    the agent holds its goal afterwards, so the goal cell is reserved through
    ``cap`` too. Each move ``frm -> to`` arriving at ``t`` forbids the reverse
    ``to -> frm`` (a swap) at ``t``. ``cap`` is the resolution window (RHCR) or a
    horizon large enough to hold goals "forever" for unwindowed PBS.
    """
    vset = set()
    eset = set()
    arrival = len(path) - 1
    for t, cell in enumerate(path):
        if t > cap:
            break
        vset.add((cell, t))
    goal_cell = path[-1]
    for t in range(arrival, cap + 1):
        vset.add((goal_cell, t))
    for t in range(len(path) - 1):
        if t + 1 > cap:
            break
        frm, to = path[t], path[t + 1]
        eset.add((to, frm, t + 1))
    return vset, eset


def _resv_horizon(grid: GridWorld) -> int:
    """Goal-hold horizon for unwindowed PBS: big enough to cover any path."""
    return 2 * grid.width * grid.height + grid.width + grid.height + 8


def _plan_under(grid, start, goal, higher_paths, window):
    """Plan one agent avoiding the ``higher_paths`` (only within ``window``).

    ``window`` bounds where conflicts are reserved (RHCR); ``None`` means hold
    the higher agents' goals for a full horizon (standard PBS). The agent itself
    still plans all the way to its goal (which may lie past the window).
    """
    cap = window if window is not None else _resv_horizon(grid)
    vset: set = set()
    eset: set = set()
    for p in higher_paths:
        v, e = _reserve(p, cap)
        vset |= v
        eset |= e
    # For unwindowed PBS, bound the search at the same horizon the goals are held
    # to, so a lower agent cannot "escape" past a higher agent parked on its only
    # route (which would be a spurious, un-resolvable plan). Windowed planning
    # deliberately lets the agent route to a goal beyond the window.
    max_time = None if window is not None else cap
    return plan_path(grid, start, goal, frozenset(vset), frozenset(eset),
                     max_time=max_time)


def _pbs_core(grid: GridWorld, agents: dict, *, window, rank, max_nodes: int):
    """The shared PBS search. Returns ``{agent: path}`` or ``None``."""
    order = list(agents)

    def replan(paths, pairs, targets):
        """Replan ``targets`` (in topological order) into a fresh ``paths`` copy."""
        paths = dict(paths)
        for a in _topo_order(targets, pairs, rank):
            higher = _reachable(pairs, a, forward=False)
            start, goal = agents[a]
            p = _plan_under(grid, start, goal,
                            [paths[h] for h in order if h in higher], window)
            if p is None:
                return None
            paths[a] = p
        return paths

    # root: everyone planned independently (no priorities)
    root_paths = replan({}, frozenset(), order)
    if root_paths is None:
        return None
    stack = [(frozenset(), root_paths)]
    nodes = 0
    while stack:
        pairs, paths = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            return None
        conflict = detect_first_conflict(paths, window=window)
        if conflict is None:
            return paths
        if isinstance(conflict, (VertexConflict, EdgeConflict)):
            a, b = conflict.agent_a, conflict.agent_b
        else:                                       # pragma: no cover - defensive
            return None
        # Try b<a first then a<b, so a<b is popped (explored) first.
        children = []
        for hi, lo in ((b, a), (a, b)):
            if hi in _reachable(pairs, lo, forward=True):
                continue                            # would close a priority cycle
            new_pairs = pairs | {(hi, lo)}
            targets = {lo} | _reachable(new_pairs, lo, forward=True)
            new_paths = replan(paths, new_pairs, targets)
            if new_paths is not None:
                children.append((new_pairs, new_paths))
        stack.extend(children)
    return None


def pbs(grid: GridWorld, agents: dict, *, window: int | None = None,
        order_hint=None, max_nodes: int = 10_000):
    """Solve MAPF with Priority-Based Search.

    ``agents`` maps agent id -> ``(start, goal)``. Returns a :class:`Solution`
    (sum-of-costs over the full paths) or ``None`` if the priority search is
    exhausted / the node budget ``max_nodes`` is hit. ``window`` restricts
    conflict resolution to the next ``window`` timesteps (used by RHCR);
    ``order_hint`` is an optional agent sequence whose earlier entries are
    preferred as higher priority when ties arise.
    """
    paths = pbs_paths(grid, agents, window=window, order_hint=order_hint,
                      max_nodes=max_nodes)
    if paths is None:
        return None
    return Solution(paths=paths, cost=sum_of_costs(paths))


def pbs_paths(grid: GridWorld, agents: dict, *, window: int | None = None,
              order_hint=None, max_nodes: int = 10_000):
    """PBS that returns the raw ``{agent: path}`` dict (or ``None``).

    The form RHCR consumes: it only needs the per-agent paths to step along, not
    a costed :class:`Solution`.
    """
    if not agents:
        return {}
    hint = list(order_hint) if order_hint is not None else list(agents)
    rank_of = {a: i for i, a in enumerate(hint)}
    big = len(hint) + 1

    def rank(a):
        return (rank_of.get(a, big), repr(a))

    return _pbs_core(grid, agents, window=window, rank=rank, max_nodes=max_nodes)
