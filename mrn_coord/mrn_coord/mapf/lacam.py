"""LaCAM: Lazy Constraints Addition search for MAPF (Okumura, AAAI 2023).

CBS / ECBS search a *constraint tree* and replan single agents; they are
(bounded-)optimal but their trees explode with the team size. LaCAM searches the
**configuration space** directly — a node is the joint position of *all* agents
— and uses :func:`PIBT <_pibt>` as a successor generator: from a configuration,
PIBT produces one collision-free successor in near-linear time. That alone would
be incomplete (PIBT is greedy), so LaCAM adds **lazy constraints**: each
configuration node carries a tree of low-level nodes that pin successive agents
to successive candidate cells, generating PIBT successors under those pins one at
a time. Because the constraints eventually enumerate *every* successor of a
configuration and the configuration space is finite, LaCAM is **complete** — it
finds a solution whenever one exists.

Completeness alone is cheap; *scaling* lives entirely in the order successors are
generated. LaCAM dives greedily — the empty-constraint (unconstrained) PIBT
successor is explored first, so the DFS spine *is* a PIBT rollout, and the lazy
constraints are only the backtracking fallback. If that spine reaches the goal
directly, LaCAM is fast; if it livelocks, the search drops into the
lazy-constraint enumeration, which branches-explodes (every agent's every
neighbor) and times out. A *static* per-config priority order makes the spine the
weak deterministic PIBT that livelocks. So the spine here is the **strong** PIBT,
exactly :func:`pibt_solve <mrn_coord.lifelong.pibt_solve>`'s: off-goal agents
*accumulate* priority (so a stuck agent eventually wins right of way), and a stall
— the team's summed distance-to-goal failing to reach a new low — bumps a
deterministic escape ``salt`` that scrambles equal-distance ties until the
symmetry breaks. The spine reaches the goal directly far more often, so the
exponential fallback is rarely touched. The constraint enumeration is *untouched*,
so completeness still holds — only the successor order changes.

This is the satisficing variant (any valid solution, collision-free by PIBT
construction); it is not cost-optimal. Pure and deterministic.
"""

from __future__ import annotations

from collections import deque

from .grid import Cell, GridWorld
from .solution import Solution, sum_of_costs


def _bfs_dist(grid: GridWorld, goal: Cell) -> dict:
    """4-connected BFS distance from every free cell to ``goal``."""
    dist = {goal: 0}
    q = deque([goal])
    while q:
        cell = q.popleft()
        d = dist[cell]
        for nb in grid.neighbors(cell):
            if nb not in dist:
                dist[nb] = d + 1
                q.append(nb)
    return dist


def _mix(cell, salt: int) -> int:
    """Deterministic integer hash of ``(cell, salt)`` — the twin of
    :func:`mrn_coord.lifelong.lifelong._mix`, duplicated here so :mod:`mapf` need
    not import :mod:`lifelong` (which imports back from :mod:`mapf`). Pure
    arithmetic so the scramble is bit-reproducible (no ``PYTHONHASHSEED``); only
    reached when an escape ``salt`` is active, so ``salt == 0`` is unchanged.
    """
    return ((cell[0] * 73856093) ^ (cell[1] * 19349663) ^ (salt * 83492791)) & 0xFFFFFFFF


def _pibt(grid, config, order, dist_to, forced, *, salt: int = 0):
    """One PIBT step: return ``{agent: next_cell}`` or ``None`` if infeasible.

    ``config`` maps agent -> current cell; ``order`` is the priority order to
    decide agents in; ``dist_to`` maps agent -> {cell: dist-to-goal}; ``forced``
    pins some agents to a specific next cell (the LaCAM constraint). Pushing
    (priority inheritance) and swap prevention are exactly PIBT's. A non-zero
    ``salt`` scrambles equal-distance candidate ties via :func:`_mix` (the
    deterministic livelock escape); ``salt == 0`` breaks ties by raw coordinate.
    """
    occupant = {config[a]: a for a in config}
    next_pos: dict = {}
    next_occ: dict = {}

    def candidates(a):
        if a in forced:
            return [forced[a]]
        d = dist_to[a]
        big = len(d) + 1
        here = config[a]
        if salt:
            return sorted(grid.neighbors(here),
                          key=lambda c: (d.get(c, big), c == here, _mix(c, salt)))
        return sorted(grid.neighbors(here),
                      key=lambda c: (d.get(c, big), c == here, c))

    def decide(a, pusher=None) -> bool:
        for c in candidates(a):
            if c in next_occ:
                continue
            if pusher is not None and c == config[pusher]:
                continue                          # would swap with the pusher
            next_pos[a] = c
            next_occ[c] = a
            other = occupant.get(c)
            if other is not None and other != a and other not in next_pos:
                if decide(other, pusher=a):
                    return True
                del next_occ[c]
                del next_pos[a]
                continue
            return True
        return False

    for a in order:
        if a not in next_pos:
            if not decide(a):
                return None
    return next_pos


def lacam(grid: GridWorld, agents: dict, *,
          max_iterations: int = 1_000_000, stall_patience: int = 3):
    """Solve a MAPF instance (satisficing, complete) by LaCAM.

    ``agents`` maps an agent id to a ``(start, goal)`` tuple. Returns a
    collision-free :class:`Solution` (not cost-optimal), or ``None`` if the
    instance is infeasible or the iteration budget is exhausted. The greedy DFS
    spine runs the strong PIBT — accumulating priorities plus a deterministic
    livelock escape that engages after ``stall_patience`` non-improving steps; see
    the module docstring.
    """
    ids = sorted(agents)
    starts = {a: agents[a][0] for a in ids}
    goals = {a: agents[a][1] for a in ids}
    for a in ids:
        if not grid.is_free(starts[a]) or not grid.is_free(goals[a]):
            return None
    dist_to = {a: _bfs_dist(grid, goals[a]) for a in ids}
    for a in ids:
        if starts[a] not in dist_to[a]:
            return None                           # goal unreachable from start

    init = tuple(starts[a] for a in ids)
    goal_config = tuple(goals[a] for a in ids)

    n = len(ids)
    size = grid.width * grid.height
    big = size + 1
    cfg_cache: dict = {}

    def cfg_of(config):
        c = cfg_cache.get(config)
        if c is None:
            c = {ids[i]: config[i] for i in range(n)}
            cfg_cache[config] = c
        return c

    def sum_dist(config):
        return sum(dist_to[ids[i]].get(config[i], big) for i in range(n))

    # Per-config dive state, mirroring pibt_solve: an accumulating priority vector
    # (off-goal agents gain +1, fractional reset on arrival) and a running-minimum
    # stall counter (best summed distance-to-goal along the path to this config).
    # `depth` doubles as the escape salt source so the perturbation varies down the
    # spine, exactly as pibt_solve's per-step salt does.
    def priority_order(config):
        prio = state[config][0]
        return [ids[k] for k in sorted(range(n), key=lambda k: (-prio[k], ids[k]))]

    prio0 = [dist_to[ids[i]].get(init[i], big) / size for i in range(n)]
    state = {init: (prio0, sum_dist(init), 0, 0)}   # config -> (prio, best, stuck, depth)
    expand = {init: 0}                # how many times each config has been expanded

    parent = {init: None}
    trees: dict = {init: [{}]}        # per-config stack of partial constraints
    explored = {init}
    open_stack = [init]               # DFS over configurations

    iterations = 0
    while open_stack:
        iterations += 1
        if iterations > max_iterations:
            return None
        config = open_stack[-1]
        if config == goal_config:
            return _reconstruct(parent, config, ids)

        tree = trees[config]
        if not tree:
            open_stack.pop()
            continue

        constraint = tree.pop()
        cfg = cfg_of(config)
        # Lazily branch: pin the next (so-far-unconstrained) agent to each of its
        # candidate cells, pushing those low-level nodes for later expansion.
        if len(constraint) < n:
            agent = ids[len(constraint)]
            for v in grid.neighbors(cfg[agent]):
                child = dict(constraint)
                child[agent] = v
                tree.append(child)

        prio, best, stuck, depth = state[config]
        # The escape salt mixes in `expand[config]` so a stuck config gets a
        # *distinct* salted dive every time it is re-expanded — not just the single
        # unconstrained one. This is what lets the spine recover where pibt_solve's
        # oscillating walk would: there, a livelock is reseeded with a fresh salt
        # each step; here, `explored` forbids revisiting a config, so the diversity
        # has to come from re-expanding the *same* node under new salts instead.
        salt = (depth + 1 + expand[config]) if stuck >= stall_patience else 0
        expand[config] += 1
        if constraint:
            order = ([a for a in ids if a in constraint]
                     + [a for a in priority_order(config) if a not in constraint])
        else:
            order = priority_order(config)
        nxt = _pibt(grid, cfg, order, dist_to, constraint, salt=salt)
        if nxt is None:
            continue
        new_config = tuple(nxt[ids[i]] for i in range(n))
        if new_config in explored:
            continue
        explored.add(new_config)
        parent[new_config] = config
        trees[new_config] = [{}]
        expand[new_config] = 0
        # Accumulate priority and roll the running-minimum stall forward, so the
        # child's spine expansion behaves like pibt_solve's next step.
        new_prio = [(prio[i] - int(prio[i])) if new_config[i] == goal_config[i]
                    else prio[i] + 1 for i in range(n)]
        s = sum_dist(new_config)
        if s < best:
            state[new_config] = (new_prio, s, 0, depth + 1)
        else:
            state[new_config] = (new_prio, best, stuck + 1, depth + 1)
        open_stack.append(new_config)

    return None


def _reconstruct(parent, goal_config, ids) -> Solution:
    seq = []
    c = goal_config
    while c is not None:
        seq.append(c)
        c = parent[c]
    seq.reverse()
    paths = {ids[i]: [step[i] for step in seq] for i in range(len(ids))}
    return Solution(paths=paths, cost=sum_of_costs(paths))
