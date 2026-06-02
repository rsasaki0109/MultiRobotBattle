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

import heapq
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
          max_iterations: int = 1_000_000, stall_patience: int = 3,
          optimize: bool = False, guide: dict | None = None,
          history: list | None = None):
    """Solve a MAPF instance (satisficing, complete) by LaCAM.

    ``agents`` maps an agent id to a ``(start, goal)`` tuple. Returns a
    collision-free :class:`Solution` (not cost-optimal), or ``None`` if the
    instance is infeasible or the iteration budget is exhausted. The greedy DFS
    spine runs the strong PIBT — accumulating priorities plus a deterministic
    livelock escape that engages after ``stall_patience`` non-improving steps; see
    the module docstring.

    With ``optimize=True`` it runs the **anytime** variant (LaCAM\\*): instead of
    returning the first solution, it keeps searching, tracking the best cost from
    the start to each configuration (``g``) and *rewiring* a configuration's parent
    whenever a cheaper route to it is found, while pruning any node that cannot beat
    the best solution so far (``g`` + the admissible sum-of-remaining-distances).
    It returns the cheapest solution found before ``max_iterations`` runs out.

    Scope, measured honestly: on **small** instances this reaches the true optimum
    — it matches CBS's sum-of-costs agent-for-agent on 200/200 random 4x4-6x6 / 2-4
    agent cases (where the satisficing default lands ~1.13x optimal). It does **not**
    scale as a cost optimizer: on 16-30-agent open grids the configuration space is
    astronomically large, lower-bound pruning barely bites, and a 200k-iteration
    budget (~10s/instance) returns the *same* cost as the first dive. For cost at
    scale use :func:`mapf_lns <mrn_coord.mapf.mapf_lns>` — local search drives those
    instances to ~1.13x the lower bound in a fraction of the time. The default
    (``optimize=False``) is unchanged: first solution, fast, at any team size.

    ``guide`` (default ``None``) overrides the per-agent distance map that drives
    PIBT's candidate ordering and the priority seed — :func:`lacam_ltm` passes
    congestion-weighted distances here. The admissible heuristic used for
    optimize pruning and the stall detector keeps using the *true* BFS distance,
    so guidance steers the dive without breaking the cost bound. ``history``
    (default ``None``), if a list, collects every committed agent move
    ``(from, to)`` across PIBT executions — the raw signal :func:`lacam_ltm`
    accumulates into its traffic map. With both ``None`` the search is
    byte-for-byte unchanged.
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
    # `guide_to` drives PIBT ordering/priority (LaCAM*+LTM passes weighted maps);
    # `dist_to` stays the true BFS distance for the admissible h and stall.
    guide_to = guide if guide is not None else dist_to

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

    prio0 = [guide_to[ids[i]].get(init[i], big) / size for i in range(n)]
    state = {init: (prio0, sum_dist(init), 0, 0)}   # config -> (prio, best, stuck, depth)
    expand = {init: 0}                # how many times each config has been expanded

    parent = {init: None}
    trees: dict = {init: [{}]}        # per-config stack of partial constraints
    explored = {init}
    open_stack = [init]               # DFS over configurations

    # The per-timestep sum-of-costs increment: an agent pays 1 unless it is parked
    # at its goal across the whole move. Summed over a path this is exactly the SOC.
    def edge_cost(a, b):
        return sum(1 for i in range(n)
                   if a[i] != goal_config[i] or b[i] != goal_config[i])

    g = {init: 0}                     # best known cost from init (LaCAM* only)
    best_cost = None                  # cost of the cheapest solution found so far

    iterations = 0
    while open_stack:
        iterations += 1
        if iterations > max_iterations:
            if optimize:
                break                 # return the best solution found so far
            return None
        config = open_stack[-1]
        if optimize:
            # Prune: a node that cannot beat the incumbent (g + admissible h) is dead.
            if best_cost is not None and g[config] + sum_dist(config) >= best_cost:
                open_stack.pop()
                continue
            if config == goal_config:
                if best_cost is None or g[config] < best_cost:
                    best_cost = g[config]
                open_stack.pop()      # keep searching for something cheaper
                continue
        elif config == goal_config:
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
        nxt = _pibt(grid, cfg, order, guide_to, constraint, salt=salt)
        if nxt is None:
            continue
        new_config = tuple(nxt[ids[i]] for i in range(n))

        if history is not None:        # committed actions for the traffic map
            for i in range(n):
                if config[i] != new_config[i]:
                    history.append((config[i], new_config[i]))

        if optimize:
            ng = g[config] + edge_cost(config, new_config)
            # Skip a successor that already cannot beat the incumbent.
            if best_cost is not None and ng + sum_dist(new_config) >= best_cost:
                continue
            if new_config in explored:
                if ng < g[new_config]:        # cheaper route found: rewire + re-open
                    g[new_config] = ng
                    parent[new_config] = config
                    open_stack.append(new_config)
                continue
            g[new_config] = ng
        elif new_config in explored:
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

    if optimize and goal_config in explored:
        return _reconstruct(parent, goal_config, ids)
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


def _soc(solution: Solution) -> int:
    """True sum-of-costs by arrival time: per agent, the last step it is *not*
    parked at its goal, plus one. ``sum_of_costs`` (= sum of path lengths) instead
    over-counts LaCAM's makespan padding (the trailing rest at the goal), so two
    solutions of different makespan aren't comparable by it; this is."""
    total = 0
    for path in solution.paths.values():
        goal = path[-1]
        arrival = 0
        for i, cell in enumerate(path):
            if cell != goal:
                arrival = i + 1
        total += arrival
    return total


def _weighted_dist(grid: GridWorld, goal: Cell, edge_w) -> dict:
    """Backward Dijkstra from ``goal``: shortest *weighted* distance from every
    cell to the goal, where moving ``p -> v`` costs ``edge_w(p, v)``. Used to turn
    a congestion-weighted traffic map into per-agent guidance for PIBT."""
    dist = {goal: 0.0}
    pq = [(0.0, goal)]
    while pq:
        d, v = heapq.heappop(pq)
        if d > dist[v]:
            continue
        for p in grid.neighbors(v):               # p is a predecessor of v
            nd = d + edge_w(p, v)
            if nd < dist.get(p, float("inf")):
                dist[p] = nd
                heapq.heappush(pq, (nd, p))
    return dist


def lacam_ltm(grid: GridWorld, agents: dict, *, rounds: int = 6,
              max_iterations: int = 200_000, stall_patience: int = 3,
              optimize: bool = True, w_max: float = 10.0):
    """LaCAM\\*+LTM — a Python reproduction of "A Lightweight Traffic Map for
    Efficient Anytime LaCAM\\*" (arXiv:2603.07891; only a C++ reference exists).

    Plain LaCAM\\* guides PIBT with a *static* shortest-path distance, so every
    dive re-walks the same congested corridors — which is why
    :func:`lacam`'s ``optimize=True`` stalls at scale (it returns the same cost
    as the first dive on 16-30-agent grids). LaCAM\\*+LTM instead builds a
    **lightweight traffic map** during the search: a directed-edge weight that
    accumulates the agent moves actually committed by PIBT. Between bounded runs
    it normalizes those counts into ``[0, w_max]``, recomputes each agent's
    guidance distance on the congestion-weighted graph (so dives route *around*
    the busy edges), and restarts. The cheapest solution over all rounds (by true
    :func:`_soc`) is returned.

    This is a faithful *subset* of the paper: it accumulates **committed** moves
    only — the blocked-action and wait-propagation terms are omitted — and
    restarts each round from the root (one-shot mode). The admissible heuristic is
    untouched, so within a round ``optimize=True`` keeps its cost guarantees;
    guidance only changes which dive PIBT takes. Deterministic.
    """
    ids = sorted(agents)
    goals = {a: agents[a][1] for a in ids}

    raw: dict = {}                                # directed edge -> commit count
    best: Solution | None = None
    best_soc = None
    for r in range(rounds):
        if raw:
            max_count = max(raw.values())
            norm = {e: w_max * c / max_count for e, c in raw.items()}

            def edge_w(p, v, _norm=norm):
                return 1.0 + _norm.get((p, v), 0.0)

            guide = {a: _weighted_dist(grid, goals[a], edge_w) for a in ids}
        else:
            guide = None                          # round 0: plain BFS guidance

        history: list = []
        sol = lacam(grid, agents, max_iterations=max_iterations,
                    stall_patience=stall_patience, optimize=optimize,
                    guide=guide, history=history)
        if sol is not None:
            soc = _soc(sol)
            if best_soc is None or soc < best_soc:
                best, best_soc = sol, soc
        for u, v in history:                      # accumulate committed moves
            raw[(u, v)] = raw.get((u, v), 0) + 1
    return best
