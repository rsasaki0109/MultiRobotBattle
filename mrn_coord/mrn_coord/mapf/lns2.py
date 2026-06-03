"""MAPF-LNS2: fast repairing to feasibility by collision-minimizing LNS.

Li, Chen, Harabor, Stuckey & Koenig, *"MAPF-LNS2: Fast Repairing for Multi-Agent
Path Finding via Large Neighborhood Search"* (AAAI 2022).

The plain :func:`mrn_coord.mapf.lns.mapf_lns` is an *optimizer*: it starts from a
**feasible** (collision-free) solution and polishes its sum-of-costs, every
repair staying collision-free by construction. MAPF-LNS2 attacks the prior,
harder question — *finding a feasible solution at all* on instances so dense that
prioritized planning and CBS give up. It does so by turning feasibility itself
into an optimization: start from each agent's individual shortest path (ignoring
everyone, so the start state is riddled with collisions), and **minimize the
number of collisions** with Large Neighborhood Search until it hits zero.

The two pieces that differ from the cost-LNS:

- A **collision-minimizing low-level planner** (:func:`_plan_min_collision`).
  Where the optimizer's low level treats other agents' paths as *hard*
  obstacles, here they are *soft*: a replanned agent may pass through an occupied
  cell, but each such overlap counts a collision. A lexicographic space-time A*
  then finds the path with the **fewest collisions**, and the shortest among
  those. This is what lets repair make progress on a tangle that has no
  collision-free completion *yet*.

- The objective is the **number of colliding agent-pairs** over time (the size of
  the collision graph's edge multiset), not the sum-of-costs. A round destroys a
  neighborhood drawn from a **colliding** connected component, repairs those
  agents one by one against everyone else's current path, and keeps the result
  when it has no more collisions than before. When the count reaches zero the
  solution is feasible — collision-free — and is returned.

Like the optimizer it is anytime and deterministic given the seed; unlike it, the
returned solution is only *guaranteed* collision-free when ``stats["feasible"]``
is true (the count reached zero within the iteration budget). The truth of that
flag is decided by the exact global collision count, so the soft low level can be
approximate without affecting the feasibility guarantee.
"""

from __future__ import annotations

import heapq
import random

from .conflicts import cell_at
from .grid import Cell, GridWorld, manhattan
from .lns import _bfs_dist_from
from .solution import Solution, sum_of_costs
from .space_time_astar import plan_path


def _count_collisions(paths: dict) -> int:
    """Total colliding agent-pair instances over time (vertex + edge swaps).

    Zero iff the solution is collision-free. Agents are held at their goal past
    the end of their path (``cell_at`` clamps), so goal-hold conflicts count."""
    ids = list(paths)
    if len(ids) < 2:
        return 0
    horizon = max(len(p) for p in paths.values())
    total = 0
    for t in range(horizon):
        # vertex: every unordered pair sharing a cell at time t
        bucket: dict = {}
        for a in ids:
            bucket.setdefault(cell_at(paths[a], t), []).append(a)
        for occupants in bucket.values():
            k = len(occupants)
            if k > 1:
                total += k * (k - 1) // 2
        # edge: every unordered pair swapping across t -> t+1
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if (cell_at(paths[a], t) == cell_at(paths[b], t + 1)
                        and cell_at(paths[a], t + 1) == cell_at(paths[b], t)
                        and cell_at(paths[a], t) != cell_at(paths[a], t + 1)):
                    total += 1
    return total


def _collision_graph(paths: dict) -> dict:
    """Map each agent to the set of agents it collides with (any time)."""
    ids = list(paths)
    horizon = max((len(p) for p in paths.values()), default=0)
    graph: dict = {a: set() for a in ids}
    for t in range(horizon):
        bucket: dict = {}
        for a in ids:
            bucket.setdefault(cell_at(paths[a], t), []).append(a)
        for occupants in bucket.values():
            for i in range(len(occupants)):
                for j in range(i + 1, len(occupants)):
                    graph[occupants[i]].add(occupants[j])
                    graph[occupants[j]].add(occupants[i])
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if (cell_at(paths[a], t) == cell_at(paths[b], t + 1)
                        and cell_at(paths[a], t + 1) == cell_at(paths[b], t)
                        and cell_at(paths[a], t) != cell_at(paths[a], t + 1)):
                    graph[a].add(b)
                    graph[b].add(a)
    return graph


def _soft_reservations(paths: dict, subset: set, horizon: int):
    """Soft vertex/edge occupancy counts from every agent NOT in ``subset``.

    ``soft_v[(cell, t)]`` is how many frozen agents occupy ``cell`` at ``t``
    (each is a collision the replanned agent would incur there); goal-holds are
    expanded to ``horizon``. ``soft_e[(frm, to, t)]`` counts frozen swaps."""
    soft_v: dict = {}
    soft_e: dict = {}
    for agent, path in paths.items():
        if agent in subset:
            continue
        for t, cell in enumerate(path):
            soft_v[(cell, t)] = soft_v.get((cell, t), 0) + 1
        goal = path[-1]
        for t in range(len(path), horizon + 1):
            soft_v[(goal, t)] = soft_v.get((goal, t), 0) + 1
        for t in range(len(path) - 1):
            key = (path[t + 1], path[t], t + 1)
            soft_e[key] = soft_e.get(key, 0) + 1
    return soft_v, soft_e


def _plan_min_collision(grid, start, goal, soft_v, soft_e, horizon, dist):
    """Space-time A* minimizing (collisions, then length) against soft occupancy.

    Soft reservations are penalties, not walls: entering ``(cell, t)`` adds
    ``soft_v[(cell, t)]`` collisions and a swap adds ``soft_e``. Returns the
    fewest-collision path (shortest among ties), or ``None`` if the goal is
    unreachable within ``horizon``. ``dist`` is the obstacle-aware BFS distance
    to the goal (the length heuristic)."""
    if start not in dist:
        return None
    # Wait at the goal until after the last time a frozen agent transiently
    # passes through it, so a settled agent need not keep colliding there.
    last_goal = max((t for (c, t) in soft_v if c == goal), default=0)

    start_col = soft_v.get((start, 0), 0)
    counter = 0
    # (collisions, g+h, tie, collisions, g, cell, t)
    open_heap = [(start_col + dist[start], start_col, counter, start, 0)]
    # best (collisions, g) reached per (cell, t) for dominance pruning
    best: dict = {}

    while open_heap:
        _, col, _, cell, t = heapq.heappop(open_heap)
        prev = best.get((cell, t))
        if prev is not None and (prev[0] < col or (prev[0] == col and prev[1] <= t)):
            continue
        best[(cell, t)] = (col, t)

        if cell == goal and t >= last_goal:
            return _trace((cell, t), came_from)

        if t >= horizon:
            continue
        nt = t + 1
        for ncell in grid.neighbors(cell):
            add = soft_v.get((ncell, nt), 0)
            if ncell != cell:
                add += soft_e.get((cell, ncell, nt), 0)
            ncol = col + add
            key = (ncell, nt)
            prev = best.get(key)
            if prev is not None and (prev[0] < ncol or
                                     (prev[0] == ncol and prev[1] <= nt)):
                continue
            counter += 1
            came_from[key] = (cell, t)
            heapq.heappush(
                open_heap,
                (ncol + dist.get(ncell, manhattan(ncell, goal)),
                 ncol, counter, ncell, nt),
            )
    return None


# ``came_from`` is module-level scratch reset per planner call (keeps the A* body
# small); the planner is single-threaded and deterministic so this is safe.
came_from: dict = {}


def _trace(state, parents):
    path = [state]
    while state in parents:
        state = parents[state]
        path.append(state)
    path.reverse()
    return [cell for (cell, _) in path]


def mapf_lns2(
    grid: GridWorld,
    agents: dict,
    *,
    neighborhood_size: int = 5,
    iterations: int = 200,
    seed: int = 0,
    stats: dict | None = None,
):
    """Find a feasible MAPF solution by collision-minimizing LNS.

    ``agents`` maps an agent id to ``(start, goal)``. Starts from individual
    shortest paths (collision-ridden) and repairs toward zero collisions. Returns
    the best :class:`Solution` found; it is collision-free iff
    ``stats["feasible"]`` is true. ``stats`` records ``initial_collisions``,
    ``final_collisions``, ``iterations``, ``accepted``, and ``feasible``.
    Returns ``None`` only if some agent cannot reach its goal at all.
    """
    global came_from
    ids = sorted(agents)
    dist = {a: _bfs_dist_from(grid, agents[a][1]) for a in ids}

    paths: dict = {}
    for a in ids:
        came_from = {}
        p = plan_path(grid, agents[a][0], agents[a][1])
        if p is None:
            return None
        paths[a] = p

    formula = (2 * grid.width * grid.height + len(ids)
               + grid.width + grid.height + 5)
    horizon = max(formula, max(len(p) for p in paths.values())) + len(ids) + 5

    cur = _count_collisions(paths)
    initial = cur
    rng = random.Random(seed)
    accepted = 0
    k = max(2, min(neighborhood_size, len(ids)))

    for _ in range(iterations):
        if cur == 0:
            break
        subset = _select_neighborhood(paths, rng, k)
        if not subset:
            break
        saved = {a: paths[a] for a in subset}
        soft_v, soft_e = _soft_reservations(paths, subset, horizon)
        # repair the subset one by one, each against everyone else's current
        # path (frozen non-subset plus already-repaired subset members)
        order = sorted(subset)
        rng.shuffle(order)
        ok = True
        for a in order:
            came_from = {}
            p = _plan_min_collision(grid, agents[a][0], agents[a][1],
                                    soft_v, soft_e, horizon, dist[a])
            if p is None:
                ok = False
                break
            paths[a] = p
            for t, cell in enumerate(p):
                soft_v[(cell, t)] = soft_v.get((cell, t), 0) + 1
            for t in range(len(p), horizon + 1):
                soft_v[(p[-1], t)] = soft_v.get((p[-1], t), 0) + 1
            for t in range(len(p) - 1):
                key = (p[t + 1], p[t], t + 1)
                soft_e[key] = soft_e.get(key, 0) + 1
        new = _count_collisions(paths) if ok else cur + 1
        if ok and new <= cur:
            if new < cur:
                accepted += 1
            cur = new
        else:
            for a, p in saved.items():            # revert: repair did not help
                paths[a] = p

    if stats is not None:
        stats["initial_collisions"] = initial
        stats["final_collisions"] = cur
        stats["iterations"] = iterations
        stats["accepted"] = accepted
        stats["feasible"] = cur == 0
    return Solution(paths=paths, cost=sum_of_costs(paths))


def _select_neighborhood(paths, rng, k):
    """A neighborhood of up to ``k`` agents grown from a colliding component.

    Pick a random colliding agent and BFS its collision graph; if that yields
    fewer than ``k`` agents, pad with other colliding agents, then any agents."""
    graph = _collision_graph(paths)
    colliding = [a for a in paths if graph[a]]
    if not colliding:
        return set()
    seed_agent = colliding[rng.randrange(len(colliding))]
    chosen = [seed_agent]
    frontier = [seed_agent]
    seen = {seed_agent}
    while frontier and len(chosen) < k:
        nbrs = sorted(graph[frontier.pop()])
        rng.shuffle(nbrs)
        for b in nbrs:
            if b not in seen:
                seen.add(b)
                chosen.append(b)
                frontier.append(b)
                if len(chosen) >= k:
                    break
    if len(chosen) < k:
        # Pad with ANY remaining agents (not just colliding ones): resolving a
        # collision often needs a non-colliding bystander to step aside.
        rest = [a for a in paths if a not in seen]
        rng.shuffle(rest)
        chosen += rest[:k - len(chosen)]
    return set(chosen[:k])
