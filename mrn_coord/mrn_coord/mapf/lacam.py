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
finds a solution whenever one exists — while staying fast enough to scale to
teams CBS cannot touch.

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


def _pibt(grid, config, order, dist_to, forced):
    """One PIBT step: return ``{agent: next_cell}`` or ``None`` if infeasible.

    ``config`` maps agent -> current cell; ``order`` is the priority order to
    decide agents in; ``dist_to`` maps agent -> {cell: dist-to-goal}; ``forced``
    pins some agents to a specific next cell (the LaCAM constraint). Pushing
    (priority inheritance) and swap prevention are exactly PIBT's.
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


def lacam(grid: GridWorld, agents: dict, *, max_iterations: int = 1_000_000):
    """Solve a MAPF instance (satisficing, complete) by LaCAM.

    ``agents`` maps an agent id to a ``(start, goal)`` tuple. Returns a
    collision-free :class:`Solution` (not cost-optimal), or ``None`` if the
    instance is infeasible or the iteration budget is exhausted.
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
    cfg_cache: dict = {}
    order_cache: dict = {}

    def cfg_of(config):
        c = cfg_cache.get(config)
        if c is None:
            c = {ids[i]: config[i] for i in range(n)}
            cfg_cache[config] = c
        return c

    def priority_order(config):
        # far-from-goal first (a PIBT heuristic); ties by id for determinism.
        o = order_cache.get(config)
        if o is None:
            cfg = cfg_of(config)
            o = sorted(ids, key=lambda a: (-dist_to[a].get(cfg[a], 0), a))
            order_cache[config] = o
        return o

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

        if constraint:
            order = ([a for a in ids if a in constraint]
                     + [a for a in priority_order(config) if a not in constraint])
        else:
            order = priority_order(config)
        nxt = _pibt(grid, cfg, order, dist_to, constraint)
        if nxt is None:
            continue
        new_config = tuple(nxt[ids[i]] for i in range(n))
        if new_config in explored:
            continue
        explored.add(new_config)
        parent[new_config] = config
        trees[new_config] = [{}]
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
