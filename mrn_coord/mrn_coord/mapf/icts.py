"""The Increasing Cost Tree Search (ICTS) for optimal MAPF.

Sharon, Stern, Goldenberg & Felner, *"The increasing cost tree search for
optimal multi-agent pathfinding"* (Artificial Intelligence, 2013; IJCAI 2011).
A different optimal paradigm from Conflict-Based Search: instead of branching on
*constraints* (CBS / :mod:`mrn_coord.mapf.cbs`), ICTS branches on *costs*.

Two levels:

1. **High level — the Increasing Cost Tree (ICT).** A node is a vector of
   per-agent path costs ``(C_1, ..., C_k)``. The root gives every agent its
   individual shortest-path cost. A child increments one agent's cost by ``1``.
   The tree is searched in order of increasing total cost ``sum C_i``, so the
   first node that admits a conflict-free joint solution is optimal in
   sum-of-costs — exactly what :func:`mrn_coord.mapf.cbs.cbs` returns.

2. **Low level — the MDD goal test.** For a cost vector, build each agent's MDD
   (the union of all cost-``C_i`` paths; :func:`mrn_coord.mapf.mdd.build_mdd`)
   and search the *cross-product* of the MDDs for one assignment of paths that
   is free of vertex and swap conflicts. If one exists, the node is a goal.

The cross-product search is exponential in the number of agents — that is the
known weakness of ICTS, and why it shines on few-but-tightly-coupled agents
rather than large open teams. Its key accelerator, reused here verbatim, is
**pairwise pruning**: before the full ``k``-agent search, check every *pair* of
agents in isolation (:func:`mrn_coord.mapf.mdd.are_dependent`) — if any two
agents have no conflict-free pair of cost-``C_i`` paths, the full node cannot
either, so it is pruned without the expensive joint search. ``are_dependent``,
written for CBSH's dependency graph, is precisely this 2-agent MDD test.

``prune=None`` disables pairwise pruning (a clean ablation: same answer, but
every node pays for the full joint search), the way ``cbsh(heuristic=None)``
ablates the heuristic.
"""

from __future__ import annotations

from collections import deque

from .grid import GridWorld
from .mdd import are_dependent, build_mdd
from .solution import Solution, sum_of_costs
from .space_time_astar import plan_path


def icts(
    grid: GridWorld,
    agents: dict,
    *,
    prune: str | None = "pairwise",
    max_nodes: int = 20_000,
    stats: dict | None = None,
):
    """Solve a MAPF instance optimally with the Increasing Cost Tree Search.

    ``agents`` maps an agent id to ``(start, goal)``. ``prune`` is ``"pairwise"``
    (the default — pairwise-dependency pruning before each joint search) or
    ``None`` (no pruning; same optimum, more joint searches). Returns an optimal
    :class:`Solution` (the same sum-of-costs as :func:`cbs`) or ``None`` if the
    instance is infeasible or the ``max_nodes`` budget is exhausted.

    When ``stats`` is given it is populated with ``"nodes"`` (ICT nodes dequeued),
    ``"joint_searches"`` (full ``k``-agent searches actually run), and
    ``"pruned"`` (nodes eliminated by the pairwise test before any joint search).
    """
    agent_ids = list(agents)

    # Root: each agent's individual shortest-path cost (unconstrained).
    base: list[int] = []
    for a in agent_ids:
        start, goal = agents[a]
        path = plan_path(grid, start, goal)
        if path is None:
            return None
        base.append(len(path) - 1)
    root = tuple(base)

    mdd_memo: dict = {}
    dep_memo: dict = {}

    def mdd_for(idx: int, cost: int):
        key = (idx, cost)
        if key not in mdd_memo:
            start, goal = agents[agent_ids[idx]]
            mdd_memo[key] = build_mdd(grid, start, goal, cost)
        return mdd_memo[key]

    nodes = joint_searches = pruned = 0
    visited = {root}
    queue: deque = deque([root])

    while queue:
        costs = queue.popleft()
        nodes += 1
        if nodes > max_nodes:
            _record(stats, nodes, joint_searches, pruned)
            return None

        mdds = [mdd_for(i, costs[i]) for i in range(len(agent_ids))]
        if any(m is None for m in mdds):
            # No path of exactly this length for some agent (cannot happen for
            # cost >= individual optimum on a grid with waits) — skip the node.
            _enqueue_children(costs, visited, queue)
            continue

        if prune == "pairwise" and _pairwise_blocked(
            grid, agents, agent_ids, mdds, dep_memo
        ):
            pruned += 1
            _enqueue_children(costs, visited, queue)
            continue

        joint_searches += 1
        paths = _joint_search(grid, agents, agent_ids, mdds, costs)
        if paths is not None:
            trimmed = {a: _trim_goal(p, agents[a][1]) for a, p in paths.items()}
            _record(stats, nodes, joint_searches, pruned)
            return Solution(paths=trimmed, cost=sum_of_costs(trimmed))

        _enqueue_children(costs, visited, queue)

    _record(stats, nodes, joint_searches, pruned)
    return None


def _record(stats, nodes, joint_searches, pruned):
    if stats is not None:
        stats["nodes"] = nodes
        stats["joint_searches"] = joint_searches
        stats["pruned"] = pruned


def _enqueue_children(costs, visited, queue):
    """Children of an ICT node: add ``1`` to one agent's cost. The BFS queue,
    fed children of total ``T+1`` only after all total-``T`` nodes, dequeues in
    non-decreasing total cost — so the first conflict-free node is optimal."""
    for i in range(len(costs)):
        child = costs[:i] + (costs[i] + 1,) + costs[i + 1:]
        if child not in visited:
            visited.add(child)
            queue.append(child)


def _pairwise_blocked(grid, agents, agent_ids, mdds, dep_memo) -> bool:
    """True if some pair of agents is *dependent* at these costs — i.e. has no
    conflict-free pair of paths — so the whole node is hopeless (ICTS's pairwise
    pruning, via the same 2-agent MDD test CBSH uses for its dependency graph)."""
    n = len(agent_ids)
    for i in range(n):
        for j in range(i + 1, n):
            key = (i, mdds[i].cost, j, mdds[j].cost)
            dep = dep_memo.get(key)
            if dep is None:
                start_a = agents[agent_ids[i]][0]
                start_b = agents[agent_ids[j]][0]
                dep = are_dependent(grid, mdds[i], mdds[j], start_a, start_b)
                dep_memo[key] = dep
            if dep:
                return True
    return False


def _joint_search(grid, agents, agent_ids, mdds, costs):
    """Search the cross-product of the MDDs for a conflict-free joint path.

    A state is the tuple of cells the agents occupy at a timestep. We step every
    agent along its own MDD (an agent past its cost is parked at its goal),
    forbid vertex and swap conflicts, and dedupe states per level. Returns the
    per-agent paths (length ``horizon + 1``) or ``None`` if no joint path exists.
    """
    n = len(agent_ids)
    horizon = max(costs)
    start_state = tuple(agents[a][0] for a in agent_ids)
    goal_state = tuple(agents[a][1] for a in agent_ids)

    # parents[t] maps each reachable state at time t to its predecessor at t-1.
    parents: list[dict] = [{start_state: None}]
    frontier = {start_state}

    for t in range(horizon):
        nxt: dict = {}
        next_cells = [mdds[i].cells(t + 1) for i in range(n)]
        for state in frontier:
            for combo in _joint_successors(grid, state, next_cells):
                if combo not in nxt:
                    nxt[combo] = state
        if not nxt:
            return None
        parents.append(nxt)
        frontier = nxt.keys()

    if goal_state not in parents[horizon]:
        return None

    # Backtrack the chosen joint path, then split it into per-agent paths.
    states = [goal_state]
    cur = goal_state
    for t in range(horizon, 0, -1):
        cur = parents[t][cur]
        states.append(cur)
    states.reverse()
    return {
        agent_ids[i]: [states[t][i] for t in range(horizon + 1)]
        for i in range(n)
    }


def _joint_successors(grid, state, next_cells):
    """All conflict-free joint successors of ``state``, assigning agents one at a
    time and pruning vertex/swap conflicts against already-placed agents (far
    cheaper than enumerating the full Cartesian product then filtering)."""
    n = len(state)
    out: list = []
    placed: list = []
    occupied: set = set()

    def rec(idx: int):
        if idx == n:
            out.append(tuple(placed))
            return
        here = state[idx]
        for w in grid.neighbors(here):
            if w not in next_cells[idx]:
                continue
            if w in occupied:  # vertex conflict
                continue
            # swap conflict: some placed agent j moved here<-w while we move w
            if any(w == state[j] and placed[j] == here for j in range(idx)):
                continue
            placed.append(w)
            occupied.add(w)
            rec(idx + 1)
            occupied.discard(w)
            placed.pop()

    rec(0)
    return out


def _trim_goal(path: list, goal):
    """Drop trailing goal-parked cells so ``len(path) - 1`` is the agent's true
    cost (the last time it occupies its goal), matching CBS's path lengths."""
    end = len(path)
    while end > 1 and path[end - 1] == goal and path[end - 2] == goal:
        end -= 1
    return path[:end]
