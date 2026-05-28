"""Travel-cost computation and multi-robot frontier allocation.

Cost is the shortest distance through known-free space (BFS, 4-connected).
Two assignment strategies are provided:

- :func:`greedy_auction` — repeatedly commit the globally cheapest
  (robot, frontier) pair; fast and simple, not always optimal.
- :func:`hungarian_assignment` / :func:`min_cost_assignment` — the optimal
  minimum-total-cost assignment (Kuhn–Munkres), handling rectangular cost
  matrices by transposing so rows <= cols.

:func:`allocate_frontiers` ties them together: BFS cost from each robot to each
frontier representative, then an assignment.
"""

from __future__ import annotations

from collections import deque

from .occupancy import Cell, OccupancyGrid

INF = float("inf")


def bfs_free_distances(grid: OccupancyGrid, start: Cell) -> dict:
    """Distances from ``start`` to every reachable free cell (4-connected).

    ``start`` itself must be free. Unreachable cells are simply absent from the
    returned dict.
    """
    if not grid.is_free(start):
        return {}
    dist = {start: 0}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for n in grid.free_neighbors(cur):
            if n not in dist:
                dist[n] = dist[cur] + 1
                queue.append(n)
    return dist


def greedy_auction(robots, frontiers, cost_fn) -> dict:
    """Assign frontiers to robots by repeatedly taking the cheapest pair.

    ``robots`` and ``frontiers`` are id lists; ``cost_fn(robot, frontier)``
    returns a finite cost or ``inf`` if unreachable. Returns ``robot -> frontier``
    for as many pairs as possible (``min(len(robots), len(frontiers))`` at most),
    skipping infinite-cost pairs.
    """
    free_robots = list(robots)
    free_frontiers = list(frontiers)
    assignment: dict = {}

    while free_robots and free_frontiers:
        best = None
        for r in free_robots:
            for f in free_frontiers:
                c = cost_fn(r, f)
                if c == INF:
                    continue
                key = (c, str(r), str(f))
                if best is None or key < best[0]:
                    best = (key, r, f)
        if best is None:
            break                       # no reachable pairs remain
        _, r, f = best
        assignment[r] = f
        free_robots.remove(r)
        free_frontiers.remove(f)

    return assignment


def hungarian_assignment(cost) -> list:
    """Optimal min-cost assignment for an ``n x m`` matrix with ``n <= m``.

    Returns a list of ``(row, col)`` pairs matching every row to a distinct
    column (Kuhn–Munkres / Jonker shortest-augmenting-path, O(n^2 m)).
    """
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    if n > m:
        raise ValueError("hungarian_assignment requires rows <= cols")

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)        # p[j] = row matched to column j (1-indexed; 0 = none)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    return [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]


def min_cost_assignment(cost) -> list:
    """Optimal assignment for any rectangular matrix.

    Returns ``(row, col)`` pairs. Transposes internally when there are more
    rows than columns so :func:`hungarian_assignment` always sees rows <= cols.
    """
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    if n <= m:
        return hungarian_assignment(cost)
    transposed = [[cost[i][j] for i in range(n)] for j in range(m)]
    return [(i, j) for (j, i) in hungarian_assignment(transposed)]


def allocate_frontiers(
    grid: OccupancyGrid, robot_positions: dict, frontier_targets, *, method="hungarian"
) -> dict:
    """Assign frontier cells to robots minimizing travel cost.

    ``robot_positions`` maps robot id to its (free) cell; ``frontier_targets`` is
    a list of frontier cells (e.g. cluster representatives). ``method`` is
    ``"hungarian"`` (optimal total cost) or ``"greedy"``. Returns
    ``robot -> frontier_cell`` for the reachable matches.
    """
    robots = list(robot_positions)
    frontiers = list(frontier_targets)
    if not robots or not frontiers:
        return {}

    # Precompute BFS distance fields once per robot.
    fields = {r: bfs_free_distances(grid, robot_positions[r]) for r in robots}

    def cost_fn(r, f):
        return fields[r].get(f, INF)

    if method == "greedy":
        return greedy_auction(robots, frontiers, cost_fn)
    if method != "hungarian":
        raise ValueError(f"unknown method: {method!r}")

    # Build a cost matrix; replace unreachable inf with a large finite penalty so
    # the optimal solver stays well-defined, then drop those padded matches.
    reachable = [
        cost_fn(r, f)
        for r in robots for f in frontiers
        if cost_fn(r, f) != INF
    ]
    penalty = (max(reachable) + 1) * (len(robots) + len(frontiers) + 1) if reachable else 1.0
    matrix = [[
        (cost_fn(r, f) if cost_fn(r, f) != INF else penalty)
        for f in frontiers
    ] for r in robots]

    assignment: dict = {}
    for ri, fi in min_cost_assignment(matrix):
        if cost_fn(robots[ri], frontiers[fi]) != INF:
            assignment[robots[ri]] = frontiers[fi]
    return assignment
