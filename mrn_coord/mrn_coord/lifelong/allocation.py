"""Task allocation for lifelong MAPF: who does which task.

The lifelong loop (:mod:`mrn_coord.lifelong.lifelong`) has to hand free robots
new tasks forever. The cheapest rule is round-robin — deal out the next task in
a fixed cycle, ignoring geometry — which routinely sends a robot clear across
the warehouse past a closer one. Smarter allocation assigns by *cost* (here the
obstacle-aware BFS travel distance), which shortens trips and lifts throughput.

Two cost-aware allocators, both pure and deterministic, both taking a
``cost[i][j]`` matrix (agent ``i`` -> task ``j``, ``inf`` = forbidden) and
returning ``{agent_row: task_col}`` for ``min(#agents, #tasks)`` pairs:

- :func:`hungarian` — the optimal solution to the linear assignment problem
  (Kuhn-Munkres with potentials, ``O(n^3)``): the assignment of minimum total
  cost. The centralized optimum to compare against.
- :func:`auction` — a **regret-based auction**: each round the still-unassigned
  agent with the most to lose (largest gap between its best and second-best
  remaining task) bids first and claims its best task. A decentralized,
  market-style heuristic — fast and close to optimal — of the kind used for
  multi-robot task allocation.
"""

from __future__ import annotations

INF = float("inf")


def hungarian(cost) -> dict:
    """Optimal min-total-cost assignment (Kuhn-Munkres with potentials).

    ``cost`` is an ``R x C`` matrix. Returns ``{row: col}`` matching every row
    (if ``R <= C``) or every column, whichever is smaller, at minimum total
    cost. Rows/cols assigned through an ``inf`` entry are dropped from the
    result (treated as forbidden).
    """
    rows = len(cost)
    cols = len(cost[0]) if rows else 0
    if rows == 0 or cols == 0:
        return {}

    # The algorithm matches every "worker" (the smaller side); transpose so
    # workers are rows.
    transposed = rows > cols
    if transposed:
        cost = [[cost[i][j] for i in range(rows)] for j in range(cols)]
        rows, cols = cols, rows

    n, m = rows, cols
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)        # p[j] = worker assigned to column j (1-indexed)
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
            for j in range(m + 1):
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

    assignment = {}
    for j in range(1, m + 1):
        if p[j] != 0:
            r, c = p[j] - 1, j - 1
            if cost[r][c] >= INF:
                continue
            if transposed:
                r, c = c, r
            assignment[r] = c
    return assignment


def auction(cost) -> dict:
    """Regret-based sequential auction assignment.

    Each round, every unassigned agent looks at its cheapest and second-cheapest
    remaining task; the agent with the largest *regret* (second − first, i.e.
    the most it loses by not getting its best) bids and is given its best task.
    Ties break toward lower cost then lower index, so it is deterministic.
    Returns ``{row: col}``.
    """
    rows = len(cost)
    cols = len(cost[0]) if rows else 0
    free_rows = set(range(rows))
    free_cols = set(range(cols))
    assignment = {}
    while free_rows and free_cols:
        best = None  # (regret, -best_cost, row, col) maximized on regret
        for i in sorted(free_rows):
            opts = sorted((cost[i][j], j) for j in free_cols)
            if opts[0][0] >= INF:
                continue                       # no reachable task for this agent
            first_cost, first_col = opts[0]
            regret = (opts[1][0] - first_cost) if len(opts) > 1 else INF
            key = (regret, -first_cost)
            if best is None or key > best[0]:
                best = (key, i, first_col)
        if best is None:
            break                              # every free agent is forbidden
        _, row, col = best
        assignment[row] = col
        free_rows.discard(row)
        free_cols.discard(col)
    return assignment


ALLOCATORS = {"hungarian": hungarian, "auction": auction}
