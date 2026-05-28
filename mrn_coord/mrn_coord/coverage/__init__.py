"""Cooperative coverage / exploration: frontiers + multi-robot task allocation.

The third coordination module. Given a partially-explored occupancy grid and a
set of robots, decide *who explores where*:

1. **Frontiers** (:mod:`frontier`) — the boundary between known-free and unknown
   space is where new information lives. Frontier cells are free cells adjacent
   to unknown cells; clustering groups them into a handful of candidate targets.
2. **Allocation** (:mod:`allocation`) — assign frontier targets to robots by
   travel cost (BFS distance over free space). Two strategies: a fast greedy
   auction and an optimal Hungarian (min-total-cost) assignment.

Everything is pure and deterministic, unit-tested in CI; the Hungarian solver
is cross-checked against brute-force optimal assignment.
"""

from .allocation import (
    bfs_free_distances,
    greedy_auction,
    hungarian_assignment,
    min_cost_assignment,
    allocate_frontiers,
)
from .frontier import (
    FrontierCluster,
    cluster_frontiers,
    frontier_cells,
    is_frontier,
)
from .occupancy import FREE, OCCUPIED, UNKNOWN, OccupancyGrid

__all__ = [
    "UNKNOWN",
    "FREE",
    "OCCUPIED",
    "OccupancyGrid",
    "is_frontier",
    "frontier_cells",
    "FrontierCluster",
    "cluster_frontiers",
    "bfs_free_distances",
    "greedy_auction",
    "hungarian_assignment",
    "min_cost_assignment",
    "allocate_frontiers",
]
