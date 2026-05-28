"""Frontier detection and clustering.

A *frontier cell* is a known-free cell adjacent to at least one unknown cell —
the boundary of explored space, and therefore where moving gains new
information. Individual frontier cells are clustered into connected groups so a
robot can be sent to one representative target per group rather than one per
cell.
"""

from __future__ import annotations

from dataclasses import dataclass

from .occupancy import Cell, OccupancyGrid


def is_frontier(grid: OccupancyGrid, cell: Cell) -> bool:
    """True if ``cell`` is free and borders unknown space."""
    if not grid.is_free(cell):
        return False
    return any(grid.is_unknown(n) for n in grid.neighbors4(cell))


def frontier_cells(grid: OccupancyGrid) -> list[Cell]:
    """All frontier cells, in a deterministic (x, then y) order."""
    cells = [
        (x, y)
        for x in range(grid.width)
        for y in range(grid.height)
        if is_frontier(grid, (x, y))
    ]
    return cells


@dataclass(frozen=True)
class FrontierCluster:
    """A connected group of frontier cells and a representative target."""

    cells: tuple
    representative: Cell

    @property
    def size(self) -> int:
        return len(self.cells)


def _centroid(cells) -> tuple[float, float]:
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def cluster_frontiers(grid: OccupancyGrid, cells=None) -> list[FrontierCluster]:
    """Group frontier cells into 4-connected clusters.

    Each cluster's representative is the member nearest its centroid (the
    medoid), so the target sits inside the frontier. Clusters are returned
    sorted by representative for determinism.
    """
    if cells is None:
        cells = frontier_cells(grid)
    remaining = set(cells)
    clusters: list[FrontierCluster] = []

    while remaining:
        seed = min(remaining)            # deterministic start
        stack = [seed]
        remaining.discard(seed)
        group = [seed]
        while stack:
            cur = stack.pop()
            for n in grid.neighbors4(cur):
                if n in remaining:
                    remaining.discard(n)
                    stack.append(n)
                    group.append(n)
        cx, cy = _centroid(group)
        representative = min(
            group, key=lambda c: ((c[0] - cx) ** 2 + (c[1] - cy) ** 2, c)
        )
        clusters.append(FrontierCluster(tuple(sorted(group)), representative))

    clusters.sort(key=lambda fc: fc.representative)
    return clusters
