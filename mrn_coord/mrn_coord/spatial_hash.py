"""Uniform-grid spatial index for O(n) neighbour queries (battle scale-out).

Deterministic: cells are scanned in sorted order and bucket members are sorted
by index so tie-breaking matches the brute-force loops.
"""

from __future__ import annotations

import math


class SpatialHash:
    """Point index for disk queries on a 2-D position list."""

    __slots__ = ("cell_size", "_inv", "_buckets")

    def __init__(self, cell_size: float = 4.0):
        self.cell_size = cell_size
        self._inv = 1.0 / cell_size
        self._buckets: dict[tuple[int, int], list[int]] = {}

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return (int(math.floor(x * self._inv)), int(math.floor(y * self._inv)))

    def build(self, positions):
        """Index ``positions`` — a list of ``(x, y)``."""
        self._buckets = {}
        for i, (x, y) in enumerate(positions):
            self._buckets.setdefault(self._key(x, y), []).append(i)
        for cell in self._buckets:
            self._buckets[cell].sort()

    def query_disk(self, x: float, y: float, radius: float, positions):
        """Indices ``j`` with ``|positions[j] - (x,y)| <= radius``, sorted."""
        r_cells = int(math.ceil(radius * self._inv)) + 1
        cx, cy = self._key(x, y)
        r2 = radius * radius
        seen = []
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                for j in self._buckets.get((cx + dx, cy + dy), ()):
                    px, py = positions[j]
                    ddx, ddy = x - px, y - py
                    if ddx * ddx + ddy * ddy <= r2:
                        seen.append(j)
        seen.sort()
        return seen
