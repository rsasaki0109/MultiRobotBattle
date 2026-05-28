"""The shared grid world for MAPF.

A 4-connected grid with blocked cells. Movement is one cell per timestep in a
cardinal direction or a *wait* in place, so :meth:`GridWorld.neighbors` always
includes the cell itself. Coordinates are ``(x, y)`` integers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Cell = tuple[int, int]


def manhattan(a: Cell, b: Cell) -> int:
    """L1 distance, the admissible heuristic for 4-connected movement."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass
class GridWorld:
    """A rectangular grid with a set of blocked cells."""

    width: int
    height: int
    blocked: frozenset = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("grid dimensions must be positive")
        # accept any iterable of cells; store as a frozenset for fast lookup
        self.blocked = frozenset(self.blocked)

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, cell: Cell) -> bool:
        return cell not in self.blocked

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and self.passable(cell)

    def neighbors(self, cell: Cell) -> list[Cell]:
        """Reachable cells in one timestep, including waiting in place."""
        x, y = cell
        candidates = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [c for c in candidates if self.is_free(c)]
