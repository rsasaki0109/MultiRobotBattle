"""A three-state occupancy grid for exploration.

Each cell is ``UNKNOWN`` (not yet observed), ``FREE`` (known traversable), or
``OCCUPIED`` (known obstacle). This is the minimal model a frontier-based
explorer needs: frontiers live exactly on the free/unknown boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Cell = tuple[int, int]

UNKNOWN = -1
FREE = 0
OCCUPIED = 1

_CHAR_TO_STATE = {"?": UNKNOWN, ".": FREE, "#": OCCUPIED}
_STATE_TO_CHAR = {UNKNOWN: "?", FREE: ".", OCCUPIED: "#"}


@dataclass
class OccupancyGrid:
    """A rectangular grid of cell states (default everything ``UNKNOWN``)."""

    width: int
    height: int
    states: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("grid dimensions must be positive")
        self.states = dict(self.states)

    @classmethod
    def from_rows(cls, rows) -> "OccupancyGrid":
        """Build from text rows using ``.`` free, ``#`` occupied, ``?`` unknown.

        Row 0 in the input is the top row (``y = height - 1``), so the grid reads
        the way it is written.
        """
        rows = list(rows)
        height = len(rows)
        width = len(rows[0]) if rows else 0
        states: dict = {}
        for r, row in enumerate(rows):
            if len(row) != width:
                raise ValueError("all rows must have equal width")
            y = height - 1 - r
            for x, ch in enumerate(row):
                if ch not in _CHAR_TO_STATE:
                    raise ValueError(f"unknown cell char: {ch!r}")
                states[(x, y)] = _CHAR_TO_STATE[ch]
        return cls(width, height, states)

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def state(self, cell: Cell) -> int:
        return self.states.get(cell, UNKNOWN)

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and self.state(cell) == FREE

    def is_unknown(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and self.state(cell) == UNKNOWN

    def is_occupied(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and self.state(cell) == OCCUPIED

    def neighbors4(self, cell: Cell) -> list[Cell]:
        x, y = cell
        cand = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [c for c in cand if self.in_bounds(c)]

    def free_neighbors(self, cell: Cell) -> list[Cell]:
        return [c for c in self.neighbors4(cell) if self.is_free(c)]

    def char(self, cell: Cell) -> str:
        return _STATE_TO_CHAR[self.state(cell)]
