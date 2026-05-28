"""Solution container and cost/rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .grid import Cell, GridWorld


def sum_of_costs(paths: dict) -> int:
    """Sum over agents of (path length - 1) — the standard MAPF cost."""
    return sum(max(0, len(p) - 1) for p in paths.values())


def makespan(paths: dict) -> int:
    """Time at which the last agent reaches its goal."""
    return max((max(0, len(p) - 1) for p in paths.values()), default=0)


def pad_paths(paths: dict) -> dict:
    """Pad every path to the makespan horizon by holding the goal cell."""
    horizon = max((len(p) for p in paths.values()), default=0)
    return {
        agent: list(p) + [p[-1]] * (horizon - len(p))
        for agent, p in paths.items()
    }


@dataclass
class Solution:
    """A set of per-agent collision-free paths and its sum-of-costs."""

    paths: dict
    cost: int

    @property
    def makespan(self) -> int:
        return makespan(self.paths)


def render_ascii(grid: GridWorld, paths: dict, t: int) -> str:
    """Render the grid at time ``t`` with each agent drawn at its cell.

    Agents are labeled by the first character of ``str(agent_id)``; blocked
    cells are ``#`` and free cells ``.``. Useful for CLI demos and quick visual
    checks in tests.
    """
    positions: dict[Cell, str] = {}
    for agent, path in paths.items():
        cell = path[t] if t < len(path) else path[-1]
        positions[cell] = str(agent)[0]

    lines = []
    for y in range(grid.height - 1, -1, -1):
        row = []
        for x in range(grid.width):
            cell = (x, y)
            if cell in positions:
                row.append(positions[cell])
            elif not grid.passable(cell):
                row.append("#")
            else:
                row.append(".")
        lines.append("".join(row))
    return "\n".join(lines)
