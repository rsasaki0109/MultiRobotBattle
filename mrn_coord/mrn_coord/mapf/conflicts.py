"""Conflict detection between planned paths.

Two paths conflict if the agents occupy the same cell at the same time (a
*vertex* conflict) or swap cells across a single timestep (an *edge* conflict).
Paths can have different lengths; an agent that has reached its goal is treated
as staying there, so :func:`cell_at` clamps past the end of a path.
"""

from __future__ import annotations

from dataclasses import dataclass

from .grid import Cell


@dataclass(frozen=True)
class VertexConflict:
    """Two agents occupy ``cell`` at the same ``time``."""

    agent_a: object
    agent_b: object
    cell: Cell
    time: int


@dataclass(frozen=True)
class EdgeConflict:
    """Two agents swap: ``a`` moves ``cell_a -> cell_b`` while ``b`` moves
    ``cell_b -> cell_a``, both arriving at ``time``."""

    agent_a: object
    agent_b: object
    cell_a: Cell
    cell_b: Cell
    time: int


def cell_at(path: list[Cell], t: int) -> Cell:
    """Cell occupied at time ``t``; agents wait at their goal past the end."""
    if t < len(path):
        return path[t]
    return path[-1]


def detect_first_conflict(paths: dict, *, window: int | None = None):
    """Return the earliest conflict between any pair of paths, or ``None``.

    Scans time forward and, within a timestep, all agent pairs; returns a
    :class:`VertexConflict` or :class:`EdgeConflict` for the first collision
    found. Determinism comes from the insertion order of ``paths``.

    ``window`` bounds the resolution horizon: when given, only conflicts at
    times ``t <= window`` are reported and anything beyond is ignored. This is
    what Rolling-Horizon Collision Resolution (RHCR) needs — resolve collisions
    inside the lookahead window, leave the rest for the next replan.
    """
    agents = list(paths)
    if len(agents) < 2:
        return None
    horizon = max(len(p) for p in paths.values())
    if window is not None:
        horizon = min(horizon, window + 1)

    for t in range(horizon):
        # vertex conflicts at time t
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                if cell_at(paths[a], t) == cell_at(paths[b], t):
                    return VertexConflict(a, b, cell_at(paths[a], t), t)
        # edge (swap) conflicts between t and t+1
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                a_t, a_t1 = cell_at(paths[a], t), cell_at(paths[a], t + 1)
                b_t, b_t1 = cell_at(paths[b], t), cell_at(paths[b], t + 1)
                if a_t != a_t1 and a_t == b_t1 and a_t1 == b_t:
                    return EdgeConflict(a, b, a_t, a_t1, t + 1)

    return None
