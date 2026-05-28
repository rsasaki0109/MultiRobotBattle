"""Multi-Agent Path Finding (MAPF): collision-free planning on a shared grid.

The pieces compose bottom-up:

- :mod:`grid` — the shared world (a 4-connected grid with obstacles) and the
  Manhattan heuristic.
- :mod:`space_time_astar` — the low-level single-agent planner over
  ``(cell, time)`` states, honoring vertex and edge constraints (including a
  wait action). This is what both high-level planners call.
- :mod:`conflicts` — vertex and edge (swap) conflict detection between planned
  paths, with stay-at-goal semantics.
- :mod:`cbs` — Conflict-Based Search: an optimal (sum-of-costs) two-level
  search that resolves conflicts by branching constraints.
- :mod:`prioritized` — prioritized planning: fast and incomplete; plans agents
  in priority order, each treating higher-priority paths as moving obstacles.
- :mod:`solution` — ``Solution`` plus cost/makespan/padding/rendering helpers.
"""

from .cbs import cbs
from .conflicts import (
    EdgeConflict,
    VertexConflict,
    cell_at,
    detect_first_conflict,
)
from .grid import Cell, GridWorld, manhattan
from .prioritized import prioritized_planning
from .solution import Solution, makespan, pad_paths, render_ascii, sum_of_costs
from .space_time_astar import plan_path

__all__ = [
    "Cell",
    "GridWorld",
    "manhattan",
    "plan_path",
    "VertexConflict",
    "EdgeConflict",
    "cell_at",
    "detect_first_conflict",
    "cbs",
    "prioritized_planning",
    "Solution",
    "sum_of_costs",
    "makespan",
    "pad_paths",
    "render_ascii",
]
