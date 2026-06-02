"""Multi-Agent Path Finding (MAPF): collision-free planning on a shared grid.

The pieces compose bottom-up:

- :mod:`grid` — the shared world (a 4-connected grid with obstacles) and the
  Manhattan heuristic.
- :mod:`space_time_astar` — the low-level single-agent planner over
  ``(cell, time)`` states, honoring vertex and edge constraints (including a
  wait action). This is what both high-level planners call.
- :mod:`sipp` — Safe Interval Path Planning: a drop-in low-level planner that
  searches ``(cell, safe interval)`` states instead, collapsing long waits into
  a single state for the same minimal-time path.
- :mod:`conflicts` — vertex and edge (swap) conflict detection between planned
  paths, with stay-at-goal semantics.
- :mod:`cbs` — Conflict-Based Search: an optimal (sum-of-costs) two-level
  search that resolves conflicts by branching constraints.
- :mod:`ecbs` — Enhanced CBS: bounded-suboptimal (``cost <= w * optimal``) via
  focal search at both levels; expands far fewer nodes than CBS for a little
  cost slack, so it scales to more agents.
- :mod:`lacam` — LaCAM: complete satisficing search over whole configurations
  using PIBT as a successor generator with lazy constraints; scales to large
  teams (not cost-optimal).
- :mod:`lns` — MAPF-LNS: anytime large-neighborhood search that destroys and
  repairs a few agents at a time, polishing any feasible solution toward the
  optimum at scale.
- :mod:`prioritized` — prioritized planning: fast and incomplete; plans agents
  in priority order, each treating higher-priority paths as moving obstacles.
- :mod:`pbs` — Priority-Based Search: searches over priority *orderings* (PP at
  the low level), resolving the head-on deadlocks fixed-order PP cannot; the
  windowed solver behind lifelong RHCR.
- :mod:`solution` — ``Solution`` plus cost/makespan/padding/rendering helpers.
"""

from .cbs import cbs
from .ecbs import ecbs
from .lacam import lacam, lacam_ltm
from .lns import mapf_lns
from .pbs import pbs, pbs_paths
from .conflicts import (
    EdgeConflict,
    VertexConflict,
    cell_at,
    detect_first_conflict,
)
from .grid import Cell, GridWorld, manhattan
from .path_follower import pure_pursuit
from .prioritized import prioritized_planning
from .sipp import plan_sipp
from .solution import Solution, makespan, pad_paths, render_ascii, sum_of_costs
from .space_time_astar import plan_path

__all__ = [
    "Cell",
    "GridWorld",
    "manhattan",
    "plan_path",
    "plan_sipp",
    "VertexConflict",
    "EdgeConflict",
    "cell_at",
    "detect_first_conflict",
    "cbs",
    "ecbs",
    "lacam",
    "lacam_ltm",
    "mapf_lns",
    "pbs",
    "pbs_paths",
    "prioritized_planning",
    "pure_pursuit",
    "Solution",
    "sum_of_costs",
    "makespan",
    "pad_paths",
    "render_ascii",
]
