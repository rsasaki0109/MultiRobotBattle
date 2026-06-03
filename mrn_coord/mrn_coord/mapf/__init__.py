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
  search that resolves conflicts by branching constraints. With
  ``disjoint=True`` it uses disjoint splitting (Li et al. 2019): it branches one
  agent positive/negative on the conflict cell so the children *partition* the
  solution space the standard two-negative split overlaps — same optimum, fewer
  expansions.
- :mod:`cbsh` — CBS with improved heuristics (Li et al. 2019): the same optimal
  search, but an admissible CG/DG/WDG heuristic plus cardinal-conflict
  prioritization cut the high-level node expansions by a large factor.
- :mod:`icts` — the Increasing Cost Tree Search (Sharon et al. 2013): an optimal
  paradigm orthogonal to CBS — branch on per-agent *costs*, not constraints, and
  test each cost vector by searching the cross-product of the agents' MDDs, with
  pairwise-dependency pruning to skip hopeless nodes.
- :mod:`rectangle` — rectangle symmetry reasoning (Li et al. 2019): a barrier
  constraint that breaks the symmetric blowup of two agents crossing an open
  region, wired into ``cbsh(rectangle=True)``.
- :mod:`mutex` — mutex propagation (Zhang et al. 2020): propagates mutexes over a
  pair of MDDs to detect cardinal conflicts and synthesize symmetry-breaking
  constraints automatically, generalizing rectangle reasoning.
- :mod:`ecbs` — Enhanced CBS: bounded-suboptimal (``cost <= w * optimal``) via
  focal search at both levels; expands far fewer nodes than CBS for a little
  cost slack, so it scales to more agents.
- :mod:`eecbs` — EECBS (Li et al. 2021): bounded-suboptimal like ECBS, but it
  reuses CBSH's admissible WDG heuristic for a tight lower bound and runs
  Explicit Estimation Search at the high level, certifying the ``w`` bound with
  fewer expansions than ECBS at the same factor.
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
from .cbsh import cbsh
from .ecbs import ecbs
from .eecbs import eecbs
from .icts import icts
from .mutex import classify_conflict, generate_mutexes, pc_constraints
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
    "cbsh",
    "ecbs",
    "eecbs",
    "icts",
    "classify_conflict",
    "generate_mutexes",
    "pc_constraints",
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
