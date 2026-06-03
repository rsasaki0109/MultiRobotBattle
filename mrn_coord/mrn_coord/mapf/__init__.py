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
- :mod:`bypass` — CBS with bypassing conflicts (Boyarski et al. 2015): before
  splitting a conflict, check if either child is a *valid bypass* — same cost and
  strictly fewer conflicts — and if so adopt its path into the node instead of
  growing the tree. Same optimum as CBS; collapses the tree on the non-cardinal
  conflicts plain CBS wastefully splits (a cardinal conflict can never bypass).
- :mod:`macbs` — Meta-Agent CBS (Sharon et al. 2012/2015): the same optimal CBS,
  but with a conflict bound ``B`` — two "agents" that conflict more than ``B``
  times are *merged* into a meta-agent solved by a coupled (joint) low level.
  ``B=∞`` is standard CBS; ``B=0`` collapses toward a single joint search;
  every ``B`` returns the same optimum, absorbing a tree-exploding bottleneck
  into one coupled solve.
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
- :mod:`ccbs` — Continuous-time CBS (Andreychuk et al. 2019): drops the discrete
  clock entirely — agents are disks on an 8-connected geometric roadmap, moves
  take real (irrational) durations, conflicts are "centres within ``2r`` at any
  real instant", and yields cost the minimal real wait. Catches mid-edge
  geometric collisions the vertex/edge model is blind to.
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
- :mod:`mstar` — M*: subdimensional expansion (Wagner & Choset 2011/2015). An
  optimal (sum-of-costs) joint-space search that keeps the search dimension low
  almost everywhere — each agent follows its individual optimal policy until a
  collision couples it, at which point only the colliding agents branch over
  their full moves. Same optimum as CBS; on instances with few, isolated
  interactions the collision set — and the search — stays low-dimensional and
  invariant to the rest of the team.
- :mod:`standley` — Standley's optimal MAPF (AAAI 2010): two ways to beat the
  ``b**n`` joint branching. ``od_astar`` is *operator decomposition* — assign a
  move to one agent at a time so the effective branching is ``b`` not ``b**n``;
  ``independence_detection`` plans agents (groups) separately and merges only the
  groups that actually collide. Both reach CBS's optimum; OD generates a small
  fraction of a joint A*'s successors, ID never searches independent agents
  together.
- :mod:`satmdd` — MDD-SAT (Surynek et al. 2016): the declarative paradigm. Encode
  "is there a collision-free plan of makespan ``mu``?" as CNF over per-agent MDD
  cells and hand it to a SAT solver; sweep ``mu`` up from the trivial lower bound
  so the first satisfiable makespan is optimal, self-certified by the UNSAT of
  every smaller one. Optimizes *labeled* makespan (≥ :mod:`flow`'s anonymous).
- :mod:`bcp` — branch-and-cut-and-price (Lam et al. 2019): the LP/duality
  paradigm. The path-based (set-partitioning) linear program is solved by
  *column generation* (a reduced-cost shortest path *prices* in only useful
  paths) with *lazy* vertex/edge conflict *cuts*, and branch-and-price closes
  the integrality gap. Same optimum as CBS, certified by the LP lower bound
  (gap zero) — the first solver here that optimizes rather than searches.
- :mod:`flow` — anonymous makespan-optimal MAPF (Yu & LaValle 2013): when targets
  are interchangeable, minimum-makespan routing reduces to integer MAX FLOW on a
  time-expanded network — polynomial, no search tree, with a self-certified
  optimum. A relaxation of labeled MAPF (its makespan lower-bounds any labeled
  solution's).
- :mod:`tswap` — Offline TSWAP (Okumura & Defago 2022): the fast, constructive
  counterpart for the same anonymous problem. From an *arbitrary* initial
  assignment it repeats one-timestep planning with **target swapping** until all
  agents sit on targets — collision-free by construction, complete by a potential
  argument, sub-optimal but near-optimal at a fraction of flow's cost.
- :mod:`lns` — MAPF-LNS: anytime large-neighborhood search that destroys and
  repairs a few agents at a time, polishing any feasible solution toward the
  optimum at scale.
- :mod:`lns2` — MAPF-LNS2 (Li et al. 2022): the feasibility counterpart, which
  minimizes the *number of collisions* with a collision-minimizing low level —
  repairing a colliding shortest-path start to a feasible solution where CBS and
  prioritized planning bust their budget.
- :mod:`push_and_rotate` — Push and Swap / Push and Rotate (Luna & Bekris 2011;
  de Wilde et al. 2014): a constructive, primitive-based solver (push / swap /
  rotate) rather than a search — collision-free and on-goal by construction,
  complete when the map has slack and, via a constructive row/column reduction
  (plus a tracked-agent BFS endgame for the single-blank case), on fully packed
  grids (the 15-puzzle regime, 1–3 empty cells) where the greedy primitives
  stall — solving crowded maps where optimal search blows up, at the cost of
  optimality.
- :mod:`prioritized` — prioritized planning: fast and incomplete; plans agents
  in priority order, each treating higher-priority paths as moving obstacles.
- :mod:`ddm` — database-driven multi-robot planning (Han & Yu 2020): a decoupled
  planner whose two heuristics are an *optimal sub-problem solution database* —
  conflicts resolved in tiny 2×3/3×3 windows by a precomputed, reused optimal
  joint motion — and *path diversification* (pick the shortest path overlapping
  others least). Collision-free by construction; incomplete.
- :mod:`whca` — Windowed Hierarchical Cooperative A* (Silver 2005): cooperative
  planning made scalable. The *hierarchical* heuristic is the true shortest-path
  distance to the goal on the static map (Reverse Resumable A*), perfect enough
  to prune the dead ends Manhattan walks into; the *window* limits cooperation to
  a ``w``-step lookahead that rolls forward each round with a rotating priority
  order — bounding the search depth and breaking the transient deadlocks a single
  fixed priority order livelocks on. Collision-free by construction, incomplete.
- :mod:`pbs` — Priority-Based Search: searches over priority *orderings* (PP at
  the low level), resolving the head-on deadlocks fixed-order PP cannot; the
  windowed solver behind lifelong RHCR.
- :mod:`solution` — ``Solution`` plus cost/makespan/padding/rendering helpers.
"""

from .bcp import bcp
from .bypass import cbs_bypass
from .cbs import cbs
from .cbsh import cbsh
from .ddm import ddm
from .macbs import macbs
from .ccbs import ccbs
from .ecbs import ecbs
from .eecbs import eecbs
from .flow import anonymous_makespan
from .tswap import tswap
from .icts import icts
from .mutex import classify_conflict, generate_mutexes, pc_constraints
from .lacam import lacam, lacam_ltm
from .lns import mapf_lns
from .lns2 import mapf_lns2
from .mstar import joint_astar, mstar
from .satmdd import satmdd
from .standley import independence_detection, od_astar
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
from .whca import whca_star
from .push_and_rotate import push_and_rotate
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
    "bcp",
    "cbs_bypass",
    "cbs",
    "cbsh",
    "ddm",
    "macbs",
    "ccbs",
    "ecbs",
    "eecbs",
    "anonymous_makespan",
    "tswap",
    "icts",
    "classify_conflict",
    "generate_mutexes",
    "pc_constraints",
    "lacam",
    "lacam_ltm",
    "mstar",
    "joint_astar",
    "od_astar",
    "independence_detection",
    "satmdd",
    "mapf_lns",
    "mapf_lns2",
    "pbs",
    "pbs_paths",
    "prioritized_planning",
    "whca_star",
    "push_and_rotate",
    "pure_pursuit",
    "Solution",
    "sum_of_costs",
    "makespan",
    "pad_paths",
    "render_ascii",
]
