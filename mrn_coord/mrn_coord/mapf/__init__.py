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
- :mod:`multi_label_astar` — Multi-Label A* (Grenouilleau et al. 2019): the
  low level for *ordered* goals (pickup then delivery), planning the whole route
  in one search over ``(cell, time, label)`` states. It passes *through* the
  pickup instead of resting there, so it finds paths the two sequential
  start->pickup, pickup->delivery searches miss or lengthen.
- :mod:`sipps` — SIPP with Soft constraints (Li et al. 2022), the low level
  behind MAPF-LNS2. Hard constraints define the safe intervals; the other agents'
  paths are *soft* — passable at one collision each (counted even while waiting) —
  and SIPPS finds the fewest-collision, then shortest, path. Same
  ``(collisions, length)`` optimum as lns2's time-expanded planner, over the far
  smaller safe-interval state space.
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
- :mod:`k_robust` — k-robust CBS (Atzmon et al. 2018): the same optimal CBS, but
  it plans for *delays*. A k-robust plan stays collision-free as long as no agent
  is delayed by more than ``k`` steps — it leaves a ``k``-step buffer at every
  shared cell (no two agents use a cell within ``k`` steps of each other). The
  high level detects a *k-delay* vertex conflict and splits it with ordinary
  negative constraints; ``k=0`` is byte-for-byte plain CBS, larger ``k`` buys
  robustness at a monotone cost.
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
- :mod:`bcbs` — Bounded CBS (Barer et al. 2014), ECBS's sibling from the same
  paper: focal at both levels too, but the high-level bound is taken against the
  best *cost* (not a lower bound), so the two factors multiply — ``w_high *
  w_low`` suboptimal, with independent knobs. ECBS's tighter ``w`` bound is what
  superseded it; kept as a gated contrast (it expands fewer nodes at higher cost).
- :mod:`eecbs` — EECBS (Li et al. 2021): bounded-suboptimal like ECBS, but it
  reuses CBSH's admissible WDG heuristic for a tight lower bound and runs
  Explicit Estimation Search at the high level, certifying the ``w`` bound with
  fewer expansions than ECBS at the same factor.
- :mod:`fecbs` — FECBS (Chan et al. 2021): ECBS with *flex distribution*. Instead
  of bounding every agent by ``w *`` its own optimum, it bounds only the *total*,
  lending each replanned agent the suboptimality budget the others left unspent so
  it can route around conflicts. Same ``w`` guarantee as ECBS; far fewer
  high-level nodes when the per-agent bound is the bottleneck (tight ``w``).
- :mod:`highway` — Highway heuristics (Cohen et al. 2015): a set of *directed
  edges* marking a preferred flow, layered on ECBS. The low-level focal search
  ranks its ``w``-bounded candidates by fewest conflicts *then* fewest
  off-highway moves, so agents flow with the highway and head-on conflicts vanish
  before the high level branches — far fewer expansions for the *same* ``w``
  guarantee (the OPEN bound is untouched). ``ecbs(grid, agents, highways=H)``;
  with no highway it is byte-for-byte plain ECBS.
- :mod:`lacam` — LaCAM: complete satisficing search over whole configurations
  using PIBT as a successor generator with lazy constraints; scales to large
  teams (not cost-optimal).
- :mod:`pibt_swap` — the **swap** operation that improves PIBT successor
  generation in LaCAM2 (Okumura 2023). Two agents that must exchange ends of a
  narrow corridor livelock plain PIBT; the swap detects a required-and-possible
  exchange and pulls the partner through a degree-≥2 pocket, vacating the corridor.
  The canonical fix for the livelocks :mod:`lacam`'s spine instead escapes with a
  deterministic salt. ``swap=False`` recovers plain PIBT.
- :mod:`mstar` — M*: subdimensional expansion (Wagner & Choset 2011/2015). An
  optimal (sum-of-costs) joint-space search that keeps the search dimension low
  almost everywhere — each agent follows its individual optimal policy until a
  collision couples it, at which point only the colliding agents branch over
  their full moves. Same optimum as CBS; on instances with few, isolated
  interactions the collision set — and the search — stays low-dimensional and
  invariant to the rest of the team.
- :mod:`rmstar` — recursive M* (Wagner & Choset 2011/2015): basic M*'s flat
  collision set unions independent collisions that share an ancestor; rM* keeps a
  *partition* and couples only agents that genuinely collide, so peak coupling is
  the largest irreducible interacting group, not the union. Coupled groups branch
  their joint optimal policy. Same optimum as CBS; on collisions that decompose,
  the peak group stays constant and expansions grow polynomially where basic M*
  grows exponentially.
- :mod:`epea` — Enhanced Partial Expansion A* (Goldenberg et al. 2014): optimal
  joint-space search that, when it expands a node, generates *only* the children
  whose ``f`` equals the node's, via an Operator Selection Function, and
  re-inserts the node at its next child ``f``. Same optimum as CBS; generates far
  fewer nodes than the fully-expanding joint A* (:func:`joint_astar`).
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
- :mod:`cbm` — Conflict-Based Min-cost-flow for TAPF (Ma & Koenig 2016): target
  assignment *and* path finding for **teams** of agents. Targets within a team are
  interchangeable, across teams distinct. The low level solves each team as an
  anonymous makespan max-flow (reusing :mod:`flow`) under the high-level
  constraints; the high level is CBS over *inter-team* conflicts. It interpolates
  the two extremes — one team is pure :mod:`flow`, singleton teams are labeled
  makespan-optimal MAPF — and is makespan-optimal throughout.
- :mod:`cbs_ta` — CBS with optimal Target Assignment (Hönig et al. 2018): when
  each agent may serve any goal from a pool, find the *jointly* optimal assignment
  *and* paths. CBS's single root becomes a **forest** of roots (one per assignment,
  unfolded lazily in increasing cost by Murty's K-best matching); searched
  best-first by sum-of-costs the first conflict-free node is jointly optimal. The
  labeled, sum-of-costs cousin of :mod:`cbm` (teams/makespan): one distinct goal
  per agent is byte-for-byte :mod:`cbs`, a shared pool is the anonymous optimum.
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
- :mod:`push_and_swap` — Push and Swap (Luna & Bekris 2011): the swap-only
  *ancestor* of :mod:`push_and_rotate`, kept as its own solver. Two primitives
  only (push / swap), no rotate and no packed-grid reduction — it reuses the
  *same* push/swap machinery but with rotate off. Complete and valid by
  construction wherever there is slack (it matches :mod:`push_and_rotate` on
  sparse maps), but it stalls on the cyclic, slack-free regions (a packed
  rectangle, a full ring) that de Wilde et al. (2014) showed the bare core
  cannot solve — the exact completeness gap the rotate primitive closes.
- :mod:`bibox` — Bibox (Surynek 2009): a constructive, polynomial-time *complete*
  solver for **biconnected** graphs with at least two blanks, built on an *open
  ear decomposition*. Derived ears (chains attached at both ends to the part built
  so far) are solved in reverse order and locked, each filled by *rotating* the
  cycle it forms with a return path; the basic cycle is closed last. Valid by
  construction and complete on its class, where optimal search (CBS) blows up.
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
- :mod:`footstep` — search-based **footstep planning** for a humanoid (Hornung
  et al., Humanoids 2012): the robot's state is its stance-foot pose plus which
  foot is the stance (the feet alternate), and a step is a displacement of the
  swing foot from a small discrete footstep set. Weighted A* (``w = 1`` optimal,
  ``w > 1`` bounded-suboptimal with far fewer expansions) plus an anytime
  decreasing-``w`` schedule, over a rectangular-footprint collision check.
- :mod:`footstep_mapf` — **multi-humanoid footstep MAPF**: lifts the footstep
  planner to a team whose *bodies* must not collide, synchronised by step index,
  resolved by prioritized planning (each humanoid avoids the higher-priority
  bodies as tick-indexed obstacles, with a stand-still *wait* primitive).
  Collision-free by construction; incomplete (fails on symmetric head-on cases,
  like any fixed priority order). A kinematic planning/coordination reproduction
  — no whole-body dynamics.
- :mod:`lipm_walk` — **biped walking pattern generation by ZMP preview control**
  (Kajita et al., ICRA 2003): turns a footstep plan into the dynamically stable
  *center-of-mass* trajectory that realises it. The biped is a Linear Inverted
  Pendulum (CoM at constant height over a "ZMP cart"); a preview controller —
  an optimal ZMP-tracking servo that looks ahead at the future reference —
  drives the CoM so the induced Zero-Moment Point stays under the support foot.
  The Riccati gains are solved in pure Python (no numpy). This is the *dynamics*
  companion to :mod:`footstep`'s kinematic planning.
- :mod:`capture_point` — **humanoid push recovery** (Pratt et al., Humanoids
  2006) on the same LIPM: the **Capture Point** ``ξ = x + ẋ/ω₀`` is where the
  foot must step to bring the CoM to rest after a push. Stepping there captures
  the fall; stepping short or long does not. A big push beyond one step's reach
  is only *N-step capturable*. Exact closed-form LIPM, pure Python.
- :mod:`dcm_walk` — **DCM walking control** (Englsberger et al., IEEE T-RO
  2015): the Capture Point made into a continuous walking controller. A backward
  recursion ``ξ_ini = p + (ξ_eos − p) e^{−ωT}`` plans a bounded DCM reference
  over a footstep plan (the CoM trails it through the full stride), and the
  tracking law ``r_cmd = r_ref + (1 + k_ξ/ω)(ξ − ξ_ref)`` drives the DCM error to
  zero at the chosen rate ``k_ξ`` — while open-loop (no feedback) blows up at
  exactly rate ``ω``. Exact closed-form LIPM, pure Python.
- :mod:`mpc_walk` — **trajectory-free MPC walking** (Wieber, Humanoids 2006):
  the constrained-QP counterpart of :mod:`lipm_walk`'s preview control. No
  tracked trajectory — the ZMP is held in the support polygon by a *hard*
  inequality while a jerk + reference-velocity objective picks the smoothest
  walk. Changing variables to the ZMP makes the support constraint a box, solved
  exactly each tick by a small active-set QP. The hard constraint keeps the ZMP
  legal under a strong push, where the unconstrained LQR-like cousin carries it
  out of the foot. Pure Python, no numpy.
- :mod:`herdt_walk` — **MPC walking with automatic footstep placement** (Herdt
  et al., Advanced Robotics 2010): the direct extension of :mod:`mpc_walk` that
  removes its honest limit. The footstep positions become decision variables of
  the same QP, so the controller automatically chooses where to step to follow a
  reference velocity. Changing variables twice — to the ZMP *and* the foot
  increments — keeps it a box QP solved by the same active-set method. Under a
  strong push that makes the fixed-foot MPC fall, this one takes a *capture step*
  and recovers; with the feet frozen it collapses bit-for-bit to :mod:`mpc_walk`.
  Pure Python, no numpy.
- :mod:`kajita_stabilizer` — **biped walking stabilization by LIPM tracking**
  (Kajita et al., IROS 2010): the closed-loop, on-the-real-robot counterpart of
  the open-loop pattern generators above. The LIPM is inherently unstable, so
  playing a precomputed ZMP back open-loop diverges under any perturbation; the
  stabilizer measures the actual CoM and commands a modified ZMP
  ``p = p^ref + k_p (x−x^ref) + k_v (ẋ−ẋ^ref)`` (``k_p>1`` overcomes the
  instability), saturated to the support foot (ankle strategy). A push within the
  capturable margin is rejected with the ZMP inside the foot; a larger push
  saturates and the robot must *step* — the escalation to :mod:`capture_point` /
  :mod:`herdt_walk`. Pure Python, no numpy.
- :mod:`push_recovery` — **ankle / hip / step decision surfaces** (Stephens,
  Humanoids 2007): the unifying analysis of the push-recovery family. All three
  are read off the capture point ``ξ = x + ẋ/ω`` — ankle recovers iff ξ is in the
  foot, the *hip* strategy (a flywheel at the CoM) widens that interval by
  ``Δ_hip = (τ_max/mg)(1 − e^{−ωT_max})²`` via a bang-bang momentum pulse, and
  past that a *step* is required (deferred to :mod:`capture_point`). The closed
  form matches an exact bang-bang LIPPF simulation to machine precision (the
  paper's printed eq. (15) is a typo). Pure Python, no numpy.
- :mod:`capturability` — **N-step capturability analysis** (Koolen et al., IJRR
  2012): the analytic backbone tying the push-recovery family together. A push is
  *N-step capturable* if the robot can stop in ``N`` steps; the N-step capture
  region is the geometric series ``ξ_N = foot + l_max·Σ_{k=1}^N e^{−kωT}``, nested
  and bounded by a finite **capturability limit** ``ξ_∞ = foot + l_max/(e^{ωT}−1)``
  past which no number of steps recovers. Koolen's three models — point foot /
  finite foot / reaction mass — are exactly :mod:`capture_point` and the
  :mod:`push_recovery` ankle / hip strategies. The closed form is certified
  against an exact greedy LIPM rollout. Pure Python, no numpy.
- :mod:`resolved_momentum` — **Resolved Momentum Control** (Kajita et al., IROS
  2003): the first *whole-body* method here — it commands the total linear and
  angular momentum of an articulated, free-floating planar humanoid. The
  **centroidal momentum matrix** ``A(q)`` makes the momentum linear in the
  generalized velocity, ``h = A(q)·q̇``; a momentum reference plus task constraints
  (pinned support foot, tracked swing foot) are resolved by the inertia-matrix
  pseudo-inverse ``q̇ = Bᵀ(BBᵀ)⁻¹ b``. Regulating the angular momentum to zero makes
  the body counter-rotate internally (the whole-body root of the reaction-mass /
  hip strategy of :mod:`push_recovery` / :mod:`capturability`), and a kick drives a
  swing foot along an arc with the support foot pinned. The momentum matrix is
  certified against a finite-difference. Pure Python, no numpy.
- :mod:`drrt` — **discrete RRT** (Solovey, Salzman & Halperin, WAFR 2014 / IJRR
  2016): multi-robot motion planning in **continuous space**. Each robot is a
  disc with its own PRM roadmap; the team's joint space is the **tensor product**
  of those roadmaps (``∏ |V_i|`` composite vertices — exponential, the
  *haystack*), explored *implicitly* by an RRT whose expansion is the
  **direction oracle** ``O_d`` (per robot, the one roadmap edge best aligned with
  the heading to a random sample). Collision checking is exact and continuous
  (quadratic disc/disc closest-approach + swept-disc/obstacle). Feasibility /
  probabilistically-complete, not cost-optimal. The same module carries
  **dRRT\\*** (Shome, Solovey, Dobson, Halperin & Bekris, AuRo 2020): the
  asymptotically-*optimal* successor that keeps the explored implicit roadmap as
  a **graph** and returns its Dijkstra shortest path (anytime, informed
  sampling), certified to converge to ``composite_optimum`` — the brute optimum
  over the full implicit roadmap. Pure Python, no numpy.
- :mod:`kcbs` — **Kinodynamic CBS** (Kottinger, Almagor & Lahijanian, IROS 2022):
  the first planner here that respects robot **dynamics**. Each robot is a
  **Dubins car** (constant speed, bounded turn rate — it cannot turn in place or
  move sideways, only follow curves of radius ``>= V/ω_max``). The low level is a
  kinodynamic RRT that forward-propagates the dynamics (exact arc integration) in
  state×time, avoiding obstacles and space–time constraint tubes; the high level
  is CBS, branching on a continuous-time collision by forbidding a robot from the
  conflict location during a short window. Returns dynamically-feasible,
  collision-free trajectories (feasibility, not cost-optimal). Pure Python.
- :mod:`solution` — ``Solution`` plus cost/makespan/padding/rendering helpers.
"""

try:
    from .bcp import bcp
except ModuleNotFoundError as _bcp_exc:  # numpy/scipy are optional extras
    _bcp_missing = _bcp_exc

    def bcp(*_args, **_kwargs):  # type: ignore[misc]
        """Placeholder when the optional LP backend (numpy/scipy) is absent.

        Only :mod:`bcp` (branch-and-cut-and-price) needs numpy/scipy; the rest
        of the zoo is pure standard-library Python. Install the backend with
        ``pip install mapf-zoo[bcp]`` to enable it.
        """
        raise ImportError(
            "mapf.bcp requires numpy and scipy — install them with "
            "`pip install mapf-zoo[bcp]`"
        ) from _bcp_missing

from .bypass import cbs_bypass
from .cbm import cbm
from .cbs import cbs
from .cbs_ta import cbs_ta
from .bcbs import bcbs
from .pibt_swap import pibt_swap
from .cbsh import cbsh
from .ddm import ddm
from .epea import epea_star
from .k_robust import k_robust_cbs
from .macbs import macbs
from .ccbs import ccbs
from .ecbs import ecbs
from .eecbs import eecbs
from .fecbs import fecbs
from .highway import ecbs_highway, keep_side_highway, ring_highway
from .flow import anonymous_makespan
from .tswap import tswap
from .icts import icts
from .mutex import classify_conflict, generate_mutexes, pc_constraints
from .lacam import lacam, lacam_ltm
from .lns import mapf_lns
from .lns2 import mapf_lns2
from .mstar import joint_astar, mstar
from .rmstar import rmstar
from .satmdd import satmdd
from .standley import independence_detection, od_astar
from .pbs import pbs, pbs_paths
from .footstep import (
    FootstepPlan,
    FootstepState,
    FootstepWorld,
    ara_star,
    plan_footsteps,
)
from .footstep_mapf import (
    bodies_collision_free,
    plan_footsteps_reserved,
    prioritized_footstep_mapf,
)
from .lipm_walk import (
    PreviewGains,
    WalkPattern,
    generate_walk,
    lipm_track,
    preview_gains,
    zmp_stability,
)
from .capture_point import (
    capture_point,
    n_step_capture,
    omega0,
    recover_step,
    simulate_lipm,
)
from .dcm_walk import (
    DCMPlan,
    plan_dcm_reference,
    track_dcm,
    vrp_command,
)
from .mpc_walk import (
    CondensedMPC,
    MPCParams,
    MPCWalkResult,
    build_condensed,
    simulate_mpc,
    solve_box_qp,
)
from .herdt_walk import (
    HerdtMPC,
    HerdtParams,
    HerdtWalkResult,
    build_herdt,
    simulate_herdt,
)
from .kajita_stabilizer import (
    StabilizerParams,
    StabilizerResult,
    gains_for_poles,
    reference_trajectory,
    simulate_stabilizer,
    stabilizer_params,
)
from .push_recovery import (
    RecoveryResult,
    StrategyParams,
    classify,
    hip_recovery_boundary,
    simulate_ankle,
    simulate_hip,
)
from .capturability import (
    CaptureParams,
    GreedyRecovery,
    capture_region,
    capturability_margin,
    inf_step_region,
    n_step_region,
    simulate_greedy,
)
from .resolved_momentum import (
    Link,
    MomentumTask,
    MomentumTrajectory,
    PlanarRobot,
    make_humanoid,
    resolve_momentum,
    simulate as simulate_momentum,
    task_nullspace,
)
from .drrt import (
    Obstacle,
    Roadmap,
    StarSolution,
    build_roadmap,
    composite_optimum,
    direction_oracle,
    drrt,
    drrt_star,
    solution_clearance,
    tensor_product_size,
)
from .kcbs import (
    Constraint,
    DubinsCar,
    KCBSSolution,
    first_conflict,
    kcbs,
    plan_trajectory,
    propagate,
    trajectory_feasible,
)
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
from .bibox import bibox, ear_decomposition
from .push_and_rotate import push_and_rotate
from .push_and_swap import push_and_swap
from .sipp import plan_sipp
from .sipps import plan_sipps
from .solution import Solution, makespan, pad_paths, render_ascii, sum_of_costs
from .multi_label_astar import mla_star, two_step_plan
from .space_time_astar import plan_path

__all__ = [
    "Cell",
    "GridWorld",
    "manhattan",
    "plan_path",
    "mla_star",
    "two_step_plan",
    "plan_sipp",
    "plan_sipps",
    "VertexConflict",
    "EdgeConflict",
    "cell_at",
    "detect_first_conflict",
    "bcp",
    "cbs_bypass",
    "cbm",
    "cbs",
    "cbs_ta",
    "cbsh",
    "ddm",
    "epea_star",
    "k_robust_cbs",
    "macbs",
    "ccbs",
    "ecbs",
    "fecbs",
    "eecbs",
    "ecbs_highway",
    "keep_side_highway",
    "ring_highway",
    "bcbs",
    "anonymous_makespan",
    "tswap",
    "icts",
    "classify_conflict",
    "generate_mutexes",
    "pc_constraints",
    "lacam",
    "lacam_ltm",
    "pibt_swap",
    "mstar",
    "joint_astar",
    "rmstar",
    "od_astar",
    "independence_detection",
    "satmdd",
    "mapf_lns",
    "mapf_lns2",
    "pbs",
    "pbs_paths",
    "prioritized_planning",
    "FootstepWorld",
    "FootstepState",
    "FootstepPlan",
    "plan_footsteps",
    "ara_star",
    "plan_footsteps_reserved",
    "prioritized_footstep_mapf",
    "bodies_collision_free",
    "preview_gains",
    "generate_walk",
    "lipm_track",
    "zmp_stability",
    "WalkPattern",
    "PreviewGains",
    "capture_point",
    "simulate_lipm",
    "recover_step",
    "n_step_capture",
    "omega0",
    "plan_dcm_reference",
    "track_dcm",
    "vrp_command",
    "DCMPlan",
    "build_condensed",
    "simulate_mpc",
    "solve_box_qp",
    "MPCParams",
    "CondensedMPC",
    "MPCWalkResult",
    "build_herdt",
    "simulate_herdt",
    "HerdtMPC",
    "HerdtParams",
    "HerdtWalkResult",
    "stabilizer_params",
    "simulate_stabilizer",
    "reference_trajectory",
    "gains_for_poles",
    "StabilizerParams",
    "StabilizerResult",
    "classify",
    "simulate_ankle",
    "simulate_hip",
    "hip_recovery_boundary",
    "StrategyParams",
    "RecoveryResult",
    "n_step_region",
    "inf_step_region",
    "capture_region",
    "capturability_margin",
    "simulate_greedy",
    "CaptureParams",
    "GreedyRecovery",
    "make_humanoid",
    "resolve_momentum",
    "task_nullspace",
    "simulate_momentum",
    "PlanarRobot",
    "Link",
    "MomentumTask",
    "MomentumTrajectory",
    "drrt",
    "drrt_star",
    "composite_optimum",
    "build_roadmap",
    "direction_oracle",
    "solution_clearance",
    "tensor_product_size",
    "Roadmap",
    "Obstacle",
    "StarSolution",
    "kcbs",
    "plan_trajectory",
    "propagate",
    "trajectory_feasible",
    "first_conflict",
    "DubinsCar",
    "Constraint",
    "KCBSSolution",
    "whca_star",
    "push_and_rotate",
    "push_and_swap",
    "bibox",
    "ear_decomposition",
    "pure_pursuit",
    "Solution",
    "sum_of_costs",
    "makespan",
    "pad_paths",
    "render_ascii",
]
