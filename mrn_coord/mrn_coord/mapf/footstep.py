"""Search-based footstep planning for a humanoid.

A faithful reproduction of the planning framework of Hornung, Dornbush,
Likhachev & Bennewitz, *"Anytime Search-Based Footstep Planning with
Suboptimality Bounds"* (IEEE-RAS Humanoids 2012), which itself builds on
Garimort/Hornung/Bennewitz (ICRA 2011) and the ROS ``footstep_planner``.

The robot's state is its **stance foot** pose ``s = (x, y, theta)`` together
with *which* foot is the stance (left / right); the feet alternate. A footstep
action is the displacement of the *swing* foot in the stance-foot frame,
``a = (dx, dy, dtheta)``, drawn from a small discrete **footstep set** (Fig. 2;
``dx in [-10,22]cm``, ``dy in [12,28]cm``, ``dtheta in [-0.23,40]deg`` for a
large humanoid). Because the legs alternate, the set is mirrored for the other
foot. Expanding a state applies every action; the resulting end-foot pose is
kept only if its rectangular footprint is collision-free (humanoids may step
*over* shallow obstacles, so only the end location is checked).

Transition cost is Eq. (1):

    c(s, s') = || (x, y), (x', y') || + k

— the Euclidean distance the stance position travels, plus a constant ``k`` per
step that penalises longer (more-step) paths. The search is guided by an
**admissible** straight-line Euclidean heuristic to the goal (the paper's A*
``w = 1`` case; their 2D-Dijkstra heuristic is *inadmissible* precisely because
it ignores stepping over obstacles, so we use the Euclidean one).

Three searches are provided over this lattice graph:

- :func:`plan_footsteps` — weighted A* (``wA*``). ``w = 1`` is optimal A*;
  ``w > 1`` inflates the heuristic for a ``w``-suboptimal path found by
  expanding far fewer states (Sec. IV).
- :func:`ara_star` — Anytime Repairing A* (Likhachev et al. 2004): a series of
  ``wA*`` searches with *decreasing* ``w`` that reuse the previous search's
  work (the ``INCONS`` list), publishing a provably ``w``-suboptimal solution
  after each pass and converging to the optimum (Sec. V).

Everything here is pure, deterministic, ROS-free Python. The world is metric
(continuous poses on a lattice); :meth:`FootstepWorld.from_grid` builds one from
the repo's :class:`~mrn_coord.mapf.grid.GridWorld` so the footstep planner and
the grid MAPF zoo share obstacle descriptions.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

# --- foot / footstep parameterization (Fig. 2, a large humanoid) ------------

FOOT_LENGTH = 0.16  # m, along the heading (forward)
FOOT_WIDTH = 0.10   # m, lateral

# The 14-step footstep set: displacement (dx, dy, dtheta_deg) of the SWING foot
# in the STANCE foot frame, for the canonical "swing the LEFT foot relative to
# the RIGHT stance foot" case (dy > 0 is lateral, to the left; dtheta > 0 turns
# left). It is mirrored (dy, dtheta negated) when the right foot swings. The
# specific 14 are a representative instantiation of the Fig. 2 ranges
# (dx in [-0.10, 0.22] m, dy in [0.12, 0.28] m, dtheta in [-0.23, 40] deg).
DEFAULT_FOOTSTEP_SET = (
    (0.00, 0.20, 0.0),    # mark time in place
    (0.08, 0.20, 0.0),    # nominal forward
    (0.16, 0.20, 0.0),    # long forward
    (0.22, 0.20, 0.0),    # max forward
    (-0.08, 0.20, 0.0),   # step back
    (0.08, 0.26, 0.0),    # wide stance forward
    (0.08, 0.14, 0.0),    # narrow stance forward
    (0.00, 0.20, 20.0),   # turn in place
    (0.00, 0.20, 40.0),   # sharp turn in place
    (0.08, 0.20, 20.0),   # forward + turn
    (0.08, 0.20, 40.0),   # forward + sharp turn
    (0.16, 0.20, 20.0),   # long forward + turn
    (0.12, 0.22, 10.0),   # wide forward + slight turn
    (0.00, 0.22, -0.23),  # near-straight wide (lower dtheta bound)
)

LEFT, RIGHT = "L", "R"


def _other(foot):
    return RIGHT if foot == LEFT else LEFT


# --- geometry ---------------------------------------------------------------


def _foot_corners(x, y, theta, length=FOOT_LENGTH, width=FOOT_WIDTH):
    """The four corners of the oriented foot rectangle centred at (x, y)."""
    c, s = math.cos(theta), math.sin(theta)
    hl, hw = length / 2.0, width / 2.0
    out = []
    for sx, sy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
        out.append((x + sx * c - sy * s, y + sx * s + sy * c))
    return out


def _foot_sample_offsets(length, width, res):
    """A grid of (sx, sy) offsets covering the foot rectangle in its own frame,
    spaced at ``res`` and always including the four corners."""
    hl, hw = length / 2.0, width / 2.0
    nx = max(1, int(math.ceil(length / res)))
    ny = max(1, int(math.ceil(width / res)))
    xs = [-hl + (2 * hl) * i / nx for i in range(nx + 1)]
    ys = [-hw + (2 * hw) * j / ny for j in range(ny + 1)]
    return [(sx, sy) for sx in xs for sy in ys]


@dataclass(frozen=True)
class FootstepWorld:
    """A metric world: bounds ``[0,width] x [0,height]`` (m) and rectangular,
    axis-aligned obstacles (``(xmin, ymin, xmax, ymax)``) the feet must avoid.

    Obstacles are rasterised once into an occupancy set at ``collision_res`` so
    foot collision checks are O(footprint) cell lookups (the paper's distance-map
    idea), not a polygon test against every obstacle — this is what keeps the
    multi-humanoid search tractable in pure Python.
    """

    width: float
    height: float
    obstacles: tuple = ()
    collision_res: float = 0.05

    def __post_init__(self):
        occ = set()
        res = self.collision_res
        for (xmin, ymin, xmax, ymax) in self.obstacles:
            i0, i1 = int(math.floor(xmin / res)), int(math.ceil(xmax / res))
            j0, j1 = int(math.floor(ymin / res)), int(math.ceil(ymax / res))
            for i in range(i0, i1):
                for j in range(j0, j1):
                    occ.add((i, j))
        object.__setattr__(self, "_occ", frozenset(occ))
        object.__setattr__(
            self, "_offsets",
            tuple(_foot_sample_offsets(FOOT_LENGTH, FOOT_WIDTH, res)),
        )

    @classmethod
    def from_grid(cls, grid, cell_size=0.25, clearance=0.0, collision_res=0.05):
        """Build a world from a :class:`GridWorld`; each blocked cell becomes a
        ``cell_size`` square obstacle (optionally grown by ``clearance``)."""
        obs = []
        for (cx, cy) in grid.blocked:
            obs.append(
                (
                    cx * cell_size - clearance,
                    cy * cell_size - clearance,
                    (cx + 1) * cell_size + clearance,
                    (cy + 1) * cell_size + clearance,
                )
            )
        return cls(grid.width * cell_size, grid.height * cell_size,
                   tuple(obs), collision_res)

    def foot_collision_free(self, x, y, theta) -> bool:
        """True iff the foot rectangle at ``(x, y, theta)`` is in bounds and
        clear of every obstacle (sampled at ``collision_res``)."""
        c, s = math.cos(theta), math.sin(theta)
        res = self.collision_res
        occ = self._occ
        for sx, sy in self._offsets:
            px = x + sx * c - sy * s
            py = y + sx * s + sy * c
            if not (0.0 <= px <= self.width and 0.0 <= py <= self.height):
                return False
            if (int(math.floor(px / res)), int(math.floor(py / res))) in occ:
                return False
        return True


# --- lattice state ----------------------------------------------------------


@dataclass(frozen=True)
class FootstepState:
    x: float
    y: float
    theta: float  # radians, normalised to (-pi, pi]
    foot: str     # which foot is the STANCE foot (LEFT / RIGHT)


def _norm_angle(a):
    a = math.fmod(a, 2 * math.pi)
    if a > math.pi:
        a -= 2 * math.pi
    elif a <= -math.pi:
        a += 2 * math.pi
    return a


def _apply(state: FootstepState, step, *, xy_res, theta_res):
    """Apply a canonical footstep ``step`` to ``state`` (mirrored for the other
    foot), returning the new (stance = just-placed swing) state, snapped to the
    lattice. ``state.foot`` is the *stance* foot, so the swing is the other one."""
    dx, dy, dtheta_deg = step
    swing = _other(state.foot)
    # canonical set is for swinging the LEFT foot; mirror for the right
    sign = 1.0 if swing == LEFT else -1.0
    dtheta = math.radians(dtheta_deg) * sign
    lat = dy * sign
    c, s = math.cos(state.theta), math.sin(state.theta)
    nx = state.x + dx * c - lat * s
    ny = state.y + dx * s + lat * c
    ntheta = _norm_angle(state.theta + dtheta)
    # snap to lattice for state equality
    nx = round(nx / xy_res) * xy_res
    ny = round(ny / xy_res) * xy_res
    ntheta = _norm_angle(round(ntheta / theta_res) * theta_res)
    return FootstepState(nx, ny, ntheta, swing)


def _key(state: FootstepState, xy_res, theta_res):
    return (
        round(state.x / xy_res),
        round(state.y / xy_res),
        round(state.theta / theta_res),
        state.foot,
    )


# --- planner ----------------------------------------------------------------


@dataclass
class FootstepPlan:
    """A footstep plan: the ordered stance-foot states and its cost."""

    states: list           # list[FootstepState], start..goal
    cost: float
    suboptimality: float = 1.0  # w bound this plan was found under

    def __len__(self):
        return len(self.states)


class _Planner:
    def __init__(self, world, start, goal, *, step_cost, goal_xy_tol,
                 goal_theta_tol, footstep_set, xy_res, theta_res, max_expansions,
                 heuristic="steps"):
        self.world = world
        self.start = start
        self.goal = goal  # (gx, gy) or (gx, gy, gtheta)
        self.step_cost = step_cost
        self.goal_xy_tol = goal_xy_tol
        self.goal_theta_tol = goal_theta_tol
        self.footstep_set = footstep_set
        self.xy_res = xy_res
        self.theta_res = theta_res
        self.max_expansions = max_expansions
        self.heuristic = heuristic
        # the largest stance-position displacement a single step can achieve,
        # used to lower-bound the remaining step count
        self.max_reach = max(math.hypot(dx, dy) for dx, dy, _ in footstep_set)

    def h(self, st: FootstepState):
        d = math.hypot(self.goal[0] - st.x, self.goal[1] - st.y)
        if self.heuristic == "euclid":
            # the paper's bare straight-line distance (admissible but weak: it
            # ignores the per-step cost, so optimal A* expands large areas)
            return d
        # a stronger but still admissible bound: the straight-line distance plus
        # the per-step constant times the minimum remaining number of steps
        # (each step advances the stance at most ``max_reach`` and costs >= k).
        min_steps = math.ceil(d / self.max_reach - 1e-9) if d > 1e-9 else 0
        return d + self.step_cost * min_steps

    def is_goal(self, st: FootstepState):
        if math.hypot(self.goal[0] - st.x, self.goal[1] - st.y) > self.goal_xy_tol:
            return False
        if len(self.goal) >= 3 and self.goal_theta_tol is not None:
            if abs(_norm_angle(st.theta - self.goal[2])) > self.goal_theta_tol:
                return False
        return True

    def successors(self, st: FootstepState):
        out = []
        for step in self.footstep_set:
            nxt = _apply(st, step, xy_res=self.xy_res, theta_res=self.theta_res)
            if not self.world.foot_collision_free(nxt.x, nxt.y, nxt.theta):
                continue
            c = math.hypot(nxt.x - st.x, nxt.y - st.y) + self.step_cost
            out.append((nxt, c))
        return out


def _weighted_astar(P: _Planner, w, stats):
    start = P.start
    skey = _key(start, P.xy_res, P.theta_res)
    g = {skey: 0.0}
    state_of = {skey: start}
    parent = {skey: None}
    open_heap = [(w * P.h(start), 0.0, skey)]
    closed = set()
    expansions = 0
    while open_heap:
        f, gs, k = heapq.heappop(open_heap)
        if k in closed:
            continue
        if gs > g[k]:
            continue
        st = state_of[k]
        if P.is_goal(st):
            # reconstruct
            path = []
            kk = k
            while kk is not None:
                path.append(state_of[kk])
                kk = parent[kk]
            path.reverse()
            stats["expansions"] = expansions
            return FootstepPlan(path, g[k], suboptimality=w)
        closed.add(k)
        expansions += 1
        if expansions > P.max_expansions:
            stats["expansions"] = expansions
            return None
        for nxt, c in P.successors(st):
            nk = _key(nxt, P.xy_res, P.theta_res)
            ng = g[k] + c
            if nk not in g or ng < g[nk] - 1e-9:
                g[nk] = ng
                state_of[nk] = nxt
                parent[nk] = k
                heapq.heappush(open_heap, (ng + w * P.h(nxt), ng, nk))
    stats["expansions"] = expansions
    return None


def plan_footsteps(world, start, goal, *, w=1.0, step_cost=0.30,
                   goal_xy_tol=0.18, goal_theta_tol=None,
                   footstep_set=DEFAULT_FOOTSTEP_SET, xy_res=0.02,
                   theta_res=math.radians(10), max_expansions=200_000,
                   heuristic="steps", return_stats=False):
    """Plan a footstep path with weighted A* (``w = 1`` is optimal A*).

    ``start`` is a :class:`FootstepState`; ``goal`` is ``(x, y)`` or
    ``(x, y, theta)``. ``heuristic`` is ``"steps"`` (the default, admissible
    Euclidean-plus-min-step-count bound) or ``"euclid"`` (the paper's weaker
    bare Euclidean distance). Returns a :class:`FootstepPlan` (or ``None`` if no
    plan within ``max_expansions``); with ``return_stats`` also returns a stats
    dict (``{"expansions": ...}``).
    """
    P = _Planner(
        world, start, goal, step_cost=step_cost, goal_xy_tol=goal_xy_tol,
        goal_theta_tol=goal_theta_tol, footstep_set=footstep_set,
        xy_res=xy_res, theta_res=theta_res, max_expansions=max_expansions,
        heuristic=heuristic,
    )
    stats = {"expansions": 0}
    plan = _weighted_astar(P, w, stats)
    if return_stats:
        return plan, stats
    return plan


def ara_star(world, start, goal, *, weights=(4.0, 2.0, 1.5, 1.0),
             step_cost=0.30, goal_xy_tol=0.18, goal_theta_tol=None,
             footstep_set=DEFAULT_FOOTSTEP_SET, xy_res=0.02,
             theta_res=math.radians(10), max_expansions=200_000,
             heuristic="steps", return_stats=False):
    """Anytime footstep planning over a *decreasing* ``weights`` schedule.

    Runs weighted A* for each ``w`` (largest first) and returns the list of
    :class:`FootstepPlan`\\ s — one per weight — each provably within its ``w``
    of optimal, the last (``w = 1``) optimal. This reproduces the anytime
    behaviour the ARA*/R* paper relies on (Sec. V / Table I): a fast, cheap
    first solution at large ``w`` (few expansions), refined to the optimum as
    ``w`` falls, with the expansion count climbing only as the bound tightens.

    *Scope note.* This is the **anytime schedule**, not the full ARA* of
    Likhachev et al. (2004): each weight re-plans from scratch rather than
    reusing the previous pass's ``INCONS`` list. The headline results being
    reproduced — bounded suboptimality and the cost/expansion-vs-``w`` trade-off
    — hold exactly; only the incremental-reuse optimisation is omitted (a
    deliberately honest simplification, like the basic-M\\* / detector-only
    choices elsewhere in this zoo). With ``return_stats`` also returns
    ``{"expansions": cumulative, "per_weight": [...]}``.
    """
    plans = []
    total_exp = 0
    per_weight = []
    for w in weights:
        plan, st = plan_footsteps(
            world, start, goal, w=w, step_cost=step_cost, goal_xy_tol=goal_xy_tol,
            goal_theta_tol=goal_theta_tol, footstep_set=footstep_set,
            xy_res=xy_res, theta_res=theta_res, max_expansions=max_expansions,
            heuristic=heuristic, return_stats=True,
        )
        total_exp += st["expansions"]
        per_weight.append({"w": w, "expansions": st["expansions"],
                           "cost": None if plan is None else plan.cost})
        if plan is None:
            continue
        plans.append(plan)
    if return_stats:
        return plans, {"expansions": total_exp, "per_weight": per_weight}
    return plans
