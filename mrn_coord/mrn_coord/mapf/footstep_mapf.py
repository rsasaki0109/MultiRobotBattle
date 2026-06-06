"""Multi-humanoid footstep-level Multi-Agent Path Finding.

This lifts the single-humanoid footstep planner of :mod:`footstep`
(Hornung et al., Humanoids 2012) to a *team* of humanoids that must reach their
goals without their **bodies** colliding — coordination at the footstep
resolution rather than on a coarse grid.

**Honest scope.** This is a *planning / coordination* reproduction, kinematic
and discrete-time, not a whole-body controller: there is no dynamics, ZMP, or
contact reasoning. The pieces it does model faithfully:

- Each humanoid's low level is footstep A* over ``(stance pose, which foot)``
  with the Fig. 2 footstep set and Eq. 1 cost, exactly as in :mod:`footstep`.
- Time is the **step index** (tick): at tick ``t`` every humanoid has taken
  ``t`` footsteps, as in grid MAPF where each agent moves once per timestep.
- A humanoid's **body** is approximated by a disc of radius ``body_radius``
  centred at the stance foot; two bodies conflict at a tick when their discs
  overlap. A **STAY** action (stand still one tick) gives the planner the
  "wait" primitive MAPF needs to break deadlocks; a finished humanoid holds its
  goal, blocking it for the others.

The high level is **prioritized planning** (the repo's existing incomplete-but-
fast paradigm, :mod:`prioritized`): plan humanoids in priority order, each
treating the already-planned, higher-priority bodies as moving obstacles
(tick-indexed forbidden discs). Collision-free by construction when it succeeds;
it can fail (return ``None`` for a humanoid) on the symmetric head-on cases that
defeat any fixed priority order — the honest limitation of prioritized planning.
"""

from __future__ import annotations

import heapq
import math

from .footstep import (
    DEFAULT_FOOTSTEP_SET,
    FootstepPlan,
    _apply,
    _key,
    _norm_angle,
)

# default torso disc: a large humanoid is ~0.3-0.4 m wide
DEFAULT_BODY_RADIUS = 0.22


def _h(goal, st, step_cost, max_reach):
    d = math.hypot(goal[0] - st.x, goal[1] - st.y)
    min_steps = math.ceil(d / max_reach - 1e-9) if d > 1e-9 else 0
    return d + step_cost * min_steps


def _bodies_clear(cx, cy, reservations, tick, two_r):
    """True iff a body disc centred at ``(cx, cy)`` at ``tick`` overlaps none of
    the reserved (higher-priority) body centres at that tick."""
    centres = reservations.get(tick)
    if not centres:
        return True
    for (rx, ry) in centres:
        if math.hypot(cx - rx, cy - ry) + 1e-9 < two_r:
            return False
    return True


def plan_footsteps_reserved(world, start, goal, reservations, *,
                            body_radius=DEFAULT_BODY_RADIUS, w=1.0,
                            step_cost=0.30, wait_cost=0.30, goal_xy_tol=0.18,
                            goal_theta_tol=None, footstep_set=DEFAULT_FOOTSTEP_SET,
                            xy_res=0.02, theta_res=math.radians(10),
                            max_tick=60, max_expansions=120_000,
                            return_stats=False):
    """Footstep A* over ``(stance pose, tick)`` honoring tick-indexed body
    reservations from higher-priority humanoids.

    ``reservations`` maps ``tick -> list[(cx, cy)]`` of body centres already
    occupied; the search keeps the planned body disc clear of all of them, and
    only accepts the goal when the humanoid can *hold* it (its goal disc is clear
    for every reserved tick from arrival onward). A **STAY** action lets it wait
    in place. Returns a :class:`FootstepPlan` whose ``states`` are the per-tick
    stance poses (held at the goal after arrival), or ``None``.
    """
    two_r = 2.0 * body_radius
    max_reach = max(math.hypot(dx, dy) for dx, dy, _ in footstep_set)
    last_res_tick = max(reservations) if reservations else 0

    def is_goal(st):
        if math.hypot(goal[0] - st.x, goal[1] - st.y) > goal_xy_tol:
            return False
        if len(goal) >= 3 and goal_theta_tol is not None:
            if abs(_norm_angle(st.theta - goal[2])) > goal_theta_tol:
                return False
        return True

    def can_hold(st, tick):
        # the humanoid stands at its goal from `tick` to the last reserved tick
        for t in range(tick, last_res_tick + 1):
            if not _bodies_clear(st.x, st.y, reservations, t, two_r):
                return False
        return True

    skey = (_key(start, xy_res, theta_res), 0)
    g = {skey: 0.0}
    state_of = {skey: start}
    parent = {skey: None}
    open_heap = [(w * _h(goal, start, step_cost, max_reach), 0.0, skey)]
    closed = set()
    expansions = 0

    while open_heap:
        f, gs, sk = heapq.heappop(open_heap)
        if sk in closed or gs > g[sk] + 1e-9:
            continue
        st = state_of[sk]
        tick = sk[1]
        if is_goal(st) and can_hold(st, tick):
            path = []
            kk = sk
            while kk is not None:
                path.append(state_of[kk])
                kk = parent[kk]
            path.reverse()
            # the returned plan ends at the real arrival tick; the prioritized
            # layer holds the goal for later ticks via min(t, L-1) when it
            # reserves, so no trailing padding is needed here
            stats = {"expansions": expansions}
            plan = FootstepPlan(path, g[sk], suboptimality=w)
            return (plan, stats) if return_stats else plan
        closed.add(sk)
        expansions += 1
        if expansions > max_expansions or tick >= max_tick:
            if expansions > max_expansions:
                break
            continue

        # successors: footstep moves ...
        succ = []
        for step in footstep_set:
            nxt = _apply(st, step, xy_res=xy_res, theta_res=theta_res)
            if not world.foot_collision_free(nxt.x, nxt.y, nxt.theta):
                continue
            cost = math.hypot(nxt.x - st.x, nxt.y - st.y) + step_cost
            succ.append((nxt, cost))
        # ... plus STAY (stand still one tick)
        succ.append((st, wait_cost))

        for nxt, cost in succ:
            ntick = tick + 1
            if not _bodies_clear(nxt.x, nxt.y, reservations, ntick, two_r):
                continue
            nk = (_key(nxt, xy_res, theta_res), ntick)
            ng = g[sk] + cost
            if nk not in g or ng < g[nk] - 1e-9:
                g[nk] = ng
                state_of[nk] = nxt
                parent[nk] = sk
                heapq.heappush(
                    open_heap,
                    (ng + w * _h(goal, nxt, step_cost, max_reach), ng, nk),
                )
    stats = {"expansions": expansions}
    return (None, stats) if return_stats else None


def prioritized_footstep_mapf(world, agents, *, order=None,
                              body_radius=DEFAULT_BODY_RADIUS, w=1.0,
                              step_cost=0.30, wait_cost=0.30, goal_xy_tol=0.18,
                              footstep_set=DEFAULT_FOOTSTEP_SET, xy_res=0.02,
                              theta_res=math.radians(10), max_tick=60,
                              max_expansions=120_000, return_stats=False):
    """Prioritized multi-humanoid footstep MAPF.

    ``agents`` maps ``id -> (start_state, goal)`` where ``start_state`` is a
    :class:`FootstepState` and ``goal`` is ``(x, y)`` / ``(x, y, theta)``. Plans
    humanoids in ``order`` (default: the dict order), each avoiding the bodies of
    those already planned. Returns ``id -> FootstepPlan`` (a humanoid that could
    not be placed maps to ``None``). With ``return_stats`` also returns a stats
    dict.
    """
    ids = list(order) if order is not None else list(agents)
    plans = {}
    reservations = {}  # tick -> list[(cx, cy)]
    horizon = 0
    total_exp = 0

    for aid in ids:
        start, goal = agents[aid]
        plan, st = plan_footsteps_reserved(
            world, start, goal, reservations, body_radius=body_radius, w=w,
            step_cost=step_cost, wait_cost=wait_cost, goal_xy_tol=goal_xy_tol,
            footstep_set=footstep_set, xy_res=xy_res, theta_res=theta_res,
            max_tick=max_tick, max_expansions=max_expansions, return_stats=True,
        )
        total_exp += st["expansions"]
        plans[aid] = plan
        if plan is None:
            continue
        horizon = max(horizon, len(plan.states) - 1)
        # reserve this humanoid's body disc at every tick, holding the goal
        L = len(plan.states)
        upto = max(horizon, max_tick)
        for t in range(upto + 1):
            s = plan.states[min(t, L - 1)]
            reservations.setdefault(t, []).append((s.x, s.y))

    if return_stats:
        return plans, {"expansions": total_exp, "horizon": horizon}
    return plans


def bodies_collision_free(plans, *, body_radius=DEFAULT_BODY_RADIUS):
    """Verify a set of footstep plans is body-collision-free: at no shared tick
    do any two humanoids' body discs overlap. Returns ``True`` / ``False``."""
    two_r = 2.0 * body_radius
    live = {k: v for k, v in plans.items() if v is not None}
    if not live:
        return True
    horizon = max(len(p.states) for p in live.values())
    for t in range(horizon):
        centres = []
        for p in live.values():
            s = p.states[min(t, len(p.states) - 1)]
            centres.append((s.x, s.y))
        for i in range(len(centres)):
            for j in range(i + 1, len(centres)):
                (ax, ay), (bx, by) = centres[i], centres[j]
                if math.hypot(ax - bx, ay - by) + 1e-9 < two_r:
                    return False
    return True
