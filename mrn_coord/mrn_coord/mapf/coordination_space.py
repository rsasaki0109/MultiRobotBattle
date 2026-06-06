"""Path–velocity decomposition: coordination-space scheduling.

A pure-Python reproduction of the classic **coordination diagram** / velocity-tuning
paradigm — Kant & Zucker, *"Toward Efficient Trajectory Planning: The Path-Velocity
Decomposition"* (IJRR 1986), and O'Donnell & Lozano-Pérez, *"Deadlock-Free and
Collision-Free Coordination of Two Robots"* (ICRA 1989).

Every other multi-robot planner in this zoo decides *where* each robot goes. This
one does the opposite: each robot's **geometric path is fixed** (planned in advance,
decoupled), and the planner only schedules **how fast** each robot moves along its
own path. The joint state is the tuple of path parameters ``(s_1, …, s_n) ∈ [0,1]^n``
— the **coordination space**. A pair of robots *collides* on the sub-square of
parameter values where their bodies overlap; those sub-squares are the obstacles.
A collision-free schedule is a **monotone** path through the coordination space from
``(0,…,0)`` to ``(1,…,1)`` that avoids every collision region — monotone because a
robot only ever moves *forward* along its path (it may wait, i.e. hold ``s_i``, but
never reverse). For two robots this is the famous 2-D coordination diagram: a
staircase routed around the collision blob.

- :func:`discretize_path` — sample a polyline into ``m`` arc-length-uniform points.
- :func:`build_collision_table` — the pairwise ``m×m`` collision masks (the obstacles
  of the coordination space).
- :func:`schedule` — A* over the index lattice ``{0,…,m-1}^n``; each step advances
  any non-empty subset of the robots by one (the rest wait → velocity tuning), never
  entering a colliding cell nor crossing one in transit. Minimises **makespan** (the
  number of steps); collision-free *by construction*.

Honest scope (see ``docs/coordination.md``): path–velocity decomposition only tunes
velocities along **fixed** paths — it can resolve *timing* conflicts (crossings,
merges) but **cannot reroute**. Two robots assigned the *same* corridor in opposite
directions have no monotone collision-free schedule (the collision band cuts the
coordination space in two); :func:`schedule` correctly returns ``None`` — the
paradigm's known incompleteness, not a bug. Collisions are checked at the chosen
resolution ``m`` (plus the simultaneous-motion clearance on each transition).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "CoordinationProblem",
    "CoordinationSchedule",
    "build_collision_table",
    "discretize_path",
    "min_clearance",
    "schedule",
    "schedule_to_trajectories",
]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def discretize_path(path, m):
    """Sample ``path`` (a polyline of points) into ``m`` arc-length-uniform points."""
    if m < 2:
        raise ValueError("m must be >= 2")
    seglens = [_dist(path[i], path[i + 1]) for i in range(len(path) - 1)]
    total = sum(seglens)
    if total <= 1e-12:
        return [tuple(path[0]) for _ in range(m)]
    out = []
    for k in range(m):
        target = total * k / (m - 1)
        acc = 0.0
        for i, sl in enumerate(seglens):
            if acc + sl >= target - 1e-12 or i == len(seglens) - 1:
                a = 0.0 if sl <= 1e-12 else (target - acc) / sl
                a = max(0.0, min(1.0, a))
                p0, p1 = path[i], path[i + 1]
                out.append((p0[0] + a * (p1[0] - p0[0]),
                            p0[1] + a * (p1[1] - p0[1])))
                break
            acc += sl
    return out


def _moving_min_distance(a0, a1, b0, b1):
    """Closest approach of two points moving simultaneously over ``t in [0,1]``."""
    r0 = (a0[0] - b0[0], a0[1] - b0[1])
    d = (a1[0] - a0[0] - (b1[0] - b0[0]), a1[1] - a0[1] - (b1[1] - b0[1]))
    dd = d[0] * d[0] + d[1] * d[1]
    if dd <= 1e-15:
        return math.hypot(r0[0], r0[1])
    t = -(r0[0] * d[0] + r0[1] * d[1]) / dd
    t = max(0.0, min(1.0, t))
    return math.hypot(r0[0] + t * d[0], r0[1] + t * d[1])


@dataclass
class CoordinationProblem:
    """Fixed geometric paths + body radii, sampled at resolution ``m``."""

    paths: list           # list of polylines (each a list of (x, y))
    radii: list           # one radius per robot
    m: int = 20

    @property
    def n(self):
        return len(self.paths)

    def samples(self):
        return [discretize_path(p, self.m) for p in self.paths]


def build_collision_table(problem, safety_margin=0.0):
    """Pairwise ``m×m`` collision masks: ``table[(a,b)][ia][ib]`` is ``True`` iff
    robot ``a`` at path index ``ia`` is within ``r_a+r_b+safety_margin`` of robot
    ``b`` at path index ``ib`` (the obstacles of the coordination space)."""
    pts = problem.samples()
    m = problem.m
    table = {}
    for a in range(problem.n):
        for b in range(a + 1, problem.n):
            thr = problem.radii[a] + problem.radii[b] + safety_margin
            mask = [[_dist(pts[a][ia], pts[b][ib]) < thr - 1e-9
                     for ib in range(m)] for ia in range(m)]
            table[(a, b)] = mask
    return table


@dataclass
class CoordinationSchedule:
    states: list          # list of index-tuples from start to goal
    makespan: int         # number of steps (== len(states) - 1)
    expansions: int


def _advance_moves(n):
    """All non-empty subsets of robots to advance by one (2**n - 1 of them)."""
    moves = []
    for mask in range(1, 1 << n):
        moves.append(tuple((mask >> i) & 1 for i in range(n)))
    return moves


def schedule(problem, *, safety_margin=0.0, max_expansions=200000):
    """A* over the coordination lattice; minimise makespan, avoid collisions.

    ``safety_margin`` inflates the collision radius so the executed schedule keeps
    real clearance at the chosen resolution.  Returns a
    :class:`CoordinationSchedule` or ``None`` if no monotone collision-free
    schedule exists (e.g. a shared corridor traversed in opposite directions —
    the velocity-tuning paradigm cannot reroute).
    """
    import heapq

    n = problem.n
    m = problem.m
    pts = problem.samples()
    table = build_collision_table(problem, safety_margin)
    moves = _advance_moves(n)
    start = tuple(0 for _ in range(n))
    goal = tuple(m - 1 for _ in range(n))

    def _cell_collision(state):
        for (a, b), mask in table.items():
            if mask[state[a]][state[b]]:
                return True
        return False

    def _transition_collision(u, v):
        # robots move simultaneously along their path segments u->v
        for a in range(n):
            for b in range(a + 1, n):
                if _moving_min_distance(pts[a][u[a]], pts[a][v[a]],
                                        pts[b][u[b]], pts[b][v[b]]) \
                        < problem.radii[a] + problem.radii[b] + safety_margin \
                        - 1e-9:
                    return True
        return False

    def _h(state):
        return max(m - 1 - state[i] for i in range(n))

    if _cell_collision(start) or _cell_collision(goal):
        return None
    open_list = [(_h(start), 0, start)]
    g = {start: 0}
    parent = {start: None}
    expansions = 0
    while open_list and expansions < max_expansions:
        _, gu, u = heapq.heappop(open_list)
        if gu > g.get(u, math.inf):
            continue
        expansions += 1
        if u == goal:
            states = [u]
            while parent[states[-1]] is not None:
                states.append(parent[states[-1]])
            states.reverse()
            return CoordinationSchedule(states, len(states) - 1, expansions)
        for mv in moves:
            v = tuple(min(m - 1, u[i] + mv[i]) for i in range(n))
            if v == u or _cell_collision(v) or _transition_collision(u, v):
                continue
            ng = gu + 1
            if ng < g.get(v, math.inf):
                g[v] = ng
                parent[v] = u
                heapq.heappush(open_list, (ng + _h(v), ng, v))
    return None


def schedule_to_trajectories(problem, sched):
    """Per-robot list of ``(x, y)`` waypoints, one per scheduled step."""
    pts = problem.samples()
    return {a: [pts[a][state[a]] for state in sched.states]
            for a in range(problem.n)}


def min_clearance(trajectories, radii):
    """Independent oracle: min pairwise clearance *minus* ``r_a+r_b`` over every
    simultaneous-motion step of the schedule (``< 0`` means a collision)."""
    ids = list(trajectories)
    horizon = len(trajectories[ids[0]])
    worst = math.inf
    for t in range(horizon - 1):
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                d = _moving_min_distance(trajectories[ids[a]][t],
                                         trajectories[ids[a]][t + 1],
                                         trajectories[ids[b]][t],
                                         trajectories[ids[b]][t + 1])
                worst = min(worst, d - (radii[ids[a]] + radii[ids[b]]))
    return worst
