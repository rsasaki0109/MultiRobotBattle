"""Kinodynamic Conflict-Based Search (K-CBS).

A pure-Python reproduction of Kottinger, Almagor & Lahijanian's *"Conflict-Based
Search for Multi-Robot Motion Planning with Kinodynamic Constraints"* (IROS 2022).

Every CBS variant in this zoo — and CCBS, and dRRT — plans **geometric** motion:
agents teleport between graph vertices, or slide along straight roadmap edges. None
of them respects a robot's **dynamics**. K-CBS does: each robot is a **Dubins car**
(constant forward speed, bounded turn rate, so it *cannot* turn in place or move
sideways — it must follow curves of radius ``>= V/ω_max``), and the planner returns
**dynamically-feasible** trajectories.

The two levels mirror CBS, lifted onto a sampling-based kinodynamic low level:

- :class:`KinodynamicRRT` (:func:`plan_trajectory`) — the **low level**: an RRT that
  grows a tree of timed states by *forward-propagating* the car's dynamics under a
  discrete set of controls (exact arc integration, :func:`propagate`). It plans in
  state×time, avoiding obstacles **and** a set of space–time
  :class:`Constraint` tubes handed down by the high level. Goal-biased; returns a
  trajectory sampled every ``dt``.
- :func:`first_conflict` — exact-at-``dt`` collision detection between two timed
  trajectories (a robot that has reached its goal *parks* there): the first instant
  two discs come within ``r_i + r_j``. This is both the conflict detector and the
  independent oracle the gate verifies solutions against.
- :func:`kcbs` — the **high level**: best-first over a constraint tree by
  sum-of-durations. On the first conflict it adds, to each robot in turn, a
  space–time constraint forbidding it from coming within the collision radius of
  the conflict location during a short window, and branches — exactly CBS, but the
  constraint is a continuous space–time tube and the low level is kinodynamic.

Honest scope (see ``docs/coordination.md``): the low level is sampling-based, so
K-CBS is a **feasibility** planner — dynamically feasible and collision-free, not
cost-optimal (the paper restores completeness with a meta-robot *merge* bound à la
MA-CBS; that is omitted here and noted). Collisions are checked on the shared ``dt``
grid (fine ``dt``); a robot stops and parks on reaching its goal (a modelling
choice, since a Dubins car cannot otherwise halt).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

__all__ = [
    "DubinsCar",
    "Constraint",
    "KinodynamicRRT",
    "first_conflict",
    "kcbs",
    "plan_trajectory",
    "propagate",
    "trajectory_feasible",
    "min_separation",
]


# --------------------------------------------------------------------------- #
# the robot and its dynamics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DubinsCar:
    """A constant-speed bounded-turn-rate car: state ``(x, y, theta)``."""

    speed: float = 1.0
    omega_max: float = 1.5
    radius: float = 0.3

    @property
    def min_turn_radius(self):
        return self.speed / self.omega_max


def propagate(car, state, omega, dt):
    """Exact arc integration of the Dubins dynamics for one step of ``dt``.

    ``x' = V cosθ``, ``y' = V sinθ``, ``θ' = ω`` — closed-form over constant ``ω``.
    """
    x, y, th = state
    v = car.speed
    if abs(omega) < 1e-9:
        return (x + v * math.cos(th) * dt, y + v * math.sin(th) * dt, th)
    th2 = th + omega * dt
    x2 = x + (v / omega) * (math.sin(th2) - math.sin(th))
    y2 = y - (v / omega) * (math.cos(th2) - math.cos(th))
    return (x2, y2, th2)


@dataclass(frozen=True)
class Constraint:
    """Space–time tube: the robot must stay ``>= radius`` from ``(cx, cy)`` while
    ``t_lo <= t <= t_hi``."""

    cx: float
    cy: float
    t_lo: float
    t_hi: float
    radius: float


def _obstacle_hit(obstacles, car, x, y):
    return any(math.hypot(x - ox, y - oy) < orad + car.radius - 1e-9
               for ox, oy, orad in obstacles)


def _constraint_hit(constraints, x, y, t):
    for c in constraints:
        if c.t_lo - 1e-9 <= t <= c.t_hi + 1e-9 and \
                math.hypot(x - c.cx, y - c.cy) < c.radius - 1e-9:
            return True
    return False


# --------------------------------------------------------------------------- #
# low level: kinodynamic RRT in state × time
# --------------------------------------------------------------------------- #
@dataclass
class KinodynamicRRT:
    car: DubinsCar
    bounds: tuple            # (xmin, xmax, ymin, ymax)
    obstacles: list = field(default_factory=list)
    dt: float = 0.1
    prim_steps: int = 5      # sub-steps per RRT edge (one held control)
    goal_radius: float = 0.4
    goal_bias: float = 0.2
    max_nodes: int = 6000

    def _controls(self):
        w = self.car.omega_max
        return (-w, -0.5 * w, 0.0, 0.5 * w, w)

    def plan(self, start, goal, constraints, rng):
        """Return a trajectory ``[(t, x, y, theta), ...]`` sampled every ``dt``
        from ``start`` to within ``goal_radius`` of ``goal``, or ``None``."""
        xmin, xmax, ymin, ymax = self.bounds
        # nodes: (state, time, parent_index, fine_states_from_parent)
        nodes = [(start, 0.0, -1, [(0.0, start[0], start[1], start[2])])]
        gx, gy = goal[0], goal[1]

        def _reached(state):
            return math.hypot(state[0] - gx, state[1] - gy) <= self.goal_radius

        if _reached(start):
            return [(0.0, start[0], start[1], start[2])]

        for _ in range(self.max_nodes):
            if rng.random() < self.goal_bias:
                sample = (gx, gy)
            else:
                sample = (rng.uniform(xmin, xmax), rng.uniform(ymin, ymax))
            # nearest node by Euclidean position of its endpoint
            ni = min(range(len(nodes)), key=lambda i: math.hypot(
                nodes[i][0][0] - sample[0], nodes[i][0][1] - sample[1]))
            nstate, ntime, _, _ = nodes[ni]
            # pick the control whose propagated endpoint is closest to sample
            best = None
            for omega in self._controls():
                s = nstate
                t = ntime
                fine = []
                ok = True
                for _k in range(self.prim_steps):
                    s = propagate(self.car, s, omega, self.dt)
                    t = t + self.dt
                    if not (xmin <= s[0] <= xmax and ymin <= s[1] <= ymax):
                        ok = False
                        break
                    if _obstacle_hit(self.obstacles, self.car, s[0], s[1]):
                        ok = False
                        break
                    if _constraint_hit(constraints, s[0], s[1], t):
                        ok = False
                        break
                    fine.append((t, s[0], s[1], s[2]))
                if not ok:
                    continue
                d = math.hypot(s[0] - sample[0], s[1] - sample[1])
                if best is None or d < best[0]:
                    best = (d, s, t, fine)
            if best is None:
                continue
            _, s, t, fine = best
            nodes.append((s, t, ni, fine))
            if _reached(s):
                return self._reconstruct(nodes, len(nodes) - 1)
        return None

    def _reconstruct(self, nodes, idx):
        # walk parents, collect fine states (each edge's fine list excludes its
        # own start, which is the parent's endpoint), then prepend the root.
        chain = []
        while idx != -1:
            state, time, parent, fine = nodes[idx]
            chain.append(fine)
            idx = parent
        chain.reverse()
        traj = list(chain[0])           # root's single (0, start)
        for fine in chain[1:]:
            traj.extend(fine)
        return traj


def plan_trajectory(car, start, goal, bounds, obstacles, constraints, *,
                    rng=None, **kw):
    """Convenience wrapper around :class:`KinodynamicRRT`."""
    rng = rng or random.Random(0)
    planner = KinodynamicRRT(car, bounds, obstacles=list(obstacles), **kw)
    return planner.plan(start, goal, list(constraints), rng)


# --------------------------------------------------------------------------- #
# trajectory sampling, feasibility, and conflict detection
# --------------------------------------------------------------------------- #
def _state_at(traj, t):
    """Position of a trajectory at time ``t`` (parks at the last state)."""
    if t <= traj[0][0]:
        return traj[0][1], traj[0][2]
    if t >= traj[-1][0]:
        return traj[-1][1], traj[-1][2]
    # trajectories are on a uniform dt grid; locate the bracketing samples
    lo, hi = 0, len(traj) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if traj[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    t0, x0, y0, _ = traj[lo]
    t1, x1, y1, _ = traj[hi]
    a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    return x0 + a * (x1 - x0), y0 + a * (y1 - y0)


def trajectory_feasible(car, traj, dt, *, tol=1e-6):
    """Independent check that ``traj`` obeys the Dubins dynamics.

    Every consecutive pair must be reproducible by *some* ``|ω| <= ω_max`` held
    for ``dt`` at the car's constant speed (parking — a repeated state — is
    allowed as the goal halt).  Returns ``True`` iff dynamically feasible.
    """
    for (t0, x0, y0, th0), (t1, x1, y1, th1) in zip(traj, traj[1:]):
        if abs((t1 - t0) - dt) > 1e-6 and abs(t1 - t0) > 1e-6:
            return False
        if (x0, y0, th0) == (x1, y1, th1):
            continue                       # parked at goal
        omega = (th1 - th0) / dt
        if abs(omega) > car.omega_max + 1e-6:
            return False
        px, py, pth = propagate(car, (x0, y0, th0), omega, dt)
        if math.hypot(px - x1, py - y1) > car.speed * dt * tol + 1e-4:
            return False
    return True


def first_conflict(traj_a, traj_b, rad_a, rad_b, dt, horizon=None):
    """First time (on the ``dt`` grid) two trajectories come within ``r_a+r_b``.

    Returns ``(t, mx, my)`` at the conflict (midpoint), or ``None``.
    """
    end = max(traj_a[-1][0], traj_b[-1][0]) if horizon is None else horizon
    thr = rad_a + rad_b
    steps = int(round(end / dt)) + 1
    for k in range(steps):
        t = k * dt
        ax, ay = _state_at(traj_a, t)
        bx, by = _state_at(traj_b, t)
        if math.hypot(ax - bx, ay - by) < thr - 1e-9:
            return (t, 0.5 * (ax + bx), 0.5 * (ay + by))
    return None


def min_separation(traj_a, traj_b, dt, horizon=None):
    """Closest approach (on the ``dt`` grid) between two trajectories."""
    end = max(traj_a[-1][0], traj_b[-1][0]) if horizon is None else horizon
    steps = int(round(end / dt)) + 1
    best = math.inf
    for k in range(steps):
        t = k * dt
        ax, ay = _state_at(traj_a, t)
        bx, by = _state_at(traj_b, t)
        best = min(best, math.hypot(ax - bx, ay - by))
    return best


# --------------------------------------------------------------------------- #
# high level: conflict-based search over space–time constraints
# --------------------------------------------------------------------------- #
@dataclass
class _Node:
    constraints: dict        # robot -> list[Constraint]
    trajs: dict              # robot -> trajectory
    cost: float

    def __lt__(self, other):
        return self.cost < other.cost


@dataclass
class KCBSSolution:
    trajectories: dict
    cost: float
    high_level_expansions: int


def kcbs(cars, starts, goals, bounds, obstacles, *, dt=0.1, window=0.3,
         max_expansions=400, rng=None, **low_level_kw):
    """Kinodynamic CBS.  ``cars/starts/goals`` are dicts keyed by robot id.

    Returns a :class:`KCBSSolution` (dynamically-feasible, collision-free
    trajectories) or ``None`` if the budget is exhausted.
    """
    import heapq

    rng = rng or random.Random(0)
    ids = list(starts)

    def _plan(robot, constraints):
        # deterministic per-(robot, constraint-count) seed so re-planning the
        # same node is reproducible yet varies as constraints accrue.
        seed = (hash(robot) ^ (len(constraints) * 2654435761)) & 0xFFFFFFFF
        return plan_trajectory(cars[robot], starts[robot], goals[robot],
                               bounds, obstacles, constraints,
                               rng=random.Random(seed), dt=dt, **low_level_kw)

    root_trajs = {}
    for i in ids:
        tr = _plan(i, [])
        if tr is None:
            return None
        root_trajs[i] = tr
    root = _Node({i: [] for i in ids}, root_trajs,
                 sum(t[-1][0] for t in root_trajs.values()))
    open_list = [root]
    expansions = 0

    while open_list and expansions < max_expansions:
        node = heapq.heappop(open_list)
        expansions += 1
        # find the first conflict across all pairs
        conflict = None
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = ids[a], ids[b]
                c = first_conflict(node.trajs[i], node.trajs[j],
                                   cars[i].radius, cars[j].radius, dt)
                if c is not None:
                    conflict = (i, j, c)
                    break
            if conflict:
                break
        if conflict is None:
            return KCBSSolution(node.trajs, node.cost, expansions)
        i, j, (tc, mx, my) = conflict
        rad = cars[i].radius + cars[j].radius
        for who in (i, j):
            cons = Constraint(mx, my, tc - window, tc + window, rad)
            new_cons = dict(node.constraints)
            new_cons[who] = node.constraints[who] + [cons]
            tr = _plan(who, new_cons[who])
            if tr is None:
                continue
            new_trajs = dict(node.trajs)
            new_trajs[who] = tr
            child = _Node(new_cons, new_trajs,
                          sum(t[-1][0] for t in new_trajs.values()))
            heapq.heappush(open_list, child)
    return None
