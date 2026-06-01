"""Continuous-space kinodynamic planning: Dubins curves + Hybrid A*.

Where :func:`mrn_sim.navigate.plan_world_path` plans on a 4-connected grid
(axis-aligned moves, blind to the robot's heading and turn radius), this plans
in the continuous ``(x, y, theta)`` state space with a **bounded turning
radius**, so the path is *kinematically feasible* for a bounded-curvature
(car-like / smoothly-steered differential-drive) robot: smooth, forward-only,
with headings the pure-pursuit follower can actually track — no axis-aligned
zig-zag a unicycle would have to pivot in place to follow.

Two pieces, both pure and deterministic, depending only on
:mod:`mrn_sim.world`:

* :func:`dubins_path` — the closed-form shortest bounded-curvature curve
  between two oriented poses (the six Dubins words ``LSL RSR LSR RSL RLR LRL``).
* :func:`plan_kinodynamic` — Hybrid A* over a discretized ``(x, y, theta)``
  lattice, expanding constant-curvature motion primitives, collision-checking
  each arc against the world, and periodically attempting a Dubins
  *analytic-expansion* shot straight to the goal pose. The heuristic combines a
  holonomic obstacle-aware grid distance with the obstacle-free Dubins length.

This is not claimed to be cost-optimal (the grid heuristic and a coarse lattice
trade optimality for speed); it returns a feasible, smooth path quickly.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from .world import World


def _mod2pi(theta: float) -> float:
    """Wrap to ``[0, 2*pi)``."""
    return theta - 2.0 * math.pi * math.floor(theta / (2.0 * math.pi))


def _norm(theta: float) -> float:
    """Wrap to ``(-pi, pi]``."""
    t = math.fmod(theta + math.pi, 2.0 * math.pi)
    if t <= 0.0:
        t += 2.0 * math.pi
    return t - math.pi


def _arc(x: float, y: float, th: float, k: float, ds: float):
    """Exact constant-curvature step: curvature ``k`` over arc length ``ds``.

    ``k > 0`` turns left, ``k < 0`` right, ``k == 0`` goes straight.
    """
    if abs(k) < 1e-9:
        return (x + ds * math.cos(th), y + ds * math.sin(th), th)
    th2 = th + k * ds
    x2 = x + (math.sin(th2) - math.sin(th)) / k
    y2 = y - (math.cos(th2) - math.cos(th)) / k
    return (x2, y2, th2)


# --------------------------------------------------------------------------- #
# Dubins shortest path                                                        #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DubinsPath:
    """A Dubins curve: three segments at the given turning ``radius``.

    ``word`` is e.g. ``("L", "S", "L")``; ``segments`` holds the three segment
    *lengths in metres* (arc length for ``L``/``R``, straight length for ``S``).
    ``length`` is their sum.
    """

    word: tuple
    segments: tuple
    radius: float
    length: float

    def sample(self, step: float, start):
        """Sample poses ``[(x, y, theta), ...]`` along the curve from ``start``."""
        x, y, th = start
        poses = [(x, y, th)]
        kind = {"S": 0.0, "L": 1.0 / self.radius, "R": -1.0 / self.radius}
        for seg_type, seg_len in zip(self.word, self.segments):
            k = kind[seg_type]
            n = max(1, int(math.ceil(seg_len / step)))
            ds = seg_len / n
            for _ in range(n):
                x, y, th = _arc(x, y, th, k, ds)
                poses.append((x, y, _norm(th)))
        return poses


def _dubins_words(alpha: float, beta: float, d: float):
    """Yield ``(t, p, q, word)`` for each feasible Dubins word (normalized).

    ``d`` is the start-goal distance divided by the turning radius; ``t, p, q``
    are segment lengths in the *normalized* frame (radians for arc segments,
    radius-multiples for the straight segment), so the path length in metres is
    ``(t + p + q) * radius``.
    """
    sa, sb = math.sin(alpha), math.sin(beta)
    ca, cb = math.cos(alpha), math.cos(beta)
    c_ab = math.cos(alpha - beta)

    # LSL
    tmp0 = d + sa - sb
    p_sq = 2.0 + d * d - 2.0 * c_ab + 2.0 * d * (sa - sb)
    if p_sq >= 0.0:
        tmp1 = math.atan2(cb - ca, tmp0)
        yield (_mod2pi(-alpha + tmp1), math.sqrt(p_sq), _mod2pi(beta - tmp1),
               ("L", "S", "L"))
    # RSR
    tmp0 = d - sa + sb
    p_sq = 2.0 + d * d - 2.0 * c_ab + 2.0 * d * (sb - sa)
    if p_sq >= 0.0:
        tmp1 = math.atan2(ca - cb, tmp0)
        yield (_mod2pi(alpha - tmp1), math.sqrt(p_sq), _mod2pi(-beta + tmp1),
               ("R", "S", "R"))
    # LSR
    p_sq = -2.0 + d * d + 2.0 * c_ab + 2.0 * d * (sa + sb)
    if p_sq >= 0.0:
        p = math.sqrt(p_sq)
        tmp1 = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
        yield (_mod2pi(-alpha + tmp1), p, _mod2pi(-beta + tmp1),
               ("L", "S", "R"))
    # RSL
    p_sq = -2.0 + d * d + 2.0 * c_ab - 2.0 * d * (sa + sb)
    if p_sq >= 0.0:
        p = math.sqrt(p_sq)
        tmp1 = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
        yield (_mod2pi(alpha - tmp1), p, _mod2pi(beta - tmp1),
               ("R", "S", "L"))
    # RLR
    tmp = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (sa - sb)) / 8.0
    if abs(tmp) <= 1.0:
        p = _mod2pi(2.0 * math.pi - math.acos(tmp))
        t = _mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + p / 2.0)
        yield (t, p, _mod2pi(alpha - beta - t + p), ("R", "L", "R"))
    # LRL
    tmp = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (sb - sa)) / 8.0
    if abs(tmp) <= 1.0:
        p = _mod2pi(2.0 * math.pi - math.acos(tmp))
        t = _mod2pi(-alpha + math.atan2(-ca + cb, d + sa - sb) + p / 2.0)
        yield (t, p, _mod2pi(_mod2pi(beta) - alpha + 2.0 * p) - t, ("L", "R", "L"))


def dubins_path(start, goal, radius: float) -> DubinsPath:
    """Shortest Dubins curve from pose ``start`` to pose ``goal`` (no obstacles).

    ``start`` / ``goal`` are ``(x, y, theta)``; ``radius`` is the (minimum)
    turning radius. Returns the minimum-length :class:`DubinsPath` over the six
    words. Always succeeds (a Dubins path between any two poses exists).
    """
    if radius <= 0.0:
        raise ValueError("turning radius must be positive")
    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    dist = math.hypot(dx, dy)
    d = dist / radius
    theta = _mod2pi(math.atan2(dy, dx))
    alpha = _mod2pi(start[2] - theta)
    beta = _mod2pi(goal[2] - theta)

    best = None
    for t, p, q, word in _dubins_words(alpha, beta, d):
        cost = t + p + q
        if best is None or cost < best[0]:
            best = (cost, (t, p, q), word)
    cost, segs_norm, word = best
    # de-normalize: arc segments scale by radius (radians -> arc length),
    # the straight segment is already a radius-multiple -> also scale by radius.
    segments = tuple(s * radius for s in segs_norm)
    return DubinsPath(word=word, segments=segments, radius=radius,
                      length=cost * radius)


# --------------------------------------------------------------------------- #
# Hybrid A*                                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class KinoResult:
    """A kinodynamic plan: feasible poses, length, and search effort."""

    poses: list          # [(x, y, theta), ...]
    length: float        # path length in metres
    expansions: int      # nodes popped from the open set

    @property
    def waypoints(self) -> list:
        """``[(x, y), ...]`` — drop-in for the pure-pursuit follower."""
        return [(p[0], p[1]) for p in self.poses]


def _holonomic_heuristic(world: World, goal, *, cell_size: float, robot_radius: float):
    """Dijkstra (8-connected) distance to ``goal`` over free cells.

    Returns ``(dist, nx, ny)`` where ``dist`` maps free cell ``(cx, cy)`` to the
    shortest grid distance (metres) to the goal cell, accounting for obstacles.
    Cells not reachable (or blocked) are simply absent from the dict.
    """
    nx = max(1, int(math.ceil(world.width / cell_size)))
    ny = max(1, int(math.ceil(world.height / cell_size)))

    def free(cx, cy):
        wx, wy = (cx + 0.5) * cell_size, (cy + 0.5) * cell_size
        return world.is_free(wx, wy, robot_radius)

    gx = min(nx - 1, max(0, int(math.floor(goal[0] / cell_size))))
    gy = min(ny - 1, max(0, int(math.floor(goal[1] / cell_size))))
    dist = {}
    if not free(gx, gy):
        return dist, nx, ny
    diag = math.sqrt(2.0) * cell_size
    straight = cell_size
    heap = [(0.0, gx, gy)]
    while heap:
        d, cx, cy = heapq.heappop(heap)
        if (cx, cy) in dist:
            continue
        dist[(cx, cy)] = d
        for dxc in (-1, 0, 1):
            for dyc in (-1, 0, 1):
                if dxc == 0 and dyc == 0:
                    continue
                ax, ay = cx + dxc, cy + dyc
                if not (0 <= ax < nx and 0 <= ay < ny) or (ax, ay) in dist:
                    continue
                if not free(ax, ay):
                    continue
                step_cost = diag if (dxc != 0 and dyc != 0) else straight
                heapq.heappush(heap, (d + step_cost, ax, ay))
    return dist, nx, ny


def _segment_free(world: World, poses, robot_radius: float, clearance: float) -> bool:
    """True if every sampled pose clears the world by ``robot_radius + clearance``."""
    margin = robot_radius + clearance
    for (x, y, _th) in poses:
        if not world.is_free(x, y, margin):
            return False
    return True


def plan_kinodynamic(
    world: World,
    start,
    goal,
    *,
    turn_radius: float = 1.2,
    robot_radius: float = 0.25,
    clearance: float = 0.1,
    xy_resolution: float = 0.5,
    yaw_resolution: float = math.radians(15.0),
    step_size: float | None = None,
    n_curvatures: int = 5,
    steer_penalty: float = 0.0,
    goal_xy_tol: float = 0.5,
    goal_yaw_tol: float | None = None,
    analytic_every: int = 5,
    max_expansions: int = 30000,
) -> KinoResult | None:
    """Plan a kinematically feasible path with Hybrid A*.

    ``start`` and ``goal`` are oriented poses ``(x, y, theta)``. The robot moves
    forward at a bounded curvature (``|kappa| <= 1 / turn_radius``); the planner
    expands ``n_curvatures`` constant-curvature primitives of length
    ``step_size`` (default ``1.5 * xy_resolution``), collision-checks each arc,
    and every ``analytic_every`` expansions tries a closed-form Dubins shot to
    the goal pose. The open set is keyed by a discretized ``(x, y, theta)``
    lattice (``xy_resolution`` / ``yaw_resolution``).

    Returns a :class:`KinoResult` (its ``.waypoints`` feed the pure-pursuit
    follower) or ``None`` if no path is found within ``max_expansions``. The
    final pose's heading is matched only when ``goal_yaw_tol`` is set; otherwise
    the goal is a position disk of radius ``goal_xy_tol``.
    """
    if step_size is None:
        step_size = 1.5 * xy_resolution
    max_kappa = 1.0 / turn_radius
    if n_curvatures < 1:
        raise ValueError("n_curvatures must be >= 1")
    if n_curvatures == 1:
        curvatures = [0.0]
    else:
        curvatures = [
            -max_kappa + 2.0 * max_kappa * i / (n_curvatures - 1)
            for i in range(n_curvatures)
        ]

    start = (float(start[0]), float(start[1]), _norm(float(start[2])))
    goal = (float(goal[0]), float(goal[1]), _norm(float(goal[2])))

    hdist, _nx, _ny = _holonomic_heuristic(
        world, goal, cell_size=xy_resolution, robot_radius=robot_radius
    )

    def heuristic(pose) -> float:
        cx = int(math.floor(pose[0] / xy_resolution))
        cy = int(math.floor(pose[1] / xy_resolution))
        grid = hdist.get((cx, cy))
        dub = dubins_path(pose, goal, turn_radius).length
        euclid = math.hypot(goal[0] - pose[0], goal[1] - pose[1])
        return max(dub, grid if grid is not None else euclid)

    def key(pose):
        return (
            int(math.floor(pose[0] / xy_resolution)),
            int(math.floor(pose[1] / xy_resolution)),
            int(math.floor(_mod2pi(pose[2]) / yaw_resolution)),
        )

    def at_goal(pose) -> bool:
        if math.hypot(pose[0] - goal[0], pose[1] - goal[1]) > goal_xy_tol:
            return False
        if goal_yaw_tol is not None and abs(_norm(pose[2] - goal[2])) > goal_yaw_tol:
            return False
        return True

    if not world.is_free(start[0], start[1], robot_radius + clearance):
        return None

    sub = max(2, int(math.ceil(step_size / (0.5 * xy_resolution))))
    ds = step_size / sub

    # nodes[i] = (pose, parent_index, primitive_poses_into_this_node)
    nodes = [(start, -1, [start])]
    counter = 0
    open_heap = [(heuristic(start), counter, 0.0, 0)]   # (f, tie, g, node_index)
    g_best = {key(start): 0.0}
    expansions = 0

    while open_heap and expansions < max_expansions:
        _f, _tie, g, idx = heapq.heappop(open_heap)
        pose = nodes[idx][0]
        if g > g_best.get(key(pose), g) + 1e-9:
            continue                                    # stale heap entry
        expansions += 1

        # analytic expansion: closed-form Dubins shot straight to the goal pose
        if expansions % analytic_every == 0 or at_goal(pose):
            dp = dubins_path(pose, goal, turn_radius)
            shot = dp.sample(0.5 * step_size, pose)
            if _segment_free(world, shot, robot_radius, clearance) and (
                goal_yaw_tol is None
                or abs(_norm(shot[-1][2] - goal[2])) <= goal_yaw_tol
            ):
                return _reconstruct(nodes, idx, shot, g + dp.length, expansions)

        if at_goal(pose):
            return _reconstruct(nodes, idx, [pose], g, expansions)

        for kappa in curvatures:
            px, py, pth = pose
            prim = [(px, py, pth)]
            for _ in range(sub):
                px, py, pth = _arc(px, py, pth, kappa, ds)
                prim.append((px, py, _norm(pth)))
            if not _segment_free(world, prim, robot_radius, clearance):
                continue
            npose = prim[-1]
            nkey = key(npose)
            ng = g + step_size + steer_penalty * abs(kappa) * step_size
            if nkey in g_best and g_best[nkey] <= ng + 1e-9:
                continue
            g_best[nkey] = ng
            nodes.append((npose, idx, prim))
            counter += 1
            heapq.heappush(open_heap, (ng + heuristic(npose), counter, ng, len(nodes) - 1))

    return None


def _reconstruct(nodes, end_index, tail_poses, length, expansions):
    """Walk parent links from ``end_index`` to the start, then append ``tail_poses``."""
    chain = []
    idx = end_index
    while idx >= 0:
        _pose, parent, prim = nodes[idx]
        chain.append(prim)
        idx = parent
    chain.reverse()

    poses: list = []
    for prim in chain:
        if poses and prim and poses[-1] == prim[0]:
            poses.extend(prim[1:])
        else:
            poses.extend(prim)
    if poses and tail_poses and poses[-1] == tail_poses[0]:
        poses.extend(tail_poses[1:])
    else:
        poses.extend(tail_poses)
    return KinoResult(poses=poses, length=length, expansions=expansions)
