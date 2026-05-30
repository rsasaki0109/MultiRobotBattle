"""Pure-pursuit path following for a unicycle robot.

Turns a planned path (a list of world points, e.g. from the MAPF planner) plus
the robot's current pose into a unicycle command ``(v, omega)`` that drives it
along the path. Pure and ROS-free, so it is unit-tested in CI; the follower
node is a thin shell that calls :func:`pure_pursuit` each tick.

This is the unicycle-compatible controller the simulator wants: the MAPF planner
guarantees a collision-free path, and this drives the (non-holonomic) robot
along it — closing planning → world.
"""

from __future__ import annotations

import math

Pose = tuple[float, float, float]


def _normalize(angle: float) -> float:
    a = math.fmod(angle, 2.0 * math.pi)
    if a <= -math.pi:
        a += 2.0 * math.pi
    elif a > math.pi:
        a -= 2.0 * math.pi
    return a


def _closest_index(pose: Pose, path) -> int:
    px, py = pose[0], pose[1]
    best_i, best_d = 0, float("inf")
    for i, (x, y) in enumerate(path):
        d = (x - px) ** 2 + (y - py) ** 2
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def _target_point(pose: Pose, path, lookahead: float):
    """The carrot: the first path point at least ``lookahead`` ahead of the
    closest point, or the goal if none is far enough."""
    start = _closest_index(pose, path)
    px, py = pose[0], pose[1]
    for i in range(start, len(path)):
        x, y = path[i]
        if math.hypot(x - px, y - py) >= lookahead:
            return (x, y)
    return path[-1]


def pure_pursuit(
    pose: Pose,
    path,
    *,
    lookahead: float = 1.0,
    v_nominal: float = 1.0,
    goal_tolerance: float = 0.3,
    max_omega: float = 2.5,
) -> tuple:
    """Compute ``(v, omega, reached)`` to follow ``path`` from ``pose``.

    Returns ``reached=True`` (and zero command) once within ``goal_tolerance`` of
    the final point. When the carrot is behind the robot it turns in place toward
    it; otherwise it applies the pure-pursuit curvature ``2*y_body / L^2``.
    """
    if not path:
        return (0.0, 0.0, True)

    x, y, theta = pose
    gx, gy = path[-1]
    if math.hypot(gx - x, gy - y) <= goal_tolerance:
        return (0.0, 0.0, True)

    tx, ty = _target_point(pose, path, lookahead)
    dx, dy = tx - x, ty - y
    c, s = math.cos(theta), math.sin(theta)
    x_body = c * dx + s * dy
    y_body = -s * dx + c * dy

    if x_body <= 0.0:
        # carrot is behind: rotate in place toward it
        omega = max(-max_omega, min(max_omega, 2.0 * _normalize(math.atan2(dy, dx) - theta)))
        return (0.0, omega, False)

    dist = math.hypot(x_body, y_body)
    curvature = 2.0 * y_body / (dist * dist) if dist > 1e-6 else 0.0
    v = v_nominal
    omega = max(-max_omega, min(max_omega, v * curvature))
    return (v, omega, False)
