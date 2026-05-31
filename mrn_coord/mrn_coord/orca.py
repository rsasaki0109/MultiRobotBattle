"""Optimal Reciprocal Collision Avoidance (ORCA) — local multi-agent avoidance.

ORCA (van den Berg, Guy, Lin & Manocha, 2011) is the de-facto standard for
distributed local collision avoidance: each agent picks, every tick, the
velocity closest to its preferred velocity that *guarantees* collision-free
motion for a time horizon, assuming every other agent reasons symmetrically.
That reciprocity is what dissolves the oscillation and deadlock you get from
naive pairwise repulsion (the ``mutual_avoidance`` baseline elsewhere in this
repo).

For each neighbour, ORCA forbids a half-plane of velocities; the admissible set
is the intersection of those half-planes (plus the max-speed disc). Picking the
closest point to the preferred velocity is a 2-D linear program, solved
incrementally (``_linear_program2``) with a distance-minimising fallback when
the constraints are jointly infeasible (``_linear_program3``) — both ported
faithfully from the reference RVO2 implementation.

Pure and deterministic. :func:`orca_velocity` is the whole interface: give it an
agent's state, its preferred velocity, and its neighbours / static obstacles,
and it returns the new velocity. Convert it to a unicycle command with
:func:`mrn_coord.flocking.velocity_to_unicycle`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_EPS = 1e-9


@dataclass(frozen=True)
class Line:
    """A directed line; the feasible half-plane is to its left.

    ``(px, py)`` is a point on the line and ``(dx, dy)`` its unit direction. A
    velocity ``v`` is admissible w.r.t. this line iff ``det(dir, v - point) >= 0``.
    """

    px: float
    py: float
    dx: float
    dy: float


def _det(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _orca_line(rel_pos, rel_vel, combined_r, inv_tau, inv_dt, responsibility, vel):
    """Build the ORCA half-plane a single neighbour imposes on the agent.

    ``rel_pos`` / ``rel_vel`` are (other - self) position and (self - other)
    velocity; ``responsibility`` is the share of the avoidance this agent takes
    (0.5 for a reciprocal agent, 1.0 for a static obstacle). Mirrors the cut-off
    circle / legs / collision cases of the RVO2 reference.
    """
    px, py = rel_pos
    vx, vy = rel_vel
    dist_sq = px * px + py * py
    combined_sq = combined_r * combined_r

    if dist_sq > combined_sq:
        # No collision yet — VO is a cone truncated by a circle at the horizon.
        wx = vx - inv_tau * px
        wy = vy - inv_tau * py
        w_len_sq = wx * wx + wy * wy
        dot1 = wx * px + wy * py
        if dot1 < 0.0 and dot1 * dot1 > combined_sq * w_len_sq:
            # Project on the cut-off circle.
            w_len = math.sqrt(w_len_sq)
            ux, uy = wx / w_len, wy / w_len
            dir_x, dir_y = uy, -ux
            scale = combined_r * inv_tau - w_len
            u = (ux * scale, uy * scale)
        else:
            # Project on the nearer leg of the cone.
            leg = math.sqrt(dist_sq - combined_sq)
            if _det(px, py, wx, wy) > 0.0:
                dir_x = (px * leg - py * combined_r) / dist_sq
                dir_y = (px * combined_r + py * leg) / dist_sq
            else:
                dir_x = -(px * leg + py * combined_r) / dist_sq
                dir_y = -(-px * combined_r + py * leg) / dist_sq
            dot2 = vx * dir_x + vy * dir_y
            u = (dot2 * dir_x - vx, dot2 * dir_y - vy)
    else:
        # Already overlapping — project onto the cut-off circle for this step.
        wx = vx - inv_dt * px
        wy = vy - inv_dt * py
        w_len = math.hypot(wx, wy) or _EPS
        ux, uy = wx / w_len, wy / w_len
        dir_x, dir_y = uy, -ux
        scale = combined_r * inv_dt - w_len
        u = (ux * scale, uy * scale)

    return Line(vel[0] + responsibility * u[0], vel[1] + responsibility * u[1], dir_x, dir_y)


def _linear_program1(lines, i, radius, opt, result):
    """Optimise along line ``i`` subject to constraints ``0..i-1`` and the disc.

    Returns ``(ok, result)``; ``ok`` is False if line ``i`` is infeasible given
    the earlier constraints. ``opt`` is the preferred velocity.
    """
    li = lines[i]
    dot = li.px * li.dx + li.py * li.dy
    discriminant = dot * dot + radius * radius - (li.px * li.px + li.py * li.py)
    if discriminant < 0.0:
        return False, result
    sqrt_disc = math.sqrt(discriminant)
    t_left = -dot - sqrt_disc
    t_right = -dot + sqrt_disc
    for k in range(i):
        lk = lines[k]
        denom = _det(li.dx, li.dy, lk.dx, lk.dy)
        numer = _det(lk.dx, lk.dy, li.px - lk.px, li.py - lk.py)
        if abs(denom) <= _EPS:
            if numer < 0.0:
                return False, result
            continue
        t = numer / denom
        if denom >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return False, result
    # Optimise closest point to the preferred velocity along the line segment.
    t = li.dx * (opt[0] - li.px) + li.dy * (opt[1] - li.py)
    t = max(t_left, min(t_right, t))
    return True, (li.px + t * li.dx, li.py + t * li.dy)


def _linear_program2(lines, radius, opt):
    """Closest admissible velocity to ``opt`` within the disc + all half-planes.

    Returns ``(fail, result)`` where ``fail`` is the number of lines satisfied
    before the first infeasible one (== ``len(lines)`` means full success).
    """
    speed_sq = opt[0] * opt[0] + opt[1] * opt[1]
    if speed_sq > radius * radius:
        s = radius / math.sqrt(speed_sq)
        result = (opt[0] * s, opt[1] * s)
    else:
        result = (opt[0], opt[1])
    for i, li in enumerate(lines):
        if _det(li.dx, li.dy, li.px - result[0], li.py - result[1]) > 0.0:
            ok, candidate = _linear_program1(lines, i, radius, opt, result)
            if not ok:
                return i, result
            result = candidate
    return len(lines), result


def _linear_program3(lines, begin, radius, result):
    """Distance-minimising fallback when the half-planes are jointly infeasible.

    Pushes ``result`` out of every violated constraint by the least amount,
    keeping it on the max-speed disc. Mirrors the RVO2 dense fallback.
    """
    distance = 0.0
    for i in range(begin, len(lines)):
        li = lines[i]
        if _det(li.dx, li.dy, li.px - result[0], li.py - result[1]) <= distance:
            continue
        proj = []
        for j in range(i):
            lj = lines[j]
            determinant = _det(li.dx, li.dy, lj.dx, lj.dy)
            if abs(determinant) <= _EPS:
                if li.dx * lj.dx + li.dy * lj.dy > 0.0:
                    continue
                px = 0.5 * (li.px + lj.px)
                py = 0.5 * (li.py + lj.py)
            else:
                f = _det(lj.dx, lj.dy, li.px - lj.px, li.py - lj.py) / determinant
                px = li.px + f * li.dx
                py = li.py + f * li.dy
            ddx, ddy = lj.dx - li.dx, lj.dy - li.dy
            dlen = math.hypot(ddx, ddy)
            if dlen <= _EPS:
                continue
            proj.append(Line(px, py, ddx / dlen, ddy / dlen))
        opt_dir = (-li.dy, li.dx)
        # Optimise *direction* (push out maximally) along the line normal.
        fail, candidate = _linear_program2_dir(proj, radius, opt_dir, result)
        if fail >= len(proj):
            result = candidate
        distance = _det(li.dx, li.dy, li.px - result[0], li.py - result[1])
    return result


def _linear_program2_dir(lines, radius, opt_dir, fallback):
    """``_linear_program2`` variant that optimises a direction (for LP3)."""
    result = (opt_dir[0] * radius, opt_dir[1] * radius)
    for i, li in enumerate(lines):
        if _det(li.dx, li.dy, li.px - result[0], li.py - result[1]) > 0.0:
            lj = li
            dot = lj.px * lj.dx + lj.py * lj.dy
            discriminant = dot * dot + radius * radius - (lj.px * lj.px + lj.py * lj.py)
            if discriminant < 0.0:
                return i, fallback
            sqrt_disc = math.sqrt(discriminant)
            t_left = -dot - sqrt_disc
            t_right = -dot + sqrt_disc
            ok = True
            for k in range(i):
                lk = lines[k]
                denom = _det(lj.dx, lj.dy, lk.dx, lk.dy)
                numer = _det(lk.dx, lk.dy, lj.px - lk.px, lj.py - lk.py)
                if abs(denom) <= _EPS:
                    if numer < 0.0:
                        ok = False
                        break
                    continue
                t = numer / denom
                if denom >= 0.0:
                    t_right = min(t_right, t)
                else:
                    t_left = max(t_left, t)
                if t_left > t_right:
                    ok = False
                    break
            if not ok:
                return i, fallback
            t = lj.dx * opt_dir[0] + lj.dy * opt_dir[1]
            t = t_right if t > 0.0 else t_left
            result = (lj.px + t * lj.dx, lj.py + t * lj.dy)
    return len(lines), result


def orca_velocity(
    position,
    velocity,
    pref_velocity,
    neighbors=(),
    obstacles=(),
    *,
    radius: float = 0.25,
    max_speed: float = 1.5,
    time_horizon: float = 2.0,
    time_horizon_obst: float = 1.0,
    time_step: float = 0.1,
) -> tuple:
    """Return the new ORCA velocity for one agent.

    - ``neighbors``: iterable of ``(position, velocity, radius)`` moving agents
      (reciprocal — each side takes half the avoidance).
    - ``obstacles``: iterable of ``(x, y, radius)`` static circles (this agent
      takes the full avoidance, with a shorter horizon).

    The result is the admissible velocity closest to ``pref_velocity``, capped
    at ``max_speed``. Pure and deterministic.
    """
    inv_tau = 1.0 / time_horizon
    inv_tau_obst = 1.0 / time_horizon_obst
    inv_dt = 1.0 / time_step
    lines = []

    # Static obstacles first (full responsibility, shorter horizon).
    for (ox, oy, orad) in obstacles:
        rel_pos = (ox - position[0], oy - position[1])
        rel_vel = (velocity[0], velocity[1])  # obstacle velocity is zero
        lines.append(_orca_line(rel_pos, rel_vel, radius + orad,
                                inv_tau_obst, inv_dt, 1.0, velocity))
    num_obst = len(lines)

    for (npos, nvel, nrad) in neighbors:
        rel_pos = (npos[0] - position[0], npos[1] - position[1])
        rel_vel = (velocity[0] - nvel[0], velocity[1] - nvel[1])
        lines.append(_orca_line(rel_pos, rel_vel, radius + nrad,
                                inv_tau, inv_dt, 0.5, velocity))

    fail, result = _linear_program2(lines, max_speed, pref_velocity)
    if fail < len(lines):
        result = _linear_program3(lines, fail, max_speed, result)
    return result
