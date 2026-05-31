"""Boids flocking — decentralized swarm behavior.

Each agent steers from only its local neighbors via the three classic rules:

- **separation**: steer away from neighbors that are too close;
- **alignment**: match the average heading (velocity) of nearby neighbors;
- **cohesion**: steer toward the average position of nearby neighbors.

:func:`flock_velocities` is a pure, ROS-free reactive step: given all positions
and velocities it returns each agent's new velocity (clamped to ``max_speed``).
It scales to tens or hundreds of agents and is the swarm counterpart to the
small-team coordination in :mod:`mrn_coord.mapf` / :mod:`mrn_coord.formation`.
"""

from __future__ import annotations

import math

Vec2 = tuple[float, float]


def _normalize_angle(angle: float) -> float:
    a = math.fmod(angle, 2.0 * math.pi)
    if a <= -math.pi:
        a += 2.0 * math.pi
    elif a > math.pi:
        a -= 2.0 * math.pi
    return a


def velocity_to_unicycle(
    yaw: float, vx: float, vy: float, *,
    max_v: float = 1.5, max_omega: float = 2.5, k_omega: float = 3.0,
) -> Vec2:
    """Convert a desired holonomic velocity into a unicycle ``(v, omega)``.

    Drives forward in proportion to how well the body x-axis already faces the
    desired direction (``v = speed * max(0, cos(heading_error))``) and turns
    toward it (``omega = k_omega * heading_error``, clamped). A near-zero desired
    velocity yields a zero command. This is what lets a holonomic controller
    (e.g. Boids) steer a differential-drive robot.
    """
    speed = math.hypot(vx, vy)
    if speed < 1e-6:
        return (0.0, 0.0)
    err = _normalize_angle(math.atan2(vy, vx) - yaw)
    v = min(max_v, speed) * max(0.0, math.cos(err))
    omega = max(-max_omega, min(max_omega, k_omega * err))
    return (v, omega)


def goal_seek(positions, goal, *, gain: float = 1.0, max_speed: float = 2.0) -> list:
    """Per-agent migration velocity toward a shared ``goal`` point.

    Each agent gets ``gain * (goal - position)`` clamped to ``max_speed`` — a
    directed pull that turns aimless flocking into migration. Near the goal the
    pull shrinks to zero. ``goal`` is an ``(x, y)`` tuple.
    """
    gx, gy = goal
    out = []
    for (px, py) in positions:
        out.append(_clamp_speed(gain * (gx - px), gain * (gy - py), max_speed))
    return out


def obstacle_avoidance(
    positions,
    obstacles,
    *,
    influence: float = 2.0,
    strength: float = 2.0,
    max_accel: float = 6.0,
) -> list:
    """Repulsion from circular obstacles, per agent.

    ``obstacles`` is a list of ``(x, y, radius)``. For each agent, every
    obstacle whose surface is within ``influence`` contributes an outward push
    that grows as the clearance shrinks (``strength / clearance**2``), clamped
    per agent to ``max_accel``. Returns a list of ``(ax, ay)`` to add to the
    flocking velocity. Pure — takes plain tuples, not world objects.
    """
    out = []
    for (px, py) in positions:
        ax = ay = 0.0
        for (ox, oy, r) in obstacles:
            dx, dy = px - ox, py - oy
            d = math.hypot(dx, dy)
            clearance = d - r
            if clearance < influence:
                w = strength / (max(clearance, 0.2) ** 2)
                if d > 1e-9:
                    ax += (dx / d) * w
                    ay += (dy / d) * w
                else:
                    ax += w  # exactly at center: push +x arbitrarily
        mag = math.hypot(ax, ay)
        if mag > max_accel and mag > 0.0:
            ax, ay = ax / mag * max_accel, ay / mag * max_accel
        out.append((ax, ay))
    return out


def predator_evasion(
    positions,
    predator,
    *,
    influence: float = 6.0,
    strength: float = 5.0,
    max_accel: float = 8.0,
) -> list:
    """Per-agent flee acceleration away from a ``predator`` point.

    Like :func:`obstacle_avoidance` but for a single, scarier point with a
    longer reach: every agent within ``influence`` of ``predator`` gets an
    outward push that grows as it gets closer (``strength / dist**2``), clamped
    to ``max_accel``. ``predator`` is an ``(x, y)`` tuple. Returns ``(ax, ay)``
    per agent (zero when out of range).
    """
    qx, qy = predator
    out = []
    for (px, py) in positions:
        dx, dy = px - qx, py - qy
        d = math.hypot(dx, dy)
        if d < influence:
            w = strength / (max(d, 0.3) ** 2)
            if d > 1e-9:
                ax, ay = (dx / d) * w, (dy / d) * w
            else:
                ax, ay = w, 0.0   # exactly on the predator: flee +x arbitrarily
            mag = math.hypot(ax, ay)
            if mag > max_accel and mag > 0.0:
                ax, ay = ax / mag * max_accel, ay / mag * max_accel
            out.append((ax, ay))
        else:
            out.append((0.0, 0.0))
    return out


def _clamp_speed(vx: float, vy: float, max_speed: float) -> Vec2:
    speed = math.hypot(vx, vy)
    if max_speed > 0.0 and speed > max_speed and speed > 0.0:
        scale = max_speed / speed
        return (vx * scale, vy * scale)
    return (vx, vy)


def flock_velocities(
    positions,
    velocities,
    *,
    perception: float = 3.0,
    separation: float = 1.0,
    w_sep: float = 1.6,
    w_ali: float = 1.0,
    w_coh: float = 1.0,
    inertia: float = 0.85,
    max_speed: float = 2.0,
) -> list:
    """One reactive Boids step → each agent's new velocity.

    ``positions`` and ``velocities`` are parallel lists of ``(x, y)``. For each
    agent the three rules are computed over neighbors within ``perception``
    (separation only over neighbors within ``separation``), combined with the
    agent's own velocity scaled by ``inertia``, and clamped to ``max_speed``.
    """
    n = len(positions)
    new_velocities = []
    for i in range(n):
        px, py = positions[i]
        vx, vy = velocities[i]
        sep_x = sep_y = 0.0
        ali_x = ali_y = 0.0
        coh_x = coh_y = 0.0
        count = 0
        for j in range(n):
            if j == i:
                continue
            qx, qy = positions[j]
            dx, dy = px - qx, py - qy
            dist = math.hypot(dx, dy)
            if dist > perception or dist == 0.0:
                continue
            count += 1
            coh_x += qx
            coh_y += qy
            ali_x += velocities[j][0]
            ali_y += velocities[j][1]
            if dist < separation:
                # push away, stronger the closer it is
                sep_x += dx / dist
                sep_y += dy / dist

        ax = inertia * vx + w_sep * sep_x
        ay = inertia * vy + w_sep * sep_y
        if count > 0:
            # cohesion: toward the neighbor centroid
            ax += w_coh * (coh_x / count - px)
            ay += w_coh * (coh_y / count - py)
            # alignment: toward the average neighbor velocity
            ax += w_ali * (ali_x / count - vx)
            ay += w_ali * (ali_y / count - vy)

        new_velocities.append(_clamp_speed(ax, ay, max_speed))
    return new_velocities
