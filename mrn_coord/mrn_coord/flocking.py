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
