"""Tiny kinematic integration shared by the coordination simulation node.

Pure and ROS-free so it is unit-tested in CI; the agent simulator node is a thin
shell that calls :func:`euler_step` once per tick.
"""

from __future__ import annotations

import math

Vec2 = tuple[float, float]


def euler_step(position: Vec2, velocity: Vec2, dt: float, max_speed: float | None = None) -> Vec2:
    """Advance a single-integrator agent one timestep.

    ``new = position + dt * velocity``. If ``max_speed`` is given, the velocity
    magnitude is clamped first so the simulated agent can't teleport on a large
    command.
    """
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    vx, vy = velocity
    if max_speed is not None and max_speed >= 0.0:
        speed = math.hypot(vx, vy)
        if speed > max_speed and speed > 0.0:
            scale = max_speed / speed
            vx, vy = vx * scale, vy * scale
    return (position[0] + dt * vx, position[1] + dt * vy)
