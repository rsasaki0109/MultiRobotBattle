"""Unicycle (differential-drive) kinematics and angle helpers.

A robot pose is ``(x, y, theta)``. The unicycle model takes a linear speed
``v`` and turn rate ``omega`` and advances one timestep. Pure and ROS-free.
"""

from __future__ import annotations

import math

Pose = tuple[float, float, float]


def normalize_angle(angle: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    a = math.fmod(angle, 2.0 * math.pi)
    if a <= -math.pi:
        a += 2.0 * math.pi
    elif a > math.pi:
        a -= 2.0 * math.pi
    return a


def unicycle_step(pose: Pose, v: float, omega: float, dt: float) -> Pose:
    """Advance a unicycle one timestep.

    Uses the midpoint heading over the step so a simultaneous turn-and-drive
    integrates cleanly; reduces to straight-line motion when ``omega`` is zero.
    """
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    x, y, theta = pose
    heading = theta + 0.5 * omega * dt
    x += v * math.cos(heading) * dt
    y += v * math.sin(heading) * dt
    theta = normalize_angle(theta + omega * dt)
    return (x, y, theta)
