"""Geometric sensor models over the true world state.

These produce the measurements the localization stack consumes. The geometry is
pure and noiseless (so it is exactly testable); :func:`add_gaussian_noise`
applies reproducible noise on top using a caller-supplied ``random.Random``.

- :func:`range_bearing` — range and body-frame bearing to a target point
  (a range-bearing radio, like the UWB constraint source).
- :func:`relative_pose_body` — the full SE(2) relative pose of another robot in
  the observer's body frame (what a V2V ``RelativePoseConstraint`` carries).
- :func:`gnss_observation` — the absolute ``(x, y)`` of a robot (a GNSS fix).
"""

from __future__ import annotations

import math

from .kinematics import Pose, normalize_angle


def range_bearing(observer: Pose, target_xy: tuple) -> tuple:
    """Range and bearing (in the observer's body frame) to ``target_xy``."""
    ox, oy, otheta = observer
    dx = target_xy[0] - ox
    dy = target_xy[1] - oy
    rng = math.hypot(dx, dy)
    bearing = normalize_angle(math.atan2(dy, dx) - otheta)
    return (rng, bearing)


def relative_pose_body(observer: Pose, target: Pose) -> Pose:
    """SE(2) pose of ``target`` expressed in ``observer``'s body frame.

    Returns ``(x_body, y_body, dtheta)``: rotate the world displacement into the
    observer frame and take the relative heading. This is the cooperative V2V
    measurement.
    """
    ox, oy, otheta = observer
    tx, ty, ttheta = target
    dx = tx - ox
    dy = ty - oy
    c = math.cos(otheta)
    s = math.sin(otheta)
    x_body = c * dx + s * dy
    y_body = -s * dx + c * dy
    return (x_body, y_body, normalize_angle(ttheta - otheta))


def gnss_observation(pose: Pose) -> tuple:
    """The absolute ``(x, y)`` position of a robot (noiseless geometry)."""
    return (pose[0], pose[1])


def add_gaussian_noise(value: float, sigma: float, rng) -> float:
    """Add reproducible zero-mean Gaussian noise using ``rng`` (a Random)."""
    if sigma <= 0.0:
        return value
    return value + rng.gauss(0.0, sigma)
