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


# Diagonal indices into a row-major 6x6 covariance.
_XX, _YY, _ZZ, _RR, _PP, _YAW = 0, 7, 14, 21, 28, 35


def relative_pose_observation(
    observer: Pose,
    target: Pose,
    xy_sigma: float = 0.1,
    yaw_sigma: float = 0.05,
    rng=None,
) -> tuple:
    """A noisy V2V relative-pose measurement with covariance.

    Returns ``(x, y, yaw, covariance)`` where the pose is ``target`` in the
    observer's body frame (see :func:`relative_pose_body`) and ``covariance`` is
    a row-major 6x6 list. When ``rng`` (a ``random.Random``) is given, zero-mean
    Gaussian noise is added to each measured component; the covariance always
    reflects the configured sigmas. The off-SE(2) axes (z, roll, pitch) get a
    finite placeholder variance so the covariance is valid.
    """
    x, y, yaw = relative_pose_body(observer, target)
    if rng is not None:
        x = add_gaussian_noise(x, xy_sigma, rng)
        y = add_gaussian_noise(y, xy_sigma, rng)
        yaw = normalize_angle(add_gaussian_noise(yaw, yaw_sigma, rng))

    covariance = [0.0] * 36
    covariance[_XX] = xy_sigma * xy_sigma
    covariance[_YY] = xy_sigma * xy_sigma
    covariance[_ZZ] = 1.0
    covariance[_RR] = 1.0
    covariance[_PP] = 1.0
    covariance[_YAW] = yaw_sigma * yaw_sigma
    return (x, y, yaw, covariance)


def add_gaussian_noise(value: float, sigma: float, rng) -> float:
    """Add reproducible zero-mean Gaussian noise using ``rng`` (a Random)."""
    if sigma <= 0.0:
        return value
    return value + rng.gauss(0.0, sigma)
