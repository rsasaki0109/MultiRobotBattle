"""Dynamic Window Approach (DWA): a reactive local controller.

Where pure-pursuit (``mrn_coord.mapf.path_follower``) blindly steers at the
carrot and leans on a separate repulsion term to dodge obstacles, DWA chooses
the command by **forward-simulating candidate velocities** and scoring the
resulting short trajectories — so obstacle avoidance and goal progress are
decided together, with the robot's acceleration limits respected.

Each tick it samples ``(v, omega)`` pairs from the *dynamic window* — the
velocities reachable from the current ``(v, omega)`` within one control step
given ``accel_v`` / ``accel_omega`` — rolls each out over a short horizon under
the unicycle model, discards any rollout that hits an obstacle or leaves the
world, and scores the survivors by a weighted sum of three classic DWA terms:

* **heading** — progress toward the (local) goal,
* **clearance** — distance kept from the nearest obstacle,
* **velocity** — preference for moving fast.

Pure and deterministic, depending only on :mod:`mrn_sim.world` /
:mod:`mrn_sim.kinematics` (no ``mrn_coord``). Pair it with a global planner by
feeding the carrot on a planned path as the local goal: the plan gives the
route, DWA handles smooth, accel-limited, obstacle-reactive tracking — including
obstacles that were not in the plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .kinematics import unicycle_step
from .world import World


@dataclass(frozen=True)
class DWAConfig:
    """Tuning for :func:`dwa_command`. Defaults suit the 2D demo world."""

    max_v: float = 1.6
    min_v: float = 0.0
    max_omega: float = 3.0
    accel_v: float = 2.5            # m/s^2
    accel_omega: float = 6.0        # rad/s^2
    v_samples: int = 7
    omega_samples: int = 15
    dt: float = 0.1                 # control period (matches world step)
    predict_time: float = 1.2       # rollout horizon (s)
    robot_radius: float = 0.25
    clearance_margin: float = 0.15  # extra buffer beyond robot_radius
    w_heading: float = 0.8
    w_clearance: float = 0.25
    w_velocity: float = 0.15
    goal_tolerance: float = 0.3


def _rollout(pose, v, omega, cfg: DWAConfig):
    """Forward-simulate a constant ``(v, omega)`` over the horizon; return poses."""
    n = max(1, int(round(cfg.predict_time / cfg.dt)))
    poses = []
    p = pose
    for _ in range(n):
        p = unicycle_step(p, v, omega, cfg.dt)
        poses.append(p)
    return poses


def _min_clearance(poses, obstacles, world: World, robot_radius: float) -> float:
    """Smallest surface clearance over a rollout; negative means a collision."""
    best = float("inf")
    for (x, y, _th) in poses:
        if not world.in_bounds(x, y, robot_radius):
            return -1.0
        for (ox, oy, r) in obstacles:
            c = math.hypot(x - ox, y - oy) - r - robot_radius
            if c < best:
                best = c
    return best


def dwa_command(
    pose,
    v_cur: float,
    omega_cur: float,
    goal,
    obstacles,
    world: World,
    cfg: DWAConfig = DWAConfig(),
):
    """Pick the best accel-limited ``(v, omega)`` toward ``goal`` avoiding obstacles.

    ``pose`` is ``(x, y, theta)``; ``v_cur`` / ``omega_cur`` the current command
    (defining the dynamic window); ``goal`` a ``(x, y)`` local target (e.g. the
    carrot on a global path); ``obstacles`` a list of ``(x, y, radius)``;
    ``world`` supplies the bounds. Returns ``(v, omega)``. If no sampled
    trajectory is collision-free, returns a decelerating in-place rotation toward
    the goal so the robot slows and reorients rather than charging ahead.
    """
    gx, gy = goal[0], goal[1]

    # dynamic window: reachable velocities this control step, clamped to limits.
    v_lo = max(cfg.min_v, v_cur - cfg.accel_v * cfg.dt)
    v_hi = min(cfg.max_v, v_cur + cfg.accel_v * cfg.dt)
    w_lo = max(-cfg.max_omega, omega_cur - cfg.accel_omega * cfg.dt)
    w_hi = min(cfg.max_omega, omega_cur + cfg.accel_omega * cfg.dt)

    margin = cfg.robot_radius + cfg.clearance_margin

    def lin(a, b, n):
        if n <= 1:
            return [0.5 * (a + b)]
        return [a + (b - a) * i / (n - 1) for i in range(n)]

    best = None  # (score, v, omega)
    for v in lin(v_lo, v_hi, cfg.v_samples):
        for omega in lin(w_lo, w_hi, cfg.omega_samples):
            poses = _rollout(pose, v, omega, cfg)
            clear = _min_clearance(poses, obstacles, world, margin)
            if clear < 0.0:
                continue  # rollout collides or leaves the world
            ex, ey, eth = poses[-1]
            # heading term: 1 when the end pose points straight at the goal.
            to_goal = math.atan2(gy - ey, gx - ex)
            heading = 1.0 - abs(_wrap(to_goal - eth)) / math.pi
            # progress term: how much closer the end pose is to the goal.
            d0 = math.hypot(gx - pose[0], gy - pose[1])
            d1 = math.hypot(gx - ex, gy - ey)
            progress = (d0 - d1) / (cfg.max_v * cfg.predict_time + 1e-9)
            clearance = min(1.0, clear / 1.0)        # saturate at 1 m
            velocity = v / cfg.max_v if cfg.max_v > 0 else 0.0
            score = (
                cfg.w_heading * (0.5 * heading + 0.5 * max(0.0, progress))
                + cfg.w_clearance * clearance
                + cfg.w_velocity * velocity
            )
            if best is None or score > best[0]:
                best = (score, v, omega)

    if best is None:
        # everything collides: brake to a stop while rotating toward the goal.
        to_goal = math.atan2(gy - pose[1], gx - pose[0])
        turn = _wrap(to_goal - pose[2])
        omega = max(-cfg.max_omega, min(cfg.max_omega, 2.0 * turn))
        return (0.0, omega)

    return (best[1], best[2])


def _wrap(a: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    a = math.fmod(a + math.pi, 2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi
