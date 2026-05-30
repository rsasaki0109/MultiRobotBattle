"""The 2D world: robots, circular obstacles, bounds, and a collision-aware step.

The world holds the *true* state of every robot. :func:`step` advances all
robots by their commanded ``(v, omega)`` under the unicycle model and rejects
any move that would leave the bounds or enter an obstacle (the robot holds its
previous pose for that step). Deterministic and pure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .kinematics import Pose, unicycle_step


@dataclass(frozen=True)
class Robot:
    """A robot's true pose and physical radius."""

    robot_id: str
    pose: Pose
    radius: float = 0.25


@dataclass(frozen=True)
class Obstacle:
    """A circular obstacle."""

    x: float
    y: float
    radius: float

    def blocks(self, px: float, py: float, robot_radius: float) -> bool:
        return math.hypot(px - self.x, py - self.y) <= self.radius + robot_radius


@dataclass
class World:
    """Robots, obstacles, and rectangular bounds ``[0, width] x [0, height]``."""

    width: float
    height: float
    robots: dict = field(default_factory=dict)        # id -> Robot
    obstacles: list = field(default_factory=list)

    def in_bounds(self, px: float, py: float, robot_radius: float) -> bool:
        return (
            robot_radius <= px <= self.width - robot_radius
            and robot_radius <= py <= self.height - robot_radius
        )

    def is_free(self, px: float, py: float, robot_radius: float) -> bool:
        if not self.in_bounds(px, py, robot_radius):
            return False
        return not any(o.blocks(px, py, robot_radius) for o in self.obstacles)


def step(world: World, commands: dict, dt: float) -> World:
    """Advance every robot by its ``(v, omega)`` command, honoring collisions.

    ``commands`` maps robot id to ``(v, omega)`` (missing robots hold still). A
    proposed pose that leaves the bounds or enters an obstacle is rejected and
    the robot keeps its previous pose for that step. Returns a new :class:`World`.
    """
    new_robots: dict = {}
    for rid, robot in world.robots.items():
        v, omega = commands.get(rid, (0.0, 0.0))
        nx, ny, ntheta = unicycle_step(robot.pose, v, omega, dt)
        if world.is_free(nx, ny, robot.radius):
            new_robots[rid] = Robot(rid, (nx, ny, ntheta), robot.radius)
        else:
            # blocked: keep position, but still allow the turn to take effect
            _, _, turned = unicycle_step(robot.pose, 0.0, omega, dt)
            new_robots[rid] = Robot(rid, (robot.pose[0], robot.pose[1], turned), robot.radius)
    return World(world.width, world.height, new_robots, list(world.obstacles))
