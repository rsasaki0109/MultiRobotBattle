"""Point-to-point navigation in the continuous world.

The classic single-robot navigation pipeline, built from the pieces already in
the repo: discretize the world's obstacles into an occupancy grid, plan a
shortest path on it with the MAPF low-level grid A* (``plan_path`` with no
constraints), and convert the cells back to world waypoints. A pure-pursuit
follower (``mrn_coord.mapf.path_follower.pure_pursuit``) then drives the unicycle
robot along the path through the collision-aware ``world.step``.

Pure and ROS-free (it does import ``mrn_coord``); not imported by
``mrn_sim.__init__`` so the sim core stays standalone.
"""

from __future__ import annotations

import math

from mrn_coord.flocking import (
    mutual_avoidance,
    obstacle_avoidance,
    velocity_to_unicycle,
)
from mrn_coord.mapf.grid import GridWorld
from mrn_coord.mapf.path_follower import carrot_point
from mrn_coord.mapf.space_time_astar import plan_path

from .world import World, step


def world_to_cell(xy, cell_size: float) -> tuple:
    """World point -> integer grid cell (floor)."""
    return (int(math.floor(xy[0] / cell_size)), int(math.floor(xy[1] / cell_size)))


def cell_to_world(cell, cell_size: float) -> tuple:
    """Grid cell -> world point at the cell center."""
    return ((cell[0] + 0.5) * cell_size, (cell[1] + 0.5) * cell_size)


def occupancy_from_world(world: World, *, cell_size: float = 0.5, inflation: float = 0.35) -> GridWorld:
    """Discretize the world's circular obstacles into a 4-connected grid.

    A cell is blocked if its center lies within ``radius + inflation`` of any
    obstacle (the inflation accounts for the robot's footprint, so a planned
    path keeps clearance). The grid spans ``[0, width] x [0, height]``.
    """
    nx = max(1, int(math.ceil(world.width / cell_size)))
    ny = max(1, int(math.ceil(world.height / cell_size)))
    blocked = set()
    for cx in range(nx):
        for cy in range(ny):
            wx, wy = cell_to_world((cx, cy), cell_size)
            for o in world.obstacles:
                if math.hypot(wx - o.x, wy - o.y) <= o.radius + inflation:
                    blocked.add((cx, cy))
                    break
    return GridWorld(nx, ny, blocked=blocked)


def plan_world_path(
    world: World, start, goal, *, cell_size: float = 0.5, inflation: float = 0.35
) -> list | None:
    """Plan a world-space path from ``start`` to ``goal`` around the obstacles.

    Returns a list of world ``(x, y)`` waypoints (cell centers) ending at the
    exact ``goal``, or ``None`` if no path exists (or start/goal lie in an
    inflated obstacle cell). The first waypoint is the start cell center.
    """
    grid = occupancy_from_world(world, cell_size=cell_size, inflation=inflation)
    s = world_to_cell(start, cell_size)
    g = world_to_cell(goal, cell_size)
    if not grid.is_free(s) or not grid.is_free(g):
        return None
    cells = plan_path(grid, s, g)
    if cells is None:
        return None
    points = [cell_to_world(c, cell_size) for c in cells]
    points.append((float(goal[0]), float(goal[1])))   # finish at the exact goal
    return points


def navigate_step(
    world: World,
    paths: dict,
    *,
    dt: float = 0.1,
    lookahead: float = 0.9,
    max_speed: float = 1.6,
    goal_tolerance: float = 0.3,
    w_obstacle: float = 1.2,
    obstacle_influence: float = 1.5,
    obstacle_strength: float = 2.0,
    w_mutual: float = 1.5,
    mutual_radius: float = 1.4,
    max_v: float = 1.6,
    max_omega: float = 3.0,
):
    """Advance multi-robot navigation one step with reciprocal avoidance.

    Each robot is pulled toward the carrot on its own ``paths[id]`` while being
    pushed away from obstacles and from the *other robots* (so independent
    navigators don't collide). The combined velocity is realized as a unicycle
    command through the collision-aware ``step``. ``paths`` maps robot id to a
    world-waypoint list (or ``None``). Returns ``(new_world, reached)`` where
    ``reached`` maps id -> bool. Deterministic.
    """
    ids = list(world.robots)
    positions = [(world.robots[a].pose[0], world.robots[a].pose[1]) for a in ids]
    obstacles = [(o.x, o.y, o.radius) for o in world.obstacles]
    obs = obstacle_avoidance(positions, obstacles,
                             influence=obstacle_influence, strength=obstacle_strength)
    mut = mutual_avoidance(positions, radius=mutual_radius)

    reached = {}
    commands = {}
    for i, a in enumerate(ids):
        pose = world.robots[a].pose
        path = paths.get(a)
        if not path:
            reached[a] = True
            commands[a] = (0.0, 0.0)
            continue
        gx, gy = path[-1]
        if math.hypot(gx - pose[0], gy - pose[1]) <= goal_tolerance:
            reached[a] = True
            commands[a] = (0.0, 0.0)
            continue
        reached[a] = False
        cx, cy = carrot_point(pose, path, lookahead)
        dx, dy = cx - pose[0], cy - pose[1]
        d = math.hypot(dx, dy) or 1.0
        vx = dx / d * max_speed + w_obstacle * obs[i][0] + w_mutual * mut[i][0]
        vy = dy / d * max_speed + w_obstacle * obs[i][1] + w_mutual * mut[i][1]
        commands[a] = velocity_to_unicycle(pose[2], vx, vy, max_v=max_v, max_omega=max_omega)

    return step(world, commands, dt), reached
