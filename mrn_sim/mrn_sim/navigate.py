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

from mrn_coord.mapf.grid import GridWorld
from mrn_coord.mapf.space_time_astar import plan_path

from .world import World


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
