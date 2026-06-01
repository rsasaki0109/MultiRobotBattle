"""Prioritized planning: fast, incomplete MAPF.

Agents are planned one at a time in priority order. Each agent treats the
already-planned higher-priority paths as moving obstacles: it cannot occupy a
reserved cell at a reserved time, cannot enter a higher-priority agent's goal
once that agent has settled there, and cannot swap against a reserved move.

This is cheap and often good enough, but unlike :func:`cbs` it is incomplete —
it can fail (return ``None``) on instances that do have a solution, because a
bad priority order can paint a later agent into a corner.
"""

from __future__ import annotations

from .grid import GridWorld
from .solution import Solution, sum_of_costs
from .space_time_astar import plan_path


def _reservation_horizon(grid: GridWorld, agents: dict) -> int:
    return 2 * grid.width * grid.height + len(agents) + grid.width + grid.height + 5


def prioritized_planning(
    grid: GridWorld, agents: dict, order=None, *, horizon: int | None = None,
    low_level=plan_path,
):
    """Plan agents in ``order`` (default: insertion order), reserving paths.

    ``agents`` maps agent id to ``(start, goal)``. Returns a :class:`Solution`
    or ``None`` if some agent cannot be routed around the higher-priority
    reservations. ``low_level`` is the single-agent planner each agent calls
    with ``(grid, start, goal, vertex_constraints, edge_constraints)``; the
    default is time-expanded A* (:func:`plan_path`), but
    :func:`mrn_coord.mapf.sipp.plan_sipp` is a drop-in alternative that explores
    far fewer states on wait-heavy instances.
    """
    order = list(order) if order is not None else list(agents)
    if horizon is None:
        horizon = _reservation_horizon(grid, agents)

    vertex_reservations: set = set()
    edge_reservations: set = set()
    paths: dict = {}

    for agent in order:
        start, goal = agents[agent]
        # Cap the path at the reservation horizon: a path longer than the window
        # over which higher-priority agents hold their goals could slip *past*
        # that window and collide with a settled agent. Within the horizon, the
        # goal holds cover every timestep, so the result is genuinely conflict-free.
        path = low_level(
            grid,
            start,
            goal,
            frozenset(vertex_reservations),
            frozenset(edge_reservations),
            max_time=horizon,
        )
        if path is None:
            return None
        paths[agent] = path

        # Reserve every occupied (cell, time) along the path.
        for t, cell in enumerate(path):
            vertex_reservations.add((cell, t))
        # The agent waits at its goal forever — block it for all later times.
        arrival = len(path) - 1
        goal_cell = path[-1]
        for t in range(arrival, horizon + 1):
            vertex_reservations.add((goal_cell, t))
        # Forbid lower-priority agents from swapping against each move: a move
        # frm->to arriving at t+1 reserves the reverse transition to->frm at t+1.
        for t in range(len(path) - 1):
            frm, to = path[t], path[t + 1]
            edge_reservations.add((to, frm, t + 1))

    return Solution(paths=paths, cost=sum_of_costs(paths))
