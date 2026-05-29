"""Pure helpers bridging the MAPF core to ROS, kept free of rclpy.

These do the parsing and grid-to-world conversion that the planner node needs,
but contain no ROS imports, so they are unit-tested in CI just like the rest of
the MAPF core. The node (:mod:`planner_node`) is a thin shell over them.
"""

from __future__ import annotations

from .cbs import cbs
from .grid import Cell, GridWorld
from .prioritized import prioritized_planning


def safe_topic_token(agent_id: str) -> str:
    """A ROS-valid topic token for an agent id (must not start with a digit)."""
    s = str(agent_id)
    if s and (s[0].isalpha() or s[0] == "_"):
        return s
    return "a_" + s


def parse_cell(text: str) -> Cell:
    """Parse a ``"x,y"`` string into an integer cell."""
    parts = str(text).split(",")
    if len(parts) != 2:
        raise ValueError(f"expected 'x,y', got {text!r}")
    return (int(parts[0]), int(parts[1]))


def parse_cells(items) -> set:
    """Parse an iterable of ``"x,y"`` strings into a set of cells."""
    return {parse_cell(item) for item in items}


def build_agents(agent_ids, starts, goals) -> dict:
    """Zip parallel id/start/goal lists into a ``{id: (start, goal)}`` dict."""
    if not (len(agent_ids) == len(starts) == len(goals)):
        raise ValueError("agent_ids, starts, and goals must have equal length")
    return {
        str(a): (parse_cell(s), parse_cell(g))
        for a, s, g in zip(agent_ids, starts, goals)
    }


def solve_scenario(
    width: int,
    height: int,
    blocked,
    agents: dict,
    *,
    solver: str = "cbs",
    max_expansions: int = 100_000,
):
    """Build the grid and solve with the named solver.

    Returns a :class:`Solution` or ``None``. ``solver`` is ``"cbs"`` (optimal)
    or ``"prioritized"`` (fast, incomplete).
    """
    grid = GridWorld(width, height, blocked=set(blocked))
    if solver == "cbs":
        return cbs(grid, agents, max_expansions=max_expansions)
    if solver == "prioritized":
        return prioritized_planning(grid, agents)
    raise ValueError(f"unknown solver: {solver!r}")


def path_to_world_points(
    cells, cell_size: float = 1.0, origin=(0.0, 0.0)
) -> list:
    """Convert a list of grid cells to world ``(x, y)`` points.

    ``world = origin + cell * cell_size`` (cell centers if ``origin`` is the
    grid origin). Suitable for filling a ``nav_msgs/Path``.
    """
    ox, oy = origin
    return [(ox + c[0] * cell_size, oy + c[1] * cell_size) for c in cells]


def yaw_along(points) -> list:
    """Heading at each point, facing the next one (last holds the previous)."""
    import math

    if not points:
        return []
    yaws = []
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        yaws.append(yaws[-1] if (dx == 0.0 and dy == 0.0) and yaws else math.atan2(dy, dx))
    yaws.append(yaws[-1] if yaws else 0.0)
    return yaws
