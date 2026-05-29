"""Pure helpers bridging the coverage allocator to ROS (no rclpy).

Parses robot start cells from ROS parameter strings and converts grid cells to
world points. Frontier detection and allocation are reused unchanged from the
coverage core.
"""

from __future__ import annotations

from .occupancy import Cell


def parse_robot_positions(robot_ids, position_strs) -> dict:
    """Build ``{id: (x, y)}`` from parallel id and ``"x,y"`` cell lists."""
    if len(robot_ids) != len(position_strs):
        raise ValueError("robot_ids and robot_positions must have equal length")
    out: dict = {}
    for rid, text in zip(robot_ids, position_strs):
        parts = str(text).split(",")
        if len(parts) != 2:
            raise ValueError(f"expected 'x,y' cell, got {text!r}")
        out[str(rid)] = (int(parts[0]), int(parts[1]))
    return out


def cell_to_world(cell: Cell, cell_size: float = 1.0, origin=(0.0, 0.0)) -> tuple:
    """Convert a grid cell to a world ``(x, y)`` point."""
    return (origin[0] + cell[0] * cell_size, origin[1] + cell[1] * cell_size)
