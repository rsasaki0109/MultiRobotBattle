"""Pure helpers bridging the formation controller to ROS (no rclpy).

Parses the formation spec and edge list from ROS parameter strings. The control
law itself lives in :mod:`mrn_coord.formation.control` and is reused unchanged.
"""

from __future__ import annotations

from .spec import FormationSpec


def parse_offsets(agent_ids, offset_strs) -> FormationSpec:
    """Build a :class:`FormationSpec` from parallel id and ``"x,y"`` lists."""
    if len(agent_ids) != len(offset_strs):
        raise ValueError("agent_ids and formation_offsets must have equal length")
    offsets: dict = {}
    for agent, text in zip(agent_ids, offset_strs):
        parts = str(text).split(",")
        if len(parts) != 2:
            raise ValueError(f"expected 'x,y' offset, got {text!r}")
        offsets[str(agent)] = (float(parts[0]), float(parts[1]))
    return FormationSpec(offsets)


def parse_edges(edge_strs) -> list:
    """Parse ``"i,j"`` strings into undirected agent-id edges (blanks skipped)."""
    edges = []
    for text in edge_strs:
        if not str(text):
            continue
        parts = str(text).split(",")
        if len(parts) != 2:
            raise ValueError(f"expected 'i,j' edge, got {text!r}")
        edges.append((parts[0].strip(), parts[1].strip()))
    return edges
