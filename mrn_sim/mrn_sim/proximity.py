"""Proximity queries over the world — who can sense whom.

Pure and ROS-free: given a world and a sensing radius, list the directed agent
pairs within range. Used both for emitting V2V measurements and for drawing the
sensing links.
"""

from __future__ import annotations

import math

from .world import World


def in_range_pairs(world: World, radius: float) -> list:
    """Directed ``(observer, target)`` pairs whose centers are within ``radius``.

    Both directions are returned (``(i, j)`` and ``(j, i)``) since each agent
    observes the other. Deterministic order: sorted by id.
    """
    ids = sorted(world.robots)
    pairs = []
    for i in range(len(ids)):
        for j in range(len(ids)):
            if i == j:
                continue
            a, b = world.robots[ids[i]].pose, world.robots[ids[j]].pose
            if math.hypot(a[0] - b[0], a[1] - b[1]) <= radius:
                pairs.append((ids[i], ids[j]))
    return pairs


def undirected_in_range(world: World, radius: float) -> list:
    """Unique unordered pairs within ``radius`` (for drawing links once)."""
    ids = sorted(world.robots)
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = world.robots[ids[i]].pose, world.robots[ids[j]].pose
            if math.hypot(a[0] - b[0], a[1] - b[1]) <= radius:
                pairs.append((ids[i], ids[j]))
    return pairs
