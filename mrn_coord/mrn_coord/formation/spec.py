"""Desired formation shape: per-agent offsets in a formation frame.

A :class:`FormationSpec` stores, for each agent, its target position in an
abstract *formation frame*. Only **relative** offsets matter to the controller
(the absolute frame origin is unobservable from relative measurements), so the
spec is used through :meth:`FormationSpec.desired_relative`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Vec2 = tuple[float, float]


@dataclass(frozen=True)
class FormationSpec:
    """Target offsets ``agent -> (x, y)`` defining the formation shape."""

    offsets: dict

    def agents(self) -> list:
        return list(self.offsets)

    def desired_relative(self, i, j) -> Vec2:
        """Desired relative position of ``j`` as seen from ``i`` (``c_j - c_i``)."""
        ci = self.offsets[i]
        cj = self.offsets[j]
        return (cj[0] - ci[0], cj[1] - ci[1])


def line_formation(agents, spacing: float = 1.0, axis: Vec2 = (1.0, 0.0)) -> FormationSpec:
    """Place agents evenly along a line through the origin.

    Agent ``k`` (in iteration order) sits at ``k * spacing`` along the unit
    ``axis`` direction.
    """
    ax, ay = axis
    norm = math.hypot(ax, ay)
    if norm == 0.0:
        raise ValueError("axis must be non-zero")
    ux, uy = ax / norm, ay / norm
    offsets = {
        agent: (k * spacing * ux, k * spacing * uy)
        for k, agent in enumerate(agents)
    }
    return FormationSpec(offsets)


def polygon_formation(agents, radius: float = 1.0, phase: float = 0.0) -> FormationSpec:
    """Place agents on a regular polygon (circle) of the given ``radius``.

    Agent ``k`` is at angle ``phase + 2*pi*k/n``. With three agents this is an
    equilateral triangle — a recognizable formation for demos.
    """
    agents = list(agents)
    n = len(agents)
    if n == 0:
        return FormationSpec({})
    offsets = {}
    for k, agent in enumerate(agents):
        theta = phase + 2.0 * math.pi * k / n
        offsets[agent] = (radius * math.cos(theta), radius * math.sin(theta))
    return FormationSpec(offsets)
