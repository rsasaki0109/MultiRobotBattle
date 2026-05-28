"""Decentralized formation control over V2V relative-pose measurements.

This is the coordination-layer counterpart that *reuses the localization
stack's output*: the cooperative graph already exchanges relative-pose
constraints between agents, and a displacement-based formation controller needs
exactly that — the relative position of each neighbor. Nothing here needs a
global frame; each agent acts on what it can measure of its neighbors.

The control law is the classic displacement-based consensus

    u_i = gain * sum_{j in N(i)} ( r_ij - r*_ij )

where ``r_ij = p_j - p_i`` is the measured relative position of neighbor ``j``
(what a V2V ``RelativePoseConstraint`` carries) and ``r*_ij`` is the desired
relative offset from the :class:`FormationSpec`. On a connected graph this
drives the agents into the desired shape; with no leader the formation centroid
stays put, and with a fixed leader the shape anchors to it.

Pieces:

- :mod:`spec` — the desired shape (per-agent offsets) and shape builders.
- :mod:`control` — relative measurements, the control law, and the error metric.
- :mod:`simulate` — Euler integration of the closed loop for tests and demos.
"""

from .control import (
    formation_control_from_relative,
    formation_error,
    relative_measurement,
    relative_measurements,
)
from .simulate import simulate
from .spec import FormationSpec, line_formation, polygon_formation

__all__ = [
    "FormationSpec",
    "line_formation",
    "polygon_formation",
    "relative_measurement",
    "relative_measurements",
    "formation_control_from_relative",
    "formation_error",
    "simulate",
]
