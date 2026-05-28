"""The displacement-based formation control law and its error metric.

Everything here is decentralized: an agent's command depends only on the
relative positions of its neighbors (``r_ij = p_j - p_i``) and the desired
offsets from the :class:`FormationSpec`. That relative measurement is exactly
what a V2V ``RelativePoseConstraint`` provides, so the same input that feeds the
cooperative-localization graph also drives the formation.
"""

from __future__ import annotations

import math

from .spec import FormationSpec, Vec2


def relative_measurement(positions: dict, i, j) -> Vec2:
    """Relative position of ``j`` seen from ``i``: ``p_j - p_i``."""
    pi = positions[i]
    pj = positions[j]
    return (pj[0] - pi[0], pj[1] - pi[1])


def relative_measurements(positions: dict, edges) -> dict:
    """Build both directed measurements for every undirected edge.

    Returns a dict keyed by ``(i, j)`` so the control law can look up each
    agent's neighbors. ``r_ji == -r_ij`` by construction.
    """
    meas: dict = {}
    for i, j in edges:
        meas[(i, j)] = relative_measurement(positions, i, j)
        meas[(j, i)] = relative_measurement(positions, j, i)
    return meas


def _neighbors(meas: dict) -> dict:
    adjacency: dict = {}
    for (i, j) in meas:
        adjacency.setdefault(i, []).append(j)
    return adjacency


def formation_control_from_relative(
    meas: dict, spec: FormationSpec, gain: float = 1.0, *, fixed=()
) -> dict:
    """Command velocity per agent from relative measurements.

    ``meas`` maps ``(i, j) -> r_ij``; ``spec`` provides the desired offsets.
    Agents in ``fixed`` (e.g. a leader) are commanded zero so they follow their
    own motion instead of the formation law.
    """
    fixed = set(fixed)
    adjacency = _neighbors(meas)
    commands: dict = {}
    for i, neighbors in adjacency.items():
        if i in fixed:
            commands[i] = (0.0, 0.0)
            continue
        ux = uy = 0.0
        for j in neighbors:
            rx, ry = meas[(i, j)]
            dx, dy = spec.desired_relative(i, j)
            ux += rx - dx
            uy += ry - dy
        commands[i] = (gain * ux, gain * uy)
    return commands


def formation_error(positions: dict, spec: FormationSpec, edges) -> float:
    """RMS displacement error ``||r_ij - r*_ij||`` over the undirected edges.

    Zero exactly when the formation shape is achieved (the absolute position and
    a global translation are irrelevant — only relative offsets are scored).
    """
    edges = list(edges)
    if not edges:
        return 0.0
    total = 0.0
    for i, j in edges:
        rx, ry = relative_measurement(positions, i, j)
        dx, dy = spec.desired_relative(i, j)
        total += (rx - dx) ** 2 + (ry - dy) ** 2
    return math.sqrt(total / len(edges))
