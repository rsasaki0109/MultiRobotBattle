"""Closed-loop simulation of the formation controller (Euler integration).

Pure and deterministic: given start positions it integrates the displacement
control law forward, optionally with one agent acting as a free-moving leader.
Used by the unit tests (convergence) and the demo CLI.
"""

from __future__ import annotations

from .control import (
    formation_control_from_relative,
    formation_error,
    relative_measurements,
)
from .spec import FormationSpec, Vec2


def simulate(
    positions: dict,
    spec: FormationSpec,
    edges,
    *,
    gain: float = 1.0,
    dt: float = 0.1,
    steps: int = 200,
    leader=None,
    leader_velocity: Vec2 = (0.0, 0.0),
):
    """Integrate the closed loop and return ``(trajectory, errors)``.

    ``trajectory`` is a list of ``agent -> (x, y)`` snapshots (length
    ``steps + 1``); ``errors`` is the matching list of
    :func:`formation_error` values. A ``leader`` agent (if given) ignores the
    formation law and moves at ``leader_velocity`` so the shape tracks it.
    """
    edges = list(edges)
    pos = {a: (float(p[0]), float(p[1])) for a, p in positions.items()}
    trajectory = [dict(pos)]
    errors = [formation_error(pos, spec, edges)]
    fixed = {leader} if leader is not None else set()

    for _ in range(steps):
        meas = relative_measurements(pos, edges)
        commands = formation_control_from_relative(meas, spec, gain, fixed=fixed)
        new_pos: dict = {}
        for agent, p in pos.items():
            if agent == leader:
                new_pos[agent] = (
                    p[0] + dt * leader_velocity[0],
                    p[1] + dt * leader_velocity[1],
                )
            else:
                ux, uy = commands.get(agent, (0.0, 0.0))
                new_pos[agent] = (p[0] + dt * ux, p[1] + dt * uy)
        pos = new_pos
        trajectory.append(dict(pos))
        errors.append(formation_error(pos, spec, edges))

    return trajectory, errors
