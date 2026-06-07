"""Control Barrier Function (CBF) safety certificates for multi-robot systems.

A pure-Python reproduction of Wang, Ames & Egerstedt's *"Safety Barrier Certificates
for Collisions-Free Multirobot Systems"* (IEEE T-RO 2017). The zoo already has two
reciprocal-avoidance methods — ORCA (velocity-obstacle) and BVC (buffered Voronoi).
CBF is the third and a control-theoretic one: a **minimally-invasive safety filter**.
Given any nominal controller `û` (here a go-to-goal proportional law), it solves a
small QP that perturbs `û` as little as possible while *guaranteeing* the swarm stays
collision-free, by keeping each pairwise barrier non-negative for all time.

Single-integrator robots ``ṗ_i = u_i``.  For each pair define the barrier
``h_ij = ‖p_i − p_j‖² − (2r)²`` (``≥ 0`` ⟺ no overlap).  Forward invariance of the
safe set is enforced by the **barrier certificate** ``ḣ_ij ≥ −γ·h_ij``, i.e.

    −2 (p_i − p_j)·(u_i − u_j) ≤ γ·h_ij,

a *linear* inequality in the stacked control ``u``.  The safety filter is the QP

    min ½‖u − û‖²   s.t.  every pairwise certificate holds,

whose Hessian is the identity — so its solution is simply the **Euclidean projection**
of ``û`` onto the polyhedron of safe controls.  :func:`safe_control` computes that
projection by Dykstra's alternating projection onto the half-spaces (the same engine
as :mod:`bvc`'s cell projection).  Stopping (``u = 0``) always satisfies every
certificate when ``h_ij ≥ 0``, so the QP is always feasible.

- :func:`barrier_constraints` — the half-spaces ``a·u ≤ b`` of the certificate.
- :func:`safe_control` — the minimally-invasive projection (the QP solution).
- :func:`simulate` — run the filtered controller to the goals.
- :func:`min_separation` — the independent oracle the gate verifies against.

Honest scope (see ``docs/coordination.md``): the certificate is continuous-time; we
integrate it at a finite ``dt`` with a small safety margin folded into the barrier, so
the guarantee is enforced up to discretization (the gate checks real separation stays
``≥ 2r``).  Like every reactive filter it is **not complete** — symmetric standoffs
deadlock safely (both robots decelerate to a stop), shown honestly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "CBFResult",
    "barrier_constraints",
    "min_separation",
    "nominal_control",
    "safe_control",
    "simulate",
]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nominal_control(positions, goals, *, gain=1.0, v_max=1.0):
    """Go-to-goal proportional controller, clipped to ``v_max``."""
    u = []
    for p, g in zip(positions, goals):
        ux, uy = gain * (g[0] - p[0]), gain * (g[1] - p[1])
        s = math.hypot(ux, uy)
        if s > v_max and s > 1e-12:
            ux, uy = ux * v_max / s, uy * v_max / s
        u.append((ux, uy))
    return u


def barrier_constraints(positions, radius, *, gamma=1.0, margin=0.0):
    """Half-spaces ``a·u <= b`` of the pairwise barrier certificates.

    ``a`` is a stacked ``2n`` vector with ``-2(p_i-p_j)`` in robot ``i``'s block and
    ``+2(p_i-p_j)`` in robot ``j``'s; ``b = gamma * h_ij`` with a safety ``margin``
    folded into the effective radius.
    """
    n = len(positions)
    safe = radius + 0.5 * margin
    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            h = dx * dx + dy * dy - (2 * safe) ** 2
            a = [0.0] * (2 * n)
            a[2 * i], a[2 * i + 1] = -2 * dx, -2 * dy
            a[2 * j], a[2 * j + 1] = 2 * dx, 2 * dy
            cons.append((a, gamma * h))
    return cons


def _project_halfspace(x, a, b):
    val = sum(a[k] * x[k] for k in range(len(x))) - b
    if val <= 0.0:
        return x
    nn = sum(ai * ai for ai in a)
    if nn <= 1e-15:
        return x
    s = val / nn
    return [x[k] - s * a[k] for k in range(len(x))]


def safe_control(positions, u_nominal, radius, *, gamma=1.0, margin=0.0,
                 iterations=80, tol=1e-12):
    """Minimally-invasive safe control: Euclidean projection of ``u_nominal`` onto
    the safe polyhedron (the CBF-QP solution) via Dykstra's alternating projection.
    """
    cons = barrier_constraints(positions, radius, gamma=gamma, margin=margin)
    x = [c for uv in u_nominal for c in uv]   # flatten to 2n
    if not cons:
        return [(x[2 * i], x[2 * i + 1]) for i in range(len(u_nominal))]
    corr = [[0.0] * len(x) for _ in cons]
    for _ in range(iterations):
        moved = 0.0
        for k, (a, b) in enumerate(cons):
            y = [x[t] + corr[k][t] for t in range(len(x))]
            px = _project_halfspace(y, a, b)
            corr[k] = [y[t] - px[t] for t in range(len(x))]
            moved += sum(abs(px[t] - x[t]) for t in range(len(x)))
            x = px
        if moved < tol:
            break
    return [(x[2 * i], x[2 * i + 1]) for i in range(len(u_nominal))]


@dataclass
class CBFResult:
    paths: dict
    arrived: bool
    steps: int
    num_arrived: int
    max_intervention: float    # largest ‖u - û‖ over the run


def simulate(starts, goals, radius, *, dt=0.05, gain=1.5, v_max=1.0,
             gamma=1.0, margin=0.1, goal_radius=0.08, max_steps=600):
    """Run the CBF-filtered go-to-goal controller; collision-free by the barrier
    certificate (up to the integration step).  Returns a :class:`CBFResult`."""
    positions = [tuple(s) for s in starts]
    paths = {i: [positions[i]] for i in range(len(positions))}
    max_interv = 0.0
    steps = 0
    for steps in range(1, max_steps + 1):
        if all(_dist(positions[i], goals[i]) <= goal_radius
               for i in range(len(positions))):
            steps -= 1
            break
        u_nom = nominal_control(positions, goals, gain=gain, v_max=v_max)
        u = safe_control(positions, u_nom, radius, gamma=gamma, margin=margin)
        for i in range(len(positions)):
            max_interv = max(max_interv, math.hypot(u[i][0] - u_nom[i][0],
                                                    u[i][1] - u_nom[i][1]))
        positions = [(positions[i][0] + u[i][0] * dt,
                      positions[i][1] + u[i][1] * dt)
                     for i in range(len(positions))]
        for i in range(len(positions)):
            paths[i].append(positions[i])
    num_arrived = sum(_dist(positions[i], goals[i]) <= goal_radius
                      for i in range(len(positions)))
    return CBFResult(paths, num_arrived == len(positions), steps, num_arrived,
                     max_interv)


def min_separation(paths, radius):
    """Independent oracle: closest approach (minus ``2r``) over the run."""
    ids = list(paths)
    horizon = len(paths[ids[0]])
    worst = math.inf
    for t in range(horizon):
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                worst = min(worst,
                            _dist(paths[ids[a]][t], paths[ids[b]][t]) - 2 * radius)
    return worst
