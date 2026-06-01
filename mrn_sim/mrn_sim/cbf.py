"""Control Barrier Function (CBF) safety filter — minimal-deviation safety as a QP.

A safety filter sits between a nominal controller (pure pursuit, DWA, MPC, …)
and the robot: it passes the nominal command through unchanged when it is safe,
and otherwise returns the *closest* command that provably keeps the robot out of
collision. "Provably" means a **control barrier function**: for each obstacle
define ``h(x) >= 0`` on the safe set; enforcing ``ḣ(x, u) >= -alpha * h(x)``
makes the safe set *forward invariant* — once safe, the robot stays safe — for
any class-K gain ``alpha``. That single inequality per obstacle, plus the
actuation limits, bounds a polytope of safe commands, and the filter returns the
point in it nearest the nominal command:

    minimize  ½ ||u - u_nom||²   subject to   A u >= b.

For a unicycle a *first-order* CBF is degenerate — ``ḣ`` is linear in ``v`` but
independent of ``omega`` (turning is second-order), so it could only brake, not
steer. The standard fix is to regulate a **look-ahead point**
``p_L = p + L (cosθ, sinθ)`` a distance ``L`` ahead: its velocity maps to
``(v, omega)`` through an invertible matrix, so both controls enter ``ḣ`` and the
filter can steer around obstacles, not just slow down.

The QP has two variables, so it is solved exactly by enumerating active sets
(the optimum touches 0, 1, or 2 constraints): project the nominal point onto
each constraint and each pair-intersection, and keep the feasible candidate
closest to nominal. Pure and deterministic; depends only on math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CBFConfig:
    """Tuning for :func:`cbf_filter`."""

    look_ahead: float = 0.2         # L: regulated point ahead of the wheel axis
    alpha: float = 4.0              # class-K gain on h (larger = act later)
    robot_radius: float = 0.25
    safety_margin: float = 0.1      # extra buffer beyond the two radii
    max_v: float = 1.6
    min_v: float = -0.2             # allow a little reverse so it is never stuck
    max_omega: float = 3.0


def _feasible(u, rows, tol=1e-7):
    return all(a0 * u[0] + a1 * u[1] >= b - tol for (a0, a1, b) in rows)


def _project_line(u_nom, row):
    """Closest point to ``u_nom`` on the line ``a·u = b`` (row = (a0, a1, b))."""
    a0, a1, b = row
    nn = a0 * a0 + a1 * a1
    if nn < 1e-12:
        return None
    t = (b - (a0 * u_nom[0] + a1 * u_nom[1])) / nn
    return (u_nom[0] + t * a0, u_nom[1] + t * a1)


def _intersect(r1, r2):
    """Intersection of two lines ``a·u = b``; ``None`` if parallel."""
    a0, a1, b0 = r1
    c0, c1, b1 = r2
    det = a0 * c1 - a1 * c0
    if abs(det) < 1e-12:
        return None
    return ((b0 * c1 - a1 * b1) / det, (a0 * b1 - b0 * c0) / det)


def _solve_qp(u_nom, rows):
    """Closest point to ``u_nom`` in ``{u : A u >= b}`` (2-D, identity Hessian).

    Enumerates active sets: the unconstrained optimum (u_nom) if feasible, else
    the projection onto each single constraint, else each pair intersection.
    Returns the feasible candidate nearest ``u_nom`` (or ``u_nom`` clamped if the
    polytope is empty, which the box limits prevent in practice).
    """
    if _feasible(u_nom, rows):
        return u_nom

    def dist2(u):
        return (u[0] - u_nom[0]) ** 2 + (u[1] - u_nom[1]) ** 2

    best = None
    best_d = float("inf")
    for i, row in enumerate(rows):
        cand = _project_line(u_nom, row)
        if cand is not None and _feasible(cand, rows):
            d = dist2(cand)
            if d < best_d:
                best, best_d = cand, d
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            cand = _intersect(rows[i], rows[j])
            if cand is not None and _feasible(cand, rows):
                d = dist2(cand)
                if d < best_d:
                    best, best_d = cand, d
    return best if best is not None else u_nom


def cbf_filter(pose, u_nom, obstacles, cfg: CBFConfig = CBFConfig()):
    """Return the safe command nearest ``u_nom`` (= ``(v, omega)``).

    ``pose`` is ``(x, y, theta)``; ``obstacles`` a list of ``(x, y, radius)`` or
    ``(x, y, radius, vx, vy)`` (a moving obstacle's velocity is accounted for in
    the barrier rate). Builds one CBF inequality per obstacle plus the velocity
    box, then solves the minimal-deviation QP.
    """
    x, y, theta = pose
    L = cfg.look_ahead
    ct, st = math.cos(theta), math.sin(theta)
    # look-ahead point and the map  ṗ_L = M · (v, omega)
    px, py = x + L * ct, y + L * st
    m00, m01 = ct, -L * st
    m10, m11 = st, L * ct

    rows = []
    for obs in obstacles:
        ox, oy, r = obs[0], obs[1], obs[2]
        ovx, ovy = (obs[3], obs[4]) if len(obs) >= 5 else (0.0, 0.0)
        dx, dy = px - ox, py - oy
        safe = r + cfg.robot_radius + cfg.safety_margin
        h = dx * dx + dy * dy - safe * safe
        # ḣ = 2 (p_L - o)·(ṗ_L - v_o) = (2 (dx,dy)ᵀ M) u - 2 (dx,dy)·v_o
        a0 = 2.0 * (dx * m00 + dy * m10)
        a1 = 2.0 * (dx * m01 + dy * m11)
        rhs = -cfg.alpha * h + 2.0 * (dx * ovx + dy * ovy)
        rows.append((a0, a1, rhs))            # a·u >= rhs

    # actuation box as inequalities a·u >= b
    rows.append((1.0, 0.0, cfg.min_v))        # v >= min_v
    rows.append((-1.0, 0.0, -cfg.max_v))      # v <= max_v
    rows.append((0.0, 1.0, -cfg.max_omega))   # omega >= -max_omega
    rows.append((0.0, -1.0, -cfg.max_omega))  # omega <= max_omega

    v, omega = _solve_qp((u_nom[0], u_nom[1]), rows)
    v = min(cfg.max_v, max(cfg.min_v, v))
    omega = min(cfg.max_omega, max(-cfg.max_omega, omega))
    return (v, omega)
