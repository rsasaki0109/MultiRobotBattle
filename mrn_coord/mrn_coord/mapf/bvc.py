"""Buffered Voronoi Cells (BVC): decentralized, position-space collision avoidance.

A pure-Python reproduction of Zhou, Wang, Bandyopadhyay & Schwager's *"Fast, On-line
Collision Avoidance for Dynamic Vehicles using Buffered Voronoi Cells"* (IEEE RA-L
2017). The zoo already has ORCA — reciprocal avoidance in **velocity** space. BVC is
its **position**-space cousin and a distinct mechanism: each robot, knowing the
others' positions, restricts its next move to its own **buffered Voronoi cell** and
steps toward its goal within it.

For each pair the **Voronoi boundary** is the perpendicular bisector — the set of
points equidistant from `p_i` and `p_j`; robot `i`'s side is the half-plane of points
nearer to `p_i`. Retract that half-plane inward by the body radius `r` and you get
the **buffered** half-plane; the intersection over all neighbours is the **BVC**
(`buffered_voronoi_cell`). The key property: two robots' buffered cells are separated
by at least `2r`, so **if every robot stays inside its own BVC, no two robots ever
collide** — collision-free *by construction* (`step_bvc` / `simulate`). And whenever
the robots are currently `≥ 2r` apart, each robot's *own* position lies in its BVC,
so the cell is non-empty and the robot can always at least hold still.

- :func:`buffered_voronoi_cell` — the half-planes `a·x ≤ b` of robot `i`'s BVC.
- :func:`project_to_cell` — the closest point in a convex polygon (intersection of
  half-planes) to a target, by Dykstra's alternating projection. The BVC controller
  steps each robot toward the projection of its goal.
- :func:`simulate` — run all robots in lockstep to their goals; returns the paths
  and whether everyone arrived.
- :func:`min_separation` — the independent oracle the gate verifies against.

Honest scope (see ``docs/coordination.md``): BVC is a **reactive, decentralized**
controller — collision-free always, but **not complete**. In perfectly symmetric
configurations (robots evenly on a circle heading to antipodal goals) it **deadlocks**
at a symmetric standoff: every robot stays collision-free but none reaches its goal.
A tiny deterministic perturbation breaks the symmetry and they converge — the
paradigm's known limitation, shown explicitly rather than hidden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "BVCResult",
    "buffered_voronoi_cell",
    "min_separation",
    "project_to_cell",
    "simulate",
    "step_bvc",
]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def buffered_voronoi_cell(positions, i, radius):
    """Half-planes ``(a, b)`` (meaning ``a·x <= b``) of robot ``i``'s BVC.

    For each neighbour ``j``, the buffered bisector retracted by ``radius`` toward
    ``p_i``.  ``a`` is ``p_j - p_i`` (so larger ``a·x`` is *toward* the neighbour).
    """
    pi = positions[i]
    planes = []
    for j, pj in enumerate(positions):
        if j == i:
            continue
        nx, ny = pj[0] - pi[0], pj[1] - pi[1]
        norm = math.hypot(nx, ny)
        if norm <= 1e-12:
            continue
        mx, my = 0.5 * (pi[0] + pj[0]), 0.5 * (pi[1] + pj[1])
        b = nx * mx + ny * my - radius * norm
        planes.append(((nx, ny), b))
    return planes


def _project_halfplane(point, a, b):
    ax, ay = a
    val = ax * point[0] + ay * point[1] - b
    if val <= 0.0:
        return point
    nn = ax * ax + ay * ay
    if nn <= 1e-15:
        return point
    s = val / nn
    return (point[0] - s * ax, point[1] - s * ay)


def project_to_cell(target, planes, *, iterations=50, tol=1e-10):
    """Closest point to ``target`` inside the polygon ``a·x <= b`` (Dykstra).

    With no planes, returns ``target``.  Dykstra's alternating projection converges
    to the exact projection onto the intersection of the half-planes.
    """
    if not planes:
        return target
    x = target
    corrections = [(0.0, 0.0) for _ in planes]
    for _ in range(iterations):
        moved = 0.0
        for k, (a, b) in enumerate(planes):
            cx, cy = corrections[k]
            y = (x[0] + cx, x[1] + cy)
            px = _project_halfplane(y, a, b)
            corrections[k] = (y[0] - px[0], y[1] - px[1])
            moved += abs(px[0] - x[0]) + abs(px[1] - x[1])
            x = px
        if moved < tol:
            break
    return x


def step_bvc(positions, goals, radius, *, step_size, perturb=None):
    """One synchronous BVC step: each robot moves toward the projection of its goal
    into its buffered Voronoi cell, bounded by ``step_size``.

    ``perturb`` (optional ``{i: (dx, dy)}``) nudges a robot's goal to break
    symmetric deadlocks deterministically.
    """
    new = []
    for i, p in enumerate(positions):
        planes = buffered_voronoi_cell(positions, i, radius)
        goal = goals[i]
        if perturb and i in perturb:
            goal = (goal[0] + perturb[i][0], goal[1] + perturb[i][1])
        tgt = project_to_cell(goal, planes)
        dx, dy = tgt[0] - p[0], tgt[1] - p[1]
        d = math.hypot(dx, dy)
        if d <= step_size or d <= 1e-12:
            new.append(tgt)
        else:
            new.append((p[0] + step_size * dx / d, p[1] + step_size * dy / d))
    return new


@dataclass
class BVCResult:
    paths: dict           # robot -> list of (x, y)
    arrived: bool         # every robot within goal_radius of its goal
    steps: int
    num_arrived: int


def simulate(starts, goals, radius, *, step_size=0.1, goal_radius=0.08,
             max_steps=400, perturb=None):
    """Run the BVC controller to convergence (or ``max_steps``).

    ``starts``/``goals`` are lists of ``(x, y)``.  Returns a :class:`BVCResult`
    whose paths are collision-free by construction.
    """
    positions = [tuple(s) for s in starts]
    paths = {i: [positions[i]] for i in range(len(positions))}
    steps = 0
    for steps in range(1, max_steps + 1):
        if all(_dist(positions[i], goals[i]) <= goal_radius
               for i in range(len(positions))):
            steps -= 1
            break
        positions = step_bvc(positions, goals, radius,
                             step_size=step_size, perturb=perturb)
        for i in range(len(positions)):
            paths[i].append(positions[i])
    num_arrived = sum(_dist(positions[i], goals[i]) <= goal_radius
                      for i in range(len(positions)))
    arrived = num_arrived == len(positions)
    return BVCResult(paths, arrived, steps, num_arrived)


def min_separation(paths, radius):
    """Independent oracle: closest approach (minus ``2r``) over every synchronous
    step of ``paths`` (robot -> position list).  ``< 0`` means a collision."""
    ids = list(paths)
    horizon = len(paths[ids[0]])
    worst = math.inf
    for t in range(horizon):
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                worst = min(worst,
                            _dist(paths[ids[a]][t], paths[ids[b]][t]) - 2 * radius)
    return worst
