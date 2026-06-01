"""Execute a discrete MAPF plan as a *bodied* AMR — where the point guarantee breaks.

MAPF (``mrn_coord.mapf``, including lifelong PIBT) proves a plan collision-free on
the grid: no two agents share a cell or swap across a step. But that proof is for a
*point* moving in discrete time. A real autonomous mobile robot is a
differential-drive vehicle with a **rectangular footprint** and an orientation: it
cannot strafe sideways between cells, so it must turn in place to face the next one
(turning costs time the grid plan never counts), and its rectangle **sweeps area** —
at a junction turn or a tight aisle pass the corners can intrude into a shelf or
another robot even though the cell plan said the centres never coincide.

This module executes a grid plan under that AMR model and measures the gap:

- **turning cost** — continuous time-to-finish vs the discrete makespan. The plan
  assumes free, instantaneous reorientation; a real AMR pays for every turn.
- **footprint clearance** — the smallest gap between any robot footprint and a shelf,
  and between any two robot footprints (negative where bodies overlap). With
  ``cell_size`` comfortably larger than the footprint these stay positive; squeeze
  the cell toward the body and the overlaps appear — quantifying exactly where the
  point guarantee stops being enough.

Pure and deterministic: same plan + same parameters → identical metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .kinematics import normalize_angle, unicycle_step


@dataclass(frozen=True)
class Footprint:
    """A rectangular robot body: ``length`` along the heading, ``width`` across."""

    length: float
    width: float

    def corners(self, pose):
        """The four world-frame corners for ``pose = (x, y, theta)``."""
        x, y, th = pose
        c, s = math.cos(th), math.sin(th)
        hl, hw = self.length / 2.0, self.width / 2.0
        out = []
        for sx, sy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
            out.append((x + sx * c - sy * s, y + sx * s + sy * c))
        return out


@dataclass
class AmrExecResult:
    """Metrics from executing a MAPF plan as bodied differential-drive AMRs."""

    success: bool                  # every robot reached its final cell
    discrete_makespan: int         # the grid plan's makespan (steps)
    continuous_steps: int          # steps the bodied execution actually took
    makespan_sec: float            # continuous time to the last arrival
    turn_time_frac: float          # fraction of robot-steps spent turning in place
    min_shelf_clearance: float     # closest any footprint came to a shelf (m)
    min_robot_clearance: float     # closest any two footprints came (m)
    footprint_collisions: int      # steps with any footprint overlap (shelf or robot)

    def as_dict(self) -> dict:
        return {
            "success": self.success,
            "discrete_makespan": self.discrete_makespan,
            "continuous_steps": self.continuous_steps,
            "makespan_sec": round(self.makespan_sec, 3),
            "turn_time_frac": round(self.turn_time_frac, 3),
            "min_shelf_clearance": round(self.min_shelf_clearance, 4),
            "min_robot_clearance": round(self.min_robot_clearance, 4),
            "footprint_collisions": self.footprint_collisions,
        }


# --- convex-polygon distance / overlap (separating-axis) -----------------------

def _project(poly, ax):
    ds = [p[0] * ax[0] + p[1] * ax[1] for p in poly]
    return min(ds), max(ds)


def _edge_normals(poly):
    axes = []
    n = len(poly)
    for i in range(n):
        ex = poly[(i + 1) % n][0] - poly[i][0]
        ey = poly[(i + 1) % n][1] - poly[i][1]
        length = math.hypot(ex, ey)
        if length > 1e-12:
            axes.append((-ey / length, ex / length))
    return axes


def _point_seg_dist(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    seg = dx * dx + dy * dy
    if seg < 1e-12:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / seg))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def poly_clearance(a, b):
    """Signed clearance between two convex polygons.

    Positive = exact Euclidean separation; ``<= 0`` = the bodies overlap (the
    value is then the negated separating-axis penetration depth). Used both for
    robot-vs-shelf (oriented box vs axis box) and robot-vs-robot.
    """
    axes = _edge_normals(a) + _edge_normals(b)
    separated = False
    min_overlap = math.inf
    for ax in axes:
        a0, a1 = _project(a, ax)
        b0, b1 = _project(b, ax)
        overlap = min(a1, b1) - max(a0, b0)
        if overlap < 0.0:
            separated = True
            break
        min_overlap = min(min_overlap, overlap)
    if not separated:
        return -min_overlap                       # penetration depth (<= 0)
    # disjoint: exact distance is the min vertex-to-edge distance both ways.
    best = math.inf
    for poly, other in ((a, b), (b, a)):
        for p in poly:
            for i in range(len(other)):
                best = min(best, _point_seg_dist(p, other[i],
                                                 other[(i + 1) % len(other)]))
    return best


# --- bodied execution ----------------------------------------------------------

def _center(c, cell_size):
    return ((c[0] + 0.5) * cell_size, (c[1] + 0.5) * cell_size)


def _shelf_polys(blocked, cell_size):
    polys = []
    for (x, y) in blocked:
        x0, y0 = x * cell_size, y * cell_size
        x1, y1 = x0 + cell_size, y0 + cell_size
        polys.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    return polys


def execute_amr(paths, blocked, *, cell_size=1.0, footprint=None, max_v=1.0,
                max_omega=2.5, dt=0.1):
    """Drive each agent's grid ``path`` as a bodied differential-drive AMR.

    The plan is replayed **synchronized to its own discrete schedule** — every
    robot makes its step-``t`` move together — so the only thing under test is
    the *body*, not timing (which the Temporal Plan Graph in :mod:`mrn_sim.mapf_exec`
    already handles). Each discrete step is executed in two honest phases a
    differential-drive robot cannot avoid: **turn in place** to face the next
    cell, then **drive** forward. The step's wall-clock is the slowest robot's
    turn+drive, so the turning a point plan never counted shows up as makespan
    stretch, and the rectangular sweep shows up as footprint clearance.

    ``paths`` maps id -> list of cells (e.g. ``solution.paths`` from a MAPF
    solver, or a lifelong history slice); ``blocked`` is the shelf cells.
    ``footprint`` is an **absolute** size in metres (default 0.7 x 0.45 m), so
    shrinking ``cell_size`` toward it is what tightens the aisles. Returns an
    :class:`AmrExecResult`.
    """
    if footprint is None:
        footprint = Footprint(0.7, 0.45)
    ids = list(paths)
    if not ids:
        return AmrExecResult(True, 0, 0, 0.0, 0.0, 0.0, 0.0, 0)
    horizon = max(len(p) for p in paths.values())
    padded = {a: list(paths[a]) + [paths[a][-1]] * (horizon - len(paths[a]))
              for a in ids}
    centers = {a: [_center(c, cell_size) for c in padded[a]] for a in ids}
    shelves = _shelf_polys(blocked, cell_size)
    disc_makespan = horizon - 1

    pose = {}
    for a in ids:
        x0, y0 = centers[a][0]
        th = 0.0
        for t in range(1, horizon):                  # face the first real move
            if padded[a][t] != padded[a][0]:
                th = math.atan2(centers[a][t][1] - y0, centers[a][t][0] - x0)
                break
        pose[a] = (x0, y0, th)

    min_shelf = [math.inf]
    min_robot = [math.inf]
    collision_frames = [0]
    sub_frames = [0]

    def sample():
        fps = {a: footprint.corners(pose[a]) for a in ids}
        bad = False
        for a in ids:
            for sh in shelves:
                cl = poly_clearance(fps[a], sh)
                if cl < min_shelf[0]:
                    min_shelf[0] = cl
                if cl <= 0.0:
                    bad = True
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                cl = poly_clearance(fps[ids[i]], fps[ids[j]])
                if cl < min_robot[0]:
                    min_robot[0] = cl
                if cl <= 0.0:
                    bad = True
        if bad:
            collision_frames[0] += 1
        sub_frames[0] += 1

    sample()                                          # initial configuration
    total_time = 0.0
    turn_time = 0.0

    for t in range(horizon - 1):
        target = {a: centers[a][t + 1] for a in ids}

        # --- turn phase: rotate in place to face the target, all synchronized ---
        desired, dtheta = {}, {}
        for a in ids:
            px, py, th = pose[a]
            tx, ty = target[a]
            desired[a] = math.atan2(ty - py, tx - px) if math.hypot(
                tx - px, ty - py) > 1e-9 else th
            dtheta[a] = normalize_angle(desired[a] - th)
        rot_dur = max(abs(dtheta[a]) for a in ids) / max_omega
        if rot_dur > 1e-9:
            nfr = max(1, math.ceil(rot_dur / dt))
            for _ in range(nfr):
                for a in ids:
                    pose[a] = unicycle_step(pose[a], 0.0, dtheta[a] / rot_dur, dt)
                sample()
            for a in ids:                             # snap out residual error
                pose[a] = (pose[a][0], pose[a][1], desired[a])
            total_time += rot_dur
            turn_time += rot_dur

        # --- drive phase: translate forward to the next cell, synchronized ---
        dist = {a: math.hypot(target[a][0] - pose[a][0],
                              target[a][1] - pose[a][1]) for a in ids}
        trans_dur = max(dist.values()) / max_v
        if trans_dur > 1e-9:
            nfr = max(1, math.ceil(trans_dur / dt))
            for _ in range(nfr):
                for a in ids:
                    pose[a] = unicycle_step(pose[a], dist[a] / trans_dur, 0.0, dt)
                sample()
            for a in ids:                             # snap to the cell centre
                pose[a] = (target[a][0], target[a][1], pose[a][2])
            total_time += trans_dur

    return AmrExecResult(
        success=True,
        discrete_makespan=disc_makespan,
        continuous_steps=sub_frames[0],
        makespan_sec=total_time,
        turn_time_frac=(turn_time / total_time) if total_time else 0.0,
        min_shelf_clearance=(min_shelf[0] if min_shelf[0] != math.inf else 0.0),
        min_robot_clearance=(min_robot[0] if min_robot[0] != math.inf else 0.0),
        footprint_collisions=collision_frames[0],
    )
