"""Biped walking pattern generation by ZMP preview control (Kajita et al.).

A faithful, pure-Python reproduction of Kajita, Kanehiro, Kaneko, Fujiwara,
Harada, Yokoi & Hirukawa, *"Biped Walking Pattern Generation by using Preview
Control of Zero-Moment Point"* (IEEE ICRA 2003), using the clean state-space
restatement from the same group's *"Biped Walking Pattern Generator allowing
Auxiliary ZMP Control"* (IROS 2006), eqs. (5)-(9).

This is the natural companion to :mod:`footstep`: footstep planning decides
*where* the feet go; this decides the **center-of-mass trajectory** that makes
those footsteps a dynamically stable walk, by keeping the **Zero-Moment Point**
(ZMP) tracking a reference that sits under the support foot.

**The model.** The biped is approximated by a Linear Inverted Pendulum — a CoM
at constant height ``z_h`` over a "ZMP cart" (Fig. 3), with sagittal dynamics
``ẍ = (g / z_h) (x - p)`` where ``p`` is the ZMP. Taking the cart speed
``v = ṗ`` as the input and discretising at ``dt`` gives (eq. 5)

    x_{k+1} = A x_k + b v_k,   p_k = c x_k

with state ``x_k = [pos, vel, accel]`` and, in the cart-table form,
``A = [[1,dt,dt²/2],[0,1,dt],[0,0,1]]``, ``b = [dt³/6, dt²/2, dt]``,
``c = [1, 0, -z_h/g]`` (so ``c x = pos - (z_h/g) accel`` is exactly the ZMP).

**The controller.** A ZMP-tracking servo with **preview** of the future
reference minimises ``J = Σ Q (p^ref - p)² + R v²`` (eq. 6). The optimal law is
(eqs. 7-9)

    v_k = -K x_k + Σ_{j=1..N} f_j · p^ref_{k+j}

where ``P`` solves the discrete Riccati equation
``P = AᵀPA + cᵀQc - AᵀPb (R + bᵀPb)⁻¹ bᵀPA``, the feedback gain is
``K = (R + bᵀPb)⁻¹ bᵀPA``, and the **preview gains** are
``f_j = (R + bᵀPb)⁻¹ bᵀ (A - bK)ᵀ^{(j-1)} cᵀQ``. The preview term lets the CoM
start shifting *before* each footfall, which is what keeps the induced ZMP on
the support foot.

x and y are independent 1-D systems. Everything is pure Python — the Riccati
equation is solved by small-matrix (3×3) fixed-point iteration, no numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

G = 9.8  # gravity (m/s^2)


# --- tiny 3x3 / 3-vector linear algebra (no numpy) --------------------------


def _matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)]
            for i in range(n)]


def _transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def _mv(A, x):
    return [sum(A[i][k] * x[k] for k in range(len(x))) for i in range(len(A))]


def _dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))


def _outer(a, b):
    return [[ai * bj for bj in b] for ai in a]


# --- preview-control gains --------------------------------------------------


@dataclass
class PreviewGains:
    K: tuple            # feedback gain (3,)
    f: tuple            # preview gains f_1..f_N
    z_h: float
    dt: float
    g: float


def _cart_table_system(z_h, dt, g):
    A = [[1.0, dt, dt * dt / 2.0],
         [0.0, 1.0, dt],
         [0.0, 0.0, 1.0]]
    b = [dt ** 3 / 6.0, dt * dt / 2.0, dt]
    c = [1.0, 0.0, -z_h / g]
    return A, b, c


def _solve_dare(A, b, c, Q, R, iters=10000, tol=1e-12):
    """Solve P = AᵀPA + cᵀQc - AᵀPb (R + bᵀPb)⁻¹ bᵀPA by fixed-point iteration."""
    At = _transpose(A)
    cTc = _outer(c, c)                          # cᵀ c (3x3); scaled by Q below
    cTQc = [[Q * cTc[i][j] for j in range(3)] for i in range(3)]
    P = [[0.0] * 3 for _ in range(3)]
    for _ in range(iters):
        AtP = _matmul(At, P)                    # 3x3
        AtPA = _matmul(AtP, A)                  # 3x3
        Pb = _mv(P, b)                          # 3
        bPb = _dot(b, Pb)                       # scalar
        AtPb = _mv(AtP, b)                      # 3
        denom = R + bPb
        corr = _outer(AtPb, AtPb)               # (AᵀPb)(bᵀPA), 3x3
        newP = [[AtPA[i][j] + cTQc[i][j] - corr[i][j] / denom
                 for j in range(3)] for i in range(3)]
        diff = max(abs(newP[i][j] - P[i][j]) for i in range(3) for j in range(3))
        P = newP
        if diff < tol:
            break
    return P


def preview_gains(z_h=0.8, dt=0.02, preview_steps=80, Q=1.0, R=1e-8, g=G):
    """Compute the ZMP-preview-control gains for a CoM height ``z_h``.

    Returns :class:`PreviewGains` with the feedback gain ``K`` (3,) and the
    ``preview_steps`` preview gains ``f`` (eqs. 8-9).
    """
    A, b, c = _cart_table_system(z_h, dt, g)
    P = _solve_dare(A, b, c, Q, R)
    Pb = _mv(P, b)
    denom = R + _dot(b, Pb)                      # R + bᵀ P b
    PA = _matmul(P, A)
    bPA = [_dot(b, [PA[r][j] for r in range(3)]) for j in range(3)]  # bᵀ P A
    K = tuple(bPA[j] / denom for j in range(3))

    # preview gains: f_j = invR * bᵀ (A - bK)ᵀ^{(j-1)} cᵀ Q
    Ac = [[A[i][j] - b[i] * K[j] for j in range(3)] for i in range(3)]
    AcT = _transpose(Ac)
    xi = [Q * c[i] for i in range(3)]           # ξ_1 = cᵀ Q
    f = []
    for _ in range(preview_steps):
        f.append(_dot(b, xi) / denom)
        xi = _mv(AcT, xi)                        # ξ_{j+1} = Acᵀ ξ_j
    return PreviewGains(K=K, f=tuple(f), z_h=z_h, dt=dt, g=g)


# --- pattern generation -----------------------------------------------------


def lipm_track(zmp_ref, gains, x0=None):
    """Run the preview controller over a 1-D ZMP reference ``zmp_ref``.

    The CoM starts at ``x0`` (default ``[zmp_ref[0], 0, 0]`` — already over the
    first support point, so there is no startup transient). Returns
    ``(com, zmp)`` lists: the CoM position trajectory and the ZMP it actually
    induces (``c x_k``), same length as ``zmp_ref``.
    """
    A, b, c = _cart_table_system(gains.z_h, gains.dt, gains.g)
    K, f, N = gains.K, gains.f, len(gains.f)
    x = list(x0) if x0 is not None else [zmp_ref[0] if zmp_ref else 0.0, 0.0, 0.0]
    n = len(zmp_ref)
    com, zmp = [], []
    for k in range(n):
        preview = 0.0
        for j in range(1, N + 1):
            preview += f[j - 1] * zmp_ref[min(k + j, n - 1)]
        v = -_dot(K, x) + preview
        com.append(x[0])
        zmp.append(_dot(c, x))
        x = [A[0][0] * x[0] + A[0][1] * x[1] + A[0][2] * x[2] + b[0] * v,
             A[1][1] * x[1] + A[1][2] * x[2] + b[1] * v,
             A[2][2] * x[2] + b[2] * v]
    return com, zmp


@dataclass
class WalkPattern:
    """A generated walking pattern."""

    com_x: list
    com_y: list
    zmp_x: list           # induced ZMP
    zmp_y: list
    ref_x: list           # reference ZMP
    ref_y: list
    support: list         # index of the support foot at each sample
    foot_poses: list      # (x, y, theta) of each support foot
    dt: float

    def zmp_rms_error(self):
        """RMS distance between the induced ZMP and the reference ZMP."""
        n = len(self.zmp_x)
        s = sum((self.zmp_x[k] - self.ref_x[k]) ** 2
                + (self.zmp_y[k] - self.ref_y[k]) ** 2 for k in range(n))
        return math.sqrt(s / n)


def zmp_reference_from_footsteps(states, step_duration, dt,
                                 final_hold=1.6):
    """Build a piecewise-constant ZMP reference from a footstep plan.

    ``states`` is a sequence of stance-foot poses (objects with ``.x``, ``.y``,
    ``.theta``, e.g. :class:`~mrn_coord.mapf.footstep.FootstepState`). The ZMP
    reference holds at each stance foot's centre for ``step_duration`` seconds;
    a ``final_hold`` keeps it under the last foot so the CoM settles (and the
    preview window has reference to look ahead into). Returns
    ``(ref_x, ref_y, support_index, foot_poses)``.
    """
    per_step = max(1, int(round(step_duration / dt)))
    hold = max(1, int(round(final_hold / dt)))
    ref_x, ref_y, support = [], [], []
    foot_poses = [(s.x, s.y, s.theta) for s in states]
    for i, s in enumerate(states):
        reps = per_step + (hold if i == len(states) - 1 else 0)
        for _ in range(reps):
            ref_x.append(s.x)
            ref_y.append(s.y)
            support.append(i)
    return ref_x, ref_y, support, foot_poses


def generate_walk(states, *, gains=None, step_duration=0.7, dt=0.02,
                  z_h=0.8, preview_steps=None, Q=1.0, R=1e-8, g=G):
    """Generate a walking pattern that realises a footstep plan.

    ``states`` is the stance-foot sequence (e.g. ``FootstepPlan.states``).
    Returns a :class:`WalkPattern` with the CoM trajectory and the induced ZMP.
    """
    if gains is None:
        if preview_steps is None:
            preview_steps = int(round(1.6 / dt))
        gains = preview_gains(z_h=z_h, dt=dt, preview_steps=preview_steps,
                              Q=Q, R=R, g=g)
    ref_x, ref_y, support, foot_poses = zmp_reference_from_footsteps(
        states, step_duration, dt)
    com_x, zmp_x = lipm_track(ref_x, gains)
    com_y, zmp_y = lipm_track(ref_y, gains)
    return WalkPattern(com_x=com_x, com_y=com_y, zmp_x=zmp_x, zmp_y=zmp_y,
                       ref_x=ref_x, ref_y=ref_y, support=support,
                       foot_poses=foot_poses, dt=dt)


# --- stability: the ZMP must stay in the support polygon --------------------


def _point_in_foot(px, py, fx, fy, ftheta, length, width, margin=0.0):
    """True iff (px, py) is inside the oriented foot rectangle (+ margin)."""
    c, s = math.cos(ftheta), math.sin(ftheta)
    dx, dy = px - fx, py - fy
    # rotate the point into the foot frame
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    return (abs(lx) <= length / 2.0 + margin
            and abs(ly) <= width / 2.0 + margin)


def zmp_stability(pattern, *, foot_length, foot_width, margin=0.0):
    """Fraction of samples whose induced ZMP lies in the current support foot.

    The ZMP-stability criterion: a walk is dynamically stable while its ZMP
    stays inside the support polygon (here, the support foot's rectangle). The
    reference sits at the foot centre, so a well-tracked ZMP stays inside.
    Returns ``(fraction_inside, num_outside)``.
    """
    inside = 0
    outside = 0
    for k in range(len(pattern.zmp_x)):
        fx, fy, ft = pattern.foot_poses[pattern.support[k]]
        if _point_in_foot(pattern.zmp_x[k], pattern.zmp_y[k], fx, fy, ft,
                          foot_length, foot_width, margin):
            inside += 1
        else:
            outside += 1
    n = inside + outside
    return (inside / n if n else 1.0), outside
