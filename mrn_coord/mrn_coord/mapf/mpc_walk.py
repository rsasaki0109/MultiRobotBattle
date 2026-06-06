"""Trajectory-free MPC walking control (Wieber 2006).

A faithful, pure-Python reproduction of Wieber, *"Trajectory Free Linear Model
Predictive Control for Stable Walking in the Presence of Strong Perturbations"*
(IEEE-RAS Humanoids 2006).

This is the **constrained-QP counterpart** of :mod:`lipm_walk`'s preview control.
Kajita's preview controller is an *unconstrained* infinite-horizon LQR that
*tracks a precomputed ZMP trajectory* under the support feet. Wieber turns the
problem inside out: there is **no reference trajectory to track** ("trajectory
free"). Instead the Zero-Moment Point is kept inside the support polygon by a
**hard inequality constraint**, and a small jerk + reference-velocity objective
picks the smoothest forward-walking motion that satisfies it. Solving that
constrained quadratic program at every control tick (receding horizon) is what
buys robustness: under a strong push the controller saturates the ZMP at the
edge of the foot — the most it can legally do — instead of blindly tracking a
trajectory that would carry the ZMP out of the support polygon (i.e. tip over).

**The model** is the same cart-table Linear Inverted Pendulum as
:mod:`lipm_walk`: state ``x = [pos, vel, acc]``, control ``u`` = CoM **jerk**,

    x_{k+1} = A x_k + b u_k,   A = [[1,T,T²/2],[0,1,T],[0,0,1]], b = [T³/6,T²/2,T]

and the ZMP is the linear output ``z_k = c x_k`` with ``c = [1, 0, -z_h/g]``.

**Condensed MPC.** Over a horizon of ``N`` samples, stack the jerks
``U = [u_0 … u_{N-1}]``. Propagating the dynamics gives the future ZMP and CoM
velocity as affine functions of ``U`` (eqs. in the paper):

    Z = P_zs x + P_zu U,      V = P_vs x + P_vu U

with ``P_zu`` lower-triangular (diagonal ``c·b ≠ 0`` ⇒ invertible). Wieber's QP

    min_U  (α/2)‖U‖² + (β/2)‖V − v_ref‖²    s.t.   z_min ≤ Z ≤ z_max

minimises jerk and tracks a reference walking velocity subject to the ZMP
staying in the (possibly moving) support polygon.

**Why a box QP.** Because ``P_zu`` is invertible we change variables to the ZMP
itself, ``U = P_zu⁻¹ (Z − P_zs x)``. The support-polygon constraint becomes a
plain **box** ``z_min ≤ Z ≤ z_max``, and the objective stays strictly convex in
``Z``. The QP is then solved exactly by a small box-constrained **active-set**
method (each working-set subproblem is a Gaussian solve on the free block) — no
numpy, no external QP solver, exact KKT satisfaction in a handful of iterations.

Everything is pure Python. The condensed matrices are built by rolling the 3×3
dynamics; the per-tick Hessian is constant (state-independent) so it is built
once and reused across the whole receding-horizon run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

DEFAULT_G = 9.8  # gravity (m/s^2)


# --- tiny linear algebra (no numpy) -----------------------------------------


def _matvec(M, x):
    return [sum(M[i][k] * x[k] for k in range(len(x))) for i in range(len(M))]


def _matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)]
            for i in range(n)]


def _transpose(M):
    return [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]


def _dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))


def _inv_lower(L):
    """Inverse of a lower-triangular matrix by forward substitution."""
    n = len(L)
    M = [[0.0] * n for _ in range(n)]
    for j in range(n):
        M[j][j] = 1.0 / L[j][j]
        for i in range(j + 1, n):
            M[i][j] = -sum(L[i][k] * M[k][j] for k in range(j, i)) / L[i][i]
    return M


def _fwd_sub(L, rhs):
    """Solve ``L u = rhs`` for lower-triangular ``L`` (forward substitution)."""
    n = len(L)
    u = [0.0] * n
    for i in range(n):
        u[i] = (rhs[i] - sum(L[i][k] * u[k] for k in range(i))) / L[i][i]
    return u


def _gauss_solve(A, rhs):
    """Solve the dense linear system ``A z = rhs`` (partial-pivot elimination)."""
    n = len(A)
    M = [list(A[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        p = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[p] = M[p], M[col]
        piv = M[col][col]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col] / piv
                for cc in range(col, n + 1):
                    M[r][cc] -= f * M[col][cc]
    return [M[i][n] / M[i][i] for i in range(n)]


# --- box-constrained QP (active set) ----------------------------------------


def solve_box_qp(H, g, lo, hi, *, max_iter=None, tol=1e-10):
    """Minimise ``½ zᵀ H z + gᵀ z`` subject to ``lo ≤ z ≤ hi`` (``H`` SPD).

    A primal active-set method: solve the unconstrained problem on the free
    coordinates exactly (Gaussian elimination on the free block), clamp the
    worst box violation, and free any clamped coordinate whose multiplier
    (the objective gradient) points back into the interior. Returns the exact
    optimum ``z`` — KKT satisfied to machine precision in a few iterations,
    independent of the conditioning of ``H``.
    """
    n = len(g)
    if max_iter is None:
        max_iter = 8 * n + 40
    clamped = {}  # index -> bound value it is pinned to
    z = [0.0] * n
    for _ in range(max_iter):
        free = [i for i in range(n) if i not in clamped]
        if free:
            HFF = [[H[i][j] for j in free] for i in free]
            rhs = [-(g[i] + sum(H[i][ci] * cv for ci, cv in clamped.items()))
                   for i in free]
            zF = _gauss_solve(HFF, rhs)
        z = [0.0] * n
        for k, i in enumerate(free):
            z[i] = zF[k]
        for i, cv in clamped.items():
            z[i] = cv
        # clamp the single worst box violation among the free coordinates
        worst, wmag = None, tol
        for i in free:
            if lo[i] - z[i] > wmag:
                worst, wmag = (i, lo[i]), lo[i] - z[i]
            elif z[i] - hi[i] > wmag:
                worst, wmag = (i, hi[i]), z[i] - hi[i]
        if worst is not None:
            clamped[worst[0]] = worst[1]
            continue
        # all free feasible: free a clamped coord if its multiplier is wrong-signed
        grad = [sum(H[i][j] * z[j] for j in range(n)) + g[i] for i in range(n)]
        release = None
        for i, cv in clamped.items():
            if cv == lo[i] and grad[i] < -tol:
                release = i
                break
            if cv == hi[i] and grad[i] > tol:
                release = i
                break
        if release is not None:
            del clamped[release]
            continue
        return z
    return z


# --- MPC parameters & condensed model ---------------------------------------


@dataclass
class MPCParams:
    """Configuration of the trajectory-free walking MPC."""

    z_h: float = 0.8        # CoM height (m)
    dt: float = 0.1         # control period / horizon sample (s)
    horizon: int = 16       # number of look-ahead samples N
    alpha: float = 1e-5     # jerk regularisation weight
    beta: float = 1.0       # CoM-velocity tracking weight
    g: float = DEFAULT_G

    @property
    def omega(self):
        """LIPM natural frequency ``ω = sqrt(g / z_h)``."""
        return math.sqrt(self.g / self.z_h)


def _cart_table(z_h, dt, g):
    A = [[1.0, dt, dt * dt / 2.0],
         [0.0, 1.0, dt],
         [0.0, 0.0, 1.0]]
    b = [dt ** 3 / 6.0, dt * dt / 2.0, dt]
    c = [1.0, 0.0, -z_h / g]
    ev = [0.0, 1.0, 0.0]
    return A, b, c, ev


@dataclass
class CondensedMPC:
    """The condensed (state-eliminated) MPC matrices for fixed parameters.

    ``Z = Pzs x + Pzu U`` and ``V = Pvs x + Pvu U`` are the future ZMP and CoM
    velocity as affine functions of the jerk sequence ``U``. The Hessian ``HZ``
    of the objective re-expressed in the ZMP variable ``Z`` is constant, so it
    is built once here and reused at every control tick.
    """

    params: MPCParams
    A: list
    b: list
    c: list
    ev: list
    Pzs: list
    Pvs: list
    Pzu: list
    Pvu: list
    M: list           # Pzu^{-1}
    W: list           # Pvu @ M
    Wt: list
    HZ: list          # Hessian in the Z variable: alpha M'M + beta W'W

    def solve(self, x, vref_window, z_lo, z_hi):
        """Solve one MPC step from state ``x`` (3-vector).

        ``vref_window`` is the reference CoM velocity over the horizon, ``z_lo``
        / ``z_hi`` the per-sample ZMP bounds (the support polygon). Returns
        ``(Z, U, u0)``: the optimal future ZMP, the jerk sequence, and the
        first jerk to apply.
        """
        N = self.params.horizon
        s = _matvec(self.Pzs, x)                       # Pzs x
        r0 = [_matvec(self.Pvs, x)[i] - vref_window[i] for i in range(N)]
        Wtr0 = _matvec(self.Wt, r0)
        HZs = _matvec(self.HZ, s)
        g = [-HZs[i] + self.params.beta * Wtr0[i] for i in range(N)]
        Z = solve_box_qp(self.HZ, g, z_lo, z_hi)
        U = _fwd_sub(self.Pzu, [Z[i] - s[i] for i in range(N)])
        return Z, U, U[0]


def build_condensed(params=None):
    """Build the condensed MPC matrices for ``params`` (default :class:`MPCParams`)."""
    if params is None:
        params = MPCParams()
    A, b, c, ev = _cart_table(params.z_h, params.dt, params.g)
    N = params.horizon
    I3 = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    Apow = [I3]
    for _ in range(N):
        Apow.append(_matmul(A, Apow[-1]))
    AB = [_matvec(Apow[m], b) for m in range(N)]        # A^m b
    Pzs = [[_dot(c, [Apow[i][r][col] for r in range(3)]) for col in range(3)]
           for i in range(1, N + 1)]
    Pvs = [[_dot(ev, [Apow[i][r][col] for r in range(3)]) for col in range(3)]
           for i in range(1, N + 1)]
    Pzu = [[(_dot(c, AB[i - 1 - j]) if j <= i - 1 else 0.0) for j in range(N)]
           for i in range(1, N + 1)]
    Pvu = [[(_dot(ev, AB[i - 1 - j]) if j <= i - 1 else 0.0) for j in range(N)]
           for i in range(1, N + 1)]
    M = _inv_lower(Pzu)
    W = _matmul(Pvu, M)
    Mt, Wt = _transpose(M), _transpose(W)
    MtM, WtW = _matmul(Mt, M), _matmul(Wt, W)
    HZ = [[params.alpha * MtM[i][j] + params.beta * WtW[i][j]
           for j in range(N)] for i in range(N)]
    return CondensedMPC(params=params, A=A, b=b, c=c, ev=ev, Pzs=Pzs, Pvs=Pvs,
                        Pzu=Pzu, Pvu=Pvu, M=M, W=W, Wt=Wt, HZ=HZ)


# --- receding-horizon simulation --------------------------------------------


@dataclass
class MPCWalkResult:
    """The closed-loop trajectory of a receding-horizon MPC walk."""

    t: list
    com: list             # CoM position
    com_vel: list         # CoM velocity
    zmp: list             # induced ZMP (c x_k)
    support_center: list  # support-polygon centre at each tick
    support_half: list    # support-polygon half-width
    jerk: list            # applied jerk u0
    params: MPCParams

    def zmp_feasible(self, tol=1e-6):
        """True iff the ZMP stayed inside the support polygon at every tick."""
        return all(abs(self.zmp[k] - self.support_center[k])
                   <= self.support_half[k] + tol for k in range(len(self.zmp)))

    def max_zmp_excess(self):
        """Largest signed distance of the ZMP beyond the support polygon edge."""
        return max(abs(self.zmp[k] - self.support_center[k]) - self.support_half[k]
                   for k in range(len(self.zmp)))

    def recovered(self, vel_tol=1e-2):
        """True iff the CoM velocity has returned to (near) zero — balanced."""
        return abs(self.com_vel[-1]) < vel_tol

    def com_advance(self):
        """Net CoM travel over the run (forward-walking distance)."""
        return self.com[-1] - self.com[0]

    def mean_vel(self):
        return sum(self.com_vel) / len(self.com_vel)


def simulate_mpc(x0, centers, halves, vrefs, *, params=None, condensed=None,
                 n_steps=None, push_tick=None, push_dv=0.0, constrained=True):
    """Run the trajectory-free MPC in closed loop.

    ``centers`` / ``halves`` describe the support polygon at each tick (foot
    centre and half-width); ``vrefs`` the reference CoM velocity. All three must
    extend at least ``n_steps + horizon`` samples (the controller looks ahead).
    A push (sudden CoM-velocity jump ``push_dv``) can be injected at
    ``push_tick``. With ``constrained=False`` the ZMP box is dropped (the
    unconstrained LQR-like cousin), which is what lets the ZMP leave the foot.
    Returns an :class:`MPCWalkResult`.
    """
    if condensed is None:
        condensed = build_condensed(params)
    p = condensed.params
    N = p.horizon
    if n_steps is None:
        n_steps = len(centers) - N
    A, b, c = condensed.A, condensed.b, condensed.c
    x = list(x0)
    t, com, vel, zmp, sc, sh, jerk = [], [], [], [], [], [], []
    for k in range(n_steps):
        if push_tick is not None and k == push_tick:
            x[1] += push_dv
        vref_w = [vrefs[k + 1 + i] for i in range(N)]
        if constrained:
            z_lo = [centers[k + 1 + i] - halves[k + 1 + i] for i in range(N)]
            z_hi = [centers[k + 1 + i] + halves[k + 1 + i] for i in range(N)]
        else:
            z_lo = [-1e9] * N
            z_hi = [1e9] * N
        _, _, u0 = condensed.solve(x, vref_w, z_lo, z_hi)
        t.append(k * p.dt)
        com.append(x[0])
        vel.append(x[1])
        zmp.append(_dot(c, x))
        sc.append(centers[k])
        sh.append(halves[k])
        jerk.append(u0)
        x = [A[0][0] * x[0] + A[0][1] * x[1] + A[0][2] * x[2] + b[0] * u0,
             A[1][1] * x[1] + A[1][2] * x[2] + b[1] * u0,
             A[2][2] * x[2] + b[2] * u0]
    return MPCWalkResult(t=t, com=com, com_vel=vel, zmp=zmp, support_center=sc,
                         support_half=sh, jerk=jerk, params=p)


def standing_support(half, length, *, dt=0.1, horizon=16):
    """Constant single-support polygon (centre 0) for a standing-balance run.

    Returns ``(centers, halves)`` of ``length + horizon`` samples.
    """
    n = length + horizon + 2
    return [0.0] * n, [half] * n


def stepping_support(step_len, step_ticks, n_feet, half, *, horizon=16):
    """A forward-stepping support schedule: ``n_feet`` feet ``step_len`` apart.

    Each foot is in support for ``step_ticks`` ticks. Returns
    ``(centers, halves, n_steps)`` padded to cover the look-ahead horizon.
    """
    centers, halves = [], []
    for f in range(n_feet):
        for _ in range(step_ticks):
            centers.append(f * step_len)
            halves.append(half)
    n_steps = n_feet * step_ticks
    while len(centers) < n_steps + horizon + 2:
        centers.append(centers[-1])
        halves.append(halves[-1])
    return centers, halves, n_steps
