"""MPC walking with automatic footstep placement (Herdt et al. 2010).

A faithful, pure-Python reproduction of Herdt, Diedam, Wieber, Mombaur, Kheddar,
Yokoi, *"Online Walking Motion Generation with Automatic Footstep Placement"*
(Advanced Robotics 24(5–6), 2010).

This is the **direct extension** of :mod:`mpc_walk` (Wieber's trajectory-free
MPC). Wieber keeps the Zero-Moment Point inside *prescribed* support feet; the
honest limit recorded there is that a push beyond the in-place capturable margin
makes the CoM fall, because a fixed foot cannot be re-placed — *it needs a step*.
Herdt removes exactly that limit: the **footstep positions become decision
variables of the same QP**. The controller then automatically places where to
step so as to follow a reference walking velocity, and under a strong push it
takes a *capture step* in the push direction instead of tipping over.

**Model.** Identical cart-table LIPM to :mod:`mpc_walk`: state ``x=[pos,vel,acc]``,
control ``u`` = CoM jerk, ``x_{k+1}=Ax+bu``, ZMP ``z=cx``. Over a horizon of
``N`` samples the future ZMP / CoM velocity are affine in the jerk sequence
``U``: ``Z = Pzs x + Pzu U``, ``V = Pvs x + Pvu U`` (``Pzu`` lower-triangular,
invertible).

**The Herdt QP.** Decision variables are the jerks ``U`` **and** the next ``m``
footstep positions ``X^f`` falling inside the horizon. A 0/1 selection schedule
says which foot supports each sample: the current (committed) foot ``f_c`` or one
of the future feet. With ``U_c`` (current-foot selector) and ``U_f`` (future-foot
one-hot matrix) the support-foot location at each sample is
``foot = U_c f_c + U_f X^f``. The objective

    min  (α/2)‖U‖² + (β/2)‖V − v_ref‖² + (γ/2)‖Z − U_c f_c − U_f X^f‖²

minimises jerk, tracks a reference velocity, and keeps the ZMP near the
(variable) support-foot centre; subject to the ZMP staying in the foot,
``|Z − U_c f_c − U_f X^f| ≤ ℓ``, and to foot reachability ``Δ ≤ X^f_j − X^f_{j-1}
≤ Δ̄``.

**Why it is still a box QP (the reduction).** :mod:`mpc_walk` already changes
variables to the ZMP because ``Pzu`` is invertible. Herdt does it **twice**:

  d_i = Z_i − foot_i          (ZMP relative to its support foot)
  δ_j = X^f_j − X^f_{j-1}      (foot increments, X^f_0 = f_c)

In the variables ``y = [d ; δ]`` the support constraint is the plain box
``−ℓ ≤ d ≤ ℓ``, the reachability constraint is the plain box ``Δ ≤ δ ≤ Δ̄``, and
the objective is strictly convex (``α>0`` makes the jerk block SPD, ``γ>0`` makes
the foot block SPD — without the ZMP-centring term the foot variables would be a
nullspace direction). So the whole automatic-footstep QP is solved exactly by the
*same* :func:`mpc_walk.solve_box_qp` active-set method — no numpy, no external QP.

**Receding horizon.** At every tick the QP is re-solved from the current state;
the first jerk is applied. When a single-support phase ends, the support foot is
**committed to the first automatically-placed footstep** ``X^f_1`` — that
commitment is the automatic footstep placement realised in closed loop.

Everything is pure Python and reuses :mod:`mpc_walk`'s condensed model and box-QP
solver. Like the rest of the humanoid sub-thread this is a sagittal (1-D,
point-mass) reduction; the real paper also plans the lateral footstep and a
double-support phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .mpc_walk import (DEFAULT_G, MPCParams, CondensedMPC, build_condensed,
                       solve_box_qp, _matvec, _matmul, _transpose, _dot,
                       _fwd_sub)


@dataclass
class HerdtParams:
    """Configuration of the automatic-footstep walking MPC."""

    z_h: float = 0.8          # CoM height (m)
    dt: float = 0.1           # control period / horizon sample (s)
    horizon: int = 16         # look-ahead samples N
    alpha: float = 1e-5       # jerk regularisation weight
    beta: float = 1.0         # CoM-velocity tracking weight
    gamma: float = 1e-3       # ZMP-centring weight (keeps ZMP near foot centre)
    g: float = DEFAULT_G
    foot_half: float = 0.05   # support-foot half-length ℓ
    step_ticks: int = 8       # samples per single-support phase
    step_lo: float = -0.05    # min foot increment Δ (backward reach)
    step_hi: float = 0.40     # max foot increment Δ̄ (forward reach)

    @property
    def omega(self):
        """LIPM natural frequency ``ω = sqrt(g / z_h)``."""
        return math.sqrt(self.g / self.z_h)

    def mpc_params(self):
        """The underlying :class:`mpc_walk.MPCParams` (the condensed model)."""
        return MPCParams(z_h=self.z_h, dt=self.dt, horizon=self.horizon,
                         alpha=self.alpha, beta=self.beta, g=self.g)


def _selection(r, N, step_ticks):
    """Support-foot schedule over the horizon.

    ``r`` ticks remain on the current (committed) foot, then future feet take
    over in groups of ``step_ticks`` samples. Returns ``(support, m)`` where
    ``support[i]`` is 0 for the current foot or ``j`` for future foot ``j``, and
    ``m`` is the number of future feet falling inside the horizon.
    """
    support = [0] * min(max(r, 0), N)
    foot = 1
    while len(support) < N:
        support += [foot] * min(step_ticks, N - len(support))
        foot += 1
    support = support[:N]
    return support, max(support)


@dataclass
class HerdtMPC:
    """The automatic-footstep MPC for fixed parameters (wraps the condensed model)."""

    params: HerdtParams
    condensed: CondensedMPC

    def solve(self, x, f_c, r, vref_window, *, foot_vars=True, nominal_inc=0.0,
              return_qp=False):
        """Solve one MPC step.

        ``x`` is the 3-state, ``f_c`` the current committed support foot, ``r``
        the ticks left in the current support phase, ``vref_window`` the
        reference CoM velocity over the horizon. With ``foot_vars=True`` the
        future footsteps are free (box ``[step_lo, step_hi]`` on the increments)
        and the ZMP-centring weight ``γ`` is active. With ``foot_vars=False`` the
        future feet are *prescribed* at nominal increments and only the ZMP is a
        variable, so the problem collapses to the fixed-foot :mod:`mpc_walk` box
        QP. Returns ``(u0, feet, Z, d, delta)``; with ``return_qp`` (free feet)
        also the assembled ``(H, grad, lo, hi, y)`` box QP for diagnostics.
        """
        cm = self.condensed
        p = self.params
        N = p.horizon
        support, m = _selection(r, N, step_ticks=p.step_ticks)
        Uc = [1.0 if support[i] == 0 else 0.0 for i in range(N)]
        Uf = [[1.0 if support[i] == j + 1 else 0.0 for j in range(m)]
              for i in range(N)]
        Uf1 = [sum(Uf[i]) for i in range(N)]
        s = _matvec(cm.Pzs, x)
        Pvsx = _matvec(cm.Pvs, x)
        r0 = [Pvsx[i] - vref_window[i] for i in range(N)]
        g_e = [p.beta * v for v in _matvec(cm.Wt, r0)]   # gradient wrt (Z−s)

        if not foot_vars:
            # feet PRESCRIBED at nominal increments -> only the ZMP is variable.
            # This is exactly the fixed-foot mpc_walk box QP (d in [-ℓ, ℓ]).
            presc = [f_c + (support[i]) * nominal_inc for i in range(N)]
            c0 = [presc[i] - s[i] for i in range(N)]
            rhs_lin = [_matvec(cm.HZ, c0)[i] + g_e[i] for i in range(N)]
            lo = [-p.foot_half] * N
            hi = [p.foot_half] * N
            d = solve_box_qp(cm.HZ, rhs_lin, lo, hi)
            Z = [d[i] + presc[i] for i in range(N)]
            U = _fwd_sub(cm.Pzu, [Z[i] - s[i] for i in range(N)])
            feet = [f_c + j * nominal_inc for j in range(1, m + 1)]
            return U[0], feet, Z, d, [nominal_inc] * m

        L = [[1.0 if l <= j else 0.0 for l in range(m)] for j in range(m)]
        Sdelta = _matmul(Uf, L) if m else [[] for _ in range(N)]
        z_const = [(Uc[i] + Uf1[i]) * f_c for i in range(N)]
        c0 = [z_const[i] - s[i] for i in range(N)]

        nT = N + m
        # Z = T y + z_const,  y = [d ; δ],  T = [ I_N | Sδ ]
        T = [[(1.0 if j == i else 0.0) for j in range(N)] +
             [Sdelta[i][k] for k in range(m)] for i in range(N)]
        Tt = _transpose(T)
        H_ab = _matmul(Tt, _matmul(cm.HZ, T))            # T'(αM'M+βW'W)T
        rhs_lin = [_matvec(cm.HZ, c0)[i] + g_e[i] for i in range(N)]
        grad = _matvec(Tt, rhs_lin)                      # T'(HZ c0 + g_e)
        H = [[H_ab[i][j] + (p.gamma if (i == j and i < N) else 0.0)
              for j in range(nT)] for i in range(nT)]
        lo = [-p.foot_half] * N + [p.step_lo] * m
        hi = [p.foot_half] * N + [p.step_hi] * m

        y = solve_box_qp(H, grad, lo, hi)
        d, delta = y[:N], y[N:]
        feet, acc = [], f_c
        for j in range(m):
            acc += delta[j]
            feet.append(acc)
        Z = [d[i] + z_const[i] + sum(Sdelta[i][k] * delta[k] for k in range(m))
             for i in range(N)]
        U = _fwd_sub(cm.Pzu, [Z[i] - s[i] for i in range(N)])
        if return_qp:
            return U[0], feet, Z, d, delta, (H, grad, lo, hi, y)
        return U[0], feet, Z, d, delta


def build_herdt(params=None):
    """Build the automatic-footstep MPC for ``params`` (default :class:`HerdtParams`)."""
    if params is None:
        params = HerdtParams()
    return HerdtMPC(params=params, condensed=build_condensed(params.mpc_params()))


@dataclass
class HerdtWalkResult:
    """The closed-loop trajectory of an automatic-footstep MPC walk."""

    t: list
    com: list
    com_vel: list
    zmp: list
    foot: list             # committed support foot at each tick
    foot_half: float
    support_lo: list       # support-polygon lower edge (double-support hull at a switch)
    support_hi: list       # support-polygon upper edge
    planned_first: list    # the automatically-placed next footstep each tick
    committed_feet: list   # the sequence of committed footsteps
    jerk: list
    vref: float
    params: HerdtParams

    def zmp_feasible(self, tol=1e-6):
        """True iff the ZMP stayed inside the support polygon every tick.

        At a foot switch the polygon is the convex hull of the outgoing and
        incoming feet (the instantaneous-switch stand-in for double support),
        so the ZMP transferring between feet is correctly counted feasible.
        """
        return all(self.support_lo[k] - tol <= self.zmp[k] <= self.support_hi[k] + tol
                   for k in range(len(self.zmp)))

    def max_zmp_excess(self):
        return max(max(self.support_lo[k] - self.zmp[k],
                       self.zmp[k] - self.support_hi[k])
                   for k in range(len(self.zmp)))

    def recovered(self, vel_tol=2e-2):
        """True iff the CoM velocity has converged to the reference (balanced)."""
        return abs(self.com_vel[-1] - self.vref) < vel_tol

    def diverged(self, vel_cap=5.0):
        """True iff the CoM velocity blew up (fell)."""
        return abs(self.com_vel[-1]) > vel_cap or any(
            v != v or abs(v) > 1e6 for v in self.com_vel)

    def foot_displacement(self):
        """Net travel of the committed support foot (capture-step magnitude)."""
        return self.committed_feet[-1] - self.committed_feet[0]

    def n_committed_steps(self):
        return len(self.committed_feet) - 1

    def com_advance(self):
        return self.com[-1] - self.com[0]

    def mean_vel(self):
        return sum(self.com_vel) / len(self.com_vel)


def simulate_herdt(x0, *, params=None, herdt=None, n_steps=60, vref_val=0.0,
                   push_tick=None, push_dv=0.0, foot_vars=True, f_c0=0.0):
    """Run the automatic-footstep MPC in closed loop.

    Walks at reference velocity ``vref_val`` starting from state ``x0`` and
    committed foot ``f_c0``. A push (CoM-velocity jump ``push_dv``) can be
    injected at ``push_tick``. With ``foot_vars=True`` the footsteps are free
    (Herdt); with ``foot_vars=False`` they are frozen (the fixed-foot
    :mod:`mpc_walk` cousin, for isolation). Returns a :class:`HerdtWalkResult`.
    """
    if herdt is None:
        herdt = build_herdt(params)
    p = herdt.params
    N = p.horizon
    A, b, c = herdt.condensed.A, herdt.condensed.b, herdt.condensed.c
    # frozen-foot nominal increment: march at the reference velocity per phase
    nominal_inc = vref_val * p.step_ticks * p.dt

    x = list(x0)
    f_c = f_c0
    prev_f = f_c
    phase_t = 0
    just_switched = False
    committed = [f_c]
    t, com, vel, zmp, foot, planned_first, jerk = [], [], [], [], [], [], []
    s_lo, s_hi = [], []
    for k in range(n_steps):
        if push_tick is not None and k == push_tick:
            x[1] += push_dv
        r = p.step_ticks - phase_t
        vref_w = [vref_val] * N
        u0, feet, _, _, _ = herdt.solve(x, f_c, r, vref_w, foot_vars=foot_vars,
                                         nominal_inc=nominal_inc)
        plan = feet[0] if feet else f_c
        t.append(k * p.dt)
        com.append(x[0])
        vel.append(x[1])
        zmp.append(_dot(c, x))
        foot.append(f_c)
        # support polygon: a single foot, or the hull of both feet at a switch
        # (the instantaneous-switch stand-in for the double-support phase)
        lo_edge = min(f_c, prev_f) - p.foot_half if just_switched else f_c - p.foot_half
        hi_edge = max(f_c, prev_f) + p.foot_half if just_switched else f_c + p.foot_half
        s_lo.append(lo_edge)
        s_hi.append(hi_edge)
        just_switched = False
        planned_first.append(plan)
        jerk.append(u0)
        # advance the cart-table dynamics
        nx0 = A[0][0] * x[0] + A[0][1] * x[1] + A[0][2] * x[2] + b[0] * u0
        nx1 = A[1][1] * x[1] + A[1][2] * x[2] + b[1] * u0
        nx2 = A[2][2] * x[2] + b[2] * u0
        if abs(nx1) > 1e8:                  # clamp once clearly diverged
            nx0, nx1, nx2 = (math.copysign(1e8, nx0), math.copysign(1e8, nx1),
                             math.copysign(1e8, nx2))
        x = [nx0, nx1, nx2]
        phase_t += 1
        if phase_t >= p.step_ticks:
            phase_t = 0
            prev_f = f_c
            f_c = plan                      # commit the automatic footstep
            just_switched = (f_c != prev_f)
            committed.append(f_c)
    return HerdtWalkResult(
        t=t, com=com, com_vel=vel, zmp=zmp, foot=foot, foot_half=p.foot_half,
        support_lo=s_lo, support_hi=s_hi, planned_first=planned_first,
        committed_feet=committed, jerk=jerk, vref=vref_val, params=p)
