"""Biped walking stabilization by LIPM tracking (Kajita et al. 2010).

A faithful, pure-Python reproduction of Kajita, Morisawa, Miura, Nakaoka,
Harada, Kaneko, Kanehiro & Yokoi, *"Biped Walking Stabilization Based on Linear
Inverted Pendulum Tracking"* (IEEE/RSJ IROS 2010).

This is the **closed-loop, on-the-real-robot** counterpart of every open-loop
pattern generator in this humanoid sub-thread (:mod:`lipm_walk` ZMP-preview,
:mod:`capture_point`, :mod:`dcm_walk`, :mod:`mpc_walk` / :mod:`herdt_walk`).
Those all compute a *reference* CoM/ZMP trajectory **ahead of time**. The catch
the 2010 paper attacks: the Linear Inverted Pendulum is **inherently unstable**
(``ẍ = ω²(x − p)`` has an eigenvalue ``+ω``), so on a real robot the smallest
perturbation — a push, a modelling error, even discretisation — makes the actual
CoM diverge from the reference like ``e^{ω t}``. Playing the precomputed ZMP back
open-loop does **not** reject it. A *stabilizer* must close the loop.

**LIPM tracking.** Measure the actual CoM state ``(x, ẋ)``, compare with the
reference ``(x^ref, ẋ^ref)``, and command a **modified ZMP**

    p^cmd = p^ref + k_p (x − x^ref) + k_v (ẋ − ẋ^ref)                      (eq.)

The error ``e = x − x^ref`` then obeys (subtracting the reference LIP dynamics)

    ë = ω²(e − (p^cmd − p^ref)) = ω²(1 − k_p) e − ω² k_v ė,

a second-order system with characteristic polynomial
``s² + ω²k_v s + ω²(k_p − 1)``. It is **stable iff k_p > 1 and k_v > 0** — i.e.
the position gain must *over-shift* the ZMP past the error to overcome the
pendulum's instability. :func:`gains_for_poles` places a double real pole at
``−λ`` by ``k_v = 2λ/ω²``, ``k_p = 1 + λ²/ω²``.

**The honest limit (and the link back to stepping).** The commanded ZMP can only
be realised while it stays inside the **support foot**. We saturate ``p^cmd`` to
``p^ref ± foot_half``. A push within the in-place-capturable margin
(``|ẋ|/ω ≤ foot_half``, the capture-point condition) is rejected with the ZMP
inside the foot; a larger push saturates the ankle — the stabilizer alone can no
longer recover and the robot **must take a step** (which is exactly what
:mod:`capture_point` / :mod:`herdt_walk` do). So this module is the ankle-strategy
feedback layer; the stepping modules are its escalation.

The "real" robot is integrated with the **exact** LIPM solution over each tick
(constant ZMP held by zero-order hold), not the cart-table triple integrator the
reference used — so the reproduction also exercises the model mismatch the
stabilizer has to reject. Everything is pure Python, sagittal (1-D) like the rest
of the thread; it reuses :mod:`lipm_walk`'s preview controller to build the
reference walk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import lipm_walk

DEFAULT_G = 9.8  # gravity (m/s^2)


# --- gains ------------------------------------------------------------------


def gains_for_poles(lam, omega):
    """Place the error dynamics' poles at a double real ``−lam``.

    The closed-loop error obeys ``ë + ω²k_v ė + ω²(k_p−1) e = 0`` with
    characteristic polynomial ``s² + ω²k_v s + ω²(k_p−1)``. Matching ``(s+λ)²``
    gives ``ω²k_v = 2λ`` and ``ω²(k_p−1) = λ²``. Returns ``(k_p, k_v)`` — note
    ``k_p = 1 + λ²/ω² > 1`` always, the condition that overcomes the pendulum's
    instability.
    """
    w2 = omega * omega
    k_v = 2.0 * lam / w2
    k_p = 1.0 + lam * lam / w2
    return k_p, k_v


@dataclass
class StabilizerParams:
    """Configuration of the LIPM-tracking stabilizer."""

    z_h: float = 0.8        # CoM height (m)
    dt: float = 0.02        # control period (s)
    g: float = DEFAULT_G
    k_p: float = 2.306      # CoM position-error gain (> 1 for stability)
    k_v: float = 0.653      # CoM velocity-error gain (> 0)
    foot_half: float = 0.05  # support half-width for ZMP saturation (m)

    @property
    def omega(self):
        """LIPM natural frequency ``ω = sqrt(g / z_h)``."""
        return math.sqrt(self.g / self.z_h)


def stabilizer_params(lam=4.0, *, z_h=0.8, dt=0.02, g=DEFAULT_G, foot_half=0.05):
    """Build :class:`StabilizerParams` with gains placing the poles at ``−lam``."""
    omega = math.sqrt(g / z_h)
    k_p, k_v = gains_for_poles(lam, omega)
    return StabilizerParams(z_h=z_h, dt=dt, g=g, k_p=k_p, k_v=k_v,
                            foot_half=foot_half)


# --- exact LIPM integrator (the "real" robot) -------------------------------


def lip_step(x, v, p, omega, dt):
    """One exact LIPM step under a constant ZMP ``p`` (zero-order hold).

    Solves ``ẍ = ω²(x − p)`` analytically over ``dt``:
    ``x⁺ = (x−p)cosh(ωdt) + (ẋ/ω)sinh(ωdt) + p``,
    ``ẋ⁺ = (x−p)ω sinh(ωdt) + ẋ cosh(ωdt)``.
    """
    ch = math.cosh(omega * dt)
    sh = math.sinh(omega * dt)
    xn = (x - p) * ch + (v / omega) * sh + p
    vn = (x - p) * omega * sh + v * ch
    return xn, vn


# --- closed-loop spectral analysis (standing / linear regime) ---------------


def closed_loop_matrix(params):
    """The exact 2×2 discrete error-transition matrix in the linear regime.

    For the standing reference (``x^ref = ẋ^ref = p^ref = 0``) and an
    unsaturated command ``p = k_p x + k_v ẋ``, one exact LIPM step is linear in
    ``[x, ẋ]``; this returns that matrix. Its eigenvalues are the realised
    closed-loop poles (sampled-data), which approach ``e^{−λ dt}`` as ``dt→0``.
    """
    omega, dt = params.omega, params.dt
    ch, sh = math.cosh(omega * dt), math.sinh(omega * dt)
    kp, kv = params.k_p, params.k_v
    return [[ch + kp * (1.0 - ch), sh / omega + kv * (1.0 - ch)],
            [omega * sh * (1.0 - kp), ch - kv * omega * sh]]


def _eig2_mag(M):
    """Magnitudes of the two eigenvalues of a 2×2 matrix."""
    tr = M[0][0] + M[1][1]
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    disc = tr * tr - 4.0 * det
    if disc >= 0.0:
        r = math.sqrt(disc)
        return [abs((tr + r) / 2.0), abs((tr - r) / 2.0)]
    return [math.sqrt(det), math.sqrt(det)]  # complex pair: |λ| = sqrt(det)


def spectral_radius(M):
    return max(_eig2_mag(M))


def continuous_rate(params):
    """Realised decay rate ``−ln(ρ)/dt`` of the closed-loop error (ρ = spectral radius)."""
    rho = spectral_radius(closed_loop_matrix(params))
    return -math.log(rho) / params.dt


# --- reference trajectory (reuse the preview controller) --------------------


def reference_trajectory(zmp_ref, *, params, preview_steps=None, Q=1.0, R=1e-8):
    """Run :mod:`lipm_walk`'s ZMP-preview controller to build a reference walk.

    Returns ``(com_ref, vel_ref, zmp_ref_induced)``: the CoM position, CoM
    velocity, and the *induced* ZMP (``c x``) that, together, form a
    dynamically-consistent LIPM reference for the stabilizer to track.
    """
    if preview_steps is None:
        preview_steps = int(round(1.6 / params.dt))
    g = lipm_walk.preview_gains(z_h=params.z_h, dt=params.dt,
                                preview_steps=preview_steps, Q=Q, R=R,
                                g=params.g)
    A, b, c = lipm_walk._cart_table_system(params.z_h, params.dt, params.g)
    K, f, N = g.K, g.f, len(g.f)
    n = len(zmp_ref)
    x = [zmp_ref[0] if zmp_ref else 0.0, 0.0, 0.0]
    com, vel, zmp = [], [], []
    for k in range(n):
        preview = 0.0
        for j in range(1, N + 1):
            preview += f[j - 1] * zmp_ref[min(k + j, n - 1)]
        u = -lipm_walk._dot(K, x) + preview
        com.append(x[0])
        vel.append(x[1])
        zmp.append(lipm_walk._dot(c, x))
        x = [A[0][0] * x[0] + A[0][1] * x[1] + A[0][2] * x[2] + b[0] * u,
             A[1][1] * x[1] + A[1][2] * x[2] + b[1] * u,
             A[2][2] * x[2] + b[2] * u]
    return com, vel, zmp


def stepping_zmp_reference(step_len, step_ticks, n_feet, *, settle_ticks=80):
    """A forward-stepping piecewise-constant ZMP reference (one entry per tick)."""
    ref = []
    for fidx in range(n_feet):
        ref.extend([fidx * step_len] * step_ticks)
    ref.extend([(n_feet - 1) * step_len] * settle_ticks)
    return ref


# --- closed-loop result -----------------------------------------------------


@dataclass
class StabilizerResult:
    """The closed-loop trajectory of a stabilized (or open-loop) run."""

    t: list
    com: list             # actual CoM position
    com_vel: list         # actual CoM velocity
    zmp: list             # realised (saturated) ZMP
    zmp_cmd: list         # commanded (unsaturated) ZMP
    com_ref: list
    vel_ref: list
    zmp_ref: list
    foot_half: float
    saturated: list       # per-tick: was the command clipped to the foot?
    stabilize: bool
    params: StabilizerParams

    def error(self):
        return [self.com[k] - self.com_ref[k] for k in range(len(self.com))]

    def final_error(self):
        return abs(self.com[-1] - self.com_ref[-1])

    def max_error(self):
        return max(abs(e) for e in self.error())

    def rms_error(self):
        e = self.error()
        return math.sqrt(sum(ei * ei for ei in e) / len(e))

    def converged(self, tol=5e-3):
        """True iff the CoM has tracked back to the reference by the end."""
        return self.final_error() < tol

    def diverged(self, cap=1.0):
        """True iff the tracking error grew past ``cap`` (the robot fell)."""
        return self.max_error() > cap

    def ever_saturated(self):
        """True iff the commanded ZMP ever exceeded the support foot."""
        return any(self.saturated)

    def realised_zmp_in_support(self, tol=1e-9):
        """The realised ZMP is clipped to the foot, so this is the guarantee."""
        return all(abs(self.zmp[k] - self.zmp_ref[k]) <= self.foot_half + tol
                   for k in range(len(self.zmp)))

    def steady_error(self, last=40):
        """Mean tracking error over the final ``last`` ticks (for a bias)."""
        e = self.error()[-last:]
        return sum(e) / len(e)


def simulate_stabilizer(zmp_ref, com_ref, vel_ref, *, params, stabilize=True,
                        push_tick=None, push_dv=0.0, zmp_bias=0.0, x0=None,
                        n_steps=None):
    """Run the LIPM-tracking stabilizer in closed loop over a reference walk.

    ``zmp_ref`` / ``com_ref`` / ``vel_ref`` are the reference triple (e.g. from
    :func:`reference_trajectory`). The "real" robot starts on the reference and
    is integrated with the exact LIPM solution. A push (CoM-velocity jump
    ``push_dv`` at ``push_tick``) and/or a persistent ZMP modelling error
    ``zmp_bias`` can be injected. With ``stabilize=False`` the command is the
    bare reference ZMP (open loop), which lets the unstable pendulum diverge.
    Returns a :class:`StabilizerResult`.
    """
    omega, dt = params.omega, params.dt
    if n_steps is None:
        n_steps = len(com_ref)
    n_steps = min(n_steps, len(com_ref))
    x = com_ref[0] if x0 is None else x0[0]
    v = vel_ref[0] if x0 is None else x0[1]
    t, com, vel, zmp, zmp_cmd, sat = [], [], [], [], [], []
    for k in range(n_steps):
        if push_tick is not None and k == push_tick:
            v += push_dv
        e = x - com_ref[k]
        edot = v - vel_ref[k]
        if stabilize:
            p_cmd = zmp_ref[k] + params.k_p * e + params.k_v * edot
        else:
            p_cmd = zmp_ref[k]
        lo = zmp_ref[k] - params.foot_half
        hi = zmp_ref[k] + params.foot_half
        p_sat = min(max(p_cmd, lo), hi)
        t.append(k * dt)
        com.append(x)
        vel.append(v)
        zmp.append(p_sat)
        zmp_cmd.append(p_cmd)
        sat.append(abs(p_sat - p_cmd) > 1e-12)
        # the real robot experiences the realised ZMP plus the modelling error
        x, v = lip_step(x, v, p_sat + zmp_bias, omega, dt)
        if abs(x) > 1e6:  # overflow guard for a diverging open-loop run
            x = math.copysign(1e6, x)
            v = math.copysign(1e6, v)
    return StabilizerResult(
        t=t, com=com, com_vel=vel, zmp=zmp, zmp_cmd=zmp_cmd,
        com_ref=com_ref[:n_steps], vel_ref=vel_ref[:n_steps],
        zmp_ref=zmp_ref[:n_steps], foot_half=params.foot_half, saturated=sat,
        stabilize=stabilize, params=params)


def standing_reference(n, *, p=0.0):
    """The standing-balance reference: a constant ZMP/CoM at ``p`` for ``n`` ticks."""
    return [p] * n, [p] * n, [0.0] * n
