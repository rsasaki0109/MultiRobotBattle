"""Humanoid push recovery: ankle / hip / step decision surfaces (Stephens 2007).

A faithful, pure-Python reproduction of Benjamin Stephens, *"Humanoid Push
Recovery"* (IEEE-RAS Humanoids 2007).

This is the **unifying analysis** of the push-recovery family this humanoid
sub-thread already builds piecewise. :mod:`capture_point` answers *where to step*;
:mod:`kajita_stabilizer` is the *ankle* feedback that saturates and then needs a
step. Stephens ties the three classical balance strategies — **ankle**, **hip**,
**step** — into a single picture and derives the **decision surfaces** in the CoM
phase plane ``(x, ẋ)`` that say *which* strategy can still avert a fall.

All three are read off the **instantaneous capture point** ``ξ = x + ẋ/ω``
(``ω² = g/L``, ``L`` the CoM height):

* **Ankle (CoP balancing).** Only ankle torque ⇒ the Center of Pressure stays
  inside the foot ``[δ⁻, δ⁺]``. The CoM can be brought to rest **iff the capture
  point is inside the foot** — Stephens eq. (4):

      δ⁻ < x + ẋ/ω < δ⁺.

* **Hip (CMP balancing).** Internal joints (torso/arms, modelled as a **flywheel**
  at the CoM, the *Linear Inverted Pendulum Plus Flywheel*) apply a bounded torque
  ``τ_max`` for a bounded angle ``θ_max``. A momentum-generating torque moves the
  effective pivot — the **Centroidal Moment Point** — *beyond* the foot edge by
  ``τ/(mg)`` (eqs. 5-7). A bang-bang torque profile (eq. 8) spins the flywheel up
  and back to rest within the angle budget; the longest such pulse lasts
  ``T_max = √(I θ_max / τ_max)`` (eq. 12). Integrating the dynamics over it widens
  the capturable interval by

      Δ_hip = (τ_max / mg) · (1 − e^{−ω T_max})²

  so the hip strategy recovers **iff** ``δ⁻ − Δ_hip < ξ < δ⁺ + Δ_hip``.

  *Note on eq. (15).* The paper's printed eq. (15) writes this widening as
  ``(τ_max/mg)(e^{ω T_max} − 1)²``. Re-deriving from the paper's own eq. (13) — and
  confirming against an **exact** integration of the bang-bang LIPPF dynamics
  (:func:`hip_recovery_boundary`) — gives ``(1 − e^{−ω T_max})²``; the two differ
  by a factor ``e^{2ω T_max}``. We use the dynamically-consistent form (it matches
  the simulation to machine precision) and flag the printed one as a typo.

* **Step.** Past the hip interval, no amount of body torquing helps — the support
  must move. The foot is placed at the capture point (this is exactly
  :mod:`capture_point`'s job); a step recovers while ``|ξ|`` is within the leg's
  reach ``max_step`` (then double/CoP balancing finishes), and beyond that the
  fall is inevitable for a single step.

So the strategies **nest**: ``ankle ⊂ hip ⊂ step``, with strictly ordered
boundaries ``δ⁺ < δ⁺ + Δ_hip < max_step``. :func:`classify` returns which one the
state needs. Everything is pure Python, sagittal (1-D point-mass) like the rest of
the thread; it reuses :mod:`kajita_stabilizer`'s exact LIPM integrator and defers
the step geometry to :mod:`capture_point`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .kajita_stabilizer import lip_step

DEFAULT_G = 9.81  # gravity (m/s^2)


@dataclass
class StrategyParams:
    """Model + actuation limits for the three balance strategies."""

    g: float = DEFAULT_G
    m: float = 60.0          # body mass (kg)
    z_com: float = 1.0       # CoM height L (m)
    delta_back: float = -0.10   # heel edge δ⁻ (m, behind the CoM-over-ankle origin)
    delta_front: float = 0.10   # toe edge δ⁺ (m)
    flywheel_inertia: float = 5.0   # I (kg·m²)
    tau_max: float = 100.0          # flywheel torque limit (N·m)
    theta_max: float = 0.5          # flywheel angle limit (rad)
    max_step: float = 0.5           # longest single step the leg can reach (m)

    @property
    def omega(self):
        """LIPM natural frequency ``ω = sqrt(g / L)``."""
        return math.sqrt(self.g / self.z_com)

    @property
    def t_max(self):
        """Longest bang-bang half-pulse ``T_max = √(I θ_max / τ_max)`` (eq. 12)."""
        return math.sqrt(self.flywheel_inertia * self.theta_max / self.tau_max)

    @property
    def cmp_shift(self):
        """How far the flywheel can push the CMP past the foot: ``τ_max/(mg)``."""
        return self.tau_max / (self.m * self.g)

    @property
    def delta_hip(self):
        """Capturable-interval widening from the hip strategy (see module docs).

        ``Δ_hip = (τ_max/mg)·(1 − e^{−ω T_max})²`` — the dynamically-consistent
        form (the paper's printed ``(e^{ωT_max}−1)²`` is a typo; see
        :func:`hip_recovery_boundary`, which matches this to machine precision).
        """
        return self.cmp_shift * (1.0 - math.exp(-self.omega * self.t_max)) ** 2


# --- capture point & strategy classification --------------------------------


def capture_point(x, x_dot, params):
    """The instantaneous capture point ``ξ = x + ẋ/ω``."""
    return x + x_dot / params.omega


def classify(x, x_dot, params):
    """Return the minimal strategy that can recover state ``(x, ẋ)``.

    One of ``"ankle"`` (capture point in the foot), ``"hip"`` (within the
    flywheel-widened interval), ``"step"`` (within a leg's reach), or ``"fall"``
    (capture point beyond the longest step — a single step cannot catch it).
    """
    xi = capture_point(x, x_dot, params)
    if params.delta_back <= xi <= params.delta_front:
        return "ankle"
    if (params.delta_back - params.delta_hip <= xi
            <= params.delta_front + params.delta_hip):
        return "hip"
    if abs(xi) <= params.max_step:
        return "step"
    return "fall"


# --- ankle strategy: CoP balancing ------------------------------------------


@dataclass
class RecoveryResult:
    """A simulated balance-recovery run."""

    t: list
    x: list                 # CoM position
    x_dot: list             # CoM velocity
    cop: list               # center of pressure (ankle) / CMP (hip)
    theta: list             # flywheel angle (hip only; zeros for ankle)
    strategy: str
    params: StrategyParams

    def capture_points(self):
        w = self.params.omega
        return [self.x[k] + self.x_dot[k] / w for k in range(len(self.x))]

    def final_capture_point(self):
        return self.capture_points()[-1]

    def captured(self, vel_tol=2e-2):
        """True iff the CoM has come (near) to rest with ξ inside the foot."""
        p = self.params
        return (abs(self.x_dot[-1]) < vel_tol
                and p.delta_back - 1e-9 <= self.final_capture_point()
                <= p.delta_front + 1e-9)

    def peak_theta(self):
        return max(abs(th) for th in self.theta)

    def theta_within_limit(self, tol=1e-9):
        return self.peak_theta() <= self.params.theta_max + tol

    def max_excursion(self):
        return max(abs(xi) for xi in self.x)


def simulate_ankle(x0, x_dot0, params, *, dt=0.005, duration=3.0):
    """CoP balancing: place the CoP at the (clamped) capture point and hold.

    Setting the CoP to the instantaneous capture point freezes ``ξ`` and lets the
    CoM converge to it (the convergent component decays). Clamped to the foot, it
    captures **iff** the capture point is inside the foot — i.e. it realises the
    eq.-(4) decision surface. Returns a :class:`RecoveryResult`.
    """
    w = params.omega
    x, v = float(x0), float(x_dot0)
    n = int(round(duration / dt))
    t, xs, vs, cops = [], [], [], []
    for k in range(n):
        xi = x + v / w
        p = min(max(xi, params.delta_back), params.delta_front)
        t.append(k * dt)
        xs.append(x)
        vs.append(v)
        cops.append(p)
        x, v = lip_step(x, v, p, w, dt)
        if abs(x) > 1e3:
            break
    return RecoveryResult(t=t, x=xs, x_dot=vs, cop=cops,
                          theta=[0.0] * len(xs), strategy="ankle",
                          params=params)


# --- hip strategy: bang-bang flywheel (LIPPF) -------------------------------


def hip_recovery_boundary(params, *, samples=80):
    """The largest capture point the bang-bang flywheel recovers, by simulation.

    Exactly integrates the LIPPF over the worst-case bang-bang pulse
    (``+τ_max`` then ``−τ_max``, each for ``T_max``, eqs. 6-8) from ``(0, v₀)`` and
    bisects ``v₀`` for the boundary where the post-pulse capture point lands on the
    front foot edge ``δ⁺`` (eq. 9). Returns that boundary capture point ``ξ*``; it
    equals ``δ⁺ + Δ_hip`` to machine precision, certifying the closed form.
    """
    w, T = params.omega, params.t_max
    a = params.cmp_shift
    p1 = params.delta_front + a    # τ = +τ_max  (CMP pushed forward of the toe)
    p2 = params.delta_front - a    # τ = −τ_max  (flywheel spun back to rest)

    def xi_after_pulse(v0):
        x1, v1 = lip_step(0.0, v0, p1, w, T)
        x2, v2 = lip_step(x1, v1, p2, w, T)
        return x2 + v2 / w

    lo, hi = 0.0, 5.0 * params.max_step * w
    for _ in range(samples):
        mid = 0.5 * (lo + hi)
        if xi_after_pulse(mid) < params.delta_front:
            lo = mid
        else:
            hi = mid
    v_star = 0.5 * (lo + hi)
    return v_star / w  # ξ* = x0 + v0/ω with x0 = 0


def simulate_hip(x0, x_dot0, params, *, dt=0.002, duration=3.0):
    """Hip strategy: a bang-bang flywheel pulse, then CoP balancing finishes.

    For a forward push the ankle saturates at the toe ``δ⁺`` while the flywheel
    applies ``+τ_max`` for ``T_max`` then ``−τ_max`` for ``T_max`` (the CMP rides
    ``δ⁺ ± τ_max/mg``), returning the flywheel to rest within ``θ_max`` (eqs.
    8-12). The pulse drives the capture point back toward the foot; CoP balancing
    then drives the CoM to rest. Mirrored for a backward push. Returns a
    :class:`RecoveryResult`.
    """
    w, T = params.omega, params.t_max
    a = params.cmp_shift
    I, tau = params.flywheel_inertia, params.tau_max
    xi0 = x0 + x_dot0 / w
    fwd = xi0 >= 0.0
    edge = params.delta_front if fwd else params.delta_back
    sign = 1.0 if fwd else -1.0   # torque sign that pushes the CMP outward

    x, v, th, th_dot = float(x0), float(x_dot0), 0.0, 0.0
    t, xs, vs, cops, ths = [], [], [], [], []
    n_pulse = int(round(T / dt))
    tnow = 0.0
    # two bang-bang segments: +τ_max then −τ_max, each for T_max
    for seg, tau_seg in enumerate((sign * tau, -sign * tau)):
        pivot = edge + sign * a if seg == 0 else edge - sign * a
        for _ in range(n_pulse):
            t.append(tnow)
            xs.append(x)
            vs.append(v)
            cops.append(pivot)
            ths.append(th)
            x, v = lip_step(x, v, pivot, w, dt)
            th_dot += (tau_seg / I) * dt
            th += th_dot * dt
            tnow += dt
    # flywheel back at rest; hand over to CoP balancing for the remainder
    n_rest = int(round((duration - tnow) / dt))
    for _ in range(max(0, n_rest)):
        xi = x + v / w
        p = min(max(xi, params.delta_back), params.delta_front)
        t.append(tnow)
        xs.append(x)
        vs.append(v)
        cops.append(p)
        ths.append(th)
        x, v = lip_step(x, v, p, w, dt)
        tnow += dt
        if abs(x) > 1e3:
            break
    return RecoveryResult(t=t, x=xs, x_dot=vs, cop=cops, theta=ths,
                          strategy="hip", params=params)
