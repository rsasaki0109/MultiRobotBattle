"""N-step capturability analysis (Koolen, de Boer, Rebula, Goswami & Pratt 2012).

*Capturability-based analysis and control of legged locomotion, Part 1: Theory
and application to three simple gait models* (IJRR 31(9)). This is the analytic
backbone of the whole push-recovery sub-thread: it makes precise the question
:mod:`capture_point` and :mod:`push_recovery` answered piecewise — *given a push,
in how few steps can the robot come to a complete stop, and is it recoverable at
all?*

A state is **N-step capturable** if the robot can come to rest without falling by
taking ``N`` or fewer steps. The set of capture points from which that is possible
is the **N-step capture region**; its size is the **N-step capturability margin**.
Koolen analyses three nested gait models, all on the Linear Inverted Pendulum:

* **model 1 — point foot.** Balance only by where you *step*. The Center of
  Pressure is pinned under the (point) foot, so standing still you cannot capture
  any disturbance — you *must* step. ``model="point"``.
* **model 2 — finite foot.** A foot of half-length ``foot_half`` lets the CoP
  travel within the sole — the ankle strategy of :mod:`push_recovery`. Adds a
  constant ``foot_half`` to every region. ``model="foot"``.
* **model 3 — reaction mass.** A spinnable flywheel (centroidal angular momentum)
  shifts the *Centroidal Moment Pivot* beyond the foot by ``reaction_shift`` — the
  hip strategy of :mod:`push_recovery`. ``model="reaction"``.

**The dynamics.** Split the LIPM through the Divergent Component of Motion
``ξ = x + ẋ/ω`` (the instantaneous capture point), which runs away from the CoP as
``ξ̇ = ω(ξ − p)``. A step takes a swing time ``T``; while the swing foot is in the
air the *old* (point) foot is the pivot, so over one step period the capture point
diverges by the growth factor ``e^{ωT}`` before the new foot can act. Measuring the
capture point offset from the current stance, the per-step recursion is

    ηₖ₊₁ = ηₖ · e^{ωT} − sₖ₊₁ ,   |sₖ| ≤ l_max ,   capture when |η_N| ≤ foot_eff,

with ``foot_eff`` the model's effective CoP reach (0 / ``foot_half`` /
``foot_half + reaction_shift``), available *every* support phase. Maximising the
recoverable push (CoP at the foot edge each phase, step as far as possible every
time, ``sₖ = l_max``) collapses to a **geometric series** — the closed-form N-step
capture region:

    ξ_N = foot_eff + l_max·(e^{−ωT} + e^{−2ωT} + ⋯ + e^{−NωT})
        = foot_eff + l_max·e^{−ωT}·(1 − e^{−NωT})/(1 − e^{−ωT}).

The regions are **nested** (ξ₀ ⊂ ξ₁ ⊂ ⋯) and, crucially, **bounded**: even with
infinitely many steps the capture point grows faster than the feet can chase it
once it is too far, so

    ξ_∞ = foot_eff + l_max/(e^{ωT} − 1)

is a hard **capturability limit** — a push past it is unrecoverable by *any* number
of steps. Longer swing time ``T`` (CP diverges more between steps) or shorter steps
``l_max`` shrink the regions.

Everything is exact and pure Python: the closed form is *certified* against an
exact greedy LIPM rollout (:func:`simulate_greedy`), and the three models reduce
to :mod:`capture_point` (point) and :mod:`push_recovery` ankle/hip (foot/reaction).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_G = 9.81

MODELS = ("point", "foot", "reaction")


@dataclass
class CaptureParams:
    """Parameters of the N-step capturability analysis (sagittal, 1-D)."""

    g: float = DEFAULT_G
    z_com: float = 1.0          # CoM height -> ω = sqrt(g/z)
    step_time: float = 0.35     # swing time T between steps
    l_max: float = 0.5          # maximum step length
    foot_half: float = 0.08     # foot half-length (model 2)
    reaction_shift: float = 0.0  # extra CMP reach from the flywheel (model 3)

    @property
    def omega(self):
        """LIPM natural frequency ``ω = sqrt(g/z_com)``."""
        return math.sqrt(self.g / self.z_com)

    @property
    def growth(self):
        """Capture-point growth factor over one step period, ``e^{ωT}``."""
        return math.exp(self.omega * self.step_time)

    def foot_eff(self, model="point"):
        """Effective CoP reach for a model: 0 / foot / foot+reaction."""
        if model == "point":
            return 0.0
        if model == "foot":
            return self.foot_half
        if model == "reaction":
            return self.foot_half + self.reaction_shift
        raise ValueError(f"unknown model {model!r}; use one of {MODELS}")


def n_step_region(params, n, *, model="point"):
    """Half-width of the **N-step capture region** (Koolen's closed form).

    The capture point ``ξ`` (offset from the current stance foot) is ``n``-step
    capturable iff ``|ξ| <= n_step_region(params, n, model=...)``. ``n == 0`` is the
    in-place region (the foot/CMP can absorb it without stepping).
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    w, T = params.omega, params.step_time
    r = math.exp(-w * T)
    geo = 0.0 if n == 0 else params.l_max * r * (1.0 - r ** n) / (1.0 - r)
    return params.foot_eff(model) + geo


def inf_step_region(params, *, model="point"):
    """The **capturability limit** ``ξ_∞ = foot_eff + l_max/(e^{ωT}−1)``.

    A push whose capture point lies beyond this cannot be stopped by *any* number
    of steps — the capture point outruns the feet. The N-step regions converge up
    to it from below.
    """
    return params.foot_eff(model) + params.l_max / (params.growth - 1.0)


def capture_region(params, n, *, model="point"):
    """The N-step capture region as a symmetric interval ``(-half, half)``."""
    half = n_step_region(params, n, model=model)
    return (-half, half)


def capturability_margin(xi, params, *, model="point"):
    """Smallest ``N`` for which capture point ``ξ`` is N-step capturable.

    Returns ``0`` if already balanced (``|ξ| <= foot_eff``), a positive integer
    ``N`` if recoverable in exactly ``N`` steps, or ``math.inf`` if ``|ξ|`` is at
    or beyond the capturability limit (unrecoverable). Closed form: invert the
    geometric series ``|ξ| = foot_eff·e^{−NωT} + A(1 − e^{−NωT})`` for ``N``.
    """
    a = abs(xi)
    foot = params.foot_eff(model)
    if a <= foot + 1e-12:
        return 0
    A = params.l_max / (params.growth - 1.0)  # geometric part of ξ_∞ = foot + A
    # |ξ| = foot + A(1 − e^{−NωT})  ⇒  e^{−NωT} = 1 − (|ξ|−foot)/A
    frac = (a - foot) / A
    if frac >= 1.0 - 1e-12:
        return math.inf
    return math.ceil(-math.log(1.0 - frac) / (params.omega * params.step_time) - 1e-9)


@dataclass
class GreedyRecovery:
    """Outcome of an exact greedy max-step LIPM recovery rollout."""

    captured: bool
    num_steps: int            # steps actually taken (math.inf-safe: capped count)
    feet: list                # absolute stance-foot positions placed
    cp_offsets: list          # capture-point offset from stance before each step

    def margin(self):
        """Steps needed, or ``math.inf`` if it never captured within the cap."""
        return self.num_steps if self.captured else math.inf


def simulate_greedy(xi0, params, *, model="point", max_steps=200):
    """Exact greedy recovery: step as far as possible toward the capture point.

    This is the *certifying simulation* for the closed-form regions. Starting from
    capture-point offset ``xi0`` (from the initial stance), each support phase puts
    the CoP at the foot edge toward the capture point (the most arresting use of
    the foot/flywheel), lets the CP diverge over the swing period, then places the
    new foot at most ``l_max`` away. Captures when the diverged CP can be brought
    within the model's effective foot. The step count it returns equals
    :func:`capturability_margin` to the integer.
    """
    w, T = params.omega, params.step_time
    foot = params.foot_eff(model)
    grow = math.exp(w * T)
    if abs(xi0) <= foot + 1e-12:
        return GreedyRecovery(True, 0, [], [xi0])     # already balanced
    eta = xi0          # CP offset from current stance at the decision instant
    stance = 0.0       # absolute stance-foot position
    feet, offsets = [], []
    for k in range(1, max_steps + 1):
        offsets.append(eta)
        pivot = math.copysign(foot, eta)               # CoP at foot edge toward CP
        land = pivot + (eta - pivot) * grow            # CP diverges over the swing
        if abs(land) <= params.l_max + foot + 1e-9:
            # place the k-th foot to bring the (diverged) CP within the foot
            step = math.copysign(min(params.l_max, abs(land) - foot), land)
            feet.append(stance + step)
            return GreedyRecovery(True, k, feet, offsets)
        step = math.copysign(params.l_max, eta)        # step as far as possible
        stance += step
        feet.append(stance)
        eta = land - step                              # residual becomes next CP
    return GreedyRecovery(False, max_steps, feet, offsets)
