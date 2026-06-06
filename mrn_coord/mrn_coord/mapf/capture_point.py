"""Capture Point — a step toward humanoid push recovery (Pratt et al. 2006).

Builds on the same Linear Inverted Pendulum as :mod:`lipm_walk`. The question
this answers: after a push, *where should the foot step* to come to a complete
stop? The answer is the **Capture Point**.

For the LIPM (point mass at height ``z_h`` over a support point ``p``) the
sagittal dynamics are ``ẍ = ω₀² (x − p)`` with ``ω₀ = sqrt(g / z_h)``. Split the
state through the **Divergent Component of Motion** (DCM) ``ξ = x + ẋ/ω₀``. Then

    ξ̇ = ω₀ (ξ − p),

so ``ξ`` runs *away* from the support point exponentially — it is the unstable
part, the thing that makes the robot fall — while the CoM ``x`` is dragged along
behind it (``x`` converges to ``ξ``). To stop, you must arrest ``ξ``: place the
foot **at** ``ξ``. Then ``ξ̇ = 0``, ``ξ`` is frozen, and ``x → ξ``: the robot is
*captured*. That point ``ξ = x + ẋ/ω₀`` is the **(instantaneous) Capture Point**.

So push recovery is one formula plus a step: measure the CoM state, compute the
capture point, step there. Step **short** and ``ξ`` keeps diverging — the robot
topples forward over the new foot; step **long** and it falls back. If the
capture point lies beyond the longest reachable step, one step cannot stop the
fall — it is only *N-step capturable*, needing a sequence of steps.

Everything here is pure Python and exact: the LIPM has a closed-form
(``cosh``/``sinh``) solution, so the simulation is analytic, not integrated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_G = 9.8  # gravity (m/s^2)


def omega0(z_h, g=DEFAULT_G):
    """LIPM natural frequency ``ω₀ = sqrt(g / z_h)``."""
    return math.sqrt(g / z_h)


def capture_point(x, x_dot, z_h, g=DEFAULT_G):
    """Instantaneous Capture Point ``ξ = x + ẋ / ω₀`` (one coordinate).

    Stepping the support point to ``ξ`` brings the CoM asymptotically to rest
    over the new foot.
    """
    return x + x_dot / omega0(z_h, g)


@dataclass
class LIPMTrajectory:
    """A closed-form LIPM rollout over a fixed support point ``foot``."""

    t: list
    x: list           # CoM position
    x_dot: list        # CoM velocity
    xi: list           # divergent component / instantaneous capture point
    foot: float
    z_h: float

    def captured(self, pos_tol=0.03, vel_tol=0.05):
        """True iff the CoM has come to rest over the foot (push absorbed)."""
        return (abs(self.x_dot[-1]) < vel_tol
                and abs(self.x[-1] - self.foot) < pos_tol)

    def max_excursion(self):
        """Largest CoM distance from the foot over the rollout (fall = large)."""
        return max(abs(xx - self.foot) for xx in self.x)


def simulate_lipm(x0, x_dot0, foot, z_h, *, g=DEFAULT_G, dt=0.01, duration=3.0):
    """Roll out ``ẍ = ω₀²(x − foot)`` from ``(x0, x_dot0)`` over a fixed foot.

    Uses the exact solution
    ``x(t) = foot + (x0 − foot) cosh(ω₀ t) + (ẋ0/ω₀) sinh(ω₀ t)``.
    """
    w = omega0(z_h, g)
    n = int(round(duration / dt))
    t, x, v, xi = [], [], [], []
    for k in range(n + 1):
        tt = k * dt
        ch, sh = math.cosh(w * tt), math.sinh(w * tt)
        xx = foot + (x0 - foot) * ch + (x_dot0 / w) * sh
        vv = (x0 - foot) * w * sh + x_dot0 * ch
        t.append(tt)
        x.append(xx)
        v.append(vv)
        xi.append(xx + vv / w)
    return LIPMTrajectory(t, x, v, xi, foot, z_h)


@dataclass
class RecoveryStep:
    """The outcome of a one-step capture-point recovery."""

    capture_point: float    # the ideal step target ξ
    foot: float             # where the foot was actually placed (clamped)
    one_step_capturable: bool
    trajectory: LIPMTrajectory


def recover_step(x, x_dot, z_h, *, max_step, g=DEFAULT_G, dt=0.01, duration=3.0):
    """Plan a single capture-point recovery step from CoM state ``(x, x_dot)``.

    The foot is placed at the capture point ``ξ`` if it is within ``max_step``
    of the current CoM; otherwise it is clamped to the longest reachable step
    (then the push is not one-step capturable). Returns a :class:`RecoveryStep`
    with the resulting LIPM rollout over the placed foot.
    """
    xi = capture_point(x, x_dot, z_h, g)
    reach = x + max(-max_step, min(max_step, xi - x))
    capturable = abs(xi - x) <= max_step + 1e-9
    traj = simulate_lipm(x, x_dot, reach, z_h, g=g, dt=dt, duration=duration)
    return RecoveryStep(capture_point=xi, foot=reach,
                        one_step_capturable=capturable, trajectory=traj)


def n_step_capture(x, x_dot, z_h, *, max_step, step_time, g=DEFAULT_G,
                   max_steps=6):
    """Recover from a push with as many capture-point steps as it takes.

    Each step places the foot at the capture point clamped to ``max_step``,
    advances the LIPM by ``step_time``, and repeats until the residual capture
    point is within a foot's reach (captured) or ``max_steps`` is exhausted.
    Returns ``(num_steps, captured, foot_positions)`` — the smallest ``N`` for
    which the push is ``N``-step capturable (the capturability margin).
    """
    w = omega0(z_h, g)
    feet = []
    cx, cv = x, x_dot
    for i in range(max_steps):
        xi = cx + cv / w
        if abs(xi - cx) <= max_step + 1e-9:
            feet.append(xi)
            return i + 1, True, feet
        foot = cx + (max_step if xi > cx else -max_step)
        feet.append(foot)
        # advance the LIPM by one step period over this foot
        ch, sh = math.cosh(w * step_time), math.sinh(w * step_time)
        nx = foot + (cx - foot) * ch + (cv / w) * sh
        nv = (cx - foot) * w * sh + cv * ch
        cx, cv = nx, nv
    return max_steps, False, feet
