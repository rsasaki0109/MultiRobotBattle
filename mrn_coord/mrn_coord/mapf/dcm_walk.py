"""DCM walking control — Divergent Component of Motion (Englsberger et al. 2015).

Where :mod:`capture_point` answers *"after one push, where do I step to stop?"*,
this module answers the continuous-walking version: *"given a whole footstep
plan, what Center-of-Pressure trajectory drives the robot through it, and how do
I track it back to the plan after a disturbance?"* It reproduces Englsberger,
Ott & Albu-Schäffer, *"Three-Dimensional Bipedal Walking Control Based on
Divergent Component of Motion"* (IEEE Transactions on Robotics, 2015) — the
control framework that made the **Capture Point** into a full walking controller.

The same Linear Inverted Pendulum underlies both. Over a Virtual Repellent Point
``r`` (on the ground, the CoP / support point) the DCM ``ξ = x + ẋ/ω`` obeys

    ξ̇ = ω (ξ − r),                         (divergent — runs away from r)
    ẋ = −ω (x − ξ).                          (convergent — CoM trails the DCM)

The CoM equation is *stable* (eigenvalue ``−ω``): the CoM low-passes the DCM and
can never diverge on its own. All the instability is packed into the one scalar
``ξ``. So walking control reduces to **driving ``ξ``**, and ``ξ`` is driven by
where you put the foot — exactly the capture-point insight, now run continuously.

**Planning a DCM reference.** Hold a constant VRP ``p_i`` (the i-th footstep) for
a step time ``T``. Over that step the DCM is the pure exponential

    ξ(τ) = p_i + (ξ_ini,i − p_i) e^{ω τ},     τ ∈ [0, T].

Stitch steps together by continuity (``ξ_eos,i = ξ_ini,i+1``) and end at rest on
the last foot (``ξ_eos,N-1 = p_{N-1}``). Inverting the exponential gives the
**backward recursion** that is the heart of DCM planning:

    ξ_ini,i = p_i + (ξ_eos,i − p_i) e^{−ω T}.

Run it from the last step back to the first and you get a DCM reference that is
continuous and *bounded* — it weaves between the feet instead of diverging,
because each step's foot is placed to catch the previous step's divergence. The
CoM that trails it (closed form below) is a smooth, dynamically consistent
walking trajectory.

**Tracking it.** A real robot is pushed off the reference. The DCM tracking law

    r_cmd = r_ref + (1 + k_ξ/ω) (ξ − ξ_ref),     k_ξ > 0

makes the DCM error ``e = ξ − ξ_ref`` obey ``ė = −k_ξ e`` — it decays
exponentially at the rate *you* choose. Without the feedback term (``r_cmd =
r_ref``) the same error obeys ``ė = +ω e`` and blows up: the open-loop LIPM is
unstable, and the one feedback term is what stabilises the walk.

Everything is the exact LIPM solution (``e^{±ω t}``), so the rollouts are
analytic, not integrated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mrn_coord.mapf.capture_point import DEFAULT_G, omega0


@dataclass
class DCMPlan:
    """A planned DCM reference over a footstep sequence.

    ``vrp`` are the per-step constant Virtual Repellent Points (the footstep
    positions / CoPs); ``xi_ini``/``xi_eos`` are the per-step boundary DCMs from
    the backward recursion. ``t``/``xi``/``vrp_t``/``com`` are the densely
    sampled reference trajectories (DCM, the VRP held over each step, and the
    trailing CoM).
    """

    vrp: list           # per-step VRP (foot) positions
    xi_ini: list         # per-step initial DCM (start of step)
    xi_eos: list         # per-step end-of-step DCM
    step_time: float
    z_h: float
    t: list
    xi: list             # sampled DCM reference
    vrp_t: list          # sampled VRP (piecewise constant per step)
    com: list            # sampled CoM (trails the DCM)

    def dcm_excursion(self):
        """Largest DCM distance outside the span of the feet (boundedness)."""
        lo, hi = min(self.vrp), max(self.vrp)
        return max(0.0, max(self.xi) - hi, lo - min(self.xi))

    def com_in_support_span(self, margin=1e-9):
        """True iff the CoM never leaves the span of the planned feet."""
        lo, hi = min(self.vrp), max(self.vrp)
        return all(lo - margin <= c <= hi + margin for c in self.com)


def plan_dcm_reference(vrp, step_time, z_h, *, g=DEFAULT_G, dt=0.01, com0=None):
    """Plan a continuous, bounded DCM reference over constant-VRP footsteps.

    ``vrp`` is the list of per-step Virtual Repellent Points (footstep / CoP
    positions, one coordinate), each held for ``step_time``. Returns a
    :class:`DCMPlan` with the boundary DCMs from the backward recursion and the
    densely sampled DCM / VRP / CoM reference trajectories. ``com0`` defaults to
    the first foot (start of support).
    """
    if not vrp:
        raise ValueError("need at least one VRP / footstep")
    w = omega0(z_h, g)
    decay = math.exp(-w * step_time)

    n = len(vrp)
    xi_ini = [0.0] * n
    xi_eos = [0.0] * n
    # terminal: come to rest on the last foot
    xi_eos[n - 1] = vrp[n - 1]
    for i in range(n - 1, -1, -1):
        if i < n - 1:
            xi_eos[i] = xi_ini[i + 1]            # continuity across steps
        xi_ini[i] = vrp[i] + (xi_eos[i] - vrp[i]) * decay

    # dense sampling: DCM is exponential per step, CoM is its exact trailing sol.
    t, xi_s, vrp_s, com_s = [], [], [], []
    cx = vrp[0] if com0 is None else com0
    clock = 0.0
    steps = int(round(step_time / dt))
    for i in range(n):
        p = vrp[i]
        xi0 = xi_ini[i]
        # CoM closed form for ẋ = ω(ξ − x) with ξ(τ) = p + (xi0−p) e^{ωτ}:
        #   x(τ) = p + (xi0−p)/2 · e^{ωτ} + B e^{−ωτ},  B = x0 − p − (xi0−p)/2
        B = cx - p - (xi0 - p) / 2.0
        for k in range(steps):
            tau = k * dt
            ch_p = math.exp(w * tau)
            ch_m = math.exp(-w * tau)
            xi_v = p + (xi0 - p) * ch_p
            cx_v = p + (xi0 - p) / 2.0 * ch_p + B * ch_m
            t.append(clock)
            xi_s.append(xi_v)
            vrp_s.append(p)
            com_s.append(cx_v)
            clock += dt
        # advance CoM to the end of this step (continuity into the next)
        tau = steps * dt
        cx = p + (xi0 - p) / 2.0 * math.exp(w * tau) + B * math.exp(-w * tau)

    return DCMPlan(vrp=list(vrp), xi_ini=xi_ini, xi_eos=xi_eos,
                   step_time=step_time, z_h=z_h, t=t, xi=xi_s, vrp_t=vrp_s,
                   com=com_s)


def vrp_command(xi, xi_ref, vrp_ref, k_xi, w):
    """DCM tracking control law ``r_cmd = r_ref + (1 + k_ξ/ω)(ξ − ξ_ref)``.

    Drives the DCM tracking error ``ξ − ξ_ref`` to zero as ``e^{−k_ξ t}``.
    """
    return vrp_ref + (1.0 + k_xi / w) * (xi - xi_ref)


@dataclass
class TrackingResult:
    """A closed-loop DCM tracking rollout under an initial disturbance."""

    t: list
    xi: list             # actual DCM
    xi_ref: list          # reference DCM
    vrp_cmd: list         # commanded VRP
    err: list             # ξ − ξ_ref
    k_xi: float
    z_h: float

    def decay_rate(self):
        """Fitted exponential rate of |error| (≈ ``k_ξ``; negative = divergence).

        Returns ``−d/dt log|e|`` from the endpoints, i.e. the error envelope is
        ``≈ |e₀| e^{−rate·t}``. A positive rate means convergence.
        """
        e0, e1 = abs(self.err[0]), abs(self.err[-1])
        if e0 <= 0 or e1 <= 0:
            return float("inf")
        return -(math.log(e1) - math.log(e0)) / (self.t[-1] - self.t[0])

    def converged(self, tol=0.02):
        return abs(self.err[-1]) < tol


def track_dcm(plan, xi0, *, k_xi, g=DEFAULT_G, dt=0.01, feedback=True):
    """Simulate DCM tracking of ``plan`` from a disturbed initial DCM ``xi0``.

    At each tick the controller reads the reference DCM/VRP and the LIPM-DCM
    plant ``ξ̇ = ω(ξ − r_cmd)`` advances exactly over ``dt`` under the constant
    command ``r_cmd``. With ``feedback`` the command comes from
    :func:`vrp_command` and the error obeys ``ė = −k_ξ e`` (converges for
    ``k_ξ > 0``; ``k_ξ = 0`` *cancels the natural ω-divergence* and freezes the
    error — marginally stable). With ``feedback=False`` the command is the bare
    reference VRP (``r_cmd = r_ref``), so ``ė = +ω e`` and the error blows up:
    the open-loop LIPM is unstable. Returns a :class:`TrackingResult`.
    """
    w = omega0(plan.z_h, g)
    t, xi_s, ref_s, cmd_s, err_s = [], [], [], [], []
    xi = xi0
    n = min(len(plan.t), len(plan.xi))
    for i in range(n):
        xr = plan.xi[i]
        pr = plan.vrp_t[i]
        r_cmd = vrp_command(xi, xr, pr, k_xi, w) if feedback else pr
        t.append(plan.t[i])
        xi_s.append(xi)
        ref_s.append(xr)
        cmd_s.append(r_cmd)
        err_s.append(xi - xr)
        # exact advance of ξ̇ = ω(ξ − r_cmd) over dt with constant r_cmd
        xi = r_cmd + (xi - r_cmd) * math.exp(w * dt)
    return TrackingResult(t=t, xi=xi_s, xi_ref=ref_s, vrp_cmd=cmd_s, err=err_s,
                          k_xi=k_xi, z_h=plan.z_h)
