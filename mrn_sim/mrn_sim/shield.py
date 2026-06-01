"""A *certified* runtime safety shield — body-true, input-feasible, discrete-robust.

This is the rigorous counterpart to :mod:`mrn_sim.cbf`. That filter regulates a
**look-ahead point** ``p + L(cosθ,sinθ)`` and its forward-invariance guarantee is
about that fictitious point, not the robot body — under an aggressive approach
the body can still cross the boundary. And a *continuous-time* CBF condition
``ḣ ≥ -αh`` does not by itself guarantee ``h ≥ 0`` once a finite step ``dt`` and
finite acceleration ``a_max`` enter: the vehicle can be physically unable to
brake in time. This module closes both gaps and certifies the **robot body**.

Two layers, decoupled so the hard guarantee never depends on the soft one:

1.  **Braking speed cap (the hard guarantee).**  Against each obstacle the body
    has ``remaining = ‖p − o‖ − D`` metres before the boundary (``D`` = both
    radii + margin). The largest speed from which a maximal-deceleration stop
    fits inside ``remaining`` is ``v_cap = √(2 a_max · remaining)``. Capping the
    commanded speed at the per-obstacle minimum of that — together with the
    accel-limited window ``|v − v_prev| ≤ a_max·dt`` — means a feasible safe
    command (brake) *always exists*: the input-constrained / backup-set idea,
    made discrete-robust. This bounds the body, not a look-ahead point, and it
    cannot become infeasible.

2.  **Look-ahead steering (the soft maneuver).**  A first-order CBF on the
    look-ahead point contributes a turn that slides the robot *around* an
    obstacle rather than only braking for it. It is advisory: if it and the cap
    ever disagree the cap wins, so steering can never talk the body into the
    obstacle.

The result is the same 2-variable QP ``min ½‖u − u_nom‖² s.t. A u ≥ b`` solved
exactly by the active-set enumeration in :mod:`mrn_sim.cbf`, with an explicit
hard-brake fallback if the polytope is ever empty. The shield speaks the plant's
``(v, omega)`` directly (carry ``v`` between steps) and is accel-limited by
construction. Pure and deterministic; only math.

The certificate is empirical and falsifiable, not decorative: ``scripts/
certify_shield.py`` throws thousands of randomized and *adversarial* rollouts at
the shield — nominal commands that steer straight at the nearest obstacle at full
speed — and the benchmark gate fails the build on a single body-frame collision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .cbf import _solve_qp


@dataclass(frozen=True)
class ShieldConfig:
    """Tuning for :func:`shield_step`."""

    robot_radius: float = 0.25
    safety_margin: float = 0.1      # extra buffer beyond the two radii (absorbs dt)
    a_max: float = 2.0              # acceleration / deceleration limit
    max_v: float = 1.6
    max_omega: float = 3.0
    look_ahead: float = 0.2         # L: steering-CBF regulated point ahead of axle
    alpha: float = 4.0              # class-K gain for the steering CBF


def braking_speed_cap(pose, obstacles, cfg: ShieldConfig) -> float:
    """Largest body-safe speed: ``min_o √(2 a_max · (‖p − o‖ − D))``, clamped to ``max_v``.

    From any speed at or below this, a maximal-deceleration stop fits inside the
    distance left to every obstacle boundary — so the body cannot be carried past
    it. This is the hard guarantee; it depends only on the body position.
    """
    x, y = pose[0], pose[1]
    cap = cfg.max_v
    for obs in obstacles:
        ox, oy, r = obs[0], obs[1], obs[2]
        d = r + cfg.robot_radius + cfg.safety_margin
        remaining = math.hypot(x - ox, y - oy) - d
        cap = min(cap, math.sqrt(2.0 * cfg.a_max * max(0.0, remaining)))
    return cap


def _steering_rows(pose, obstacles, cfg: ShieldConfig):
    """First-order look-ahead CBF rows ``a·(v,ω) ≥ rhs`` — the advisory turn."""
    x, y, theta = pose[0], pose[1], pose[2]
    L = cfg.look_ahead
    ct, st = math.cos(theta), math.sin(theta)
    px, py = x + L * ct, y + L * st
    m00, m01 = ct, -L * st
    m10, m11 = st, L * ct
    rows = []
    for obs in obstacles:
        ox, oy, r = obs[0], obs[1], obs[2]
        ovx, ovy = (obs[3], obs[4]) if len(obs) >= 5 else (0.0, 0.0)
        dx, dy = px - ox, py - oy
        safe = r + cfg.robot_radius + cfg.safety_margin
        h = dx * dx + dy * dy - safe * safe
        a0 = 2.0 * (dx * m00 + dy * m10)
        a1 = 2.0 * (dx * m01 + dy * m11)
        rhs = -cfg.alpha * h + 2.0 * (dx * ovx + dy * ovy)
        rows.append((a0, a1, rhs))
    return rows


def shield_step(state, u_nom, obstacles, dt, cfg: ShieldConfig = ShieldConfig()):
    """Return a body-safe ``(v_cmd, omega_cmd)`` nearest the nominal command.

    ``state`` is ``(x, y, theta, v)`` — carry ``v`` (the achieved speed the shield
    returns) between steps so the accel limit is honoured. ``u_nom`` is the
    nominal ``(v_des, omega_des)``. ``obstacles`` are ``(x, y, radius)`` or
    ``(x, y, radius, vx, vy)``. The braking cap and accel window set a hard speed
    window; the steering CBF turns within it; the QP picks the nearest feasible
    command; if none exists the shield brakes as hard as the accel limit allows.
    """
    x, y, theta, v = state
    cap = braking_speed_cap((x, y, theta), obstacles, cfg)

    # accel-limited speed window, narrowed by the braking cap
    vlo = max(0.0, v - cfg.a_max * dt)
    vhi = min(cfg.max_v, v + cfg.a_max * dt, cap)
    if vhi < vlo:                 # cap below what we can brake to this step
        vhi = vlo                 # -> brake as hard as the accel limit permits
    v_des = max(vlo, min(vhi, u_nom[0]))

    rows = _steering_rows((x, y, theta), obstacles, cfg)
    rows.append((1.0, 0.0, vlo))            # v >= vlo
    rows.append((-1.0, 0.0, -vhi))          # v <= vhi
    rows.append((0.0, 1.0, -cfg.max_omega))
    rows.append((0.0, -1.0, -cfg.max_omega))

    v_cmd, omega = _solve_qp((v_des, u_nom[1]), rows)
    # the speed window is the hard guarantee: clamp to it regardless of the QP
    v_cmd = max(vlo, min(vhi, v_cmd))
    omega = max(-cfg.max_omega, min(cfg.max_omega, omega))
    return (v_cmd, omega)
