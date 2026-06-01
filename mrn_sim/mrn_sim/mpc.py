"""Model Predictive Control (MPC) by iterative LQR — an optimizing local controller.

Where DWA (:mod:`mrn_sim.dwa`) *samples* a grid of accel-limited ``(v, omega)``
pairs and keeps the best one-step-constant rollout, MPC *optimizes* a whole
control sequence over a receding horizon: it minimizes a smooth cost — distance
to the goal, control effort, and soft penalties for obstacles and walls —
subject to the unicycle dynamics, then applies the first command and re-solves
next tick. So instead of picking from a fixed menu it bends an entire trajectory
around obstacles, trading per-tick compute for smoother, more far-sighted motion.

The optimizer is **iterative LQR** (iLQR / DDP's Gauss-Newton cousin), the
standard trajectory optimizer for nonlinear systems:

1. roll the current control sequence out to a state trajectory;
2. **backward pass** — sweep the horizon back to front, building a local
   quadratic model of the cost-to-go and, from it, a linear feedback law
   ``du = alpha*k + K*dx`` per step (Levenberg-Marquardt regularization keeps
   the control Hessian positive-definite);
3. **forward pass** — re-roll under that law with a line search on ``alpha``,
   accepting the step only if the total cost drops;
4. repeat to convergence.

Pure and deterministic — hand-rolled 3x3/2x2 linear algebra over the unicycle
model, depending only on :mod:`mrn_sim.world` / :mod:`mrn_sim.kinematics` (no
numpy, no ``mrn_coord``). Pair it with a global planner exactly like DWA: feed
the carrot on a planned path as the local goal and inject the other robots as
discs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .kinematics import normalize_angle
from .world import World


@dataclass(frozen=True)
class MPCConfig:
    """Tuning for :func:`mpc_command`. Defaults suit the 2D demo world."""

    horizon: int = 20
    dt: float = 0.1                 # control period (matches world step)

    max_v: float = 1.6
    min_v: float = 0.0
    max_omega: float = 3.0
    accel_v: float = 2.5            # m/s^2  (clamps the applied command)
    accel_omega: float = 6.0        # rad/s^2

    q_pos: float = 1.0              # running position-tracking weight
    q_pos_terminal: float = 12.0    # terminal position weight
    r_v: float = 0.05               # control-effort weight on v
    r_omega: float = 0.05           # control-effort weight on omega

    w_obstacle: float = 40.0        # soft obstacle/wall penalty weight
    obstacle_margin: float = 0.15   # extra buffer beyond the two radii
    wall_margin: float = 0.05
    robot_radius: float = 0.25

    iterations: int = 50
    line_search_steps: int = 10
    tol: float = 1e-4
    reg_init: float = 1e-3
    reg_factor: float = 4.0
    reg_max: float = 1e6

    goal_tolerance: float = 0.3


# --- tiny linear algebra over 2- and 3-vectors / matrices ------------------

def _matT(m):
    return [[m[i][j] for i in range(len(m))] for j in range(len(m[0]))]


def _matmul(a, b):
    inner = len(b)
    cols = len(b[0])
    return [[sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)]
            for i in range(len(a))]


def _matvec(a, v):
    return [sum(a[i][k] * v[k] for k in range(len(v))) for i in range(len(a))]


def _vadd(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def _madd(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def _inv2(m):
    """Inverse of a 2x2 matrix; returns ``(inverse, determinant)``."""
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    if abs(det) < 1e-12:
        return None, det
    inv = [[m[1][1] / det, -m[0][1] / det],
           [-m[1][0] / det, m[0][0] / det]]
    return inv, det


# --- unicycle model (forward Euler) and its Jacobians ----------------------

def _step(x, u, dt):
    px, py, th = x
    v, w = u
    return [px + v * math.cos(th) * dt,
            py + v * math.sin(th) * dt,
            normalize_angle(th + w * dt)]


def _jacobians(x, u, dt):
    _, _, th = x
    v, _w = u
    fx = [[1.0, 0.0, -v * math.sin(th) * dt],
          [0.0, 1.0, v * math.cos(th) * dt],
          [0.0, 0.0, 1.0]]
    fu = [[math.cos(th) * dt, 0.0],
          [math.sin(th) * dt, 0.0],
          [0.0, dt]]
    return fx, fu


# --- cost and its derivatives ----------------------------------------------

def _stage(x, u, goal, obstacles, world: World, cfg: MPCConfig, terminal: bool):
    """Stage (or terminal) cost with gradient/Hessian in state and control."""
    px, py, th = x
    gx, gy = goal[0], goal[1]
    qp = cfg.q_pos_terminal if terminal else cfg.q_pos

    dx = px - gx
    dy = py - gy
    cost = 0.5 * qp * (dx * dx + dy * dy)
    lx = [qp * dx, qp * dy, 0.0]
    lxx = [[qp, 0.0, 0.0], [0.0, qp, 0.0], [0.0, 0.0, 0.0]]

    w = cfg.w_obstacle
    for (ox, oy, r) in obstacles:
        ex = px - ox
        ey = py - oy
        d = math.hypot(ex, ey)
        safe = r + cfg.robot_radius + cfg.obstacle_margin
        if 1e-6 < d < safe:
            rho = safe - d                      # penetration depth > 0
            cost += 0.5 * w * rho * rho
            gx_ = -ex / d                        # d(rho)/dpx
            gy_ = -ey / d
            lx[0] += w * rho * gx_
            lx[1] += w * rho * gy_
            lxx[0][0] += w * gx_ * gx_
            lxx[0][1] += w * gx_ * gy_
            lxx[1][0] += w * gy_ * gx_
            lxx[1][1] += w * gy_ * gy_

    lo = cfg.robot_radius + cfg.wall_margin
    walls = [(px, 0, lo, -1.0), (px, 0, world.width - lo, 1.0),
             (py, 1, lo, -1.0), (py, 1, world.height - lo, 1.0)]
    for coord, idx, limit, g in walls:
        viol = (limit - coord) if g < 0 else (coord - limit)
        if viol > 0.0:
            cost += 0.5 * w * viol * viol
            lx[idx] += w * viol * g
            lxx[idx][idx] += w * g * g

    if terminal:
        return cost, lx, lxx, [0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]]

    v, om = u
    cost += 0.5 * (cfg.r_v * v * v + cfg.r_omega * om * om)
    lu = [cfg.r_v * v, cfg.r_omega * om]
    luu = [[cfg.r_v, 0.0], [0.0, cfg.r_omega]]
    return cost, lx, lxx, lu, luu


def _traj_cost(xs, us, goal, obs_by_t, world, cfg):
    total = 0.0
    for t in range(len(us)):
        total += _stage(xs[t], us[t], goal, obs_by_t[t], world, cfg, False)[0]
    total += _stage(xs[-1], None, goal, obs_by_t[-1], world, cfg, True)[0]
    return total


# --- iLQR passes -----------------------------------------------------------

def _backward(xs, us, goal, obs_by_t, world, cfg, mu):
    """Backward pass; returns per-step ``(k, K)`` or ``None`` if not PD."""
    n = len(us)
    _, vx, vxx, _, _ = _stage(xs[n], None, goal, obs_by_t[n], world, cfg, True)
    ks = [None] * n
    Ks = [None] * n
    for t in range(n - 1, -1, -1):
        fx, fu = _jacobians(xs[t], us[t], cfg.dt)
        _, lx, lxx, lu, luu = _stage(
            xs[t], us[t], goal, obs_by_t[t], world, cfg, False)
        fxT, fuT = _matT(fx), _matT(fu)
        vxx_fx = _matmul(vxx, fx)
        vxx_fu = _matmul(vxx, fu)

        qx = _vadd(lx, _matvec(fxT, vx))
        qu = _vadd(lu, _matvec(fuT, vx))
        qxx = _madd(lxx, _matmul(fxT, vxx_fx))
        quu = _madd(luu, _matmul(fuT, vxx_fu))
        qux = _matmul(fuT, vxx_fx)                       # l_ux = 0

        quu_reg = [[quu[0][0] + mu, quu[0][1]],
                   [quu[1][0], quu[1][1] + mu]]
        inv, det = _inv2(quu_reg)
        if inv is None or det <= 0.0 or quu_reg[0][0] <= 0.0:
            return None

        k = [-(inv[0][0] * qu[0] + inv[0][1] * qu[1]),
             -(inv[1][0] * qu[0] + inv[1][1] * qu[1])]
        K = [[-(inv[0][0] * qux[0][j] + inv[0][1] * qux[1][j]) for j in range(3)],
             [-(inv[1][0] * qux[0][j] + inv[1][1] * qux[1][j]) for j in range(3)]]

        KT = _matT(K)                                    # 3x2
        quxT = _matT(qux)                                # 3x2
        vx = _vadd(_vadd(qx, _matvec(KT, _matvec(quu, k))),
                   _vadd(_matvec(KT, qu), _matvec(quxT, k)))
        vxx = _madd(_madd(qxx, _matmul(_matmul(KT, quu), K)),
                    _madd(_matmul(KT, qux), _matmul(quxT, K)))
        vxx = [[0.5 * (vxx[i][j] + vxx[j][i]) for j in range(3)]
               for i in range(3)]
        ks[t] = k
        Ks[t] = K
    return ks, Ks


def _forward(x0, xs, us, ks, Ks, cfg, alpha):
    n = len(us)
    new_xs = [x0]
    new_us = []
    x = x0
    for t in range(n):
        dx0 = x[0] - xs[t][0]
        dx1 = x[1] - xs[t][1]
        dx2 = normalize_angle(x[2] - xs[t][2])
        v = (us[t][0] + alpha * ks[t][0]
             + Ks[t][0][0] * dx0 + Ks[t][0][1] * dx1 + Ks[t][0][2] * dx2)
        om = (us[t][1] + alpha * ks[t][1]
              + Ks[t][1][0] * dx0 + Ks[t][1][1] * dx1 + Ks[t][1][2] * dx2)
        v = min(cfg.max_v, max(cfg.min_v, v))
        om = min(cfg.max_omega, max(-cfg.max_omega, om))
        new_us.append((v, om))
        x = _step(x, (v, om), cfg.dt)
        new_xs.append(x)
    return new_xs, new_us


def _obstacles_by_step(obstacles, moving, horizon):
    """Per-timestep obstacle lists: static obstacles plus each moving obstacle
    at its predicted position for that step (held at the last known position
    past the end of its prediction)."""
    static = list(obstacles)
    if not moving:
        return [static] * (horizon + 1)
    out = []
    for t in range(horizon + 1):
        extra = [traj[t] if t < len(traj) else traj[-1] for traj in moving]
        out.append(static + extra)
    return out


def solve_ilqr(x0, goal, obstacles, world: World, cfg: MPCConfig = MPCConfig(),
               u_init=None, moving=None):
    """Optimize a control sequence from ``x0`` toward ``goal``.

    Returns ``(us, xs, cost)``: the optimized controls, the resulting state
    trajectory, and the final total cost. ``u_init`` warm-starts the sequence
    (e.g. last tick's shifted solution); otherwise it starts from rest.
    ``moving`` is an optional list of predicted obstacle trajectories — each a
    list of ``(x, y, radius)`` indexed by timestep — so the optimizer avoids
    where a moving obstacle *will be*, not just where it is now.
    """
    n = cfg.horizon
    if u_init is not None and len(u_init) == n:
        us = [(float(v), float(w)) for (v, w) in u_init]
    else:
        us = [(0.0, 0.0)] * n

    obs_by_t = _obstacles_by_step(obstacles, moving, n)

    xs = [list(x0)]
    x = list(x0)
    for t in range(n):
        x = _step(x, us[t], cfg.dt)
        xs.append(x)
    cost = _traj_cost(xs, us, goal, obs_by_t, world, cfg)

    mu = cfg.reg_init
    for _ in range(cfg.iterations):
        bp = None
        while mu <= cfg.reg_max:
            bp = _backward(xs, us, goal, obs_by_t, world, cfg, mu)
            if bp is not None:
                break
            mu *= cfg.reg_factor
        if bp is None:
            break
        ks, Ks = bp

        improved = False
        alpha = 1.0
        for _ in range(cfg.line_search_steps):
            new_xs, new_us = _forward(x0, xs, us, ks, Ks, cfg, alpha)
            new_cost = _traj_cost(new_xs, new_us, goal, obs_by_t, world, cfg)
            if new_cost < cost - 1e-9:
                delta = cost - new_cost
                xs, us, cost = new_xs, new_us, new_cost
                mu = max(cfg.reg_init, mu / cfg.reg_factor)
                improved = True
                if delta < cfg.tol:
                    return us, xs, cost
                break
            alpha *= 0.5
        if not improved:
            mu *= cfg.reg_factor
            if mu > cfg.reg_max:
                break
    return us, xs, cost


def mpc_command(pose, v_cur, omega_cur, goal, obstacles, world: World,
                cfg: MPCConfig = MPCConfig(), u_init=None, moving=None):
    """Optimize a horizon and return the first accel-limited command.

    ``pose`` is ``(x, y, theta)``; ``goal`` a ``(x, y)`` local target (e.g. the
    carrot on a global path); ``obstacles`` a list of ``(x, y, radius)``;
    ``world`` supplies the bounds. ``moving`` optionally gives predicted
    trajectories of moving obstacles (e.g. the other robots along their paths)
    so avoidance is space-time aware. Returns ``((v, omega), us)`` — the command
    to apply now plus the full optimized sequence, so the caller can warm-start
    the next solve by passing the shifted ``us`` back as ``u_init``.
    """
    us, _xs, _cost = solve_ilqr(pose, goal, obstacles, world, cfg, u_init, moving)
    v, om = us[0]
    # Respect acceleration limits relative to the current command.
    v = min(v_cur + cfg.accel_v * cfg.dt, max(v_cur - cfg.accel_v * cfg.dt, v))
    om = min(omega_cur + cfg.accel_omega * cfg.dt,
             max(omega_cur - cfg.accel_omega * cfg.dt, om))
    v = min(cfg.max_v, max(cfg.min_v, v))
    om = min(cfg.max_omega, max(-cfg.max_omega, om))
    return (v, om), us
