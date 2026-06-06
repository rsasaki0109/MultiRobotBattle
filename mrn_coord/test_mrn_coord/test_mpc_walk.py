"""Tests for trajectory-free MPC walking control (Wieber 2006)."""

import math
import random
import unittest

from mrn_coord.mapf.mpc_walk import (
    MPCParams,
    build_condensed,
    simulate_mpc,
    solve_box_qp,
    standing_support,
    stepping_support,
    _dot,
    _matvec,
)


def _gate_params():
    return MPCParams(z_h=0.8, dt=0.1, horizon=16, alpha=1e-5, beta=1.0)


class TestCondensedModel(unittest.TestCase):
    def test_condensed_matches_rollout(self):
        # Z = Pzs x + Pzu U must equal a direct cart-table rollout's ZMP.
        cm = build_condensed(_gate_params())
        A, b, c, N = cm.A, cm.b, cm.c, 16
        rng = random.Random(7)
        x0 = [rng.uniform(-0.1, 0.1) for _ in range(3)]
        U = [rng.uniform(-1, 1) for _ in range(N)]
        x = list(x0)
        z_dir = []
        for u in U:
            x = [A[0][0] * x[0] + A[0][1] * x[1] + A[0][2] * x[2] + b[0] * u,
                 A[1][1] * x[1] + A[1][2] * x[2] + b[1] * u,
                 A[2][2] * x[2] + b[2] * u]
            z_dir.append(_dot(c, x))
        s = _matvec(cm.Pzs, x0)
        z_con = [s[i] + sum(cm.Pzu[i][j] * U[j] for j in range(N))
                 for i in range(N)]
        for a, b2 in zip(z_dir, z_con):
            self.assertAlmostEqual(a, b2, places=12)

    def test_pzu_lower_triangular_invertible(self):
        cm = build_condensed(_gate_params())
        N = 16
        for i in range(N):
            for j in range(N):
                if j > i:
                    self.assertEqual(cm.Pzu[i][j], 0.0)
        # diagonal entry is c.b and non-zero (so Pzu is invertible)
        self.assertAlmostEqual(cm.Pzu[0][0], _dot(cm.c, cm.b), places=15)
        self.assertNotAlmostEqual(cm.Pzu[0][0], 0.0)

    def test_omega(self):
        p = _gate_params()
        self.assertAlmostEqual(p.omega, math.sqrt(9.8 / 0.8))


class TestBoxQP(unittest.TestCase):
    def test_unconstrained_recovers_newton(self):
        # With wide bounds the box QP returns the unconstrained minimiser.
        H = [[4.0, 1.0], [1.0, 3.0]]
        g = [-1.0, -2.0]
        z = solve_box_qp(H, g, [-1e9, -1e9], [1e9, 1e9])
        # H z = -g  ->  exact solve
        self.assertAlmostEqual(H[0][0] * z[0] + H[0][1] * z[1], -g[0], places=9)
        self.assertAlmostEqual(H[1][0] * z[0] + H[1][1] * z[1], -g[1], places=9)

    def test_box_active_kkt(self):
        # A problem whose unconstrained optimum lies outside the box; the QP
        # must clamp and satisfy KKT exactly.
        H = [[2.0, 0.5], [0.5, 2.0]]
        g = [-5.0, -5.0]
        lo, hi = [-1.0, -1.0], [1.0, 1.0]
        z = solve_box_qp(H, g, lo, hi)
        for i in range(2):
            self.assertLessEqual(z[i], hi[i] + 1e-12)
            self.assertGreaterEqual(z[i], lo[i] - 1e-12)
        # both clamp to the upper bound here
        self.assertAlmostEqual(z[0], 1.0, places=9)
        self.assertAlmostEqual(z[1], 1.0, places=9)

    def test_kkt_residual_machine_precision_ill_conditioned(self):
        # The MPC Hessian is ill-conditioned (~1e5); the active-set solve must
        # still hit KKT to machine precision (unlike coordinate descent).
        cm = build_condensed(_gate_params())
        N = 16
        xb = [0.0, 0.16, 0.0]
        sb = _matvec(cm.Pzs, xb)
        r0 = [_matvec(cm.Pvs, xb)[i] for i in range(N)]
        g = [-_matvec(cm.HZ, sb)[i] + _matvec(cm.Wt, r0)[i] for i in range(N)]
        lo, hi = [-0.08] * N, [0.08] * N
        z = solve_box_qp(cm.HZ, g, lo, hi)
        grad = [sum(cm.HZ[i][j] * z[j] for j in range(N)) + g[i]
                for i in range(N)]
        kkt = 0.0
        for i in range(N):
            if lo[i] + 1e-9 < z[i] < hi[i] - 1e-9:
                kkt = max(kkt, abs(grad[i]))
            elif z[i] <= lo[i] + 1e-9:
                kkt = max(kkt, max(0.0, -grad[i]))
            else:
                kkt = max(kkt, max(0.0, grad[i]))
        self.assertLess(kkt, 1e-8)


class TestPushRecovery(unittest.TestCase):
    def setUp(self):
        self.cm = build_condensed(_gate_params())
        self.w = self.cm.params.omega
        self.half = 0.08
        self.cen, self.hal = standing_support(self.half, 90, horizon=16)
        self.vr = [0.0] * len(self.cen)

    def _run(self, dv, constrained, n_steps=90):
        return simulate_mpc([0, 0, 0], self.cen, self.hal, self.vr,
                            condensed=self.cm, n_steps=n_steps, push_tick=5,
                            push_dv=dv, constrained=constrained)

    def test_constrained_recovers_in_support(self):
        con = self._run(0.16, True)
        self.assertTrue(con.zmp_feasible())
        self.assertTrue(con.recovered())
        self.assertLessEqual(max(abs(z) for z in con.zmp), self.half + 1e-6)

    def test_constraint_is_load_bearing(self):
        # Same push: the unconstrained controller recovers too but drives the
        # ZMP outside the foot -- the hard constraint is what keeps it legal.
        con = self._run(0.16, True)
        unc = self._run(0.16, False)
        self.assertFalse(unc.zmp_feasible())
        self.assertGreater(max(abs(z) for z in unc.zmp), self.half)
        self.assertGreater(max(abs(z) for z in unc.zmp),
                           1.5 * max(abs(z) for z in con.zmp))

    def test_strong_push_legal_but_falls(self):
        # A push beyond in-place capturability (xi = dv/omega > foot half) keeps
        # the ZMP legal yet the CoM falls: feasibility is not balance.
        self.assertGreater(0.30 / self.w, self.half)
        strong = self._run(0.30, True, n_steps=60)
        self.assertTrue(strong.zmp_feasible())
        self.assertFalse(strong.recovered(0.05))

    def test_no_push_constrained_equals_unconstrained(self):
        c0 = simulate_mpc([0, 0, 0], self.cen, self.hal, self.vr,
                          condensed=self.cm, n_steps=40, constrained=True)
        u0 = simulate_mpc([0, 0, 0], self.cen, self.hal, self.vr,
                          condensed=self.cm, n_steps=40, constrained=False)
        for a, b in zip(c0.zmp, u0.zmp):
            self.assertAlmostEqual(a, b, places=12)

    def test_deterministic(self):
        a = self._run(0.16, True)
        b = self._run(0.16, True)
        self.assertEqual(a.zmp, b.zmp)


class TestForwardWalk(unittest.TestCase):
    def test_walk_advances_and_stays_feasible(self):
        cm = build_condensed(_gate_params())
        step_len = 0.20
        cen, hal, ns = stepping_support(step_len, 8, 12, 0.07, horizon=16)
        vref = step_len / (8 * 0.1)
        vr = [vref] * len(cen)
        walk = simulate_mpc([0, vref, 0], cen, hal, vr, condensed=cm,
                            n_steps=ns, constrained=True)
        self.assertTrue(walk.zmp_feasible())
        self.assertGreaterEqual(walk.com_advance(), 0.98 * (12 - 1) * step_len)
        self.assertLess(abs(walk.mean_vel() - vref), 0.05)


if __name__ == "__main__":
    unittest.main()
