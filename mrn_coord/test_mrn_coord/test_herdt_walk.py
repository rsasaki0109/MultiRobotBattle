"""Tests for MPC walking with automatic footstep placement (Herdt et al. 2010)."""

import unittest

from mrn_coord.mapf.herdt_walk import (
    HerdtParams,
    build_herdt,
    simulate_herdt,
    _selection,
)
from mrn_coord.mapf import mpc_walk


def _params():
    return HerdtParams(z_h=0.8, dt=0.1, horizon=16, alpha=1e-5, beta=1.0,
                       gamma=1e-3, foot_half=0.05, step_ticks=8,
                       step_lo=-0.05, step_hi=0.40)


class TestSelectionAndQP(unittest.TestCase):
    def test_selection_partitions_horizon(self):
        # r ticks on the current foot, then groups of step_ticks on future feet
        sup, m = _selection(8, 16, 8)
        self.assertEqual(sup[:8], [0] * 8)
        self.assertEqual(sup[8:16], [1] * 8)
        self.assertEqual(m, 1)
        sup2, m2 = _selection(3, 16, 8)
        self.assertEqual(sup2[:3], [0, 0, 0])
        self.assertEqual(m2, 2)
        self.assertTrue(all(0 <= s <= m2 for s in sup2))

    def test_qp_hessian_pd_and_kkt_exact(self):
        h = build_herdt(_params())
        _, _, Z, d, delta, (H, grad, lo, hi, y) = h.solve(
            [0.0, 0.30, 0.0], 0.0, 8, [0.0] * 16, foot_vars=True, return_qp=True)
        n = len(y)
        # PD via LDL pivots
        M = [row[:] for row in H]
        for k in range(n):
            self.assertGreater(M[k][k], 1e-12)
            for i in range(k + 1, n):
                f = M[i][k] / M[k][k]
                for j in range(k, n):
                    M[i][j] -= f * M[k][j]
        # KKT residual on the box QP
        kkt = 0.0
        for i in range(n):
            gi = sum(H[i][j] * y[j] for j in range(n)) + grad[i]
            if lo[i] + 1e-9 < y[i] < hi[i] - 1e-9:
                kkt = max(kkt, abs(gi))
        self.assertLess(kkt, 1e-8)

    def test_reduction_reproduces_zmp(self):
        h = build_herdt(_params())
        cm = h.condensed
        x = [0.0, 0.30, 0.0]
        _, _, Z, _, _ = h.solve(x, 0.0, 8, [0.0] * 16, foot_vars=True)
        s = mpc_walk._matvec(cm.Pzs, x)
        U = mpc_walk._fwd_sub(cm.Pzu, [Z[i] - s[i] for i in range(16)])
        A, b, c = cm.A, cm.b, cm.c
        xx = list(x)
        for i, u in enumerate(U):
            xx = [A[0][0] * xx[0] + A[0][1] * xx[1] + A[0][2] * xx[2] + b[0] * u,
                  A[1][1] * xx[1] + A[1][2] * xx[2] + b[1] * u,
                  A[2][2] * xx[2] + b[2] * u]
            self.assertAlmostEqual(mpc_walk._dot(c, xx), Z[i], places=9)


class TestPushRecovery(unittest.TestCase):
    def test_fixed_foot_falls_auto_recovers(self):
        h = build_herdt(_params())
        auto = simulate_herdt([0, 0, 0], herdt=h, n_steps=60, vref_val=0.0,
                              push_tick=5, push_dv=0.30, foot_vars=True)
        fixed = simulate_herdt([0, 0, 0], herdt=h, n_steps=60, vref_val=0.0,
                               push_tick=5, push_dv=0.30, foot_vars=False)
        self.assertTrue(auto.recovered())
        self.assertFalse(auto.diverged())
        self.assertTrue(fixed.diverged())
        self.assertGreater(auto.foot_displacement(), 0.10)  # a capture step
        self.assertTrue(auto.zmp_feasible())

    def test_frozen_equals_mpc_walk_bit_for_bit(self):
        h = build_herdt(_params())
        fixed = simulate_herdt([0, 0, 0], herdt=h, n_steps=60, vref_val=0.0,
                               push_tick=5, push_dv=0.30, foot_vars=False)
        mp = mpc_walk.MPCParams(z_h=0.8, dt=0.1, horizon=16, alpha=1e-5, beta=1.0)
        cm = mpc_walk.build_condensed(mp)
        cen, hal = mpc_walk.standing_support(0.05, 60, horizon=16)
        mw = mpc_walk.simulate_mpc([0, 0, 0], cen, hal, [0.0] * len(cen),
                                   condensed=cm, n_steps=60, push_tick=5,
                                   push_dv=0.30, constrained=True)
        self.assertEqual(fixed.zmp, mw.zmp)
        self.assertEqual(fixed.jerk, mw.jerk)

    def test_footstep_adapts_to_push_direction(self):
        h = build_herdt(_params())
        fwd = simulate_herdt([0, 0, 0], herdt=h, n_steps=20, vref_val=0.0,
                             push_tick=2, push_dv=0.20, foot_vars=True)
        bwd = simulate_herdt([0, 0, 0], herdt=h, n_steps=20, vref_val=0.0,
                             push_tick=2, push_dv=-0.20, foot_vars=True)
        self.assertGreater(fwd.committed_feet[1], 0.01)
        self.assertLess(bwd.committed_feet[1], -0.01)


class TestForwardWalking(unittest.TestCase):
    def test_walks_at_reference_velocity(self):
        h = build_herdt(_params())
        vref = 0.20
        walk = simulate_herdt([0, 0, 0], herdt=h, n_steps=56, vref_val=vref,
                              foot_vars=True)
        self.assertTrue(walk.zmp_feasible())
        self.assertLess(abs(walk.mean_vel() - vref), 0.05)
        self.assertGreaterEqual(walk.com_advance(), 0.85 * 56 * vref * 0.1)

    def test_steps_regular_after_startup(self):
        h = build_herdt(_params())
        p = h.params
        vref = 0.20
        walk = simulate_herdt([0, 0, 0], herdt=h, n_steps=56, vref_val=vref,
                              foot_vars=True)
        nominal = vref * p.step_ticks * p.dt
        incs = [walk.committed_feet[i + 1] - walk.committed_feet[i]
                for i in range(len(walk.committed_feet) - 1)]
        for di in incs[1:]:
            self.assertAlmostEqual(di, nominal, delta=0.03)

    def test_deterministic(self):
        h = build_herdt(_params())
        a = simulate_herdt([0, 0, 0], herdt=h, n_steps=40, vref_val=0.15,
                           foot_vars=True)
        b = simulate_herdt([0, 0, 0], herdt=h, n_steps=40, vref_val=0.15,
                           foot_vars=True)
        self.assertEqual(a.zmp, b.zmp)
        self.assertEqual(a.committed_feet, b.committed_feet)


if __name__ == "__main__":
    unittest.main()
