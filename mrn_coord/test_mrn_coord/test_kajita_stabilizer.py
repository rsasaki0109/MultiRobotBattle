"""Tests for biped walking stabilization by LIPM tracking (Kajita et al. 2010)."""

import dataclasses
import unittest

from mrn_coord.mapf.kajita_stabilizer import (
    closed_loop_matrix,
    continuous_rate,
    gains_for_poles,
    reference_trajectory,
    simulate_stabilizer,
    spectral_radius,
    stabilizer_params,
    standing_reference,
    stepping_zmp_reference,
)


def _params(lam=4.0):
    return stabilizer_params(lam=lam, z_h=0.8, dt=0.02, foot_half=0.05)


class TestGainsAndPoles(unittest.TestCase):
    def test_gains_place_poles_exactly(self):
        p = _params(lam=4.0)
        w2 = p.omega ** 2
        # characteristic poly s^2 + w2*k_v s + w2*(k_p-1) must equal (s+lam)^2
        self.assertAlmostEqual(w2 * p.k_v, 2.0 * 4.0, places=9)
        self.assertAlmostEqual(w2 * (p.k_p - 1.0), 4.0 ** 2, places=9)
        self.assertGreater(p.k_p, 1.0)  # the instability-overcoming condition

    def test_closed_loop_stable_open_loop_not(self):
        p = _params()
        self.assertLess(spectral_radius(closed_loop_matrix(p)), 1.0)
        p_open = dataclasses.replace(p, k_p=0.0, k_v=0.0)
        self.assertGreater(spectral_radius(closed_loop_matrix(p_open)), 1.0)

    def test_realised_rate_near_design(self):
        p = _params(lam=4.0)
        self.assertLess(abs(continuous_rate(p) - 4.0) / 4.0, 0.15)

    def test_gains_for_poles_formula(self):
        kp, kv = gains_for_poles(5.0, 3.5)
        self.assertAlmostEqual(kv, 2.0 * 5.0 / 3.5 ** 2, places=12)
        self.assertAlmostEqual(kp, 1.0 + 5.0 ** 2 / 3.5 ** 2, places=12)


class TestStandingRecovery(unittest.TestCase):
    def test_open_loop_falls_stabilizer_recovers(self):
        p = _params()
        zr, cr, vr = standing_reference(200)
        cl = simulate_stabilizer(zr, cr, vr, params=p, stabilize=True,
                                 push_tick=10, push_dv=0.10)
        ol = simulate_stabilizer(zr, cr, vr, params=p, stabilize=False,
                                 push_tick=10, push_dv=0.10)
        self.assertTrue(cl.converged())
        self.assertFalse(cl.diverged())
        self.assertTrue(cl.realised_zmp_in_support())
        self.assertTrue(ol.diverged())

    def test_small_push_no_saturation(self):
        p = _params()
        zr, cr, vr = standing_reference(200)
        small = simulate_stabilizer(zr, cr, vr, params=p, stabilize=True,
                                    push_tick=10, push_dv=0.05)
        self.assertTrue(small.converged())
        self.assertFalse(small.ever_saturated())

    def test_large_push_saturates_and_fails(self):
        # honest limit: beyond the capturable margin the ankle saturates and
        # in-place recovery fails -- the robot must take a step.
        p = _params()
        zr, cr, vr = standing_reference(200)
        big = simulate_stabilizer(zr, cr, vr, params=p, stabilize=True,
                                  push_tick=10, push_dv=0.30)
        self.assertTrue(big.ever_saturated())
        self.assertFalse(big.converged())

    def test_no_disturbance_is_noop(self):
        p = _params()
        zr, cr, vr = standing_reference(200)
        cl = simulate_stabilizer(zr, cr, vr, params=p, stabilize=True)
        ol = simulate_stabilizer(zr, cr, vr, params=p, stabilize=False)
        self.assertLess(cl.max_error(), 1e-9)
        self.assertLess(ol.max_error(), 1e-9)

    def test_model_error_rejected_to_predicted_steady_state(self):
        p = _params()
        zr, cr, vr = standing_reference(200)
        bias = 0.02
        clb = simulate_stabilizer(zr, cr, vr, params=p, stabilize=True,
                                  zmp_bias=bias)
        olb = simulate_stabilizer(zr, cr, vr, params=p, stabilize=False,
                                  zmp_bias=bias)
        self.assertAlmostEqual(clb.steady_error(), -bias / (p.k_p - 1.0),
                               places=3)
        self.assertLess(clb.max_error(), 0.05)
        self.assertTrue(olb.diverged())


class TestForwardWalk(unittest.TestCase):
    def test_stabilizer_tracks_a_pushed_walk(self):
        p = _params()
        zref = stepping_zmp_reference(0.15, 40, 6, settle_ticks=120)
        com_ref, vel_ref, zmp_ind = reference_trajectory(zref, params=p)
        cl = simulate_stabilizer(zmp_ind, com_ref, vel_ref, params=p,
                                 stabilize=True, push_tick=150, push_dv=0.12)
        ol = simulate_stabilizer(zmp_ind, com_ref, vel_ref, params=p,
                                 stabilize=False, push_tick=150, push_dv=0.12)
        self.assertTrue(cl.converged(tol=0.02))
        self.assertLess(cl.max_error(), 0.05)
        self.assertTrue(ol.diverged())

    def test_deterministic(self):
        p = _params()
        zr, cr, vr = standing_reference(120)
        a = simulate_stabilizer(zr, cr, vr, params=p, stabilize=True,
                                push_tick=10, push_dv=0.10)
        b = simulate_stabilizer(zr, cr, vr, params=p, stabilize=True,
                                push_tick=10, push_dv=0.10)
        self.assertEqual(a.com, b.com)
        self.assertEqual(a.zmp, b.zmp)


if __name__ == "__main__":
    unittest.main()
