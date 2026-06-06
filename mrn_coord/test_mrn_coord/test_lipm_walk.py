"""Tests for LIPM walking pattern generation by ZMP preview control
(Kajita et al., ICRA 2003).

Contracts: the preview controller tracks a reference ZMP tightly, and the
*preview* term is what makes it work — feedback alone lags badly; a footstep
plan turns into a dynamically stable walk whose induced ZMP never leaves the
support foot, with the CoM walking forward and swaying laterally onto each foot;
and generation is deterministic.
"""

import math
import unittest

from mrn_coord.mapf.footstep import (
    FOOT_LENGTH,
    FOOT_WIDTH,
    FootstepState,
    FootstepWorld,
    plan_footsteps,
)
from mrn_coord.mapf.lipm_walk import (
    PreviewGains,
    generate_walk,
    lipm_track,
    preview_gains,
    zmp_stability,
)

R = "R"


def _step_ref():
    ref = []
    for val, reps in [(0.0, 60), (0.3, 40), (0.6, 40), (0.9, 140)]:
        ref += [val] * reps
    return ref


class TestPreviewGains(unittest.TestCase):
    def test_gains_finite_and_steady_state(self):
        g = preview_gains(z_h=0.8, dt=0.02, preview_steps=80, Q=1.0, R=1e-8)
        self.assertTrue(all(math.isfinite(k) for k in g.K))
        self.assertEqual(len(g.f), 80)
        # steady state: for a constant reference the control should vanish, so
        # the sum of preview gains must approximately cancel the feedback K0
        self.assertAlmostEqual(sum(g.f), g.K[0], delta=0.05 * g.K[0])


class TestTracking(unittest.TestCase):
    def setUp(self):
        self.g = preview_gains(z_h=0.8, dt=0.02, preview_steps=80, Q=1.0, R=1e-8)
        self.ref = _step_ref()

    def test_preview_tracks_tightly(self):
        _, zmp = lipm_track(self.ref, self.g)
        body = range(80, len(self.ref) - 80)
        self.assertLess(max(abs(zmp[k] - self.ref[k]) for k in body), 0.015)

    def test_preview_is_load_bearing(self):
        body = range(80, len(self.ref) - 80)
        _, zmp = lipm_track(self.ref, self.g)
        err_prev = max(abs(zmp[k] - self.ref[k]) for k in body)
        g0 = PreviewGains(K=self.g.K, f=tuple(0.0 for _ in self.g.f),
                          z_h=self.g.z_h, dt=self.g.dt, g=self.g.g)
        _, zmp0 = lipm_track(self.ref, g0)
        err_fb = max(abs(zmp0[k] - self.ref[k]) for k in body)
        self.assertGreater(err_fb, 10.0 * err_prev)   # preview matters a lot


class TestWalk(unittest.TestCase):
    def setUp(self):
        world = FootstepWorld(3.0, 1.5)
        self.plan = plan_footsteps(world, FootstepState(0.4, 0.75, 0.0, R),
                                   (2.4, 0.75), w=2.0)

    def test_dynamically_stable(self):
        wp = generate_walk(self.plan.states, step_duration=0.7, dt=0.02)
        frac, out = zmp_stability(wp, foot_length=FOOT_LENGTH,
                                  foot_width=FOOT_WIDTH)
        self.assertEqual(out, 0)            # ZMP never leaves the support foot
        self.assertEqual(frac, 1.0)
        self.assertLess(wp.zmp_rms_error(), 0.030)

    def test_com_progresses_and_sways(self):
        wp = generate_walk(self.plan.states, step_duration=0.7, dt=0.02)
        comx_span = max(wp.com_x) - min(wp.com_x)
        comy_span = max(wp.com_y) - min(wp.com_y)
        footx_span = (max(p[0] for p in wp.foot_poses)
                      - min(p[0] for p in wp.foot_poses))
        self.assertGreaterEqual(comx_span, 0.9 * footx_span)  # walks forward
        self.assertGreater(comy_span, 0.10)                   # lateral sway

    def test_deterministic(self):
        a = generate_walk(self.plan.states, step_duration=0.7, dt=0.02)
        b = generate_walk(self.plan.states, step_duration=0.7, dt=0.02)
        self.assertEqual(a.com_x, b.com_x)
        self.assertEqual(a.zmp_x, b.zmp_x)


if __name__ == "__main__":
    unittest.main()
