"""Tests for Resolved Momentum Control (Kajita et al. IROS 2003)."""

import math
import random
import unittest

from mrn_coord.mapf.resolved_momentum import (
    MomentumTask,
    SHANK_TIP,
    _cross2,
    make_humanoid,
    matvec,
    resolve_momentum,
    simulate,
    task_nullspace,
)


def _rand_q(rng, nd):
    return [rng.uniform(-0.3, 0.3) for _ in range(nd)]


def _rand_qd(rng, nd):
    return [rng.uniform(-0.5, 0.5) for _ in range(nd)]


class TestMomentumMatrix(unittest.TestCase):
    def test_cmm_matches_finite_difference(self):
        R = make_humanoid()
        nd = R.ndof
        rng = random.Random(0)
        eps = 1e-6
        for _ in range(50):
            q, qd = _rand_q(rng, nd), _rand_qd(rng, nd)
            h = R.momentum(q, qd)
            q2 = [q[k] + eps * qd[k] for k in range(nd)]
            _, _, c1 = R.forward_kinematics(q)
            _, _, c2 = R.forward_kinematics(q2)
            p1, _, _ = R.forward_kinematics(q)
            p2, _, _ = R.forward_kinematics(q2)
            rc = R.com(q)
            Px = Py = L = 0.0
            for i in range(len(R.links)):
                m, I = R.links[i].mass, R.links[i].inertia
                cd = ((c2[i][0] - c1[i][0]) / eps, (c2[i][1] - c1[i][1]) / eps)
                wd = (p2[i] - p1[i]) / eps
                Px += m * cd[0]
                Py += m * cd[1]
                d = (c1[i][0] - rc[0], c1[i][1] - rc[1])
                L += I * wd + m * _cross2(d, cd)
            self.assertAlmostEqual(h[0], Px, places=4)
            self.assertAlmostEqual(h[1], Py, places=4)
            self.assertAlmostEqual(h[2], L, places=4)

    def test_linear_momentum_is_mass_times_com_velocity(self):
        R = make_humanoid()
        nd = R.ndof
        rng = random.Random(1)
        q, qd = _rand_q(rng, nd), _rand_qd(rng, nd)
        h = R.momentum(q, qd)
        eps = 1e-6
        rc1, rc2 = R.com(q), R.com([q[k] + eps * qd[k] for k in range(nd)])
        M = R.total_mass()
        self.assertAlmostEqual(h[0], M * (rc2[0] - rc1[0]) / eps, places=4)
        self.assertAlmostEqual(h[1], M * (rc2[1] - rc1[1]) / eps, places=4)

    def test_cmm_translation_invariant(self):
        R = make_humanoid()
        nd = R.ndof
        q = _rand_q(random.Random(2), nd)
        A1 = R.centroidal_momentum_matrix(q)
        q2 = list(q)
        q2[0] += 1.3
        q2[1] -= 0.7
        A2 = R.centroidal_momentum_matrix(q2)
        for r in range(3):
            for c in range(nd):
                self.assertAlmostEqual(A1[r][c], A2[r][c], places=12)


class TestResolution(unittest.TestCase):
    def test_resolve_realizes_reference_and_pins_foot(self):
        R = make_humanoid()
        q = _rand_q(random.Random(3), R.ndof)
        task = MomentumTask((5.0, -2.0, 1.5), [(2, SHANK_TIP, (0.0, 0.0))])
        qd = resolve_momentum(R, q, task)
        h = R.momentum(q, qd)
        self.assertAlmostEqual(h[0], 5.0, places=8)
        self.assertAlmostEqual(h[1], -2.0, places=8)
        self.assertAlmostEqual(h[2], 1.5, places=8)
        Jf, _ = R.point_jacobian(q, 2, SHANK_TIP)
        for v in matvec(Jf, qd):
            self.assertAlmostEqual(v, 0.0, places=8)

    def test_min_norm_and_nullspace(self):
        R = make_humanoid()
        nd = R.ndof
        rng = random.Random(4)
        q = _rand_q(rng, nd)
        task = MomentumTask((3.0, 1.0, -0.5), [(2, SHANK_TIP, (0.0, 0.0))])
        qd = resolve_momentum(R, q, task)
        N = task_nullspace(R, q, task)
        nz = matvec(N, _rand_qd(rng, nd))
        self.assertGreater(math.sqrt(sum(v * v for v in nz)), 1e-3)
        qd2 = [qd[k] + nz[k] for k in range(nd)]
        h, h2 = R.momentum(q, qd), R.momentum(q, qd2)
        for k in range(3):
            self.assertAlmostEqual(h[k], h2[k], places=9)
        self.assertLessEqual(sum(v * v for v in qd), sum(v * v for v in qd2) + 1e-9)

    def test_zero_angular_momentum_counter_rotation(self):
        R = make_humanoid()
        nd = R.ndof
        q = _rand_q(random.Random(5), nd)
        task = MomentumTask((8.0, 0.0, 0.0), [(2, SHANK_TIP, (0.0, 0.0))])
        qd = resolve_momentum(R, q, task)
        self.assertAlmostEqual(R.momentum(q, qd)[2], 0.0, places=9)
        _, _, com = R.forward_kinematics(q)
        rc = R.com(q)
        link_L = []
        for i in range(len(R.links)):
            m, I = R.links[i].mass, R.links[i].inertia
            cd = matvec(R.com_jacobian(q, i), qd)
            Jw = R.angular_jacobian(i)
            wd = sum(Jw[c] * qd[c] for c in range(nd))
            d = (com[i][0] - rc[0], com[i][1] - rc[1])
            link_L.append(I * wd + m * _cross2(d, cd))
        # the limbs individually carry angular momentum that cancels to zero
        self.assertGreater(max(abs(x) for x in link_L), 0.1)
        self.assertAlmostEqual(sum(link_L), 0.0, places=9)


class TestKick(unittest.TestCase):
    def test_kick_tracks_and_holds_momentum(self):
        R = make_humanoid()
        nd = R.ndof
        q0 = [0.0] * nd
        q0[R.joint_col(2)] = 0.3
        q0[R.joint_col(4)] = -0.3
        T, dt = 60, 0.01

        def task_fn(t, _q):
            vx = 0.4 * math.cos(math.pi * t / (T * dt))
            vy = 0.3 * math.sin(math.pi * t / (T * dt))
            return MomentumTask((0.0, 0.0, 0.0),
                                [(2, SHANK_TIP, (0.0, 0.0)),
                                 (4, SHANK_TIP, (vx, vy))])

        traj = simulate(R, q0, task_fn, dt=dt, steps=T,
                        track={"swing": (4, SHANK_TIP), "support": (2, SHANK_TIP)})
        self.assertLess(traj.max_abs_angular_momentum(), 1e-9)
        su = traj.point_path("support")
        drift = max(math.hypot(su[k][0] - su[0][0], su[k][1] - su[0][1])
                    for k in range(len(su)))
        self.assertLess(drift, 1e-2)
        sp = traj.point_path("swing")
        self.assertGreater(math.hypot(sp[-1][0] - sp[0][0], sp[-1][1] - sp[0][1]), 0.1)

    def test_deterministic(self):
        R = make_humanoid()
        q = _rand_q(random.Random(6), R.ndof)
        task = MomentumTask((2.0, -1.0, 0.5), [(2, SHANK_TIP, (0.0, 0.0))])
        self.assertEqual(resolve_momentum(R, q, task), resolve_momentum(R, q, task))


if __name__ == "__main__":
    unittest.main()
