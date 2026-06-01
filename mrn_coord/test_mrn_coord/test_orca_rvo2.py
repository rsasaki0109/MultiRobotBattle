"""Equivalence contract: our ORCA core vs. the reference RVO2 library.

Our ``mrn_coord.orca`` claims to be a faithful port of the reference RVO2
agent-agent collision-avoidance core. These tests make that claim checkable by
running identical scenarios through both and asserting agreement. The reference
(``Python-RVO2``, imported as ``rvo2``) is an *optional* dependency built into a
venv — when it is not importable every test skips cleanly, so the core suite is
unaffected. See ``scripts/compare_orca_rvo2.py`` for the harness and the
narrative; this file is the in-suite guard.
"""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_REPO, "scripts"))


def _have_rvo2():
    try:
        import rvo2  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_have_rvo2(), "rvo2 (reference Python-RVO2) not installed")
class TestOrcaMatchesRvo2(unittest.TestCase):
    def setUp(self):
        from compare_orca_rvo2 import (GAP_PARITY_TOL, VEL_DEV_TOL, run_all)
        self.rows = run_all(steps=400, dt=0.1)
        self.vel_tol = VEL_DEV_TOL
        self.gap_tol = GAP_PARITY_TOL

    def test_open_loop_velocity_matches_to_machine_precision(self):
        # The strong claim: fed the same state, our LP returns the same velocity
        # the reference does, across every scenario.
        for r in self.rows:
            self.assertLessEqual(
                r["max_vel_dev"], self.vel_tol,
                f"{r['scenario']} velocity disagrees with the reference")

    def test_safety_outcome_parity(self):
        # Tie-break-invariant: whatever side a symmetric pass resolves to, the
        # worst clearance the two implementations reach must agree.
        for r in self.rows:
            self.assertLessEqual(
                abs(r["min_gap_ours"] - r["min_gap_rvo2"]), self.gap_tol,
                f"{r['scenario']} reaches a different safety outcome")

    def test_velocity_agreement_is_tight(self):
        # Not just within tolerance — the port reproduces the reference linear
        # program, so agreement is at the 1e-4 level or better everywhere.
        worst = max(r["max_vel_dev"] for r in self.rows)
        self.assertLess(worst, 1e-4)


if __name__ == "__main__":
    unittest.main()
