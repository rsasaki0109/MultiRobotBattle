"""Bounded-suboptimality contract: our ECBS vs. the reference libMultiRobotPlanning.

Our ``mrn_coord.mapf.ecbs`` guarantees a sum-of-costs within ``w`` of the optimum.
These tests make that checkable against the canonical reference (Wolfgang Hönig's
``libMultiRobotPlanning`` ``ecbs``): on shared instances, both solvers must honor
``cost <= w · optimal`` (the optimum is our CBS cost, already pinned to the
reference ``cbs`` by the equivalence contract). The reference is an optional
dependency: build the ``ecbs`` binary and point ``LIBMRP_ECBS`` at it (or put it
on ``PATH``); absent, every test skips. See ``scripts/compare_ecbs_libmrp.py`` for
the harness; this file is the in-suite guard.
"""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_REPO, "scripts"))


def _have_ecbs():
    from compare_ecbs_libmrp import _ecbs_binary
    return _ecbs_binary() is not None


@unittest.skipUnless(_have_ecbs(),
                     "reference libMultiRobotPlanning `ecbs` binary not found "
                     "(set LIBMRP_ECBS or put it on PATH)")
class TestEcbsBoundedSuboptimal(unittest.TestCase):
    def setUp(self):
        from compare_ecbs_libmrp import WEIGHT, _ecbs_binary, run_all
        self.w = WEIGHT
        self.rows = run_all(_ecbs_binary(), w=WEIGHT)

    def test_our_ecbs_honors_the_bound(self):
        # The headline guarantee: our cost never exceeds w times the optimum.
        for r in self.rows:
            if r["optimal"] is not None and r["ratio_ours"] is not None:
                self.assertLessEqual(
                    r["ratio_ours"], self.w + 1e-9,
                    f"{r['scenario']} our ECBS exceeded w*optimal")

    def test_reference_ecbs_honors_the_bound(self):
        for r in self.rows:
            if r["optimal"] is not None and r["ratio_lib"] is not None:
                self.assertLessEqual(
                    r["ratio_lib"], self.w + 1e-9,
                    f"{r['scenario']} reference ECBS exceeded w*optimal")

    def test_both_solve_every_solvable_instance(self):
        # Guard the guard: a solvable instance must be solved by both, else the
        # bound is vacuously satisfied and proves nothing.
        self.assertTrue(self.rows, "no scenarios ran")
        for r in self.rows:
            if r["optimal"] is not None:
                self.assertIsNotNone(r["soc_ours"],
                                     f"{r['scenario']} our ECBS found no solution")
                self.assertIsNotNone(r["soc_lib"],
                                     f"{r['scenario']} reference ECBS found none")

    def test_cost_is_at_least_optimal(self):
        # Sanity on the denominator: no solver can beat the true optimum.
        for r in self.rows:
            if r["optimal"] is not None:
                if r["soc_ours"] is not None:
                    self.assertGreaterEqual(r["soc_ours"], r["optimal"])
                if r["soc_lib"] is not None:
                    self.assertGreaterEqual(r["soc_lib"], r["optimal"])


if __name__ == "__main__":
    unittest.main()
