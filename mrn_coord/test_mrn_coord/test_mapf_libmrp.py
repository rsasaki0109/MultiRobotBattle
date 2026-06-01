"""Equivalence contract: our CBS vs. the reference libMultiRobotPlanning.

Our ``mrn_coord.mapf.cbs`` is a from-scratch Conflict-Based Search. These tests
make "it finds the optimal solution" checkable by solving identical instances
with the canonical reference (Wolfgang Hönig's ``libMultiRobotPlanning`` C++
``cbs``) and asserting the optimal sum-of-costs and solvability agree. The
reference is an *optional* dependency: build the ``cbs`` binary and point
``LIBMRP_CBS`` at it (or put it on ``PATH``). When it is absent every test skips
cleanly, so the core suite is unaffected. See ``scripts/compare_mapf_libmrp.py``
for the harness and the narrative; this file is the in-suite guard.
"""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_REPO, "scripts"))


def _have_cbs():
    from compare_mapf_libmrp import _cbs_binary
    return _cbs_binary() is not None


@unittest.skipUnless(_have_cbs(),
                     "reference libMultiRobotPlanning `cbs` binary not found "
                     "(set LIBMRP_CBS or put it on PATH)")
class TestCbsMatchesLibMrp(unittest.TestCase):
    def setUp(self):
        from compare_mapf_libmrp import _cbs_binary, run_all
        self.rows = run_all(_cbs_binary())

    def test_solvability_parity(self):
        # Both solvers are complete, so they must agree on whether an instance
        # has a solution at all.
        for r in self.rows:
            self.assertEqual(
                r["solved_ours"], r["solved_lib"],
                f"{r['scenario']} solvability disagrees with the reference")

    def test_optimal_sum_of_costs_matches_exactly(self):
        # The strong claim: sum-of-costs is the optimization objective and its
        # optimum is a single number, so our optimal solver must reproduce the
        # reference's value exactly — no tolerance.
        for r in self.rows:
            if r["solved_ours"] and r["solved_lib"]:
                self.assertEqual(
                    r["soc_ours"], r["soc_lib"],
                    f"{r['scenario']} optimal sum-of-costs disagrees with the "
                    f"reference")

    def test_every_instance_is_actually_solved(self):
        # Guard the guard: a contract that silently degenerates to "both failed"
        # proves nothing. Every bundled instance must be solved by both.
        self.assertTrue(self.rows, "no scenarios ran")
        for r in self.rows:
            self.assertTrue(r["solved_ours"] and r["solved_lib"],
                            f"{r['scenario']} was not solved by both")


if __name__ == "__main__":
    unittest.main()
