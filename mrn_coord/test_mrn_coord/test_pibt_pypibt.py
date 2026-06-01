"""Equivalence contract: our PIBT step vs. the reference ``pypibt``.

``mrn_coord.lifelong._Pibt`` claims to be PIBT (Okumura et al. 2022). These tests
make that claim checkable by running identical scenarios through both our core and
the paper author's own reference (``pypibt``, imported as ``pypibt``) and judging
our output with the *reference's own* collision/validation logic. The reference is
an *optional*, pure-Python dependency built into a venv — when it is not importable
every test skips cleanly, so the core suite is unaffected. See
``scripts/compare_pibt_pypibt.py`` for the harness and the narrative; this file is
the in-suite guard.

The gated claim is the per-step invariant our code actually guarantees and the
fleet demo relies on — collision-freedom — not bit-for-bit path equality, which
PIBT's random tie-break makes meaningless between two faithful implementations.
"""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_REPO, "scripts"))


def _have_pypibt():
    try:
        import pypibt  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_have_pypibt(), "pypibt (reference PIBT) not installed")
class TestPibtMatchesPypibt(unittest.TestCase):
    def setUp(self):
        from compare_pibt_pypibt import (CONVERGENCE_FLOOR, MAKESPAN_RATIO_MAX,
                                         MAKESPAN_RATIO_MEAN, run_all)
        self.rows = run_all()
        self.ratio_max = MAKESPAN_RATIO_MAX
        self.ratio_mean = MAKESPAN_RATIO_MEAN
        self.conv_floor = CONVERGENCE_FLOOR

    def test_every_configuration_is_collision_free(self):
        # The load-bearing claim: across every instance and every timestep —
        # including the real lifelong warehouse run — our PIBT output is free of
        # vertex/edge collisions and illegal transitions, by the reference's own
        # checks.
        for r in self.rows:
            self.assertTrue(
                r["collision_free"],
                f"{r['scenario']}: reference found a collision ({r['coll_reason']})")

    def test_makespan_stays_within_bound_where_convergent(self):
        # Same algorithm family: where our deterministic solver reaches all goals,
        # its makespan tracks the reference's within a bound (not equality).
        mean_ratios = []
        for r in self.rows:
            if r["max_ratio"] is not None:
                self.assertLessEqual(
                    r["max_ratio"], self.ratio_max,
                    f"{r['scenario']} makespan wanders past the bound")
            if r["mean_ratio"] is not None:
                mean_ratios.append(r["mean_ratio"])
        if mean_ratios:
            self.assertLessEqual(sum(mean_ratios) / len(mean_ratios),
                                 self.ratio_mean)

    def test_convergence_does_not_collapse(self):
        # A regression backstop, not a completeness claim: deterministic PIBT is
        # knowingly incomplete, but it should still converge on most instances.
        tot_c = sum(r["converged"] for r in self.rows
                    if r["converged"] is not None)
        tot_s = sum(r["seeds"] for r in self.rows if r["converged"] is not None)
        self.assertGreaterEqual(tot_c / tot_s, self.conv_floor)


if __name__ == "__main__":
    unittest.main()
