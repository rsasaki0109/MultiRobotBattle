"""Tests for online LNS lifelong MAPF (repair, not replan-from-scratch).

CENTRAL replans every agent each boundary; online LNS keeps the committed plan
and repairs only what changed (a completed task) plus a small destroy
neighborhood. Its contracts: motion is collision-free *by construction* in both
modes (the repair is all-or-nothing -- a failed boundary is rejected and the
prior collision-free plan kept, never patched with an unsafe hold), it matches
CENTRAL's throughput at fewer replans on a well-formed map, it never does more
planning work than CENTRAL, and runs are deterministic.
"""

import unittest

from mrn_coord.lifelong import TaskStream, make_warehouse, run_online_lns
from mrn_coord.mapf.conflicts import detect_first_conflict


def _collision_free(res) -> bool:
    paths: dict = {}
    for snap in res.history:
        for a, c in snap.items():
            paths.setdefault(a, []).append(c)
    return detect_first_conflict(paths) is None


def _run(aisle, n, steps, period, k, mode):
    grid, eps = make_warehouse(rows=3, cols=4, aisle=aisle)
    starts = {f"r{i}": eps[i] for i in range(n)}
    stats: dict = {}
    res = run_online_lns(grid, dict(starts), TaskStream(list(eps)), mode=mode,
                         max_steps=steps, replan_period=period, neighborhood=k,
                         allocator="hungarian", keep_history=True, stats=stats)
    return res, stats


class TestOnlineLns(unittest.TestCase):
    def test_collision_free_by_construction(self):
        # Both modes, well-formed and dense maps -- the defining invariant.
        for aisle, n, steps, period, k in (
            (2, 6, 80, 3, 4), (2, 10, 100, 5, 2), (1, 6, 60, 3, 3),
        ):
            for mode in ("central", "lns"):
                res, _ = _run(aisle, n, steps, period, k, mode)
                self.assertTrue(_collision_free(res),
                                f"{mode} a{aisle} n{n} collided")

    def test_matches_throughput_with_fewer_replans(self):
        # On a roomy moderate-density map LNS serves as many tasks as CENTRAL
        # while replanning fewer agents per boundary, with no rejected boundary.
        c, cs = _run(2, 6, 80, 3, 4, "central")
        lns, ls = _run(2, 6, 80, 3, 4, "lns")
        self.assertEqual(lns.completed, c.completed)
        self.assertLess(ls["replans"], cs["replans"])
        self.assertEqual(ls["rejected"], 0)

    def test_central_wins_at_high_density(self):
        # When the team is dense, minimal repair rejects boundaries and falls
        # behind CENTRAL's full replan -- an honest loss, still collision-free.
        c, _ = _run(2, 10, 100, 5, 2, "central")
        lns, ls = _run(2, 10, 100, 5, 2, "lns")
        self.assertGreater(c.completed, lns.completed)
        self.assertGreater(ls["rejected"], 0)
        self.assertTrue(_collision_free(lns))

    def test_lns_never_does_more_work(self):
        for aisle, n, steps, period, k in ((2, 6, 80, 3, 4), (2, 10, 100, 5, 2)):
            _, cs = _run(aisle, n, steps, period, k, "central")
            _, ls = _run(aisle, n, steps, period, k, "lns")
            self.assertLessEqual(ls["replans"], cs["replans"])

    def test_deterministic(self):
        a, asd = _run(2, 6, 80, 3, 4, "lns")
        b, bsd = _run(2, 6, 80, 3, 4, "lns")
        self.assertEqual(a.as_dict(), b.as_dict())
        self.assertEqual(asd, bsd)

    def test_rejects_unknown_mode(self):
        grid, eps = make_warehouse(rows=3, cols=4, aisle=2)
        with self.assertRaises(ValueError):
            run_online_lns(grid, {"r0": eps[0]}, TaskStream(list(eps)),
                           mode="bogus", max_steps=4)


if __name__ == "__main__":
    unittest.main()
