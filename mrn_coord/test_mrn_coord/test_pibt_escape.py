"""Tests for the deterministic PIBT livelock escape (:func:`pibt_solve`).

Plain deterministic PIBT trades the completeness theorem's *random* tie-break for
a reproducible one, and so can livelock in a symmetric standoff. The escape bumps
a per-step salt once a stall is detected, scrambling equal-distance candidate ties
deterministically until the symmetry breaks — recovering near-complete
convergence with zero randomness. The contract: it converges far more often than
bare PIBT, never collides, and stays bit-reproducible.
"""

import random
import unittest

from mrn_coord.lifelong import pibt_solve
from mrn_coord.mapf import GridWorld


def _instance(w, h, n, seed):
    rng = random.Random(seed)
    cells = [(x, y) for x in range(w) for y in range(h)]
    return rng.sample(cells, n), rng.sample(cells, n)


def _collision_free(cfgs) -> bool:
    for prev, cur in zip(cfgs, cfgs[1:]):
        if len({tuple(c) for c in cur}) != len(cur):
            return False                      # vertex
        for i in range(len(cur)):
            for j in range(i + 1, len(cur)):
                if cur[i] == prev[j] and cur[j] == prev[i]:
                    return False              # swap
    return True


class TestPibtEscape(unittest.TestCase):
    def test_escape_converges_far_more_than_bare_pibt(self):
        # A fixed open-grid battery: the escape clears every instance; bare
        # deterministic PIBT livelocks on a meaningful chunk of them.
        families = [(8, 8, 16), (10, 10, 20), (12, 12, 30)]
        esc = base = total = 0
        for w, h, n in families:
            grid = GridWorld(w, h)
            for seed in range(20):
                starts, goals = _instance(w, h, n, seed)
                esc += pibt_solve(grid, starts, goals, escape=True)[1]
                base += pibt_solve(grid, starts, goals, escape=False)[1]
                total += 1
        self.assertEqual(esc, total, "escape failed to converge somewhere")
        self.assertLess(base, total, "bare PIBT unexpectedly converged everywhere")
        self.assertGreater(esc - base, 5, "escape barely helped")

    def test_converges_on_known_limit_cycle_seeds(self):
        # These exact instances livelock under a step-to-step stall detector: the
        # team's summed distance oscillates, so a previous-step comparison resets
        # the stall counter on every transient dip and the escape never engages
        # (or disengages mid-recovery). Measuring the stall against the running
        # *minimum* distance is immune to the oscillation and clears all of them.
        for w, h, n, seed in ((8, 8, 16, 27), (8, 8, 16, 31),
                              (10, 10, 20, 71), (12, 12, 30, 101)):
            grid = GridWorld(w, h)
            starts, goals = _instance(w, h, n, seed)
            cfgs, converged = pibt_solve(grid, starts, goals, escape=True)
            self.assertTrue(converged, f"limit cycle unbroken at {w}x{h} seed={seed}")
            self.assertTrue(_collision_free(cfgs), f"{w}x{h} seed={seed}")

    def test_always_collision_free(self):
        # Whatever the tie-break, PIBT yields a collision-free configuration every
        # step — the load-bearing guarantee must survive the perturbation.
        for w, h, n in ((10, 10, 20), (12, 12, 30)):
            grid = GridWorld(w, h)
            for seed in range(15):
                starts, goals = _instance(w, h, n, seed)
                for escape in (True, False):
                    cfgs, _ = pibt_solve(grid, starts, goals, escape=escape)
                    self.assertTrue(_collision_free(cfgs), f"{w}x{h} seed={seed}")

    def test_deterministic(self):
        grid = GridWorld(12, 12)
        starts, goals = _instance(12, 12, 30, 3)
        self.assertEqual(pibt_solve(grid, starts, goals),
                         pibt_solve(grid, starts, goals))

    def test_escape_is_off_by_default_in_the_core(self):
        # The lifelong engine builds _Pibt without a salt, so its dynamics (and
        # every pinned throughput baseline) are untouched by this feature. Guard
        # that the default _Pibt tie-break stays the plain coordinate order.
        from mrn_coord.lifelong.lifelong import _Pibt
        grid = GridWorld(5, 5)
        step = _Pibt(grid, {"a": (2, 2)}, {"a": (4, 2)},
                     {"a": {(4, 2): 0, (3, 2): 1, (2, 2): 2}})
        self.assertEqual(step.salt, 0)


if __name__ == "__main__":
    unittest.main()
