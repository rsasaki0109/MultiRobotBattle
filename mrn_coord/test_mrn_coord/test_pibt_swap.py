"""Tests for swap-enhanced PIBT — the LaCAM2 successor generator (Okumura, 2023).

Plain PIBT livelocks when two agents must exchange ends of a narrow corridor; the
swap operation detects a required-and-possible exchange and pulls the partner
through a pocket. The contracts: the swap resolves a corridor exchange that base
PIBT cannot; every solution it returns is collision-free; and swap=False is plain
PIBT (so it exposes the livelock the swap fixes).
"""

import random
import unittest

from mrn_coord.mapf import GridWorld
from mrn_coord.mapf.conflicts import detect_first_conflict
from mrn_coord.mapf.pibt_swap import pibt_swap


def _corridor_with_pocket():
    free = {(x, 0) for x in range(5)} | {(2, 1)}
    blocked = {(x, y) for x in range(5) for y in range(2) if (x, y) not in free}
    grid = GridWorld(5, 2, frozenset(blocked))
    agents = {0: ((0, 0), (4, 0)), 1: ((4, 0), (0, 0))}
    return grid, agents


class TestPIBTSwap(unittest.TestCase):
    def test_swap_resolves_corridor_base_livelocks(self):
        grid, agents = _corridor_with_pocket()
        base = pibt_swap(grid, agents, swap=False, max_timestep=200)
        sw = pibt_swap(grid, agents, swap=True, max_timestep=200)
        self.assertIsNone(base)            # base PIBT livelocks the exchange
        self.assertIsNotNone(sw)           # the swap resolves it
        self.assertIsNone(detect_first_conflict(sw))
        self.assertIn((2, 1), sw[0])       # an agent steps into the pocket
        for a, (s, g) in agents.items():
            self.assertEqual(sw[a][0], s)
            self.assertEqual(sw[a][-1], g)

    def test_battery_collision_free(self):
        solved = cf = 0
        for seed in range(60):
            rng = random.Random(seed)
            n = rng.randint(2, 5)
            cells = rng.sample([(x, y) for x in range(5) for y in range(5)], 2 * n)
            grid = GridWorld(5, 5)
            agents = {i: (cells[i], cells[n + i]) for i in range(n)}
            sw = pibt_swap(grid, agents, swap=True, max_timestep=500)
            if sw is not None:
                solved += 1
                self.assertIsNone(detect_first_conflict(sw), f"seed={seed}")
                cf += 1
        self.assertEqual(solved, cf)
        self.assertGreater(solved, 50)     # most solve (PIBT is incomplete)

    def test_swap_rescues_some_base_livelocks(self):
        rescues = 0
        for seed in range(120):
            rng = random.Random(seed)
            n = rng.randint(2, 5)
            cells = rng.sample([(x, y) for x in range(5) for y in range(5)], 2 * n)
            grid = GridWorld(5, 5)
            agents = {i: (cells[i], cells[n + i]) for i in range(n)}
            base = pibt_swap(grid, agents, swap=False, max_timestep=400)
            sw = pibt_swap(grid, agents, swap=True, max_timestep=400)
            if base is None and sw is not None:
                rescues += 1
        self.assertGreater(rescues, 0)     # the swap does real work

    def test_unreachable_goal(self):
        grid = GridWorld(3, 3, blocked={(1, 0), (1, 1), (1, 2)})
        self.assertIsNone(
            pibt_swap(grid, {0: ((0, 0), (2, 2))}, max_timestep=50))


if __name__ == "__main__":
    unittest.main()
