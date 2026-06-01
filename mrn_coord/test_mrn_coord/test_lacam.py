"""Tests for LaCAM: the complete, satisficing configuration-space MAPF solver.

LaCAM is not cost-optimal, so the contract is *validity* and *completeness*: it
returns collision-free paths that take every agent from its start to its goal,
and it finds one whenever a solution exists (checked against CBS — anything CBS
solves, LaCAM must solve too).
"""

import random
import unittest

from mrn_coord.mapf import GridWorld, cbs, lacam
from mrn_coord.mapf.conflicts import detect_first_conflict
from mrn_coord.mapf.solution import pad_paths


def _valid(grid, agents, sol) -> bool:
    if sol is None:
        return False
    if detect_first_conflict(pad_paths(sol.paths)) is not None:
        return False
    for a, (start, goal) in agents.items():
        path = sol.paths[a]
        if path[0] != start or path[-1] != goal:
            return False
        for t in range(1, len(path)):
            p, c = path[t - 1], path[t]
            if c != p and abs(c[0] - p[0]) + abs(c[1] - p[1]) != 1:
                return False
            if not grid.is_free(c):
                return False
    return True


class TestLacamBasics(unittest.TestCase):
    def test_simple_swap(self):
        grid = GridWorld(5, 3, blocked={(2, 0), (2, 2)})
        agents = {"a": ((0, 1), (4, 1)), "b": ((4, 1), (0, 1))}
        sol = lacam(grid, agents)
        self.assertTrue(_valid(grid, agents, sol))

    def test_single_agent(self):
        grid = GridWorld(5, 1)
        sol = lacam(grid, {"a": ((0, 0), (4, 0))})
        self.assertTrue(_valid(grid, {"a": ((0, 0), (4, 0))}, sol))

    def test_infeasible_returns_none(self):
        grid = GridWorld(3, 1, blocked={(1, 0)})       # wall splits the corridor
        self.assertIsNone(lacam(grid, {"a": ((0, 0), (2, 0))}))

    def test_dense_swap_in_corridor(self):
        # three agents reorder through a 3-wide corridor with one passing row
        grid = GridWorld(5, 2)
        agents = {"1": ((0, 0), (4, 0)), "2": ((4, 0), (0, 0)),
                  "3": ((2, 1), (2, 0))}
        sol = lacam(grid, agents)
        self.assertTrue(_valid(grid, agents, sol))


class TestLacamCompleteness(unittest.TestCase):
    def test_solves_everything_cbs_solves(self):
        rng = random.Random(7)
        checked = 0
        for _ in range(250):
            w = h = rng.randint(4, 6)
            blocked = frozenset(
                (rng.randrange(w), rng.randrange(h))
                for _ in range(rng.randint(0, 4)))
            grid = GridWorld(w, h, blocked)
            free = [(x, y) for x in range(w) for y in range(h)
                    if grid.is_free((x, y))]
            n = rng.randint(2, 4)
            if len(free) < 2 * n:
                continue
            pts = rng.sample(free, 2 * n)
            agents = {str(i): (pts[2 * i], pts[2 * i + 1]) for i in range(n)}
            if cbs(grid, agents, max_expansions=20_000) is None:
                continue
            sol = lacam(grid, agents, max_iterations=200_000)
            self.assertTrue(_valid(grid, agents, sol),
                            msg=f"LaCAM failed a CBS-solvable instance: {agents}")
            checked += 1
        self.assertGreater(checked, 80)

    def test_deterministic(self):
        grid = GridWorld(6, 6, blocked={(3, 3), (2, 3)})
        agents = {"a": ((0, 0), (5, 5)), "b": ((5, 0), (0, 5)),
                  "c": ((0, 5), (5, 0))}
        a = lacam(grid, agents)
        b = lacam(grid, agents)
        self.assertEqual(a.paths, b.paths)


if __name__ == "__main__":
    unittest.main()
