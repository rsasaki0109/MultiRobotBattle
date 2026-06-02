"""Tests for MAPF-LNS: validity, monotone anytime improvement, determinism.

LNS is anytime, not optimal, so the contract is: it returns a collision-free
solution, never worse than the one it started from, deterministic for a fixed
seed, and — given enough rounds on instances CBS can solve — it closes most of
the gap to the optimum.
"""

import random
import unittest

from mrn_coord.mapf import GridWorld, cbs
from mrn_coord.mapf.conflicts import detect_first_conflict
from mrn_coord.mapf.lns import mapf_lns
from mrn_coord.mapf.solution import pad_paths, sum_of_costs


def _valid(grid, agents, sol) -> bool:
    if sol is None:
        return False
    if detect_first_conflict(pad_paths(sol.paths)) is not None:
        return False
    for a, (start, goal) in agents.items():
        path = sol.paths[a]
        if path[0] != start or path[-1] != goal:
            return False
    return True


class TestLnsBasics(unittest.TestCase):
    def test_solves_and_is_collision_free(self):
        grid = GridWorld(5, 3, blocked={(2, 0), (2, 2)})
        agents = {"a": ((0, 1), (4, 1)), "b": ((4, 1), (0, 1))}
        sol = mapf_lns(grid, agents, iterations=30, seed=0)
        self.assertTrue(_valid(grid, agents, sol))

    def test_never_worse_than_initial(self):
        rng = random.Random(2)
        checked = 0
        for _ in range(60):
            w = h = rng.randint(4, 6)
            blocked = frozenset(
                (rng.randrange(w), rng.randrange(h))
                for _ in range(rng.randint(0, 4)))
            grid = GridWorld(w, h, blocked)
            free = [(x, y) for x in range(w) for y in range(h)
                    if grid.is_free((x, y))]
            n = rng.randint(2, 5)
            if len(free) < 2 * n:
                continue
            pts = rng.sample(free, 2 * n)
            agents = {str(i): (pts[2 * i], pts[2 * i + 1]) for i in range(n)}
            stats = {}
            sol = mapf_lns(grid, agents, iterations=40, seed=1, stats=stats)
            if sol is None:
                continue
            self.assertTrue(_valid(grid, agents, sol))
            self.assertLessEqual(stats["final_cost"], stats["initial_cost"])
            checked += 1
        self.assertGreater(checked, 20)

    def test_closes_gap_to_optimal_on_average(self):
        # Over solvable instances, LNS should reach the CBS optimum most of the
        # time and never beat it (optimal is a lower bound).
        rng = random.Random(5)
        optimal_hits = 0
        total = 0
        for _ in range(60):
            grid = GridWorld(5, 5)
            free = [(x, y) for x in range(5) for y in range(5)]
            n = 4
            pts = rng.sample(free, 2 * n)
            agents = {str(i): (pts[2 * i], pts[2 * i + 1]) for i in range(n)}
            opt = cbs(grid, agents, max_expansions=20_000)
            if opt is None:
                continue
            sol = mapf_lns(grid, agents, iterations=80, seed=3)
            self.assertGreaterEqual(sol.cost, opt.cost)      # optimal is a bound
            optimal_hits += sol.cost == opt.cost
            total += 1
        self.assertGreater(optimal_hits / total, 0.6)

    def test_deterministic(self):
        grid = GridWorld(6, 6, blocked={(3, 2), (2, 3)})
        agents = {"a": ((0, 0), (5, 5)), "b": ((5, 0), (0, 5)),
                  "c": ((0, 5), (5, 0))}
        a = mapf_lns(grid, agents, iterations=50, seed=7)
        b = mapf_lns(grid, agents, iterations=50, seed=7)
        self.assertEqual(a.paths, b.paths)

    def test_improves_at_scale_beyond_cbs_reach(self):
        # The "closes the gap" tests above only reach 5x5 / 4 agents; the docstring
        # claim is improvement "on team sizes far beyond CBS's reach". On a 16-agent
        # open grid LNS must cut the aggregate sum-of-costs (destroy-repair actually
        # biting), stay valid, and improve most instances -- the regime the
        # `lns_scaling_improvement` benchmark gate pins.
        grid = GridWorld(8, 8)
        cells = [(x, y) for x in range(8) for y in range(8)]
        total_initial = total_final = improved = checked = 0
        for seed in range(6):
            rng = random.Random(seed)
            starts, goals = rng.sample(cells, 16), rng.sample(cells, 16)
            agents = {i: (starts[i], goals[i]) for i in range(16)}
            stats = {}
            sol = mapf_lns(grid, agents, iterations=80, seed=0, stats=stats)
            self.assertTrue(_valid(grid, agents, sol))
            self.assertLessEqual(stats["final_cost"], stats["initial_cost"])
            total_initial += stats["initial_cost"]
            total_final += stats["final_cost"]
            improved += int(stats["final_cost"] < stats["initial_cost"])
            checked += 1
        # The prioritized initial solution is already decent on an open grid, so a
        # few instances start near a local optimum LNS can't escape in 80 rounds;
        # the robust claim is the *aggregate* gain plus a handful of improvements.
        self.assertLess(total_final, total_initial, "no aggregate gain at scale")
        self.assertGreaterEqual(improved, 2, "destroy-repair barely bit anywhere")

    def test_improves_a_poor_initial(self):
        # Hand a deliberately padded initial solution; LNS must shorten it.
        grid = GridWorld(7, 1)
        agents = {"a": ((0, 0), (6, 0))}
        from mrn_coord.mapf.solution import Solution
        padded = Solution(paths={"a": [(0, 0)] * 5 + [(i, 0) for i in range(7)]},
                          cost=0)
        padded.cost = sum_of_costs(padded.paths)
        stats = {}
        sol = mapf_lns(grid, agents, iterations=10, seed=0, init=padded,
                       stats=stats)
        self.assertEqual(sol.cost, 6)                         # straight line
        self.assertLess(stats["final_cost"], stats["initial_cost"])


if __name__ == "__main__":
    unittest.main()
