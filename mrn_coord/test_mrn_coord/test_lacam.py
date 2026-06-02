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


def _soc(paths) -> int:
    # standard sum-of-costs: per-agent arrival time, ignoring trailing rest at goal
    # (LaCAM pads every path to the makespan, which `sum_of_costs` would over-count).
    total = 0
    for p in paths.values():
        goal, arrival = p[-1], 0
        for t, c in enumerate(p):
            if c != goal:
                arrival = t + 1
        total += arrival
    return total


class TestLacamOptimize(unittest.TestCase):
    def test_matches_cbs_optimum_on_small_instances(self):
        # The satisficing default is cost-blind; `optimize=True` runs the anytime
        # LaCAM* (g-tracking + parent rewiring + lower-bound pruning) and, given a
        # generous budget, must reach the *optimum* — checked agent-for-agent
        # against CBS, which minimizes sum-of-costs. (It can never beat CBS; a
        # cheaper-than-CBS result would mean the cost accounting is wrong.)
        rng = random.Random(1)
        checked = optimal = 0
        for _ in range(120):
            w = h = rng.randint(4, 6)
            grid = GridWorld(w, h)
            free = [(x, y) for x in range(w) for y in range(h)
                    if grid.is_free((x, y))]
            n = rng.randint(2, 4)
            if len(free) < 2 * n:
                continue
            pts = rng.sample(free, 2 * n)
            agents = {str(i): (pts[2 * i], pts[2 * i + 1]) for i in range(n)}
            opt = cbs(grid, agents, max_expansions=20_000)
            if opt is None:
                continue
            sol = lacam(grid, agents, optimize=True, max_iterations=200_000)
            self.assertTrue(_valid(grid, agents, sol))
            self.assertGreaterEqual(_soc(sol.paths), _soc(opt.paths),
                                    msg=f"below the CBS optimum?! {agents}")
            checked += 1
            optimal += int(_soc(sol.paths) == _soc(opt.paths))
        self.assertGreater(checked, 60)
        self.assertEqual(optimal, checked,
                         msg=f"LaCAM* missed the optimum on {checked - optimal} of {checked}")

    def test_optimize_never_costs_more_than_satisficing(self):
        # The anytime search keeps the best found, so it can only match or beat the
        # first (satisficing) solution — never regress.
        grid = GridWorld(8, 8)
        cells = [(x, y) for x in range(8) for y in range(8)]
        for seed in range(8):
            rng = random.Random(seed)
            starts, goals = rng.sample(cells, 12), rng.sample(cells, 12)
            agents = {i: (starts[i], goals[i]) for i in range(12)}
            sat = lacam(grid, agents)
            opt = lacam(grid, agents, optimize=True, max_iterations=50_000)
            self.assertTrue(_valid(grid, agents, opt))
            self.assertLessEqual(_soc(opt.paths), _soc(sat.paths),
                                 msg=f"optimize regressed at seed={seed}")

    def test_optimize_is_deterministic(self):
        grid = GridWorld(6, 6, blocked={(3, 3), (2, 3)})
        agents = {"a": ((0, 0), (5, 5)), "b": ((5, 0), (0, 5)),
                  "c": ((0, 5), (5, 0))}
        a = lacam(grid, agents, optimize=True, max_iterations=100_000)
        b = lacam(grid, agents, optimize=True, max_iterations=100_000)
        self.assertEqual(a.paths, b.paths)

    def test_scales_past_the_toy_regime(self):
        # The completeness test above only ever stresses 2-4 agents on 4x4-6x6
        # grids — a regime where the greedy dive almost always reaches the goal, so
        # the *scaling* claim went untested. With a static per-config priority order
        # the dive was a weak deterministic PIBT that livelocked, dropping into the
        # exponential lazy-constraint fallback; this 16-30 agent open-grid battery
        # solved only ~0.667 even at 200k iterations and ran ~100x slower. The
        # strong-PIBT spine (accumulating priority + the deterministic escape salt,
        # reseeded per re-expansion) must clear every one of these, fast.
        for w, h, n in ((8, 8, 16), (10, 10, 20), (12, 12, 30)):
            grid = GridWorld(w, h)
            cells = [(x, y) for x in range(w) for y in range(h)]
            for seed in range(20):
                rng = random.Random(seed)
                starts, goals = rng.sample(cells, n), rng.sample(cells, n)
                agents = {i: (starts[i], goals[i]) for i in range(n)}
                sol = lacam(grid, agents, max_iterations=200_000)
                self.assertTrue(_valid(grid, agents, sol),
                                msg=f"LaCAM failed {w}x{h} n={n} seed={seed}")

    def test_recovers_the_seed_that_only_the_escape_solves(self):
        # 10x10/20 seed=66 livelocks the strong dive on its first salted attempt and
        # — because `explored` forbids revisiting a config — cannot recover the way
        # pibt_solve's oscillating walk does, UNLESS the escape salt is reseeded each
        # time the stuck config is re-expanded. Without that reseed this instance
        # fails even at 3,000,000 iterations / ~100s; with it, it solves instantly.
        grid = GridWorld(10, 10)
        cells = [(x, y) for x in range(10) for y in range(10)]
        rng = random.Random(66)
        starts, goals = rng.sample(cells, 20), rng.sample(cells, 20)
        agents = {i: (starts[i], goals[i]) for i in range(20)}
        self.assertTrue(_valid(grid, agents, lacam(grid, agents, max_iterations=200_000)))


if __name__ == "__main__":
    unittest.main()
