"""Tests for the MAPF core: grid, space-time A*, conflicts, CBS, prioritized."""

import unittest

from mrn_coord.mapf import (
    EdgeConflict,
    GridWorld,
    Solution,
    VertexConflict,
    cbs,
    cbsh,
    cell_at,
    detect_first_conflict,
    ecbs,
    eecbs,
    makespan,
    manhattan,
    pad_paths,
    plan_path,
    prioritized_planning,
    sum_of_costs,
)


class TestGrid(unittest.TestCase):
    def test_bounds_and_blocking(self):
        grid = GridWorld(3, 3, blocked={(1, 1)})
        self.assertTrue(grid.is_free((0, 0)))
        self.assertFalse(grid.is_free((1, 1)))
        self.assertFalse(grid.is_free((3, 0)))
        self.assertFalse(grid.is_free((-1, 0)))

    def test_neighbors_includes_wait_and_filters_blocked(self):
        grid = GridWorld(3, 3, blocked={(1, 0)})
        ns = set(grid.neighbors((0, 0)))
        self.assertIn((0, 0), ns)        # wait
        self.assertIn((0, 1), ns)
        self.assertNotIn((1, 0), ns)     # blocked
        self.assertNotIn((-1, 0), ns)    # out of bounds

    def test_rejects_nonpositive_dims(self):
        for bad in ((0, 3), (3, -1)):
            with self.assertRaises(ValueError):
                GridWorld(*bad)


class TestSpaceTimeAStar(unittest.TestCase):
    def test_straight_line(self):
        grid = GridWorld(5, 1)
        path = plan_path(grid, (0, 0), (4, 0))
        self.assertEqual(path, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
        self.assertEqual(len(path) - 1, 4)

    def test_start_equals_goal(self):
        grid = GridWorld(3, 3)
        self.assertEqual(plan_path(grid, (1, 1), (1, 1)), [(1, 1)])

    def test_detours_around_obstacle(self):
        # Wall at x=1 for y in {0,1} forces a detour through (1,2).
        grid = GridWorld(3, 3, blocked={(1, 0), (1, 1)})
        path = plan_path(grid, (0, 0), (2, 0))
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (2, 0))
        self.assertIn((1, 2), path)       # had to go around the top
        self.assertEqual(manhattan((0, 0), (2, 0)), 2)
        self.assertGreater(len(path) - 1, 2)

    def test_unreachable_returns_none(self):
        # Fully wall off the goal column.
        grid = GridWorld(3, 3, blocked={(1, 0), (1, 1), (1, 2)})
        self.assertIsNone(plan_path(grid, (0, 0), (2, 0)))

    def test_vertex_constraint_forces_wait(self):
        grid = GridWorld(3, 1)
        # Forbid being at (1,0) at t=1, so the agent must wait one step.
        path = plan_path(grid, (0, 0), (2, 0), vertex_constraints={((1, 0), 1)})
        self.assertIsNotNone(path)
        self.assertEqual(path[-1], (2, 0))
        self.assertEqual(len(path) - 1, 3)        # one extra step vs optimal 2
        self.assertNotEqual(path[1], (1, 0))      # not at (1,0) at t=1

    def test_edge_constraint_blocks_transition(self):
        grid = GridWorld(3, 1)
        # Forbid the move (0,0)->(1,0) arriving at t=1.
        path = plan_path(
            grid, (0, 0), (2, 0), edge_constraints={((0, 0), (1, 0), 1)}
        )
        self.assertIsNotNone(path)
        self.assertFalse(path[0] == (0, 0) and path[1] == (1, 0))

    def test_goal_constraint_delays_settling(self):
        grid = GridWorld(3, 1)
        # Goal blocked at t=2 -> must arrive later / wait.
        path = plan_path(grid, (0, 0), (2, 0), vertex_constraints={((2, 0), 2)})
        self.assertIsNotNone(path)
        self.assertEqual(path[-1], (2, 0))
        self.assertNotEqual(cell_at(path, 2), (2, 0))


class TestConflicts(unittest.TestCase):
    def test_vertex_conflict(self):
        paths = {"a": [(0, 0), (1, 0), (2, 0)], "b": [(2, 0), (1, 0), (0, 0)]}
        c = detect_first_conflict(paths)
        self.assertIsInstance(c, VertexConflict)
        self.assertEqual(c.cell, (1, 0))
        self.assertEqual(c.time, 1)

    def test_edge_swap_conflict(self):
        paths = {"a": [(0, 0), (1, 0)], "b": [(1, 0), (0, 0)]}
        c = detect_first_conflict(paths)
        self.assertIsInstance(c, EdgeConflict)
        self.assertEqual(c.time, 1)
        self.assertEqual({c.cell_a, c.cell_b}, {(0, 0), (1, 0)})

    def test_no_conflict(self):
        paths = {"a": [(0, 0), (0, 1)], "b": [(2, 0), (2, 1)]}
        self.assertIsNone(detect_first_conflict(paths))

    def test_cell_at_holds_goal(self):
        path = [(0, 0), (1, 0)]
        self.assertEqual(cell_at(path, 5), (1, 0))

    def test_stay_at_goal_collision_is_detected(self):
        # 'a' parks on (1,0); 'b' arrives there later -> vertex conflict.
        paths = {"a": [(1, 0)], "b": [(0, 0), (1, 0)]}
        c = detect_first_conflict(paths)
        self.assertIsInstance(c, VertexConflict)
        self.assertEqual(c.cell, (1, 0))
        self.assertEqual(c.time, 1)


class TestCBS(unittest.TestCase):
    def test_parallel_is_optimal(self):
        grid = GridWorld(5, 2)
        agents = {"a": ((0, 0), (4, 0)), "b": ((0, 1), (4, 1))}
        sol = cbs(grid, agents)
        self.assertIsNotNone(sol)
        self.assertIsNone(detect_first_conflict(sol.paths))
        self.assertEqual(sol.cost, 8)        # 4 + 4, no interference

    def test_crossing_resolves(self):
        grid = GridWorld(5, 5)
        agents = {"a": ((0, 2), (4, 2)), "b": ((2, 0), (2, 4))}
        sol = cbs(grid, agents)
        self.assertIsNotNone(sol)
        self.assertIsNone(detect_first_conflict(sol.paths))
        # base cost is 8; resolving the center crossing costs at least one wait
        self.assertGreaterEqual(sol.cost, 9)
        self.assertEqual(sol.paths["a"][-1], (4, 2))
        self.assertEqual(sol.paths["b"][-1], (2, 4))

    def test_swap_through_passing_row(self):
        grid = GridWorld(3, 2)
        agents = {"a": ((0, 0), (2, 0)), "b": ((2, 0), (0, 0))}
        sol = cbs(grid, agents)
        self.assertIsNotNone(sol)
        self.assertIsNone(detect_first_conflict(sol.paths))

    def test_unsolvable_corridor_returns_none(self):
        # 1-wide corridor, agents must swap ends -> impossible.
        grid = GridWorld(3, 1)
        agents = {"a": ((0, 0), (2, 0)), "b": ((2, 0), (0, 0))}
        self.assertIsNone(cbs(grid, agents, max_expansions=2000))

    def test_single_agent(self):
        grid = GridWorld(4, 1)
        sol = cbs(grid, {"a": ((0, 0), (3, 0))})
        self.assertIsNotNone(sol)
        self.assertEqual(sol.cost, 3)


def _rand_instance(w, h, n, seed, obstacle=0.0):
    import random
    rng = random.Random(seed)
    blocked = {(x, y) for x in range(w) for y in range(h)
               if rng.random() < obstacle}
    free = [(x, y) for x in range(w) for y in range(h) if (x, y) not in blocked]
    rng.shuffle(free)
    starts = free[:n]
    rng.shuffle(free)
    goals = free[:n]
    return GridWorld(w, h, frozenset(blocked)), \
        {i: (starts[i], goals[i]) for i in range(n)}


class TestCBSH(unittest.TestCase):
    """CBS with improved heuristics: same optimum as CBS, fewer expansions."""

    def test_matches_cbs_optimum_across_heuristics(self):
        # Every heuristic (and prioritization-only) must return the SAME optimal
        # sum-of-costs as plain CBS, with a valid collision-free solution.
        for seed in range(15):
            grid, agents = _rand_instance(6, 6, 5, seed)
            base = cbs(grid, agents, max_expansions=50000)
            if base is None:
                continue
            for mode in (None, "cg", "dg", "wdg"):
                sol = cbsh(grid, agents, heuristic=mode, max_expansions=50000)
                self.assertIsNotNone(sol, f"seed={seed} mode={mode}")
                self.assertIsNone(detect_first_conflict(sol.paths))
                self.assertEqual(sol.cost, base.cost,
                                 f"seed={seed} mode={mode}")
                for a, (start, goal) in agents.items():
                    self.assertEqual(sol.paths[a][0], start)
                    self.assertEqual(sol.paths[a][-1], goal)

    def test_expands_no_more_than_cbs(self):
        # The heuristic only delays expanding nodes that cannot be optimal, so
        # CBSH never expands MORE than vanilla CBS — and far fewer in aggregate.
        tot_cbs = tot_wdg = 0
        for seed in range(10):
            grid, agents = _rand_instance(6, 6, 6, seed, obstacle=0.12)
            s = {}
            base = cbs(grid, agents, stats=s, max_expansions=20000)
            if base is None:
                continue
            sh = {}
            sol = cbsh(grid, agents, heuristic="wdg", stats=sh,
                       max_expansions=20000)
            self.assertEqual(sol.cost, base.cost)
            self.assertLessEqual(sh["expansions"], s["expansions"], f"seed={seed}")
            tot_cbs += s["expansions"]
            tot_wdg += sh["expansions"]
        # On this conflict-heavy battery the reduction is dramatic (>5x).
        self.assertLess(tot_wdg * 5, tot_cbs)

    def test_heuristics_are_monotone(self):
        # Stronger heuristic -> never more expansions, in aggregate.
        tot = {"cg": 0, "dg": 0, "wdg": 0}
        for seed in range(10):
            grid, agents = _rand_instance(7, 7, 8, seed)
            if cbs(grid, agents, max_expansions=20000) is None:
                continue
            for mode in ("cg", "dg", "wdg"):
                sh = {}
                cbsh(grid, agents, heuristic=mode, stats=sh,
                     max_expansions=20000)
                tot[mode] += sh["expansions"]
        self.assertLessEqual(tot["wdg"], tot["dg"])
        self.assertLessEqual(tot["dg"], tot["cg"])

    def test_deterministic(self):
        grid, agents = _rand_instance(7, 7, 8, 3)
        runs = []
        for _ in range(2):
            sh = {}
            sol = cbsh(grid, agents, heuristic="wdg", stats=sh)
            runs.append((sol.cost, sh["expansions"]))
        self.assertEqual(runs[0], runs[1])

    def test_unsolvable_corridor_returns_none(self):
        # 1-wide corridor swap is infeasible; like CBS, CBSH exhausts its budget
        # and returns None. (heuristic=None keeps the per-node work cheap — the
        # WDG pairwise solve would otherwise re-prove infeasibility every node.)
        grid = GridWorld(3, 1)
        agents = {"a": ((0, 0), (2, 0)), "b": ((2, 0), (0, 0))}
        self.assertIsNone(cbsh(grid, agents, heuristic=None, max_expansions=500))


class TestEECBS(unittest.TestCase):
    """EECBS: bounded-suboptimal, fewer expansions than ECBS at the same w."""

    def test_respects_bound_across_heuristics(self):
        # Every heuristic (and the h=0 ablation) must return a valid solution
        # within the suboptimality factor of the CBS optimum, at a few factors.
        for seed in range(8):
            grid, agents = _rand_instance(6, 6, 5, seed)
            base = cbs(grid, agents, max_expansions=50000)
            if base is None:
                continue
            for w in (1.0, 1.5):
                for mode in (None, "cg", "dg", "wdg"):
                    sol = eecbs(grid, agents, w=w, heuristic=mode,
                                max_expansions=50000)
                    self.assertIsNotNone(sol, f"seed={seed} w={w} mode={mode}")
                    self.assertIsNone(detect_first_conflict(sol.paths))
                    self.assertLessEqual(sol.cost, w * base.cost + 1e-9,
                                         f"seed={seed} w={w} mode={mode}")
                    for a, (start, goal) in agents.items():
                        self.assertEqual(sol.paths[a][0], start)
                        self.assertEqual(sol.paths[a][-1], goal)

    def test_w_one_is_optimal(self):
        # At w=1.0 the bound forces optimality — EECBS must match CBS's cost.
        for seed in range(10):
            grid, agents = _rand_instance(6, 6, 5, seed)
            base = cbs(grid, agents, max_expansions=50000)
            if base is None:
                continue
            sol = eecbs(grid, agents, w=1.0, heuristic="wdg",
                        max_expansions=50000)
            self.assertIsNotNone(sol, f"seed={seed}")
            self.assertEqual(sol.cost, base.cost, f"seed={seed}")

    def test_admissible_bound_cuts_expansions_vs_ecbs(self):
        # Near-optimal (w=1.02) is where the admissible WDG bound bites: EECBS
        # must expand no more than ECBS per instance, and far fewer in aggregate.
        # The h=0 ablation reduces to ECBS's own expansion count.
        tot_ecbs = tot_wdg = 0
        for seed in range(8):
            grid, agents = _rand_instance(8, 8, 7, seed, obstacle=0.12)
            se = {}
            base = ecbs(grid, agents, w=1.02, stats=se, max_expansions=20000)
            if base is None:
                continue
            sh = {}
            sol = eecbs(grid, agents, w=1.02, heuristic="wdg", stats=sh,
                        max_expansions=20000)
            self.assertIsNotNone(sol, f"seed={seed}")
            self.assertLessEqual(sh["expansions"], se["expansions"],
                                 f"seed={seed}")
            tot_ecbs += se["expansions"]
            tot_wdg += sh["expansions"]
        # A clear aggregate win on this conflict-heavy battery.
        self.assertLess(tot_wdg, tot_ecbs)

    def test_heuristics_are_monotone(self):
        # Stronger admissible heuristic -> never more expansions, in aggregate.
        tot = {None: 0, "cg": 0, "dg": 0, "wdg": 0}
        for seed in range(8):
            grid, agents = _rand_instance(8, 8, 7, seed, obstacle=0.12)
            for mode in (None, "cg", "dg", "wdg"):
                sh = {}
                if eecbs(grid, agents, w=1.02, heuristic=mode, stats=sh,
                         max_expansions=20000) is None:
                    break
                tot[mode] += sh["expansions"]
        self.assertLessEqual(tot["wdg"], tot["dg"])
        self.assertLessEqual(tot["dg"], tot["cg"])
        self.assertLessEqual(tot["cg"], tot[None])

    def test_deterministic(self):
        grid, agents = _rand_instance(7, 7, 8, 3)
        runs = []
        for _ in range(2):
            sh = {}
            sol = eecbs(grid, agents, w=1.1, heuristic="wdg", stats=sh)
            runs.append((sol.cost, sh["expansions"]))
        self.assertEqual(runs[0], runs[1])

    def test_unsolvable_corridor_returns_none(self):
        # 1-wide corridor swap is infeasible; EECBS exhausts its budget and
        # returns None. (heuristic=None keeps the per-node work cheap.)
        grid = GridWorld(3, 1)
        agents = {"a": ((0, 0), (2, 0)), "b": ((2, 0), (0, 0))}
        self.assertIsNone(
            eecbs(grid, agents, w=1.5, heuristic=None, max_expansions=500))


class TestPrioritized(unittest.TestCase):
    def test_parallel_succeeds(self):
        grid = GridWorld(5, 2)
        agents = {"a": ((0, 0), (4, 0)), "b": ((0, 1), (4, 1))}
        sol = prioritized_planning(grid, agents)
        self.assertIsNotNone(sol)
        self.assertIsNone(detect_first_conflict(sol.paths))

    def test_crossing_is_collision_free(self):
        grid = GridWorld(5, 5)
        agents = {"a": ((0, 2), (4, 2)), "b": ((2, 0), (2, 4))}
        sol = prioritized_planning(grid, agents)
        self.assertIsNotNone(sol)
        self.assertIsNone(detect_first_conflict(sol.paths))

    def test_order_changes_nothing_about_safety(self):
        grid = GridWorld(5, 5)
        agents = {"a": ((0, 2), (4, 2)), "b": ((2, 0), (2, 4))}
        sol = prioritized_planning(grid, agents, order=["b", "a"])
        self.assertIsNotNone(sol)
        self.assertIsNone(detect_first_conflict(sol.paths))

    def test_unsolvable_corridor_returns_none(self):
        grid = GridWorld(3, 1)
        agents = {"a": ((0, 0), (2, 0)), "b": ((2, 0), (0, 0))}
        self.assertIsNone(prioritized_planning(grid, agents))


class TestSolutionHelpers(unittest.TestCase):
    def test_costs_and_makespan(self):
        paths = {"a": [(0, 0), (1, 0), (2, 0)], "b": [(0, 1), (1, 1)]}
        self.assertEqual(sum_of_costs(paths), 3)   # 2 + 1
        self.assertEqual(makespan(paths), 2)

    def test_pad_paths(self):
        paths = {"a": [(0, 0), (1, 0), (2, 0)], "b": [(0, 1)]}
        padded = pad_paths(paths)
        self.assertEqual(len(padded["a"]), 3)
        self.assertEqual(len(padded["b"]), 3)
        self.assertEqual(padded["b"], [(0, 1), (0, 1), (0, 1)])

    def test_solution_makespan_property(self):
        sol = Solution(paths={"a": [(0, 0), (1, 0)]}, cost=1)
        self.assertEqual(sol.makespan, 1)


if __name__ == "__main__":
    unittest.main()
