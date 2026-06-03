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
    icts,
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

    def test_positive_vertex_forces_occupancy(self):
        # The must-occupy half of disjoint splitting: pinning the agent to a
        # non-goal cell at a time must equal the path found by forbidding every
        # OTHER cell at that time (the brute-force encoding of "be here now").
        grid = GridWorld(5, 5)
        start, goal, v, t = (0, 0), (4, 0), (2, 2), 4
        pinned = plan_path(grid, start, goal, positive_vertex={(v, t)})
        self.assertIsNotNone(pinned)
        self.assertEqual(cell_at(pinned, t), v)
        forbid = {(c, t) for c in
                  ((x, y) for x in range(5) for y in range(5)) if c != v}
        brute = plan_path(grid, start, goal, vertex_constraints=frozenset(forbid))
        self.assertEqual(sum_of_costs({0: pinned}), sum_of_costs({0: brute}))

    def test_positive_on_goal_is_free_under_stay(self):
        # A positive constraint ON the goal is satisfied for free by arriving and
        # staying (stay-at-goal): it must NOT inflate the cost past the natural
        # shortest path. (Regression: an earlier version padded to the constraint
        # time and overcounted.)
        grid = GridWorld(5, 1)
        plain = plan_path(grid, (0, 0), (4, 0))
        pinned = plan_path(grid, (0, 0), (4, 0), positive_vertex={((4, 0), 9)})
        self.assertEqual(sum_of_costs({0: pinned}), sum_of_costs({0: plain}))
        self.assertEqual(cell_at(pinned, 9), (4, 0))

    def test_contradictory_positive_is_infeasible(self):
        # Two must-occupy cells at the same time can't both hold -> no path.
        grid = GridWorld(5, 1)
        self.assertIsNone(plan_path(
            grid, (0, 0), (4, 0),
            positive_vertex={((1, 0), 2), ((3, 0), 2)},
        ))


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


class TestDisjoint(unittest.TestCase):
    """Disjoint splitting: same optimum as standard CBS, fewer expansions."""

    def test_matches_standard_optimum_on_random(self):
        # Disjoint splitting must return the SAME optimal sum-of-costs as
        # standard CBS on every instance, with a valid collision-free solution
        # honoring endpoints. (Regression: an early version dropped the optimum
        # when a positive constraint landed on an agent's goal.)
        for w, h, n, obs in ((6, 6, 5, 0.12), (7, 7, 6, 0.05), (6, 6, 6, 0.0)):
            for seed in range(10):
                grid, agents = _rand_instance(w, h, n, seed, obstacle=obs)
                std = cbs(grid, agents, disjoint=False, max_expansions=50000)
                dis = cbs(grid, agents, disjoint=True, max_expansions=50000)
                if std is None:
                    self.assertIsNone(dis, f"{w}x{h} seed={seed}")
                    continue
                self.assertIsNotNone(dis, f"{w}x{h} seed={seed}")
                self.assertEqual(dis.cost, std.cost, f"{w}x{h} seed={seed}")
                self.assertIsNone(detect_first_conflict(dis.paths))
                for a, (start, goal) in agents.items():
                    self.assertEqual(dis.paths[a][0], start)
                    self.assertEqual(dis.paths[a][-1], goal)

    def test_partition_cuts_expansions_on_congestion(self):
        # On a congested battery disjoint splitting must expand no MORE in
        # aggregate than the standard two-negative split (and fewer in practice),
        # because its children partition rather than overlap the solution space —
        # while reaching the identical optimum on each instance.
        tot_std = tot_dis = 0
        for seed in range(12):
            grid, agents = _rand_instance(8, 8, 8, seed, obstacle=0.05)
            ss, sd = {}, {}
            std = cbs(grid, agents, disjoint=False, stats=ss,
                      max_expansions=60000)
            dis = cbs(grid, agents, disjoint=True, stats=sd,
                      max_expansions=60000)
            if std is None or dis is None:
                continue
            self.assertEqual(std.cost, dis.cost, f"seed={seed}")
            tot_std += ss["expansions"]
            tot_dis += sd["expansions"]
        self.assertLess(tot_dis, tot_std)

    def test_off_is_byte_identical_to_plain_cbs(self):
        # disjoint defaults OFF; the default path must be the standard split,
        # unchanged — same cost AND same expansion count as an explicit
        # disjoint=False call.
        grid, agents = _rand_instance(7, 7, 6, 4, obstacle=0.05)
        sa, sb = {}, {}
        default = cbs(grid, agents, stats=sa, max_expansions=50000)
        explicit = cbs(grid, agents, disjoint=False, stats=sb,
                       max_expansions=50000)
        self.assertEqual(default.cost, explicit.cost)
        self.assertEqual(sa["expansions"], sb["expansions"])

    def test_deterministic(self):
        grid, agents = _rand_instance(8, 8, 8, 1, obstacle=0.05)
        runs = []
        for _ in range(2):
            sd = {}
            sol = cbs(grid, agents, disjoint=True, stats=sd,
                      max_expansions=60000)
            runs.append((sol.cost, sd["expansions"]))
        self.assertEqual(runs[0], runs[1])

    def test_unsolvable_corridor_returns_none(self):
        grid = GridWorld(3, 1)
        agents = {"a": ((0, 0), (2, 0)), "b": ((2, 0), (0, 0))}
        self.assertIsNone(cbs(grid, agents, disjoint=True, max_expansions=2000))


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


class TestICTS(unittest.TestCase):
    """ICTS: cost-tree optimal search; pairwise pruning cuts joint searches."""

    def test_matches_cbs_optimum(self):
        # Both pruning settings must return the SAME optimal sum-of-costs as
        # plain CBS, with a valid collision-free solution honoring endpoints.
        for seed in range(12):
            grid, agents = _rand_instance(6, 6, 5, seed)
            base = cbs(grid, agents, max_expansions=50000)
            if base is None:
                continue
            for prune in ("pairwise", None):
                sol = icts(grid, agents, prune=prune, max_nodes=20000)
                self.assertIsNotNone(sol, f"seed={seed} prune={prune}")
                self.assertIsNone(detect_first_conflict(sol.paths))
                self.assertEqual(sol.cost, base.cost,
                                 f"seed={seed} prune={prune}")
                for a, (start, goal) in agents.items():
                    self.assertEqual(sol.paths[a][0], start)
                    self.assertEqual(sol.paths[a][-1], goal)

    def test_pairwise_pruning_cuts_joint_searches(self):
        # Pairwise pruning must never run MORE joint searches than the no-prune
        # ablation (which searches every node), and far fewer in aggregate on a
        # conflict-heavy battery — while reaching the identical optimum.
        tot_pair = tot_none = 0
        for seed in range(10):
            grid, agents = _rand_instance(6, 6, 5, seed, obstacle=0.12)
            if cbs(grid, agents, max_expansions=20000) is None:
                continue
            sp = {}
            sol = icts(grid, agents, prune="pairwise", stats=sp,
                       max_nodes=20000)
            sn = {}
            soln = icts(grid, agents, prune=None, stats=sn, max_nodes=20000)
            self.assertIsNotNone(sol)
            self.assertEqual(sol.cost, soln.cost, f"seed={seed}")
            self.assertLessEqual(sp["joint_searches"], sn["joint_searches"],
                                 f"seed={seed}")
            # Every node the no-prune ablation visits triggers a joint search.
            self.assertEqual(sn["joint_searches"], sn["nodes"], f"seed={seed}")
            # Pruned + searched accounts for every node visited under pruning.
            self.assertEqual(sp["pruned"] + sp["joint_searches"], sp["nodes"],
                             f"seed={seed}")
            tot_pair += sp["joint_searches"]
            tot_none += sn["joint_searches"]
        self.assertLess(tot_pair, tot_none)

    def test_deterministic(self):
        grid, agents = _rand_instance(6, 6, 5, 3, obstacle=0.12)
        runs = []
        for _ in range(2):
            sp = {}
            sol = icts(grid, agents, stats=sp, max_nodes=20000)
            runs.append((sol.cost, sp["nodes"], sp["joint_searches"]))
        self.assertEqual(runs[0], runs[1])

    def test_single_step_instance(self):
        # A trivial instance: two agents one step apart, no interaction. Root of
        # the increasing cost tree is already conflict-free.
        grid = GridWorld(3, 3)
        agents = {0: ((0, 0), (1, 0)), 1: ((0, 2), (1, 2))}
        sol = icts(grid, agents)
        self.assertEqual(sol.cost, 2)
        self.assertIsNone(detect_first_conflict(sol.paths))

    def test_unsolvable_corridor_returns_none(self):
        # 1-wide corridor swap is infeasible: no cost vector ever admits a
        # conflict-free joint path, so ICTS exhausts its node budget -> None.
        grid = GridWorld(3, 1)
        agents = {"a": ((0, 0), (2, 0)), "b": ((2, 0), (0, 0))}
        self.assertIsNone(icts(grid, agents, max_nodes=500))


_CROSSINGS = [
    (6, 6, {0: ((2, 0), (4, 5)), 1: ((1, 1), (4, 2))}),
    (7, 7, {0: ((1, 1), (5, 6)), 1: ((0, 2), (6, 3))}),
    (7, 7, {0: ((2, 0), (6, 6)), 1: ((0, 2), (6, 4))}),
    (7, 7, {0: ((2, 0), (4, 6)), 1: ((0, 2), (5, 6))}),
]


class TestRectangle(unittest.TestCase):
    """Rectangle symmetry: barrier splits keep the optimum, collapse the blowup."""

    def test_detects_known_rectangle(self):
        # A phase-locked same-direction crossing has a rectangle; the detector
        # returns a non-empty barrier for each agent.
        from mrn_coord.mapf.mdd import build_mdd
        from mrn_coord.mapf.rectangle import find_rectangle_barriers
        m0 = build_mdd(GridWorld(5, 5), (1, 0), (3, 2), 4)
        m1 = build_mdd(GridWorld(5, 5), (0, 1), (2, 3), 4)
        found = find_rectangle_barriers(m0, m1, 2)
        self.assertIsNotNone(found)
        barrier0, barrier1, _ = found
        self.assertTrue(barrier0 and barrier1)

    def test_barrier_split_preserves_optimum(self):
        # On each crossing, rectangle reasoning must return the SAME optimal cost
        # as CBS (and plain CBSH), with a valid collision-free solution, while
        # actually firing at least one barrier split.
        for w, h, agents in _CROSSINGS:
            grid = GridWorld(w, h)
            base = cbs(grid, agents, max_expansions=20000)
            s = {}
            sol = cbsh(grid, agents, heuristic="wdg", rectangle=True, stats=s,
                       max_expansions=20000)
            self.assertIsNotNone(sol)
            self.assertEqual(sol.cost, base.cost)
            self.assertIsNone(detect_first_conflict(sol.paths))
            self.assertGreater(s["rectangles"], 0)
            for a, (start, goal) in agents.items():
                self.assertEqual(sol.paths[a][0], start)
                self.assertEqual(sol.paths[a][-1], goal)

    def test_rectangle_collapses_expansions(self):
        # Barrier splitting must expand strictly fewer high-level nodes than the
        # same solver with rectangle reasoning off — that is the whole point.
        tot_on = tot_off = 0
        for w, h, agents in _CROSSINGS:
            grid = GridWorld(w, h)
            son = {}
            cbsh(grid, agents, heuristic="wdg", rectangle=True, stats=son,
                 max_expansions=20000)
            soff = {}
            cbsh(grid, agents, heuristic="wdg", rectangle=False, stats=soff,
                 max_expansions=20000)
            self.assertLessEqual(son["expansions"], soff["expansions"])
            tot_on += son["expansions"]
            tot_off += soff["expansions"]
        self.assertLess(tot_on * 5, tot_off)

    def test_off_matches_plain_cbsh(self):
        # The feature is opt-in: rectangle=False must behave exactly like the
        # default cbsh (same cost and same expansion count) on every instance.
        for seed in range(8):
            grid, agents = _rand_instance(6, 6, 6, seed, obstacle=0.12)
            s_default = {}
            d = cbsh(grid, agents, heuristic="wdg", stats=s_default,
                     max_expansions=20000)
            s_off = {}
            o = cbsh(grid, agents, heuristic="wdg", rectangle=False, stats=s_off,
                     max_expansions=20000)
            if d is None:
                self.assertIsNone(o)
                continue
            self.assertEqual(d.cost, o.cost)
            self.assertEqual(s_default["expansions"], s_off["expansions"])

    def test_optimum_preserved_on_random(self):
        # Even where rectangles rarely fire, enabling the feature must never
        # change the optimum or break the solution on random instances.
        for seed in range(12):
            grid, agents = _rand_instance(6, 6, 5, seed)
            base = cbs(grid, agents, max_expansions=20000)
            if base is None:
                continue
            sol = cbsh(grid, agents, heuristic="wdg", rectangle=True,
                       max_expansions=20000)
            self.assertIsNotNone(sol, f"seed={seed}")
            self.assertEqual(sol.cost, base.cost, f"seed={seed}")
            self.assertIsNone(detect_first_conflict(sol.paths))

    def test_deterministic(self):
        grid = GridWorld(7, 7)
        agents = {0: ((2, 0), (4, 6)), 1: ((0, 2), (5, 6))}
        runs = []
        for _ in range(2):
            s = {}
            sol = cbsh(grid, agents, heuristic="wdg", rectangle=True, stats=s)
            runs.append((sol.cost, s["expansions"], s["rectangles"]))
        self.assertEqual(runs[0], runs[1])


class TestMutex(unittest.TestCase):
    """Mutex propagation: the detector agrees with the direct dependency test."""

    def test_theorem2_matches_are_dependent(self):
        # The paper's Theorem 2: classify_conflict returns NC iff a conflict-free
        # pair of optimal paths exists — exactly what are_dependent computes. They
        # must never disagree.
        from mrn_coord.mapf.mdd import are_dependent, build_mdd
        from mrn_coord.mapf.mutex import classify_conflict
        import random
        disagree = checked = 0
        for seed in range(300):
            rng = random.Random(seed)
            w, h = rng.choice([(5, 5), (6, 5), (5, 6)])
            free = [(x, y) for x in range(w) for y in range(h)]
            rng.shuffle(free)
            sa, ga, sb, gb = free[:4]
            grid = GridWorld(w, h)
            pa, pb = plan_path(grid, sa, ga), plan_path(grid, sb, gb)
            if pa is None or pb is None:
                continue
            ca, cb = len(pa) - 1, len(pb) - 1
            if ca > cb:
                sa, ga, sb, gb, ca, cb = sb, gb, sa, ga, cb, ca
            mi = build_mdd(grid, sa, ga, ca)
            mj = build_mdd(grid, sb, gb, cb)
            if mi is None or mj is None:
                continue
            checked += 1
            cls = classify_conflict(grid, mi, mj)
            dep = are_dependent(grid, mi, mj, sa, sb)
            if (cls == "NC") != (not dep):
                disagree += 1
        self.assertGreater(checked, 0)
        self.assertEqual(disagree, 0)

    def test_detects_a_known_cardinal(self):
        # Two agents whose shortest paths must cross at (3,3): a pre-goal cardinal
        # conflict. classify must say PC, and the constraint sets must be
        # non-empty for both agents.
        from mrn_coord.mapf.mdd import build_mdd
        from mrn_coord.mapf.mutex import classify_conflict, pc_constraints
        grid = GridWorld(5, 5)
        mi = build_mdd(grid, (2, 3), (4, 3), 2)
        mj = build_mdd(grid, (3, 2), (3, 4), 2)
        self.assertEqual(classify_conflict(grid, mi, mj), "PC")
        ci, cj = pc_constraints(grid, mi, mj)
        self.assertTrue(ci and cj)

    def test_independent_agents_are_nc(self):
        # Two agents in far corners moving apart never conflict — NC, no
        # constraints needed.
        from mrn_coord.mapf.mdd import build_mdd
        from mrn_coord.mapf.mutex import classify_conflict
        grid = GridWorld(6, 6)
        mi = build_mdd(grid, (0, 0), (0, 2), 2)
        mj = build_mdd(grid, (5, 5), (5, 3), 2)
        self.assertEqual(classify_conflict(grid, mi, mj), "NC")


class TestCCBS(unittest.TestCase):
    """Continuous-time CBS: disk agents, geometrically collision-free plans."""

    def _sol(self, w, h, agents, radius=0.4, **kw):
        from mrn_coord.mapf import GridWorld
        from mrn_coord.mapf.ccbs import ccbs
        return ccbs(GridWorld(w, h), agents, radius=radius, **kw)

    def test_single_agent_continuous_cost(self):
        # A pure diagonal has irrational length — the discrete unit-clock cannot
        # represent it; continuous SIPP must.
        import math
        from mrn_coord.mapf.ccbs import shortest_trajectory
        from mrn_coord.mapf import GridWorld
        tr = shortest_trajectory(GridWorld(5, 5), (0, 0), (4, 4))
        self.assertAlmostEqual(tr[-1][1], 4 * math.sqrt(2), places=6)

    def test_mid_square_crossing_is_resolved(self):
        # Two paths cross the centre of a unit square: they share NO vertex and
        # NO edge, so the discrete model sees no conflict — yet the disks meet.
        # CCBS must detect and resolve it to >= 2r separation.
        from mrn_coord.mapf.ccbs import (min_separation, shortest_trajectory,
                                         first_collision)
        from mrn_coord.mapf import GridWorld
        agents = {0: ((0, 0), (2, 2)), 1: ((2, 0), (0, 2))}
        # the uncoordinated baseline geometrically collides (centres meet at 0)
        grid = GridWorld(5, 5)
        a = shortest_trajectory(grid, (0, 0), (2, 2))
        b = shortest_trajectory(grid, (2, 0), (0, 2))
        self.assertIsNotNone(first_collision(a, b, 0.4))
        sol = self._sol(5, 5, agents)
        self.assertIsNotNone(sol)
        self.assertGreaterEqual(
            min_separation(sol.trajectories[0], sol.trajectories[1]),
            0.8 - 1e-6)

    def test_solutions_are_geometrically_clear(self):
        # On a random battery every CCBS solution keeps all pairs >= 2r apart,
        # checked by the independent oracle (not the planner's own detector).
        import random
        from mrn_coord.mapf.ccbs import min_separation
        for seed in range(8):
            rng = random.Random(seed)
            free = [(x, y) for x in range(5) for y in range(5)]
            rng.shuffle(free)
            agents = {i: (free[i], free[3 + i]) for i in range(3)}
            sol = self._sol(5, 5, agents, max_expansions=20000)
            if sol is None:
                continue
            ids = list(sol.trajectories)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    self.assertGreaterEqual(
                        min_separation(sol.trajectories[ids[i]],
                                       sol.trajectories[ids[j]]),
                        0.8 - 1e-6, f"seed={seed}")

    def test_yield_is_a_fractional_wait(self):
        # The continuous signature: resolving the crossing costs LESS than the
        # whole-timestep yield the discrete clock would force. The extra cost
        # over the two uncoordinated diagonals is a real fraction < 1.
        import math
        from mrn_coord.mapf.ccbs import shortest_trajectory
        from mrn_coord.mapf import GridWorld
        grid = GridWorld(5, 5)
        free_cost = (shortest_trajectory(grid, (0, 0), (2, 2))[-1][1]
                     + shortest_trajectory(grid, (2, 0), (0, 2))[-1][1])
        sol = self._sol(5, 5, {0: ((0, 0), (2, 2)), 1: ((2, 0), (0, 2))})
        overhead = sol.cost - free_cost
        self.assertGreater(overhead, 0.0)
        self.assertLess(overhead, 2.0)  # not a pair of whole-timestep detours

    def test_deterministic(self):
        agents = {0: ((0, 0), (3, 3)), 1: ((3, 0), (0, 3))}
        runs = []
        for _ in range(2):
            st = {}
            sol = self._sol(7, 7, agents, stats=st)
            runs.append((round(sol.cost, 9), st["expansions"]))
        self.assertEqual(runs[0], runs[1])

    def test_corridor_swap_is_infeasible(self):
        # Radius 0.4 disks cannot pass in a 1-wide corridor -> no solution.
        sol = self._sol(5, 1, {0: ((0, 0), (4, 0)), 1: ((4, 0), (0, 0))},
                        max_expansions=5000)
        self.assertIsNone(sol)


class TestFlow(unittest.TestCase):
    """Anonymous makespan-optimal MAPF via network flow: certified, valid."""

    def _solve(self, w, h, starts, goals, **kw):
        from mrn_coord.mapf import GridWorld
        from mrn_coord.mapf.flow import anonymous_makespan
        return anonymous_makespan(GridWorld(w, h), starts, goals, **kw)

    def test_already_at_goal_set_is_makespan_zero(self):
        # Targets are interchangeable: if the start set equals the goal set, no
        # one moves -- makespan 0.
        st = {}
        res = self._solve(3, 3, [(0, 0), (2, 0)], [(2, 0), (0, 0)], stats=st)
        self.assertEqual(res[1], 0)
        self.assertTrue(st["certified"])

    def test_paths_are_valid_and_certified(self):
        from mrn_coord.mapf import GridWorld
        from mrn_coord.mapf.conflicts import detect_first_conflict
        starts, goals = [(0, 0), (4, 4)], [(4, 0), (0, 4)]
        st = {}
        res = self._solve(5, 5, starts, goals, stats=st)
        self.assertIsNotNone(res)
        paths, T = res
        self.assertTrue(st["certified"])
        self.assertTrue(all(len(p) == T + 1 for p in paths))
        self.assertEqual(sorted(p[0] for p in paths), sorted(starts))
        self.assertEqual(sorted(p[-1] for p in paths), sorted(goals))
        self.assertIsNone(detect_first_conflict({i: p for i, p in enumerate(paths)}))

    def test_matches_brute_force_optimum(self):
        # The self-certified makespan equals a brute-force anonymous joint-BFS
        # optimum on tiny instances.
        import itertools
        import random
        from collections import deque
        from mrn_coord.mapf import GridWorld
        from mrn_coord.mapf.flow import _neighbors4

        def brute(grid, starts, goals):
            goalset = frozenset(goals)
            start = tuple(starts)
            if frozenset(start) == goalset:
                return 0
            seen = {start}
            q = deque([(start, 0)])
            while q:
                cfg, d = q.popleft()
                opts = [[c] + _neighbors4(grid, c) for c in cfg]
                for nxt in itertools.product(*opts):
                    if len(set(nxt)) != len(nxt):
                        continue
                    if any(cfg[i] == nxt[j] and cfg[j] == nxt[i]
                           for i in range(len(cfg))
                           for j in range(i + 1, len(cfg))):
                        continue
                    if nxt in seen:
                        continue
                    if frozenset(nxt) == goalset:
                        return d + 1
                    seen.add(nxt)
                    q.append((nxt, d + 1))
            return None

        rng = random.Random(7)
        for _ in range(20):
            w, h = rng.choice([(3, 3), (4, 3), (3, 4)])
            n = rng.choice([2, 3])
            free = [(x, y) for x in range(w) for y in range(h)]
            rng.shuffle(free)
            starts, goals = free[:n], free[n:2 * n]
            grid = GridWorld(w, h)
            res = self._solve(w, h, starts, goals)
            self.assertEqual(res[1], brute(grid, starts, goals),
                             f"{w}x{h} {starts}->{goals}")

    def test_corridor_swap_anonymous_trivial_labeled_impossible(self):
        # On a 1-wide corridor the LABELED swap is impossible (cbs -> None), but
        # interchangeable targets make it trivial for the flow solver.
        from mrn_coord.mapf import GridWorld, cbs
        starts, goals = [(0, 0), (2, 0)], [(2, 0), (0, 0)]
        self.assertIsNotNone(self._solve(3, 1, starts, goals))
        self.assertIsNone(
            cbs(GridWorld(3, 1), {0: (starts[0], goals[0]),
                                  1: (starts[1], goals[1])}, max_expansions=2000))


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
