"""Tests for the Switchable ADG (Berndt et al.) robust MAPF execution.

The ADG executes a MAPF plan by passing-order precedences, collision-free
whatever the timing. The *switchable* variant flips a passing order when a ready
robot is stuck behind a delayed one -- but only when the flip keeps the graph
acyclic, so it is always deadlock-free. Contracts: the executed schedule is
collision-free in both modes, switching helps (lower makespan) only when the
delayed robot was scheduled first, it never deadlocks, the acyclicity guard
refuses cycle-closing reversals, and runs are deterministic. Needs ``mrn_coord``
(the planners); each test imports it at run time and skips cleanly when it is not
importable (so collection never touches it).
"""

import unittest


class TestSwitchableAdg(unittest.TestCase):
    def _imports(self):
        try:
            from mrn_coord.mapf import GridWorld, cbs
            from mrn_sim import switchable_adg as adg
        except ModuleNotFoundError:
            self.skipTest("mrn_coord not importable (needs colcon build)")
        return GridWorld, cbs, adg

    def _plus(self, GridWorld, n):
        mid = n // 2
        free = set()
        for x in range(n):
            free.add((x, mid))
        for y in range(n):
            free.add((mid, y))
        blocked = {(x, y) for x in range(n) for y in range(n)} - free
        return GridWorld(n, n, blocked=frozenset(blocked)), mid

    def _corridor(self, GridWorld, L, ax):
        free = set((x, 1) for x in range(L))
        free.add((ax, 2))
        blocked = {(x, y) for x in range(L) for y in range(3)} - free
        return GridWorld(L, 3, blocked=frozenset(blocked))

    def _crossing_plan(self, GridWorld, cbs):
        g, mid = self._plus(GridWorld, 9)
        sol = cbs(g, {"r_main": ((0, mid), (8, mid)),
                      "r_block": ((mid, mid - 1), (mid, mid + 1))})
        return sol.paths

    def test_switch_helps_when_first_mover_delayed(self):
        # The centre-crossing-first robot is delayed; flipping the one crossing
        # edge lets the long-haul robot go first -> lower makespan, one switch.
        GridWorld, cbs, adg = self._imports()
        paths = self._crossing_plan(GridWorld, cbs)
        cc, ce = adg.build_adg(paths)
        fix = adg.simulate(cc, ce, {"r_block": 8}, switchable=False,
                           keep_history=True)
        sc, se = adg.build_adg(paths)
        sw = adg.simulate(sc, se, {"r_block": 8}, switchable=True,
                          keep_history=True)
        self.assertLess(sw.makespan, fix.makespan)
        self.assertEqual(sw.switches, 1)
        self.assertTrue(sw.finished and fix.finished)
        self.assertFalse(sw.deadlock)
        self.assertTrue(adg.schedule_is_collision_free(sw.history))
        self.assertTrue(adg.schedule_is_collision_free(fix.history))

    def test_no_switch_when_it_cannot_help(self):
        # Delay the *second*-crossing robot: re-ordering buys nothing, so it must
        # fire zero switches and match the fixed schedule exactly.
        GridWorld, cbs, adg = self._imports()
        paths = self._crossing_plan(GridWorld, cbs)
        cc, ce = adg.build_adg(paths)
        fix = adg.simulate(cc, ce, {"r_main": 8}, switchable=False)
        sc, se = adg.build_adg(paths)
        sw = adg.simulate(sc, se, {"r_main": 8}, switchable=True)
        self.assertEqual(sw.switches, 0)
        self.assertEqual(sw.makespan, fix.makespan)

    def test_head_on_corridor_never_deadlocks(self):
        # In a single-file corridor every reversal would close a cycle; the guard
        # refuses them all, and the run still finishes on the fixed order.
        GridWorld, cbs, adg = self._imports()
        g = self._corridor(GridWorld, 7, 3)
        sol = cbs(g, {"r0": ((0, 1), (6, 1)), "r1": ((6, 1), (0, 1))})
        cc, ce = adg.build_adg(sol.paths)
        sw = adg.simulate(cc, ce, {"r0": 8}, switchable=True, keep_history=True)
        self.assertEqual(sw.switches, 0)
        self.assertTrue(sw.finished)
        self.assertFalse(sw.deadlock)
        self.assertTrue(adg.schedule_is_collision_free(sw.history))
        for e in ce:
            if e.switchable:
                self.assertFalse(adg._reversal_is_safe(cc, ce, e))

    def test_reversal_safe_when_independent(self):
        # The lone crossing edge has no competing constraint, so flipping it is
        # acyclic-safe -- the guard must permit it.
        GridWorld, cbs, adg = self._imports()
        paths = self._crossing_plan(GridWorld, cbs)
        cc, ce = adg.build_adg(paths)
        switchables = [e for e in ce if e.switchable]
        self.assertEqual(len(switchables), 1)
        self.assertTrue(adg._reversal_is_safe(cc, ce, switchables[0]))

    def test_no_delay_is_a_noop(self):
        # With no delay the planned order is already valid: zero switches, equal
        # makespan, collision-free.
        GridWorld, cbs, adg = self._imports()
        paths = self._crossing_plan(GridWorld, cbs)
        cc, ce = adg.build_adg(paths)
        fix = adg.simulate(cc, ce, {}, switchable=False)
        sc, se = adg.build_adg(paths)
        sw = adg.simulate(sc, se, {}, switchable=True)
        self.assertEqual(sw.switches, 0)
        self.assertEqual(sw.makespan, fix.makespan)

    def test_deterministic(self):
        GridWorld, cbs, adg = self._imports()
        paths = self._crossing_plan(GridWorld, cbs)
        cc, ce = adg.build_adg(paths)
        a = adg.simulate(cc, ce, {"r_block": 8}, switchable=True)
        sc, se = adg.build_adg(paths)
        b = adg.simulate(sc, se, {"r_block": 8}, switchable=True)
        self.assertEqual(a.as_dict(), b.as_dict())


class TestRhcReorder(unittest.TestCase):
    """Receding-horizon re-ordering (Berndt et al., T-RO 2024): the SADG MILP that
    minimizes cumulative completion, vs the reactive single-flip. Pure (no
    planner), so it does not need mrn_coord."""

    def _imports(self):
        from mrn_sim import switchable_adg as adg
        return adg

    # A plan whose only helpful re-ordering needs a multi-edge horizon.
    DEMO = {
        0: [(4, 0), (5, 0), (5, 1)],
        1: [(4, 2), (3, 2), (2, 2), (2, 1), (2, 0)],
        2: [(0, 2), (1, 2), (1, 3), (2, 3), (3, 3), (4, 3), (4, 4), (4, 5)],
        3: [(2, 4), (3, 4), (4, 4), (4, 3), (4, 2), (4, 1)],
    }
    # A plan with a cell shared by three agents (collapses to a transitive chain
    # under build_adg).
    THREE = {
        0: [(0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (5, 2)],
        1: [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (4, 4), (4, 3), (4, 2), (4, 1)],
        2: [(5, 4), (4, 4), (3, 4), (3, 4), (3, 3)],
        3: [(3, 5), (4, 5), (4, 4), (4, 3), (4, 4), (4, 3), (5, 3), (4, 3)],
        4: [(1, 4), (1, 4), (1, 3)],
    }

    def _rhc(self, adg, paths, delay, h):
        cells, edges = adg.build_sadg(paths)
        return cells, adg.simulate_rhc(cells, edges, delay, horizon=h,
                                       keep_history=True)

    def test_reordering_is_collision_free_and_deadlock_free(self):
        adg = self._imports()
        cells, r = self._rhc(adg, self.DEMO, {3: 4}, 8)
        self.assertTrue(adg.schedule_is_collision_free(r.history))
        self.assertTrue(r.finished)
        self.assertFalse(r.deadlock)

    def test_reordering_reduces_cumulative_completion(self):
        adg = self._imports()
        cells, fix = self._rhc(adg, self.DEMO, {3: 4}, 0)
        _, rhc = self._rhc(adg, self.DEMO, {3: 4}, 8)
        self.assertLess(adg.cumulative_completion(cells, rhc.history),
                        adg.cumulative_completion(cells, fix.history))
        self.assertLessEqual(rhc.makespan, fix.makespan)

    def test_deeper_horizon_finds_what_horizon_one_misses(self):
        adg = self._imports()
        cells, fix = self._rhc(adg, self.DEMO, {3: 4}, 0)
        _, h1 = self._rhc(adg, self.DEMO, {3: 4}, 1)
        _, h8 = self._rhc(adg, self.DEMO, {3: 4}, 8)
        fix_cc = adg.cumulative_completion(cells, fix.history)
        self.assertEqual(adg.cumulative_completion(cells, h1.history), fix_cc)
        self.assertLess(adg.cumulative_completion(cells, h8.history), fix_cc)

    def test_all_pairs_sadg_is_safe_where_consecutive_greedy_collides(self):
        # On a 3-agent-shared cell the all-pairs SADG re-orders and stays
        # collision-free; the consecutive-edge reactive greedy executes a real
        # collision (the leak build_sadg exists to close).
        adg = self._imports()
        cells, rhc = self._rhc(adg, self.THREE, {2: 3}, 8)
        self.assertGreater(rhc.switches, 0)
        self.assertTrue(adg.schedule_is_collision_free(rhc.history))
        self.assertTrue(rhc.finished)
        ac, ae = adg.build_adg(self.THREE)
        greedy = adg.simulate(ac, ae, {2: 3}, switchable=True, keep_history=True)
        self.assertFalse(adg.schedule_is_collision_free(greedy.history))

    def test_build_sadg_matches_build_adg_on_two_agent_cells(self):
        # With no cell shared by 3+, all-pairs == consecutive.
        adg = self._imports()
        two = {0: [(0, 0), (1, 0), (2, 0)], 1: [(2, 1), (1, 1), (1, 0), (0, 0)]}
        _, a = adg.build_adg(two)
        _, b = adg.build_sadg(two)

        def key(es):
            return sorted((e.cell, e.first, e.kf, e.second, e.ks) for e in es)

        self.assertEqual(key(a), key(b))

    def test_deterministic(self):
        adg = self._imports()
        _, a = self._rhc(adg, self.DEMO, {3: 4}, 8)
        _, b = self._rhc(adg, self.DEMO, {3: 4}, 8)
        self.assertEqual(a.as_dict(), b.as_dict())


if __name__ == "__main__":
    unittest.main()
