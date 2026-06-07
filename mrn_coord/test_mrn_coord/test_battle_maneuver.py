"""Tests for MAPF-planned battle maneuver."""

import unittest

from mrn_coord.battle import (
    BLUE,
    RED,
    BattleConfig,
    Bot,
    battle_scenario,
    battle_step,
    run_battle,
    simulate,
)
from mrn_coord.battle_maneuver import (
    ManeuverState,
    grid_from_battle,
    plan_maneuver,
    replan_paths,
    world_to_cell,
)


class TestManeuverGrid(unittest.TestCase):
    def test_obstacles_block_cells(self):
        obstacles = ((20.0, 12.0, 2.6),)
        cfg = BattleConfig(obstacles=obstacles, maneuver_cell_size=1.0)
        grid = grid_from_battle(cfg)
        blocked = world_to_cell((20.0, 12.0), 1.0)
        self.assertFalse(grid.is_free(blocked))

    def test_astar_routes_around_obstacle(self):
        obstacles = ((5.0, 5.0, 1.5),)
        cfg = BattleConfig(width=10.0, height=10.0, obstacles=obstacles,
                           maneuver_cell_size=1.0)
        grid = grid_from_battle(cfg)
        agents = {0: ((0, 5), (9, 5))}
        paths = plan_maneuver(grid, agents, "astar")
        self.assertIn(0, paths)
        cells = paths[0]
        for c in cells:
            self.assertTrue(grid.is_free(c))


class TestManeuverBattle(unittest.TestCase):
    def test_greedy_unchanged(self):
        a = run_battle(10, BattleConfig(maneuver="greedy"), seed=2)
        b = run_battle(10, BattleConfig(), seed=2)
        self.assertEqual(a.winner, b.winner)

    def test_cbs_battle_resolves(self):
        cfg = BattleConfig(maneuver="prioritized", tactics="count_aware",
                           maneuver_replan_ticks=15)
        res = run_battle(6, cfg, seed=1, max_ticks=500)
        resolved = (res.winner is not None or
                    all(n == 0 for n in res.survivors.values()))
        self.assertTrue(resolved)

    def test_maneuver_by_team_headline(self):
        cfg = BattleConfig(
            tactics="count_aware",
            maneuver="greedy",
            maneuver_by_team={RED: "astar", BLUE: "greedy"},
            maneuver_replan_ticks=12,
        )
        res = run_battle(6, cfg, seed=3, max_ticks=500)
        resolved = (res.winner is not None or
                    all(n == 0 for n in res.survivors.values()))
        self.assertTrue(resolved)

    def test_chokepoint_with_planned_maneuver(self):
        from dataclasses import replace
        bots, base_cfg, _ = battle_scenario("chokepoint")
        cfg = replace(base_cfg, maneuver="astar", maneuver_replan_ticks=15)
        res = simulate(bots, cfg, max_ticks=700)
        self.assertIsNotNone(res.winner)


class TestManeuverState(unittest.TestCase):
    def test_replan_is_cached(self):
        cfg = BattleConfig(maneuver="astar", maneuver_replan_ticks=20)
        bots = [Bot(0.0, 5.0, 0, 0, RED, 100, 100),
                Bot(30.0, 5.0, 0, 0, BLUE, 100, 100)]
        state = ManeuverState()
        from mrn_coord.battle_policy.count_aware import CountAwarePolicy
        policy = CountAwarePolicy("auto")
        live = bots
        d0 = [policy.decide(live, i, cfg) for i in range(2)]
        replan_paths(bots, live, d0, cfg, state, 0)
        first = dict(state.paths)
        replan_paths(bots, live, d0, cfg, state, 5)
        self.assertEqual(first, state.paths)


if __name__ == "__main__":
    unittest.main()
