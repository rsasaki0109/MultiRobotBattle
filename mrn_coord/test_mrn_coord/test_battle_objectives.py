"""Tests for hill / domination objective modes."""

import unittest

from mrn_coord.battle import (
    BLUE,
    RED,
    BattleConfig,
    Bot,
    battle_scenario,
    simulate,
)
from mrn_coord.battle_objectives import ObjectiveTracker, zone_leader


class TestObjectives(unittest.TestCase):
    def test_zone_leader_majority(self):
        cfg = BattleConfig(objective="hill", objective_radius=5.0,
                           objective_center=(20.0, 12.0))
        bots = [
            Bot(20, 12, 0, 0, RED, 100, 100),
            Bot(21, 12, 0, 0, RED, 100, 100),
            Bot(30, 12, 0, 0, BLUE, 100, 100),
        ]
        self.assertEqual(zone_leader(bots, cfg, [RED, BLUE]), RED)

    def test_zone_leader_tie_is_contested(self):
        cfg = BattleConfig(objective="hill", objective_radius=5.0,
                           objective_center=(20.0, 12.0))
        bots = [
            Bot(20, 12, 0, 0, RED, 100, 100),
            Bot(21, 12, 0, 0, BLUE, 100, 100),
        ]
        self.assertIsNone(zone_leader(bots, cfg, [RED, BLUE]))

    def test_hill_consecutive_resets_when_contested(self):
        tr = ObjectiveTracker(BattleConfig(objective="hill", objective_hold_ticks=3))
        self.assertIsNone(tr.tick(RED))
        self.assertIsNone(tr.tick(RED))
        self.assertEqual(tr.tick(RED), RED)
        tr2 = ObjectiveTracker(BattleConfig(objective="hill", objective_hold_ticks=3))
        tr2.tick(RED)
        tr2.tick(RED)
        self.assertIsNone(tr2.tick(None))
        self.assertIsNone(tr2.tick(RED))

    def test_hill_scenario_resolves(self):
        bots, cfg, _ = battle_scenario("hill")
        self.assertEqual(cfg.objective, "hill")
        res = simulate(bots, cfg, max_ticks=600)
        self.assertIsNotNone(res.winner)
        self.assertEqual(res.objective, "hill")
        self.assertTrue(len(res.objective_zone) == 3)

    def test_domination_scenario_resolves(self):
        bots, cfg, _ = battle_scenario("domination")
        res = simulate(bots, cfg, max_ticks=700)
        self.assertIsNotNone(res.winner)
        self.assertEqual(res.objective, "domination")

    def test_ctf_pickup_and_capture(self):
        from mrn_coord.battle_objectives import CtfTracker

        cfg = BattleConfig(objective="ctf", objective_radius=3.0, base_radius=4.0)
        bots = [
            Bot(20, 12, 0, 0, RED, 100, 100),
            Bot(35, 12, 0, 0, BLUE, 100, 100),
        ]
        tr = CtfTracker(cfg, [RED, BLUE])
        self.assertIsNone(tr.tick(bots, cfg))
        self.assertEqual(tr.carrier_idx, 0)
        bx, by, _ = tr.bases[RED]
        bots[0].x, bots[0].y = bx, by
        self.assertEqual(tr.tick(bots, cfg), RED)

    def test_ctf_scenario_resolves(self):
        bots, cfg, _ = battle_scenario("ctf")
        res = simulate(bots, cfg, max_ticks=900)
        self.assertIsNotNone(res.winner)
        self.assertEqual(res.objective, "ctf")
        self.assertTrue(len(res.objective_zone) >= 3)

    def test_base_capture_requires_majority(self):
        from mrn_coord.battle_objectives import base_capture_leader

        cfg = BattleConfig(objective="base_assault", base_radius=4.0)
        teams = [RED, BLUE]
        bots = [
            Bot(34.0, 12.0, 0, 0, RED, 100, 100),
            Bot(35.0, 12.0, 0, 0, RED, 100, 100),
            Bot(33.0, 12.0, 0, 0, BLUE, 100, 100),
        ]
        self.assertEqual(base_capture_leader(bots, cfg, teams, BLUE), RED)

    def test_base_assault_tracker_wins_on_hold(self):
        from mrn_coord.battle_objectives import BaseAssaultTracker

        cfg = BattleConfig(objective="base_assault", base_radius=4.0,
                           objective_hold_ticks=3)
        teams = [RED, BLUE]
        tr = BaseAssaultTracker(cfg, teams)
        bots = [
            Bot(34.0, 12.0, 0, 0, RED, 100, 100),
            Bot(35.0, 12.0, 0, 0, RED, 100, 100),
        ]
        win = None
        for _ in range(5):
            win = tr.tick(bots, cfg)
            if win is not None:
                break
        self.assertEqual(win, RED)

    def test_base_assault_scenario_resolves(self):
        bots, cfg, _ = battle_scenario("base_assault")
        self.assertEqual(cfg.objective, "base_assault")
        res = simulate(bots, cfg, max_ticks=900)
        self.assertIsNotNone(res.winner)
        self.assertEqual(res.objective, "base_assault")
        self.assertTrue(len(res.objective_zone) >= 2)

    def test_ctf_mapf_pair_resolves(self):
        for name in ("ctf_mapf_local", "ctf_mapf_mapf"):
            bots, cfg, _ = battle_scenario(name)
            res = simulate(bots, cfg, max_ticks=900)
            self.assertIsNotNone(res.winner)
            self.assertEqual(res.objective, "ctf")


if __name__ == "__main__":
    unittest.main()
