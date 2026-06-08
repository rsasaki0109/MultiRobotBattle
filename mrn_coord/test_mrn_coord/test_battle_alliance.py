"""Tests for allied multi-army battles."""

import unittest

from mrn_coord.battle import (
    BLUE,
    GREEN,
    RED,
    YELLOW,
    BattleConfig,
    Bot,
    battle_scenario,
    battle_step,
    simulate,
)
from mrn_coord.battle_teams import teams_are_enemies


class TestAlliances(unittest.TestCase):
    def test_allied_teams_are_not_enemies(self):
        al = {RED: 0, GREEN: 0, BLUE: 1, YELLOW: 1}
        self.assertFalse(teams_are_enemies(al, RED, GREEN))
        self.assertFalse(teams_are_enemies(al, BLUE, YELLOW))
        self.assertTrue(teams_are_enemies(al, RED, BLUE))

    def test_allies_do_not_damage_each_other(self):
        cfg = BattleConfig(attack_range=4.0, dps=100.0, dt=1.0,
                           alliances={RED: 0, GREEN: 0})
        bots = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(1.0, 0.0, 0, 0, GREEN, 100, 100)]
        battle_step(bots, cfg)
        self.assertTrue(all(b.hp == 100 for b in bots))

    def test_grand_alliance_scenario_resolves(self):
        bots, cfg, _ = battle_scenario("grand_alliance")
        self.assertGreaterEqual(len(bots), 500)
        self.assertEqual(cfg.alliances[RED], cfg.alliances[GREEN])
        self.assertNotEqual(cfg.alliances[RED], cfg.alliances[BLUE])
        res = simulate(bots, cfg, max_ticks=1000, frame_stride=8)
        self.assertIsNotNone(res.winning_alliance)
        self.assertIsNotNone(res.winner)


if __name__ == "__main__":
    unittest.main()
