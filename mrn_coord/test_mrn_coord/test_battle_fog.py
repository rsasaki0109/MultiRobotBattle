"""Tests for fog-of-war sensing in swarm battle."""

import unittest

from mrn_coord.battle import (
    BLUE,
    RED,
    BattleConfig,
    Bot,
    battle_scenario,
    battle_step,
    simulate,
)
from mrn_coord.battle_fog import can_see_enemy, sense_range_for, vision_clear


class TestFogOfWar(unittest.TestCase):
    def test_distant_enemy_not_visible(self):
        cfg = BattleConfig(fog_of_war=True, sense_range=5.0, fog_requires_los=False)
        live = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(20.0, 0.0, 0, 0, BLUE, 100, 100)]
        self.assertFalse(can_see_enemy(live, 0, 1, cfg))

    def test_near_enemy_visible_without_los_check(self):
        cfg = BattleConfig(fog_of_war=True, sense_range=5.0, fog_requires_los=False)
        live = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(3.0, 0.0, 0, 0, BLUE, 100, 100)]
        self.assertTrue(can_see_enemy(live, 0, 1, cfg))

    def test_obstacle_blocks_vision(self):
        cfg = BattleConfig(fog_of_war=True, sense_range=12.0, fog_requires_los=True,
                           obstacles=((5.0, 0.0, 2.0),))
        self.assertFalse(vision_clear(0.0, 0.0, 10.0, 0.0, cfg))
        live = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(10.0, 0.0, 0, 0, BLUE, 100, 100)]
        self.assertFalse(can_see_enemy(live, 0, 1, cfg))

    def test_fog_prevents_distant_fire(self):
        cfg = BattleConfig(attack_range=8.0, dps=10.0, dt=0.1, max_speed=0.0,
                           fog_of_war=True, sense_range=3.0, fog_requires_los=False)
        bots = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(6.0, 0.0, 0, 0, BLUE, 100, 100)]
        battle_step(bots, cfg)
        self.assertEqual(bots[1].hp, 100)

    def test_scout_sees_farther(self):
        cfg = BattleConfig(fog_of_war=True, sense_range=6.0, fog_requires_los=False)
        scout = Bot(0.0, 0.0, 0, 0, RED, 100, 100, kind="scout")
        soldier = Bot(0.0, 0.0, 0, 0, RED, 100, 100, kind="soldier")
        self.assertGreater(sense_range_for(scout, cfg), sense_range_for(soldier, cfg))

    def test_fog_ambush_scenario_resolves(self):
        bots, cfg, _ = battle_scenario("fog_ambush")
        self.assertTrue(cfg.fog_of_war)
        res = simulate(bots, cfg, max_ticks=900)
        self.assertIsNotNone(res.winner)
        self.assertTrue(len(res.fog_visible) == len(res.frames))


if __name__ == "__main__":
    unittest.main()
