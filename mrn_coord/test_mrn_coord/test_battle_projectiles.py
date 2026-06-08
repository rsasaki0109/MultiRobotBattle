"""Projectile fire mode — travel time, misses, and range falloff."""

from __future__ import annotations

import unittest

from mrn_coord.battle import RED, BLUE, Bot, BattleConfig, battle_step, simulate
from mrn_coord.battle_projectiles import ProjectileState, shot_accuracy


class TestProjectileCombat(unittest.TestCase):
    def test_hitscan_still_instant(self):
        cfg = BattleConfig(attack_range=4.0, dps=50.0, dt=0.1, max_speed=0.0,
                           fire_mode="hitscan")
        bots = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(2.0, 0.0, 0, 0, BLUE, 100, 100)]
        battle_step(bots, cfg)
        self.assertLess(bots[1].hp, 100)

    def test_projectile_defers_damage(self):
        cfg = BattleConfig(attack_range=4.0, dps=50.0, dt=0.1, max_speed=0.0,
                           fire_mode="projectile", fire_interval=0.05,
                           projectile_speed=10.0)
        bots = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(3.0, 0.0, 0, 0, BLUE, 100, 100)]
        state = ProjectileState()
        battle_step(bots, cfg, projectile_state=state, tick=0)
        self.assertEqual(bots[1].hp, 100)
        self.assertGreater(len(state.projectiles), 0)

    def test_projectile_hits_after_travel(self):
        cfg = BattleConfig(attack_range=4.0, dps=50.0, dt=0.1, max_speed=0.0,
                           fire_mode="projectile", fire_interval=0.05,
                           projectile_speed=20.0, accuracy_max=1.0,
                           accuracy_min=1.0)
        bots = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(2.0, 0.0, 0, 0, BLUE, 100, 100)]
        state = ProjectileState()
        for tick in range(8):
            battle_step(bots, cfg, projectile_state=state, tick=tick)
        self.assertLess(bots[1].hp, 100)

    def test_accuracy_falls_with_range(self):
        near = shot_accuracy(1.0, 5.0, acc_min=0.5, acc_max=1.0)
        far = shot_accuracy(4.5, 5.0, acc_min=0.5, acc_max=1.0)
        self.assertGreater(near, far)

    def test_obstacle_blocks_projectile_fire(self):
        cfg = BattleConfig(attack_range=5.0, dps=10.0, dt=0.1, max_speed=0.0,
                           fire_mode="projectile",
                           obstacles=((5.0, 0.0, 2.0),))
        bots = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(10.0, 0.0, 0, 0, BLUE, 100, 100)]
        state = ProjectileState()
        battle_step(bots, cfg, projectile_state=state, tick=0)
        self.assertEqual(len(state.projectiles), 0)

    def test_tracer_mode_applies_damage_on_fire(self):
        cfg = BattleConfig(attack_range=4.0, dps=50.0, dt=0.1, max_speed=0.0,
                           fire_mode="projectile", projectile_damage="on_fire",
                           fire_interval=0.1, accuracy_min=1.0)
        bots = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(2.0, 0.0, 0, 0, BLUE, 100, 100)]
        battle_step(bots, cfg)
        self.assertLess(bots[1].hp, 100)

    def test_showcase_duel_resolves_with_projectiles(self):
        from mrn_coord.battle import battle_scenario
        bots, cfg, _ = battle_scenario("duel")
        self.assertEqual(cfg.fire_mode, "projectile")
        self.assertEqual(cfg.projectile_damage, "on_hit")
        res = simulate(bots, cfg, max_ticks=1000)
        self.assertIsNotNone(res.winner)
        self.assertEqual(len(res.frames), len(res.projectiles))


if __name__ == "__main__":
    unittest.main()
