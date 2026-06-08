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
        self.assertEqual(len(res.frames), len(res.explosions))

    def test_splash_hits_cluster(self):
        from mrn_coord.battle import make_unit
        from mrn_coord.battle_projectiles import (
            Projectile, ProjectileState, advance_projectiles,
        )

        cfg = BattleConfig(fire_mode="projectile", dt=0.1, splash_friendly_fire=False)
        bots = [
            make_unit(0.0, 0.0, RED, "artillery"),
            Bot(10.0, 0.0, 0, 0, BLUE, 100, 100),
            Bot(10.5, 0.6, 0, 0, BLUE, 100, 100),
            Bot(10.2, -0.5, 0, 0, BLUE, 100, 100),
        ]
        state = ProjectileState(projectiles=[
            Projectile(x=10.0, y=0.0, vx=0.0, vy=0.0, damage=40.0, team=RED,
                       target_bot_idx=1, splash_radius=2.5, homing=False,
                       aim_x=10.0, aim_y=0.0, ttl=0.2, age=0.19),
        ])
        dmg, _, expls = advance_projectiles(state, bots, cfg, dt=0.05)
        self.assertEqual(len(expls), 1)
        self.assertGreater(sum(dmg), 0.0)
        self.assertGreaterEqual(sum(1 for d in dmg if d > 0), 2)

    def test_artillery_barrage_scenario_resolves(self):
        from mrn_coord.battle import battle_scenario
        bots, cfg, _ = battle_scenario("artillery_barrage")
        self.assertEqual(cfg.fire_mode, "projectile")
        res = simulate(bots, cfg, max_ticks=900)
        self.assertIsNotNone(res.winner)
        self.assertGreater(sum(len(e) for e in res.explosions), 0)


if __name__ == "__main__":
    unittest.main()
