"""Tests for the swarm battle — invariants of the team-combat simulation."""

import unittest

from mrn_coord.battle import (
    BLUE,
    CLASSES,
    GREEN,
    RED,
    SCENARIO_NAMES,
    BattleConfig,
    Bot,
    battle_scenario,
    battle_step,
    make_armies,
    make_unit,
    run_battle,
    simulate,
)


class TestArmies(unittest.TestCase):
    def test_counts_and_sides(self):
        cfg = BattleConfig()
        bots = make_armies(10, cfg, seed=0)
        self.assertEqual(len(bots), 20)
        red = [b for b in bots if b.team == RED]
        blue = [b for b in bots if b.team == BLUE]
        self.assertEqual(len(red), 10)
        self.assertEqual(len(blue), 10)
        # everyone starts alive at full health
        self.assertTrue(all(b.alive and b.hp == b.max_hp for b in bots))
        # red clusters left of centre, blue right of centre
        self.assertTrue(all(b.x < cfg.width / 2 for b in red))
        self.assertTrue(all(b.x > cfg.width / 2 for b in blue))


class TestCombatRules(unittest.TestCase):
    def test_damage_only_within_range(self):
        cfg = BattleConfig(attack_range=3.0, dps=10.0, dt=0.1)
        # one red, one blue, far apart (> attack_range) → nobody fires
        far = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
               Bot(20.0, 0.0, 0, 0, BLUE, 100, 100)]
        battle_step(far, cfg)
        self.assertTrue(all(b.hp == 100 for b in far))
        # close (< attack_range) → both take damage
        near = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(1.0, 0.0, 0, 0, BLUE, 100, 100)]
        battle_step(near, cfg)
        self.assertTrue(all(b.hp < 100 for b in near))

    def test_focus_fire_stacks(self):
        # a 1v1 vs a 1v3: the lone target should take ~3x the damage
        cfg = BattleConfig(attack_range=4.0, dps=10.0, dt=0.1, max_speed=0.0)
        duel = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(1.0, 0.0, 0, 0, BLUE, 100, 100)]
        battle_step(duel, cfg)
        solo_dmg = 100 - duel[1].hp
        swarmed = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                   Bot(0.0, 1.0, 0, 0, BLUE, 100, 100),
                   Bot(0.0, -1.0, 0, 0, BLUE, 100, 100),
                   Bot(1.0, 0.0, 0, 0, BLUE, 100, 100)]
        battle_step(swarmed, cfg)
        focus_dmg = 100 - swarmed[0].hp
        self.assertAlmostEqual(focus_dmg, 3 * solo_dmg, places=6)

    def test_zero_hp_is_eliminated(self):
        cfg = BattleConfig(attack_range=4.0, dps=1000.0, dt=1.0)
        bots = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(1.0, 0.0, 0, 0, BLUE, 100, 100)]
        battle_step(bots, cfg)
        self.assertTrue(all(not b.alive and b.hp == 0.0 for b in bots))


class TestEngagement(unittest.TestCase):
    def test_default_battles_are_decisive(self):
        # fight-to-the-death default → every battle reaches a winner
        for seed in range(6):
            res = run_battle(14, seed=seed)
            self.assertIsNotNone(res.winner, f"seed {seed} stalled")
            # the loser is wiped out; the winner has survivors
            self.assertEqual(res.survivors[res.winner] > 0, True)
            loser = BLUE if res.winner == RED else RED
            self.assertEqual(res.survivors[loser], 0)

    def test_counts_never_increase(self):
        res = run_battle(12, seed=3)
        reds = [c[0] for c in res.counts]
        blues = [c[1] for c in res.counts]
        self.assertEqual(reds, sorted(reds, reverse=True))
        self.assertEqual(blues, sorted(blues, reverse=True))

    def test_total_bots_conserved(self):
        res = run_battle(12, seed=5)
        # alive + dead is constant across the whole history
        for frame in res.frames:
            self.assertEqual(len(frame), 24)
        last = res.frames[-1]
        alive = sum(1 for b in last if b[4])
        self.assertEqual(alive, res.survivors[RED] + res.survivors[BLUE])

    def test_deterministic(self):
        a = run_battle(13, seed=7)
        b = run_battle(13, seed=7)
        self.assertEqual(a.winner, b.winner)
        self.assertEqual(a.ticks, b.ticks)
        self.assertEqual(a.survivors, b.survivors)
        self.assertEqual(a.frames[-1], b.frames[-1])

    def test_seed_changes_outcome_history(self):
        a = run_battle(13, seed=1)
        b = run_battle(13, seed=2)
        # different seeds → different engagements (not identical histories)
        self.assertNotEqual(a.frames[5], b.frames[5])


class TestUnitClasses(unittest.TestCase):
    def test_make_unit_applies_class_stats(self):
        tank = make_unit(0.0, 0.0, RED, "tank")
        st = CLASSES["tank"]
        self.assertEqual(tank.kind, "tank")
        self.assertEqual(tank.hp, st["hp"])
        self.assertEqual(tank.max_hp, st["hp"])
        self.assertEqual(tank.dps, st["dps"])
        self.assertEqual(tank.attack_range, st["attack_range"])
        self.assertEqual(tank.max_speed, st["max_speed"])

    def test_per_bot_range_is_honoured(self):
        # a sniper (range 6.5) hits an enemy at distance 5; a soldier (3.5) cannot
        cfg = BattleConfig(max_speed=0.0)
        far = 5.0
        sniper = [make_unit(0.0, 0.0, RED, "sniper"),
                  Bot(far, 0.0, 0, 0, BLUE, 100, 100)]
        battle_step(sniper, cfg)
        self.assertLess(sniper[1].hp, 100)        # sniper reaches it
        soldier = [make_unit(0.0, 0.0, RED, "soldier"),
                   Bot(far, 0.0, 0, 0, BLUE, 100, 100)]
        battle_step(soldier, cfg)
        self.assertEqual(soldier[1].hp, 100)      # soldier out of range


class TestFreeForAll(unittest.TestCase):
    def test_three_team_battle_has_one_or_no_winner(self):
        res = run_battle(10, seed=6, num_teams=3)
        self.assertEqual(res.teams, [0, 1, 2])
        standing = [t for t in res.teams if res.survivors[t] > 0]
        self.assertLessEqual(len(standing), 1)
        if res.winner is not None:
            self.assertEqual(standing, [res.winner])

    def test_simulate_on_custom_bots(self):
        bots = [Bot(1.0, 1.0, 0, 0, RED, 100, 100),
                Bot(2.0, 1.0, 0, 0, BLUE, 100, 100)]
        res = simulate(bots, BattleConfig(), max_ticks=300)
        self.assertIn(res.winner, (RED, BLUE, None))
        self.assertEqual(sorted(res.teams), [0, 1])


class TestTerrain(unittest.TestCase):
    def test_obstacles_are_never_penetrated(self):
        from mrn_coord.battle_terrain import point_clearance_rect

        bots, cfg, _ = battle_scenario("chokepoint")
        res = simulate(bots, cfg, max_ticks=700)
        for frame in res.frames:
            for (x, y, team, hp, alive, kind) in frame:
                if not alive:
                    continue
                for (ox, oy, r) in cfg.obstacles:
                    self.assertGreater((x - ox) ** 2 + (y - oy) ** 2, (r * 0.7) ** 2)
                for (cx, cy, hw, hh) in cfg.walls:
                    self.assertGreater(
                        point_clearance_rect(x, y, cx, cy, hw, hh), 0.35)


class TestScenarios(unittest.TestCase):
    def test_all_scenarios_are_decisive(self):
        for name in SCENARIO_NAMES:
            bots, cfg, title = battle_scenario(name)
            limit = 1000 if name in ("kingdom", "grand_alliance") else 900
            if name in ("hill", "domination"):
                limit = 700
            res = simulate(bots, cfg, max_ticks=limit)
            self.assertIsNotNone(res.winner, f"{name} stalled")
            self.assertTrue(title)


class TestLineOfSight(unittest.TestCase):
    def test_wall_blocks_fire(self):
        cfg = BattleConfig(attack_range=5.0, dps=10.0, dt=0.1, max_speed=0.0,
                           walls=((5.0, 0.0, 1.0, 2.0),))
        bots = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(10.0, 0.0, 0, 0, BLUE, 100, 100)]
        battle_step(bots, cfg)
        self.assertTrue(all(b.hp == 100 for b in bots))

    def test_obstacle_blocks_fire(self):
        # pillar between two bots — no damage despite being in range
        cfg = BattleConfig(attack_range=5.0, dps=10.0, dt=0.1, max_speed=0.0,
                           obstacles=((5.0, 0.0, 2.0),))
        bots = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(10.0, 0.0, 0, 0, BLUE, 100, 100)]
        battle_step(bots, cfg)
        self.assertTrue(all(b.hp == 100 for b in bots))

    def test_partial_cover_scales_damage(self):
        base = dict(attack_range=5.0, dps=10.0, dt=0.1, max_speed=0.0)
        clear_cfg = BattleConfig(**base)
        partial_cfg = BattleConfig(**base, cover_margin=1.0,
                                   obstacles=((2.5, 1.5, 1.0),))
        clear = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                 Bot(5.0, 0.0, 0, 0, BLUE, 100, 100)]
        battle_step(clear, clear_cfg)
        full_dmg = 100 - clear[1].hp
        partial = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                   Bot(5.0, 0.0, 0, 0, BLUE, 100, 100)]
        battle_step(partial, partial_cfg)
        partial_dmg = 100 - partial[1].hp
        self.assertAlmostEqual(partial_dmg, 0.5 * full_dmg, places=6)

    def test_open_field_without_obstacles_unchanged(self):
        cfg = BattleConfig(attack_range=3.0, dps=10.0, dt=0.1, require_los=True)
        near = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(1.0, 0.0, 0, 0, BLUE, 100, 100)]
        battle_step(near, cfg)
        self.assertTrue(all(b.hp < 100 for b in near))

    def test_body_can_block_fire(self):
        cfg = BattleConfig(attack_range=5.0, dps=10.0, dt=0.1, max_speed=0.0,
                           body_blocks_fire=True, body_radius=0.8)
        blocked = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                   Bot(2.5, 0.0, 0, 0, GREEN, 100, 100),   # body in the way
                   Bot(5.0, 0.0, 0, 0, BLUE, 100, 100)]
        battle_step(blocked, cfg)
        self.assertEqual(blocked[2].hp, 100)


class TestRetreatOption(unittest.TestCase):
    def test_retreat_pulls_wounded_away(self):
        # a wounded red bot with retreat enabled should steer away from its enemy
        cfg = BattleConfig(retreat_frac=0.5, w_retreat=5.0, w_pursue=2.0,
                           w_flock=0.0, w_sep=0.0, attack_range=0.0, max_speed=10.0)
        bots = [Bot(0.0, 0.0, 0, 0, RED, 10, 100),    # red at 10% hp (wounded)
                Bot(3.0, 0.0, 0, 0, BLUE, 100, 100)]  # blue to the +x
        battle_step(bots, cfg)
        # the wounded red bot should have moved in -x (away from the blue enemy)
        self.assertLess(bots[0].x, 0.0)


if __name__ == "__main__":
    unittest.main()
