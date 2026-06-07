"""Tests for TeamHOI-lite battle policies and token observations."""

import math
import unittest

from mrn_coord.battle import (
    BLUE,
    RED,
    BattleConfig,
    Bot,
    battle_step,
    make_unit,
    run_battle,
    simulate,
)
from mrn_coord.battle_policy import (
    CountAwarePolicy,
    build_observation,
)
from mrn_coord.battle_policy.count_aware import NearestPolicy, policy_for_name


class TestTokens(unittest.TestCase):
    def test_local_frame_places_enemy_ahead(self):
        live = [Bot(0.0, 0.0, 1.0, 0.0, RED, 100, 100),
                Bot(5.0, 0.0, 0.0, 0.0, BLUE, 100, 100)]
        obs = build_observation(live, 0)
        self.assertEqual(obs.n_allies, 1)
        self.assertEqual(obs.n_enemies, 1)
        tok = obs.enemy_tokens[0]
        self.assertAlmostEqual(tok.rel_x, 5.0, places=5)
        self.assertAlmostEqual(tok.rel_y, 0.0, places=5)

    def test_counts_include_whole_team(self):
        live = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(2.0, 0.0, 0, 0, RED, 100, 100),
                Bot(20.0, 0.0, 0, 0, BLUE, 100, 100),
                Bot(22.0, 0.0, 0, 0, BLUE, 100, 100),
                Bot(24.0, 0.0, 0, 0, BLUE, 100, 100)]
        obs = build_observation(live, 0, perception=3.0)
        self.assertEqual(obs.n_allies, 2)
        self.assertEqual(obs.n_enemies, 3)
        self.assertEqual(obs.n_local_allies, 1)
        self.assertEqual(obs.n_local_enemies, 0)


class TestCountAwarePolicy(unittest.TestCase):
    def test_outnumbered_reduces_pursue_scale(self):
        cfg = BattleConfig()
        policy = CountAwarePolicy("auto")
        # 2 allies vs 6 enemies
        live = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(1.0, 0.0, 0, 0, RED, 100, 100)]
        for i in range(6):
            live.append(Bot(20.0 + i, 0.0, 0, 0, BLUE, 100, 100))
        d_out = policy.decide(live, 0, cfg)
        # 6 allies vs 2 enemies
        live_adv = [Bot(0.0, 0.0, 0, 0, RED, 100, 100)]
        for i in range(5):
            live_adv.append(Bot(1.0 + i, 0.0, 0, 0, RED, 100, 100))
        live_adv.append(Bot(20.0, 0.0, 0, 0, BLUE, 100, 100))
        live_adv.append(Bot(21.0, 0.0, 0, 0, BLUE, 100, 100))
        d_in = policy.decide(live_adv, 0, cfg)
        self.assertLess(d_out.pursue_scale, d_in.pursue_scale)

    def test_focus_fire_prefers_wounded_enemy(self):
        cfg = BattleConfig(max_speed=0.0)
        policy = CountAwarePolicy("auto")
        live = [make_unit(0.0, 0.0, RED, "soldier"),
                Bot(3.5, 0.0, 0, 0, BLUE, 100, 100),
                Bot(3.0, 0.5, 0, 0, BLUE, 20, 100)]
        decision = policy.decide(live, 0, cfg)
        self.assertEqual(decision.target_index, 2)

    def test_sniper_kites_in_melee(self):
        cfg = BattleConfig(tactics="count_aware", max_speed=10.0,
                           w_pursue=2.0, w_retreat=3.0, w_sep=0.0, w_flock=0.0)
        sniper = make_unit(0.0, 0.0, RED, "sniper")
        enemy = Bot(1.0, 0.0, 0, 0, BLUE, 100, 100)
        bots = [sniper, enemy]
        battle_step(bots, cfg)
        self.assertLess(sniper.x, 0.0)

    def test_stance_presets(self):
        cfg = BattleConfig()
        policy = CountAwarePolicy("aggressive")
        live = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(10.0, 0.0, 0, 0, BLUE, 100, 100)]
        d = policy.decide(live, 0, cfg)
        self.assertGreater(d.pursue_scale, 1.2)


class TestBattleIntegration(unittest.TestCase):
    def test_nearest_policy_matches_default(self):
        cfg_default = BattleConfig()
        cfg_named = BattleConfig(tactics="nearest")
        a = run_battle(12, cfg_default, seed=4)
        b = run_battle(12, cfg_named, seed=4)
        self.assertEqual(a.winner, b.winner)
        self.assertEqual(a.ticks, b.ticks)

    def test_count_aware_battles_are_decisive(self):
        for seed in range(4):
            res = run_battle(12, BattleConfig(tactics="count_aware"), seed=seed)
            self.assertIsNotNone(res.winner, f"seed {seed} stalled")

    def test_transformer_battles_are_decisive(self):
        for seed in range(4):
            res = run_battle(12, BattleConfig(tactics="transformer"), seed=seed)
            self.assertIsNotNone(res.winner, f"seed {seed} stalled")

    def test_per_team_tactics(self):
        from mrn_coord.battle import RED
        cfg = BattleConfig(tactics="nearest",
                           tactics_by_team={RED: "count_aware"})
        res = run_battle(10, cfg, seed=2)
        self.assertIsNotNone(res.winner)

    def test_policy_for_name_variants(self):
        self.assertIsInstance(policy_for_name("nearest"), NearestPolicy)
        self.assertIsInstance(policy_for_name("count_aware"), CountAwarePolicy)
        self.assertEqual(policy_for_name("count_aware:defensive").stance, "defensive")
        from mrn_coord.battle_policy import TransformerPolicy
        self.assertIsInstance(policy_for_name("transformer"), TransformerPolicy)


if __name__ == "__main__":
    unittest.main()
