"""Tests for battle formation integration."""

import math
import unittest

from mrn_coord.battle import (
    BLUE,
    RED,
    BattleConfig,
    Bot,
    battle_scenario,
    battle_step,
    make_company,
    make_unit,
    run_battle,
    simulate,
)
from mrn_coord.battle_formations import (
    FORMATIONS,
    build_team_spec,
    formation_mode_for_counts,
    formation_commands,
)


class TestFormationBuilders(unittest.TestCase):
    def test_auto_mode_picks_wedge_when_outnumbering(self):
        self.assertEqual(formation_mode_for_counts(16, 5), "wedge")

    def test_auto_mode_picks_square_when_outnumbered(self):
        self.assertEqual(formation_mode_for_counts(5, 12), "square")

    def test_tanks_lead_wedge(self):
        spec = build_team_spec([10, 11, 12],
                                 ["sniper", "tank", "soldier"], "wedge", 2.0)
        tank = spec.offsets[11]
        sniper = spec.offsets[10]
        self.assertGreater(tank[0], sniper[0])

    def test_all_modes_build(self):
        for mode in FORMATIONS:
            if mode in ("none", "auto"):
                continue
            spec = build_team_spec([0, 1, 2, 3], ["", "", "", ""], mode, 1.5)
            self.assertEqual(len(spec.offsets), 4)


class TestBattleIntegration(unittest.TestCase):
    def test_formation_none_matches_default(self):
        a = run_battle(10, BattleConfig(formation="none"), seed=1)
        b = run_battle(10, BattleConfig(), seed=1)
        self.assertEqual(a.winner, b.winner)
        self.assertEqual(a.ticks, b.ticks)

    def test_wedge_battles_resolve(self):
        for seed in range(4):
            cfg = BattleConfig(formation="wedge", tactics="count_aware")
            res = run_battle(12, cfg, seed=seed)
            resolved = (res.winner is not None or
                        all(n == 0 for n in res.survivors.values()))
            self.assertTrue(resolved, f"seed {seed} stalemate")

    def test_per_team_formations(self):
        cfg = BattleConfig(formation="none",
                           formation_by_team={RED: "wedge", BLUE: "line"})
        res = run_battle(10, cfg, seed=3)
        self.assertIsNotNone(res.winner)

    def test_combined_arms_company_with_auto_formation(self):
        import random
        cfg = BattleConfig(formation="auto", tactics="count_aware")
        rng = random.Random(0)
        red = make_company(cfg, RED, (5.0, 12.0),
                           [("tank", 3), ("sniper", 2)], rng)
        blue = make_company(cfg, BLUE, (35.0, 12.0),
                            [("scout", 8)], rng)
        res = simulate(red + blue, cfg, max_ticks=900)
        self.assertIsNotNone(res.winner)

    def test_formation_commands_push_toward_shape(self):
        cfg = BattleConfig(formation_spacing=2.0, formation_gain=1.0)
        bots = [make_unit(0.0, 0.0, RED, "tank"),
                make_unit(8.0, 0.0, RED, "soldier"),
                Bot(30.0, 0.0, 0, 0, BLUE, 100, 100)]
        live = bots
        cmds = formation_commands(bots, live, [0, 1], "line",
                                  spacing=cfg.formation_spacing,
                                  gain=cfg.formation_gain)
        self.assertLess(cmds[1][0], 0.0)

    def test_scenarios_still_decisive(self):
        for name in ("duel", "chokepoint"):
            bots, cfg, _ = battle_scenario(name)
            res = simulate(bots, cfg, max_ticks=900)
            self.assertIsNotNone(res.winner, name)


if __name__ == "__main__":
    unittest.main()
