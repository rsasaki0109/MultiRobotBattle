"""Tests for Hungarian target assignment in battle."""

import math
import unittest

from mrn_coord.battle import (
    BLUE,
    RED,
    BattleConfig,
    Bot,
    battle_scenario,
    run_battle,
    simulate,
)
from mrn_coord.battle_assignment import (
    apply_assignments,
    hungarian_assignments,
    team_assignments,
)
from mrn_coord.battle_policy.count_aware import CountAwarePolicy, TacticalDecision


class TestHungarianAssignment(unittest.TestCase):
    def test_focus_fire_splits_targets(self):
        cfg = BattleConfig(assignment="hungarian")
        live = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(0.0, 2.0, 0, 0, RED, 100, 100),
                Bot(8.0, 0.0, 0, 0, BLUE, 30, 100),
                Bot(8.0, 4.0, 0, 0, BLUE, 100, 100)]
        assigns = hungarian_assignments(live, cfg)
        red = {i: t for i, t in assigns.items() if live[i].team == RED}
        wounded = 2
        self.assertIn(wounded, red.values())
        self.assertEqual(len(set(red.values())), 2)

    def test_apply_overrides_policy_target(self):
        cfg = BattleConfig(assignment="hungarian")
        live = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(10.0, 0.0, 0, 0, BLUE, 100, 100)]
        decisions = [TacticalDecision(1), TacticalDecision(0)]
        out = apply_assignments(decisions, live, cfg)
        self.assertEqual(out[0].target_index, 1)
        self.assertEqual(out[1].target_index, 0)

    def test_hungarian_battle_resolves(self):
        cfg = BattleConfig(assignment="hungarian", tactics="count_aware")
        res = run_battle(8, cfg, seed=2, max_ticks=600)
        resolved = (res.winner is not None or
                    all(n == 0 for n in res.survivors.values()))
        self.assertTrue(resolved)

    def test_maneuver_duel_scenario(self):
        bots, cfg, title = battle_scenario("maneuver_duel")
        self.assertIn("Maneuver", title)
        res = simulate(bots, cfg, max_ticks=700)
        self.assertIsNotNone(res.winner)

    def test_mapf_stack_duel_scenario(self):
        bots, cfg, title = battle_scenario("mapf_stack_duel")
        self.assertIn("MAPF stack", title)
        self.assertEqual(cfg.assignment_by_team.get(RED), "cbs_ta")
        res = simulate(bots, cfg, max_ticks=700)
        self.assertIsNotNone(res.winner)

    def test_cbs_ta_assigns_on_chokepoint(self):
        bots, cfg, _ = battle_scenario("chokepoint")
        cfg = BattleConfig(
            obstacles=cfg.obstacles,
            tactics="count_aware",
            assignment="cbs_ta",
            formation="wedge",
        )
        res = simulate(bots, cfg, max_ticks=700)
        self.assertIsNotNone(res.winner)

    def test_cbs_ta_team_assignments_resolves(self):
        cfg = BattleConfig(
            assignment="none",
            assignment_by_team={RED: "cbs_ta"},
            tactics="count_aware",
            obstacles=((20.0, 12.0, 2.6),),
        )
        live = [
            Bot(2.0, 12.0, 0, 0, RED, 100, 100),
            Bot(2.0, 14.0, 0, 0, RED, 100, 100),
            Bot(38.0, 12.0, 0, 0, BLUE, 100, 100),
            Bot(38.0, 14.0, 0, 0, BLUE, 100, 100),
        ]
        assigns = team_assignments(live, cfg)
        red = {i: t for i, t in assigns.items() if live[i].team == RED}
        self.assertEqual(len(red), 2)
        for ai, ej in red.items():
            self.assertEqual(live[ej].team, BLUE)


if __name__ == "__main__":
    unittest.main()
