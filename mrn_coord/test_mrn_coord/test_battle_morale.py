"""Morale / rout — collapsing teams flee instead of stalling."""

import unittest

from mrn_coord.battle import (
    RED,
    BLUE,
    BattleConfig,
    battle_scenario,
    simulate,
)
from mrn_coord.battle_morale import (
    apply_rout_steering,
    init_morale_state,
    team_is_routing,
    team_strength_frac,
)


class TestMorale(unittest.TestCase):
    def test_rout_triggers_below_threshold(self):
        from mrn_coord.battle import Bot, make_company
        import random

        cfg = BattleConfig(morale=True, morale_rout_frac=0.38, width=40.0, height=24.0)
        bots = make_company(cfg, BLUE, (30.0, 12.0), [("soldier", 10)], random.Random(0))
        state = init_morale_state(bots, [BLUE])
        for b in bots[3:]:
            b.alive = False
            b.hp = 0.0
        self.assertLessEqual(team_strength_frac(bots, BLUE, state), 0.38)
        self.assertTrue(team_is_routing(bots, BLUE, cfg, state))

    def test_routed_bots_steer_off_field(self):
        from mrn_coord.battle import Bot

        cfg = BattleConfig(morale=True, morale_rout_frac=0.5, width=40.0, height=24.0,
                           w_retreat=2.0, morale_rout_speed=1.2)
        bots = [Bot(20.0, 12.0, 0, 0, BLUE, 100, 100)]
        state = init_morale_state(bots, [BLUE])
        state.start_counts[BLUE] = 2
        bots.append(Bot(21.0, 12.0, 0, 0, BLUE, 0, 0))
        bots[1].alive = False
        live = [bots[0]]
        desired = [[0.0, 0.0]]
        apply_rout_steering(bots, live, desired, cfg, state)
        self.assertGreater(desired[0][0], 0.0)

    def test_morale_duel_resolves(self):
        bots, cfg, title = battle_scenario("morale_duel")
        self.assertTrue(cfg.morale)
        res = simulate(bots, cfg, max_ticks=900)
        self.assertIsNotNone(res.winner, "morale_duel stalled")
        self.assertTrue(res.morale_progress)
        self.assertIn("rout", title.lower())


if __name__ == "__main__":
    unittest.main()
