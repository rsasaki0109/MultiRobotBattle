"""ORCA / BVC charge — collision-free breakthrough on chokepoint."""

import unittest

from mrn_coord.battle import (
    RED,
    BLUE,
    CHARGE_HEADLINE_MODES,
    battle_scenario,
    charge_headline_duel,
    simulate,
)


class TestCharge(unittest.TestCase):
    def test_charge_headline_modes_resolve(self):
        for mode in ("orca", "bvc"):
            bots, cfg, title = charge_headline_duel(mode, seed=4, n=8)
            self.assertEqual(cfg.charge_by_team.get(RED), mode)
            self.assertEqual(cfg.charge_by_team.get(BLUE), "none")
            res = simulate(bots, cfg, max_ticks=650, frame_stride=4)
            self.assertGreater(res.ticks, 0)
            self.assertIn("greedy", title.lower())

    def test_orca_charge_duel_scenario(self):
        bots, cfg, title = battle_scenario("orca_charge_duel")
        self.assertEqual(cfg.charge_by_team[RED], "orca")
        res = simulate(bots, cfg, max_ticks=700)
        self.assertIsNotNone(res.winner, "orca_charge_duel stalled")
        self.assertIn("ORCA", title)

    def test_charge_modes_are_known(self):
        self.assertIn("orca", CHARGE_HEADLINE_MODES)
        self.assertIn("bvc", CHARGE_HEADLINE_MODES)


if __name__ == "__main__":
    unittest.main()
