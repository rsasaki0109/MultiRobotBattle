"""Headless smoke for the Pyodide battle demo bridge."""

from __future__ import annotations

import json
import os
import sys
import unittest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEMO = os.path.join(_REPO, "docs", "demo")
sys.path.insert(0, os.path.join(_REPO, "mrn_coord"))
sys.path.insert(0, _DEMO)

import battle_bridge  # noqa: E402


_WINNERS = frozenset(("red", "blue", "green", "yellow", None))
_ALLIANCES = frozenset(("western", "eastern", None))


class TestBattleBridge(unittest.TestCase):
    def _assert_outcome(self, data):
        if data.get("winning_alliance") is not None:
            self.assertIn(data["winning_alliance"], _ALLIANCES)
        else:
            self.assertIn(data.get("winner"), _WINNERS)

    def test_all_scenarios_return_frames(self):
        for name in battle_bridge.SCENARIOS:
            raw = battle_bridge.run(name)
            data = json.loads(raw)
            self.assertTrue(data["ok"], data.get("error"))
            if data.get("dual"):
                self.assertGreaterEqual(len(data["panels"]), 2)
                for panel in data["panels"]:
                    self.assertGreater(len(panel["frames"]), 10)
                    self.assertEqual(len(panel["frames"]), len(panel["shots"]))
                    self._assert_outcome(panel)
                continue
            self.assertGreater(len(data["frames"]), 10)
            self.assertEqual(len(data["frames"]), len(data["shots"]))
            self._assert_outcome(data)

    def test_unknown_scenario(self):
        data = json.loads(battle_bridge.run("nope"))
        self.assertFalse(data["ok"])


if __name__ == "__main__":
    unittest.main()
