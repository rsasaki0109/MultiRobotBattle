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


class TestBattleBridge(unittest.TestCase):
    def test_all_scenarios_return_frames(self):
        for name in battle_bridge.SCENARIOS:
            raw = battle_bridge.run(name)
            data = json.loads(raw)
            self.assertTrue(data["ok"], data.get("error"))
            self.assertGreater(len(data["frames"]), 10)
            self.assertEqual(len(data["frames"]), len(data["shots"]))
            self.assertIn(data["winner"], ("red", "blue", None))

    def test_unknown_scenario(self):
        data = json.loads(battle_bridge.run("nope"))
        self.assertFalse(data["ok"])


if __name__ == "__main__":
    unittest.main()
