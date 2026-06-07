#!/usr/bin/env python3
"""Install a built wheel in a temp venv and smoke-test MAPF + battle imports."""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
import tempfile


def _run(cmd, **kw):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wheel", nargs="?", help="path to .whl (default: dist/*.whl)")
    args = ap.parse_args()
    wheels = [args.wheel] if args.wheel else sorted(glob.glob("dist/mapf_zoo-*.whl"))
    if not wheels:
        print("no wheel found under dist/", file=sys.stderr)
        return 1
    wheel = wheels[-1]

    with tempfile.TemporaryDirectory() as td:
        venv = f"{td}/venv"
        py = f"{venv}/bin/python"
        pip = f"{venv}/bin/pip"
        _run([sys.executable, "-m", "venv", venv])
        _run([pip, "install", "--upgrade", "pip"])
        _run([pip, "install", wheel])
        _run([py, "-c", """
from mrn_coord.mapf import GridWorld, cbs
from mrn_coord.battle import run_battle, BattleConfig

grid = GridWorld(5, 5)
agents = {"1": ((0, 2), (4, 2)), "2": ((2, 0), (2, 4))}
sol = cbs(grid, agents)
assert sol.cost == 9, sol.cost

res = run_battle(6, BattleConfig(), seed=1, max_ticks=200)
assert res.winner is not None or res.ticks > 0
print("wheel OK:", sol.cost, res.winner)
"""])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
