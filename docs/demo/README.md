# Browser demo — solve & animate MAPF in your browser

A zero-backend page that runs the **real** pure-Python MAPF solvers in the
browser via [Pyodide](https://pyodide.org/). Pick an instance and a solver and
watch the collision-free paths animate. No install, no server-side compute.

## Files

- `index.html` — MAPF zoo page (Pyodide loader + UI + canvas animator).
- `battle.html` — swarm battle page (same wheel, different bridge).
- `battle_robot_art.js` — RoboMaster-style chassis canvas renderer for battle.html.
- `bridge.py` — MAPF glue: builds a small instance, runs the chosen solver,
  returns JSON for the animator.
- `battle_bridge.py` — battle glue: runs a showcase scenario via
  ``mrn_coord.battle``, returns subsampled frames + laser lines as JSON.
- `mapf_zoo-0.1.0-py3-none-any.whl` — the packaged core (MAPF + battle stack),
  fetched and unpacked onto Pyodide's `sys.path` at load time.

## Run it locally

`fetch()` needs HTTP (not `file://`), so serve the directory:

```bash
cd docs/demo
python3 -m http.server 8000
# open http://localhost:8000/
```

On GitHub Pages it is live at
[rsasaki0109.github.io/multirobot-battle/demo/](https://rsasaki0109.github.io/multirobot-battle/demo/)
and […/demo/battle.html](https://rsasaki0109.github.io/multirobot-battle/demo/battle.html)
(deployed by ``.github/workflows/pages.yaml`` on push to ``main``).

## Refresh the wheel after changing the core

The wheel is a build artifact checked in so the page is self-contained. After
editing `mrn_coord/` (MAPF, battle, or packaging), rebuild from the repo root:

```bash
python3 -m build --wheel
cp dist/mapf_zoo-0.1.0-py3-none-any.whl docs/demo/
```

(If the version bumps, update the filename in `index.html`'s `WHEEL` constant.)

## How it's verified

The Python side is exercised headlessly under CPython:

- MAPF: `bridge.solve(preset, solver)` — 27 solving combos + the deliberately
  skipped `ring × M*` blow-up case.
- Battle: `mrn_coord/test_mrn_coord/test_demo_battle_bridge.py` runs every
  `battle_bridge.run(scenario)` combo.

Only the canvas rendering is browser-only.
