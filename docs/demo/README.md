# Browser demo — solve & animate MAPF in your browser

A zero-backend page that runs the **real** pure-Python MAPF solvers in the
browser via [Pyodide](https://pyodide.org/). Pick an instance and a solver and
watch the collision-free paths animate. No install, no server-side compute.

## Files

- `index.html` — the page (Pyodide loader + UI + canvas animator).
- `bridge.py` — pure-Python glue: builds a small instance, runs the chosen
  solver, returns JSON for the animator. Imports only the `mapf-zoo` public API
  (no numpy/scipy), so it runs identically under CPython and Pyodide.
- `mapf_zoo-0.1.0-py3-none-any.whl` — the packaged MAPF core, fetched and
  unpacked onto Pyodide's `sys.path` at load time.

## Run it locally

`fetch()` needs HTTP (not `file://`), so serve the directory:

```bash
cd docs/demo
python3 -m http.server 8000
# open http://localhost:8000/
```

On GitHub Pages (serve `docs/`) it works as-is at `/<repo>/demo/`.

## Refresh the wheel after changing the core

The wheel is a build artifact checked in so the page is self-contained. After
editing `mrn_coord/mapf` (or the packaging), rebuild it from the repo root:

```bash
python3 -m build --wheel
cp dist/mapf_zoo-0.1.0-py3-none-any.whl docs/demo/
```

(If the version bumps, update the filename in `index.html`'s `WHEEL` constant.)

## How it's verified

The Python side is exercised headlessly under the same Pyodide interpreter the
browser uses: a node harness unpacks the wheel, imports `mrn_coord.mapf`, and
runs the full `bridge.solve(preset, solver)` matrix (27 solving combos + the one
deliberately-skipped `ring × M*`, M*'s honest blow-up case). Only the canvas
rendering is browser-only.
