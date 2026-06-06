"""Browser-demo bridge: build a small MAPF instance, run a chosen solver, and
return a JSON-serialisable result for the JavaScript animator.

This module is pure Python and ROS-free; it runs identically under CPython and
under Pyodide (the in-browser interpreter the demo uses). It only touches the
``mapf-zoo`` public API, so it never needs numpy/scipy (the LP-based ``bcp``
solver is intentionally left out of the demo).
"""

from __future__ import annotations

import json

from mrn_coord.mapf import (
    GridWorld,
    Solution,
    cbs,
    ecbs,
    lacam,
    mapf_lns,
    mstar,
    pbs,
    prioritized_planning,
    sum_of_costs,
)

# --- instance library -------------------------------------------------------


def _crossing():
    grid = GridWorld(7, 7)
    agents = {
        "1": ((0, 3), (6, 3)),
        "2": ((3, 0), (3, 6)),
        "3": ((0, 0), (6, 6)),
        "4": ((6, 0), (0, 6)),
    }
    return grid, agents


def _swap():
    # Three agents reorder through a 5-wide corridor with one passing row.
    grid = GridWorld(5, 2)
    agents = {
        "1": ((0, 0), (4, 0)),
        "2": ((4, 0), (0, 0)),
        "3": ((2, 1), (2, 0)),
    }
    return grid, agents


def _doorway():
    # A wall down the middle (x=3) with a single gap at y=2; two streams must
    # funnel through the one-cell doorway in opposite directions.
    blocked = {(3, y) for y in range(5) if y != 2}
    grid = GridWorld(7, 5, blocked=blocked)
    agents = {
        "1": ((0, 1), (6, 1)),
        "2": ((0, 3), (6, 3)),
        "3": ((6, 2), (0, 2)),
    }
    return grid, agents


def _ring():
    # Eight agents on the border of an 8x8 with a solid 2x2 core, each heading
    # to the opposite corner — lots of contention, no head-on monopoly.
    blocked = {(x, y) for x in (3, 4) for y in (3, 4)}
    grid = GridWorld(8, 8, blocked=blocked)
    corners = [(0, 0), (7, 0), (7, 7), (0, 7), (0, 3), (7, 4), (3, 0), (4, 7)]
    agents = {}
    for i, c in enumerate(corners):
        opp = (7 - c[0], 7 - c[1])
        agents[str(i + 1)] = (c, opp)
    return grid, agents


PRESETS = {
    "crossing": _crossing,
    "swap": _swap,
    "doorway": _doorway,
    "ring": _ring,
}

# --- solver registry --------------------------------------------------------


def _ecbs(grid, agents):
    return ecbs(grid, agents, w=1.5)


def _mstar(grid, agents):
    sol = mstar(grid, agents, max_expansions=30_000)
    if sol is None:
        raise RuntimeError(
            "M* hit the demo expansion budget on this instance."
        )
    return sol


# Combos that are pathological for a given solver and would hang the tab — we
# short-circuit them with an explanation instead of running. (M* couples the
# whole contended team on the ring; one joint expansion enumerates 5**8 moves.)
SKIP = {
    ("ring", "mstar"): (
        "M* couples every agent that collides into one joint search. On this "
        "contended ring the whole team couples, so a single expansion would "
        "enumerate 5**8 joint moves — intractable. This is M*'s honest failure "
        "mode; try CBS, PBS, or LaCAM here."
    ),
}


SOLVERS = {
    "cbs": cbs,
    "ecbs": _ecbs,
    "lacam": lacam,
    "pbs": pbs,
    "mstar": _mstar,
    "prioritized": prioritized_planning,
    "lns": mapf_lns,
}


def _normalise(result):
    """Coerce any solver's return into ``(paths, cost)`` or ``None``."""
    if result is None:
        return None
    if isinstance(result, Solution):
        return result.paths, result.cost
    if isinstance(result, dict):
        return result, sum_of_costs(result)
    # A few solvers return ``(paths, ...)`` tuples; take the first dict.
    if isinstance(result, tuple) and result and isinstance(result[0], dict):
        return result[0], sum_of_costs(result[0])
    raise TypeError("unrecognised solver return: %r" % type(result))


def solve(preset_name: str, solver_name: str) -> str:
    """Run ``solver_name`` on ``preset_name`` and return a JSON string.

    The shape is ``{ok, width, height, blocked, agents, paths, cost, makespan}``
    on success, or ``{ok: false, error}`` on failure / no solution.
    """
    try:
        if (preset_name, solver_name) in SKIP:
            return json.dumps({"ok": False, "error": SKIP[(preset_name, solver_name)]})
        grid, agents = PRESETS[preset_name]()
        solver = SOLVERS[solver_name]
        norm = _normalise(solver(grid, agents))
        if norm is None:
            return json.dumps({"ok": False, "error": "no solution found"})
        paths, cost = norm
        horizon = max(len(p) for p in paths.values())
        # pad every path to the horizon so the animator can index by timestep
        padded = {k: list(p) + [p[-1]] * (horizon - len(p)) for k, p in paths.items()}
        return json.dumps(
            {
                "ok": True,
                "width": grid.width,
                "height": grid.height,
                "blocked": sorted(list(c) for c in grid.blocked),
                "agents": {k: [list(s), list(g)] for k, (s, g) in agents.items()},
                "paths": {k: [list(c) for c in p] for k, p in padded.items()},
                "cost": cost,
                "makespan": horizon - 1,
                "solver": solver_name,
                "preset": preset_name,
            }
        )
    except Exception as exc:  # surface errors to the page rather than hanging
        return json.dumps({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)})
