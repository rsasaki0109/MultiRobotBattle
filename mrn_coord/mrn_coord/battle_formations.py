"""Battle formations — shape builders oriented toward the enemy.

Wraps :mod:`mrn_coord.formation` displacement consensus for swarm combat: each
team can hold a *line*, *wedge*, *screen*, or *square* while still pursuing and
firing. Shapes are built in a local frame (+x toward the enemy centroid) and
assigned front-to-back by unit class (tanks lead, snipers trail).
"""

from __future__ import annotations

import math

from .formation import (
    FormationSpec,
    formation_control_from_relative,
    relative_measurements,
)

FORMATIONS = ("none", "auto", "line", "wedge", "screen", "square")

_KIND_RANK = {"tank": 0, "soldier": 1, "scout": 2, "sniper": 3, "": 1}


def _rotate(x, y, fx, fy):
    """Rotate ``(x, y)`` so formation +x aligns with facing ``(fx, fy)``."""
    return (x * fx - y * fy, x * fy + y * fx)


def _slot_positions(mode, n, spacing):
    """Unoriented slot offsets (+x forward); front slots first in the list."""
    if n <= 0:
        return []
    if mode == "line":
        return [(k * spacing, 0.0) for k in range(n)]
    if mode == "screen":
        return [(0.0, (k - (n - 1) / 2.0) * spacing) for k in range(n)]
    if mode == "square":
        cols = max(1, int(math.ceil(math.sqrt(n))))
        slots = []
        for k in range(n):
            row, col = divmod(k, cols)
            slots.append((-row * spacing, (col - (cols - 1) / 2.0) * spacing))
        return slots
    if mode == "wedge":
        slots = []
        row = col = 0
        row_width = 1
        while len(slots) < n:
            for c in range(row_width):
                if len(slots) >= n:
                    break
                slots.append((-row * spacing, (c - (row_width - 1) / 2.0) * spacing))
            row += 1
            row_width += 1
        return slots
    raise ValueError(f"unknown formation mode {mode!r}")


def formation_mode_for_counts(n_allies, n_enemies):
    """Pick a formation from force ratio (used when ``formation='auto'``)."""
    ratio = n_allies / max(n_enemies, 1)
    if ratio >= 1.3:
        return "wedge"
    if ratio <= 0.75:
        return "square"
    return "screen"


def build_team_spec(agent_ids, kinds, mode, spacing):
    """Map stable ``agent_ids`` to oriented-formation offsets (sorted by class)."""
    order = sorted(range(len(agent_ids)),
                   key=lambda k: (_KIND_RANK.get(kinds[k], 1), agent_ids[k]))
    slots = _slot_positions(mode, len(agent_ids), spacing)
    offsets = {}
    for slot, k in zip(slots, order):
        offsets[agent_ids[k]] = slot
    return FormationSpec(offsets)


def team_facing(live, team_indices):
    """Unit vector from the team centroid toward enemy centroid."""
    cx = sum(live[i].x for i in team_indices) / len(team_indices)
    cy = sum(live[i].y for i in team_indices) / len(team_indices)
    ex = ey = 0.0
    my_team = live[team_indices[0]].team
    cnt = 0
    for j, b in enumerate(live):
        if b.team == my_team:
            continue
        ex += b.x - cx
        ey += b.y - cy
        cnt += 1
    if cnt == 0:
        return (1.0, 0.0)
    norm = math.hypot(ex, ey)
    return (ex / norm, ey / norm)


def complete_edges(agent_ids):
    edges = []
    for a in range(len(agent_ids)):
        for b in range(a + 1, len(agent_ids)):
            edges.append((agent_ids[a], agent_ids[b]))
    return edges


def formation_commands(bots, live, team_indices, mode, *, spacing, gain):
    """Displacement-consensus velocity per live index for one team."""
    if mode in (None, "", "none") or len(team_indices) < 2:
        return {}

    agent_ids = [bots.index(live[i]) for i in team_indices]
    kinds = [live[i].kind for i in team_indices]
    n_allies = len(team_indices)
    n_enemies = sum(1 for b in live if b.team != live[team_indices[0]].team)
    if mode == "auto":
        mode = formation_mode_for_counts(n_allies, n_enemies)

    spec = build_team_spec(agent_ids, kinds, mode, spacing)
    fx, fy = team_facing(live, team_indices)
    oriented = FormationSpec({
        aid: _rotate(*off, fx, fy) for aid, off in spec.offsets.items()
    })

    positions = {aid: (live[team_indices[k]].x, live[team_indices[k]].y)
                 for k, aid in enumerate(agent_ids)}
    edges = complete_edges(agent_ids)
    meas = relative_measurements(positions, edges)
    cmds = formation_control_from_relative(meas, oriented, gain)

    out = {}
    for k, i in enumerate(team_indices):
        aid = agent_ids[k]
        out[i] = cmds.get(aid, (0.0, 0.0))
    return out
