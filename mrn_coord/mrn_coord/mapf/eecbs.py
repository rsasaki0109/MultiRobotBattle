"""EECBS: a bounded-suboptimal CBS driven by an admissible heuristic and EES.

Li, Ruml & Koenig, *"EECBS: A Bounded-Suboptimal Search for Multi-Agent Path
Finding"* (AAAI 2021). EECBS combines the two ideas already in this package:

- the **focal, conflict-avoiding low level** of :func:`mrn_coord.mapf.ecbs.ecbs`
  — a per-agent A* whose bounded-suboptimal paths already dodge the other
  agents, so the constraint tree starts near a solution; and
- the **admissible WDG heuristic** of :func:`mrn_coord.mapf.cbsh.cbsh` — a tight
  lower bound on the extra cost the conflicts *force*.

Plain ECBS has only the first. Its high-level lower bound is the sum of the
agents' individual optima, which ignores conflict-forced cost, so it over-
expands the tree relative to what the suboptimality factor ``w`` actually
permits. EECBS bolts CBSH's lower bound onto ECBS and drives the high level with
**Explicit Estimation Search** (EES; Thayer & Ruml 2011): it only has to certify
``cost <= w * (global lower bound)``, and a tighter lower bound reaches that
certificate sooner — fewer high-level expansions at the same ``w``.

Each constraint-tree node carries two path sets:

- ``focal_paths`` — the bounded-suboptimal, conflict-avoiding paths. Their cost
  ``g`` is the candidate solution cost; conflicts are detected and split here.
- ``opt_paths`` — the *optimal* paths under the node's constraints (plain A*).
  Their cost is the admissible lower bound ``LB``, and the WDG heuristic ``h`` —
  reused verbatim from :mod:`cbsh` — is computed on their MDDs. ``f = LB + h``.

EES keeps three views of the open nodes:

- **cleanup**, ordered by ``f`` — its minimum is the global lower bound that
  anchors the ``w`` guarantee;
- **open**, ordered by an inadmissible estimate ``f̂ = max(f, g + ε̄ * conflicts)``
  where ``ε̄`` is the running mean rise in ``f`` per expansion (learned online);
- **focal**, the nodes with ``f̂`` within ``w`` of the best, ordered by the
  conflict count ``d̂`` — closest to a conflict-free solution.

Clamping ``f̂ >= f`` keeps the inadmissible estimate from dropping below the
admissible bound, which is what makes ``w`` hold for a goal popped from focal.

**Honest scope.** EECBS also adds cardinal-conflict *prioritization* at the
high level; this solver branches the first conflict found, exactly like ECBS, so
the measured gain is attributable purely to *EES + the admissible bound*.
Prioritization is already reproduced and measured in :mod:`cbsh`.
"""

from __future__ import annotations

import itertools

from .cbsh import _heuristic as _admissible_h
from .conflicts import VertexConflict, detect_first_conflict
from .ecbs import _count_conflicts, _focal_low_level
from .grid import GridWorld
from .solution import Solution, sum_of_costs
from .space_time_astar import plan_path


def eecbs(
    grid: GridWorld,
    agents: dict,
    *,
    w: float = 1.5,
    heuristic: str | None = "wdg",
    max_expansions: int = 100_000,
    stats: dict | None = None,
):
    """Solve a MAPF instance bounded-suboptimally (cost ``<= w * optimal``).

    ``agents`` maps an agent id to ``(start, goal)``; ``w >= 1`` is the
    suboptimality factor. ``heuristic`` selects the admissible high-level
    heuristic: ``"cg"``, ``"dg"``, ``"wdg"`` (default), or ``None`` for the EES
    skeleton with ``h = 0`` (an ablation that isolates the heuristic's effect —
    this reduces EECBS to ECBS driven by EES rather than focal search). Returns a
    :class:`Solution` with ``cost <= w * optimal``, or ``None`` if infeasible /
    the expansion budget is exhausted. ``stats["expansions"]`` is set to the
    number of high-level nodes expanded — directly comparable to
    :func:`mrn_coord.mapf.ecbs.ecbs`.
    """
    agent_ids = list(agents)
    order = itertools.count()
    memo: dict = {}

    def admissible(vertex, edge, opt_paths):
        return _admissible_h(grid, agents, agent_ids, vertex, edge, opt_paths,
                            heuristic, memo)

    def make_node(vertex, edge, focal_paths, opt_paths):
        g = sum_of_costs(focal_paths)
        lb = sum_of_costs(opt_paths)
        h = admissible(vertex, edge, opt_paths)
        return {
            "vertex": vertex,
            "edge": edge,
            "focal": focal_paths,
            "opt": opt_paths,
            "g": g,
            "f": lb + h,
            "conf": _count_conflicts(focal_paths),
            "id": next(order),
        }

    # Root: focal low level for the candidate paths, plain A* for the bound.
    vertex = {a: frozenset() for a in agents}
    edge = {a: frozenset() for a in agents}
    focal_paths: dict = {}
    opt_paths: dict = {}
    for agent, (start, goal) in agents.items():
        reserved = list(focal_paths.values())
        fp, _ = _focal_low_level(grid, start, goal, vertex[agent], edge[agent],
                                 reserved, w)
        op = plan_path(grid, start, goal, vertex[agent], edge[agent])
        if fp is None or op is None:
            if stats is not None:
                stats["expansions"] = 0
            return None
        focal_paths[agent] = fp
        opt_paths[agent] = op
    live = [make_node(vertex, edge, focal_paths, opt_paths)]

    eps_sum = 0.0     # online single-step error: mean rise in f per expansion
    eps_n = 0
    expansions = 0

    def fhat(node):
        eps = eps_sum / eps_n if eps_n else 0.0
        return max(node["f"], node["g"] + eps * node["conf"])

    while live:
        best_f = min(live, key=lambda n: (n["f"], n["id"]))
        best_fhat = min(live, key=lambda n: (fhat(n), n["id"]))
        threshold = w * fhat(best_fhat)
        focal = [n for n in live if fhat(n) <= threshold]
        best_dhat = min(focal, key=lambda n: (n["conf"], fhat(n), n["id"]))

        bound = w * best_f["f"]
        if fhat(best_dhat) <= bound:
            node = best_dhat
        elif fhat(best_fhat) <= bound:
            node = best_fhat
        else:
            node = best_f
        live.remove(node)

        expansions += 1
        if expansions > max_expansions:
            if stats is not None:
                stats["expansions"] = expansions
            return None

        conflict = detect_first_conflict(node["focal"])
        if conflict is None:
            if stats is not None:
                stats["expansions"] = expansions
            return Solution(paths=dict(node["focal"]), cost=node["g"])

        children = []
        for agent, constraint in _branches(conflict):
            child_vertex = dict(node["vertex"])
            child_edge = dict(node["edge"])
            if constraint[0] == "v":
                _, cell, time = constraint
                child_vertex[agent] = child_vertex[agent] | {(cell, time)}
            else:
                _, frm, to, time = constraint
                child_edge[agent] = child_edge[agent] | {(frm, to, time)}

            start, goal = agents[agent]
            reserved = [node["focal"][o] for o in agents if o != agent]
            fp, _ = _focal_low_level(
                grid, start, goal, child_vertex[agent], child_edge[agent],
                reserved, w)
            op = plan_path(grid, start, goal, child_vertex[agent], child_edge[agent])
            if fp is None or op is None:
                continue
            child_focal = dict(node["focal"])
            child_focal[agent] = fp
            child_opt = dict(node["opt"])
            child_opt[agent] = op
            children.append(
                make_node(child_vertex, child_edge, child_focal, child_opt))

        live.extend(children)
        if children:
            delta = min(c["f"] for c in children) - node["f"]
            if delta >= 0:
                eps_sum += delta
                eps_n += 1

    if stats is not None:
        stats["expansions"] = expansions
    return None


def _branches(conflict):
    """The two constraints a conflict splits into (same vocabulary as CBS)."""
    if isinstance(conflict, VertexConflict):
        return [
            (conflict.agent_a, ("v", conflict.cell, conflict.time)),
            (conflict.agent_b, ("v", conflict.cell, conflict.time)),
        ]
    return [
        (conflict.agent_a, ("e", conflict.cell_a, conflict.cell_b, conflict.time)),
        (conflict.agent_b, ("e", conflict.cell_b, conflict.cell_a, conflict.time)),
    ]
