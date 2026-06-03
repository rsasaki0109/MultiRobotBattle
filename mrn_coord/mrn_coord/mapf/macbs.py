"""Meta-Agent Conflict-Based Search (MA-CBS).

A reproduction of **Meta-Agent CBS**, Sharon, Stern, Felner & Sturtevant,
*"Conflict-based search for optimal multi-agent pathfinding"* (AAAI 2012; AIJ
2015, §5). Plain :func:`mrn_coord.mapf.cbs.cbs` is *fully decoupled* — it plans
each agent alone and resolves every collision by branching one constraint at a
time. That is ideal when agents barely interact, but on a tight bottleneck two
agents can collide over and over, and CBS pays for it by re-splitting the same
conflict deep into the constraint tree (an exponential blow-up). A *fully
coupled* joint search (:func:`mrn_coord.mapf.mstar.joint_astar`) has the opposite
problem: it never blows up on conflicts but its state space is the product of all
agents'.

MA-CBS interpolates between the two with a single knob, the **conflict bound**
``B``:

- The high level is CBS, but its "agents" are **meta-agents** — groups of one or
  more original agents.
- A global counter ``CM[i][j]`` tallies how many times agents ``i`` and ``j``
  have conflicted across the whole search. When the conflicts between the two
  meta-agents owning a fresh conflict exceed ``B``, instead of splitting, the two
  meta-agents are **merged** into one.
- A meta-agent's path is found by a **coupled** low level (a time-expanded joint
  A* over the group's configuration space) that is internally collision-free and
  respects every *external* constraint the group has accumulated. Conflicts
  *between* meta-agents are still resolved by CBS branching.

``B = ∞`` never merges, so MA-CBS *is* standard CBS. ``B = 0`` merges on the
first conflict, collapsing toward a single coupled search. Every ``B`` yields the
**same optimal sum-of-costs** (Sharon et al., Thm.) — what changes is *where* the
work happens: a bottleneck that would explode the CBS tree is absorbed into one
coupled solve. This is the optimal-MAPF cousin of the group-merging in
:func:`mrn_coord.mapf.mstar.mstar` and Standley's independence detection, but it
merges by *conflict frequency* rather than by a single collision.

The constraints, conflicts and sum-of-costs accounting are exactly those of
:mod:`cbs` (negative vertex/edge constraints, stay-at-goal cost), so the returned
optimum is identical; a meta-agent constraint forbids the cell/edge to *every*
member of the group.
"""

from __future__ import annotations

import heapq
import itertools

from .conflicts import EdgeConflict, VertexConflict, detect_first_conflict
from .grid import GridWorld
from .mstar import _dist_to_goal
from .solution import Solution, sum_of_costs
from .space_time_astar import plan_path


# --------------------------------------------------------------------------- #
# Coupled low level: time-expanded joint A* for one meta-agent                 #
# --------------------------------------------------------------------------- #
def _plan_group(grid, members, agents, vcon, econ, dist_cache,
                max_states=200_000):
    """Optimal collision-free paths for a *group* of agents under the group's
    external constraints.

    ``members`` is an iterable of agent ids; ``vcon`` a set of ``(cell, time)``
    and ``econ`` a set of ``(frm, to, time)`` that **no member** may violate
    (the meta-agent's accumulated constraints). Returns ``{id: path}`` minimizing
    the group's stay-at-goal sum-of-costs (matching :mod:`cbs`), or ``None``.

    A singleton group is just :func:`plan_path`; a multi-agent group is a joint
    A* over ``(config, time)`` whose successors are internally collision-free and
    honour the external constraints, with the same goal-rest edge cost as
    :func:`mrn_coord.mapf.mstar.joint_astar`."""
    ids = sorted(members, key=str)
    if len(ids) == 1:
        start, goal = agents[ids[0]]
        path = plan_path(grid, start, goal, frozenset(vcon), frozenset(econ))
        return None if path is None else {ids[0]: path}

    n = len(ids)
    starts = tuple(agents[m][0] for m in ids)
    goals = tuple(agents[m][1] for m in ids)
    dist = [dist_cache[g] for g in goals]
    for k in range(n):
        if dist[k].get(starts[k]) is None:
            return None
        if (starts[k], 0) in vcon:
            return None

    # Per-member latest time its goal cell is forbidden; a member may *settle*
    # (freeze at its goal for good, at zero further cost) only past that time.
    last_goal_time = [0] * n
    for cell, t in vcon:
        for k in range(n):
            if cell == goals[k]:
                last_goal_time[k] = max(last_goal_time[k], t)

    max_time = 2 * grid.width * grid.height + len(vcon) + len(econ) \
        + grid.width + grid.height + 5
    full = frozenset(range(n))

    def h(cfg, settled):
        return sum(dist[k][cfg[k]] for k in range(n) if k not in settled)

    # State = (config, settled, t). The settled set freezes each member at its
    # goal once it has arrived for good, so the accumulated cost (1 per member per
    # step until it settles) is exactly the stay-at-goal sum-of-costs of
    # :mod:`cbs` -- even when a member must vacate its goal and return.
    INF = float("inf")
    start_state = (starts, frozenset(), 0)
    g = {start_state: 0}
    parent: dict = {}
    counter = itertools.count()
    open_heap = [(h(starts, frozenset()), next(counter), start_state)]
    closed: set = set()
    states = 0
    while open_heap:
        _, _, state = heapq.heappop(open_heap)
        if state in closed:
            continue
        closed.add(state)
        states += 1
        if states > max_states:
            return None
        cfg, settled, t = state
        if settled == full:
            return _reconstruct_group(ids, goals, parent, state)
        if t >= max_time:
            continue
        nt = t + 1

        # Per-member options: (next_cell, newly_settled, step_cost).
        per_agent = []
        for i in range(n):
            if i in settled:
                per_agent.append([(goals[i], True, 0)])
                continue
            opts = []
            on_goal = cfg[i] == goals[i]
            for m in grid.neighbors(cfg[i]):
                if (m, nt) in vcon or (cfg[i], m, nt) in econ:
                    continue
                # Settle only by *staying* on a goal already reached (and past the
                # last time that goal is forbidden) -- never for free on arrival.
                if on_goal and m == goals[i] and nt >= last_goal_time[i]:
                    opts.append((m, True, 0))          # settle for good (free)
                opts.append((m, False, 1))             # move/wait, still active
            if not opts:
                per_agent = None
                break
            per_agent.append(opts)
        if per_agent is None:
            continue

        for combo in itertools.product(*per_agent):
            v = tuple(c[0] for c in combo)
            if len(set(v)) != n:                       # internal vertex collision
                continue
            swap = False
            for i in range(n):
                for j in range(i + 1, n):
                    if cfg[i] == v[j] and cfg[j] == v[i] and cfg[i] != cfg[j]:
                        swap = True
                        break
                if swap:
                    break
            if swap:
                continue
            newly = frozenset(i for i, c in enumerate(combo) if c[1])
            nsettled = settled | newly
            nstate = (v, nsettled, nt)
            ng = g[state] + sum(c[2] for c in combo)
            if ng < g.get(nstate, INF):
                g[nstate] = ng
                parent[nstate] = state
                heapq.heappush(open_heap,
                               (ng + h(v, nsettled), next(counter), nstate))
    return None


def _reconstruct_group(ids, goals, parent, goal_state):
    states = [goal_state]
    cur = goal_state
    while cur in parent:
        cur = parent[cur]
        states.append(cur)
    states.reverse()
    paths = {}
    for idx, a in enumerate(ids):
        seq = [cfg[idx] for (cfg, _settled, _t) in states]
        gc = goals[idx]
        while len(seq) > 1 and seq[-1] == gc and seq[-2] == gc:
            seq.pop()
        paths[a] = seq
    return paths


# --------------------------------------------------------------------------- #
# High level: CBS over meta-agents                                            #
# --------------------------------------------------------------------------- #
def macbs(grid: GridWorld, agents: dict, *, merge_bound: int = 1,
          max_expansions: int = 100_000, stats: dict | None = None):
    """Solve a MAPF instance optimally (sum-of-costs) with Meta-Agent CBS.

    ``agents`` maps an agent id to a ``(start, goal)``. ``merge_bound`` is the
    conflict bound ``B``: when two meta-agents have conflicted more than ``B``
    times they are merged and solved by a coupled low level. ``B`` large (e.g.
    ``10**9``) reproduces standard CBS; ``B = 0`` merges on first conflict. Any
    ``B`` returns the same optimum. Returns a :class:`Solution` or ``None``.

    If ``stats`` is given it records ``expansions`` (high-level nodes),
    ``merges`` (merge operations applied to the *solution* node's lineage is not
    tracked; this is the global count of merges performed), ``num_groups`` and
    ``max_group_size`` of the returned solution."""
    dist_cache: dict = {}
    for _, goal in agents.values():
        if goal not in dist_cache:
            dist_cache[goal] = _dist_to_goal(grid, goal)

    # Root: every agent is its own meta-agent, no constraints.
    root_groups = frozenset(frozenset({a}) for a in agents)
    vcon = {grp: frozenset() for grp in root_groups}
    econ = {grp: frozenset() for grp in root_groups}
    paths: dict = {}
    for grp in root_groups:
        sol = _plan_group(grid, grp, agents, frozenset(), frozenset(),
                          dist_cache)
        if sol is None:
            return None
        paths.update(sol)

    cmat: dict = {}                                   # CM[frozenset({i, j})] -> count
    merges = 0

    def group_conflicts(gi, gj):
        return sum(cmat.get(frozenset((a, b)), 0) for a in gi for b in gj)

    counter = itertools.count()
    open_heap = [(sum_of_costs(paths), next(counter),
                  root_groups, vcon, econ, paths)]
    expansions = 0
    while open_heap:
        cost, _, groups, vcon, econ, paths = heapq.heappop(open_heap)
        expansions += 1
        if expansions > max_expansions:
            break

        conflict = detect_first_conflict(paths)
        if conflict is None:
            if stats is not None:
                stats["expansions"] = expansions
                stats["merges"] = merges
                stats["num_groups"] = len(groups)
                stats["max_group_size"] = max(len(g) for g in groups)
            return Solution(paths=dict(paths), cost=cost)

        a, b = conflict.agent_a, conflict.agent_b
        cmat[frozenset((a, b))] = cmat.get(frozenset((a, b)), 0) + 1
        # Grouping is per-node (a branch that merged differs from one that did
        # not); only the conflict counter CM is global and monotone.
        group_of = {ag: grp for grp in groups for ag in grp}
        gi, gj = group_of[a], group_of[b]

        if group_conflicts(gi, gj) > merge_bound:
            merges += 1
            child = _merge_child(grid, agents, groups, vcon, econ, paths,
                                 gi, gj, dist_cache)
            if child is not None:
                _, _, _, c_paths = child
                heapq.heappush(open_heap, (sum_of_costs(c_paths), next(counter),
                                           *child))
            continue

        for child in _split_children(grid, agents, groups, vcon, econ, paths,
                                     conflict, gi, gj, dist_cache):
            _, _, _, c_paths = child
            heapq.heappush(open_heap, (sum_of_costs(c_paths), next(counter),
                                       *child))

    if stats is not None:
        stats["expansions"] = expansions
        stats["merges"] = merges
    return None


def _strip_v(vset):
    """Drop the opponent tag: ``(cell, t, opp)`` -> ``(cell, t)`` for the low level."""
    return frozenset((cell, t) for (cell, t, _opp) in vset)


def _strip_e(eset):
    return frozenset((frm, to, t) for (frm, to, t, _opp) in eset)


def _merge_child(grid, agents, groups, vcon, econ, paths, gi, gj, dist_cache):
    """Replace meta-agents ``gi`` and ``gj`` by their union, planned coupled under
    the combined *external* constraints. Constraints whose opponent is *inside* the
    merged group are **internal** — they came from a conflict between the two
    agents now being merged — and are dropped, because the coupled low level
    resolves internal collisions directly; keeping them would over-constrain the
    meta-agent and lose the optimum. Returns a child node or ``None``."""
    merged = gi | gj
    new_groups = (set(groups) - {gi, gj}) | {merged}
    m_v = frozenset(c for c in (vcon[gi] | vcon[gj]) if c[2] not in merged)
    m_e = frozenset(c for c in (econ[gi] | econ[gj]) if c[3] not in merged)
    sol = _plan_group(grid, merged, agents, _strip_v(m_v), _strip_e(m_e),
                      dist_cache)
    if sol is None:
        return None
    c_vcon = {g: vcon[g] for g in groups if g not in (gi, gj)}
    c_econ = {g: econ[g] for g in groups if g not in (gi, gj)}
    c_vcon[merged] = m_v
    c_econ[merged] = m_e
    c_paths = dict(paths)
    c_paths.update(sol)
    return (frozenset(new_groups), c_vcon, c_econ, c_paths)


def _split_children(grid, agents, groups, vcon, econ, paths, conflict, gi, gj,
                    dist_cache):
    """The CBS two-way split, but each constraint is laid on a *meta-agent*: one
    child forbids the cell/edge to all of ``gi``, the other to all of ``gj``. Each
    constraint is tagged with the *opponent* original agent so a later merge can
    tell internal constraints from external ones."""
    a, b = conflict.agent_a, conflict.agent_b      # a in gi, b in gj
    if isinstance(conflict, VertexConflict):
        branches = [
            (gi, "v", (conflict.cell, conflict.time, b)),
            (gj, "v", (conflict.cell, conflict.time, a)),
        ]
    else:  # EdgeConflict
        branches = [
            (gi, "e", (conflict.cell_a, conflict.cell_b, conflict.time, b)),
            (gj, "e", (conflict.cell_b, conflict.cell_a, conflict.time, a)),
        ]

    children = []
    for grp, kind, constraint in branches:
        c_vcon = dict(vcon)
        c_econ = dict(econ)
        if kind == "v":
            c_vcon[grp] = c_vcon[grp] | {constraint}
        else:
            c_econ[grp] = c_econ[grp] | {constraint}
        sol = _plan_group(grid, grp, agents, _strip_v(c_vcon[grp]),
                          _strip_e(c_econ[grp]), dist_cache)
        if sol is None:
            continue
        c_paths = dict(paths)
        c_paths.update(sol)
        children.append((groups, c_vcon, c_econ, c_paths))
    return children
