"""Discrete RRT (dRRT) for multi-robot motion planning in continuous space.

A pure-Python reproduction of Solovey, Salzman & Halperin's *"Finding a Needle
in an Exponential Haystack: Discrete RRT for Exploration of Implicit Roadmaps in
Multi-Robot Motion Planning"* (WAFR 2014; IJRR 2016).

Every other multi-agent planner in this zoo lives on a **discrete graph** — a
grid, an 8-connected roadmap, a hand-built adjacency. dRRT plans in **continuous
space**.  Each robot is a disc moving in a planar workspace cluttered with
circular obstacles; its free configurations are captured by a per-robot
**PRM roadmap** (random samples joined by obstacle-free straight segments).

The team's joint configuration space is the **tensor product** of the individual
roadmaps: a composite vertex is one roadmap vertex per robot, and two composite
vertices are adjacent iff every robot either stays put or traverses one of its
own roadmap edges.  This composite roadmap has ``prod_i |V_i|`` vertices —
exponential in the number of robots — and is never built explicitly (the
*haystack*).  dRRT explores it **implicitly** with an RRT-style tree:

- :func:`build_roadmap` — a PRM per robot: ``n_samples`` obstacle-free random
  points plus the robot's start and goal, joined to their ``k`` nearest
  neighbours by collision-free segments.
- :func:`direction_oracle` — the heart of dRRT (the operator ``O_d``).  Given a
  tree node and a random composite sample, it returns the *single* composite
  neighbour that best aligns with the direction to the sample: per robot, pick
  the roadmap edge whose heading maximises the cosine to that robot's desired
  direction (or *stay* if no edge points forward).  One implicit-neighbour
  lookup, no enumeration of the exponential neighbour set.
- :func:`drrt` — the search: sample a random point per robot (goal-biased),
  find the nearest tree node, expand it one composite step via the oracle, and
  add the new node iff the simultaneous straight-line motion of all robots is
  collision-free (disc/disc and disc/obstacle).  A greedy *connect-to-target*
  drives a fresh node straight at the goal.  Probabilistically complete.

Collision checking is exact and continuous: :func:`moving_min_distance` solves
the quadratic separation of two robots moving simultaneously along their
segments, and :func:`segment_point_distance` clears the swept disc of each
obstacle.  The same predicates back :func:`solution_clearance`, the independent
oracle the gate verifies every returned plan against.

Honest scope (see ``docs/coordination.md``): dRRT is a **feasibility / anytime**
planner, not a cost-optimal one — paths are collision-free and probabilistically
complete but not minimal.  The gate pins soundness (every plan clears ``2r`` and
the obstacles), the implicit-exploration signature (tree size vs the exponential
product), the oracle's value (aligned expansion solves where random-neighbour
expansion does not), and determinism — not optimality.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

__all__ = [
    "Obstacle",
    "Roadmap",
    "Solution",
    "StarSolution",
    "build_roadmap",
    "composite_optimum",
    "direction_oracle",
    "drrt",
    "drrt_star",
    "moving_min_distance",
    "segment_point_distance",
    "solution_clearance",
    "tensor_product_size",
]


# --------------------------------------------------------------------------- #
# geometry (pure scalars; no numpy)
# --------------------------------------------------------------------------- #
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def segment_point_distance(a0, a1, c):
    """Minimum distance from the segment ``a0->a1`` to the point ``c``."""
    dx, dy = a1[0] - a0[0], a1[1] - a0[1]
    dd = dx * dx + dy * dy
    if dd <= 1e-15:
        return _dist(a0, c)
    t = ((c[0] - a0[0]) * dx + (c[1] - a0[1]) * dy) / dd
    t = max(0.0, min(1.0, t))
    px, py = a0[0] + t * dx, a0[1] + t * dy
    return math.hypot(px - c[0], py - c[1])


def moving_min_distance(a0, a1, b0, b1):
    """Closest approach of two points moving simultaneously over ``t in [0,1]``.

    Point ``A`` travels ``a0->a1`` while ``B`` travels ``b0->b1`` (same clock).
    Returns ``min_t |A(t) - B(t)|`` — the exact disc-separation predicate.
    """
    r0 = _sub(a0, b0)
    d = (a1[0] - a0[0] - (b1[0] - b0[0]), a1[1] - a0[1] - (b1[1] - b0[1]))
    dd = d[0] * d[0] + d[1] * d[1]
    if dd <= 1e-15:
        return math.hypot(r0[0], r0[1])
    t = -(r0[0] * d[0] + r0[1] * d[1]) / dd
    t = max(0.0, min(1.0, t))
    rx, ry = r0[0] + t * d[0], r0[1] + t * d[1]
    return math.hypot(rx, ry)


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius: float


# --------------------------------------------------------------------------- #
# per-robot PRM roadmap
# --------------------------------------------------------------------------- #
@dataclass
class Roadmap:
    """A probabilistic roadmap for one robot: points + symmetric adjacency.

    Vertex ``0`` is always the start, vertex ``1`` always the goal.
    """

    points: list
    adj: list  # adj[i] = set of neighbour indices
    start: int = 0
    goal: int = 1

    def __len__(self):
        return len(self.points)


def _point_clear(p, obstacles, r, width, height):
    if not (0.0 <= p[0] <= width and 0.0 <= p[1] <= height):
        return False
    return all(_dist(p, (o.x, o.y)) >= o.radius + r for o in obstacles)


def _segment_clear(a, b, obstacles, r):
    return all(segment_point_distance(a, b, (o.x, o.y)) >= o.radius + r
               for o in obstacles)


def build_roadmap(start, goal, obstacles, r, width, height, *,
                  n_samples=25, k=10, rng=None, max_attempts=4000):
    """Build a PRM for a single disc robot of radius ``r``.

    ``n_samples`` obstacle-free random points (plus ``start``/``goal``) joined to
    their ``k`` nearest neighbours by collision-free straight segments.
    """
    rng = rng or random.Random(0)
    points = [tuple(start), tuple(goal)]
    attempts = 0
    while len(points) < n_samples + 2 and attempts < max_attempts:
        attempts += 1
        p = (rng.uniform(0.0, width), rng.uniform(0.0, height))
        if _point_clear(p, obstacles, r, width, height):
            points.append(p)
    n = len(points)
    adj = [set() for _ in range(n)]
    for i in range(n):
        order = sorted(range(n), key=lambda j: _dist(points[i], points[j]))
        added = 0
        for j in order:
            if j == i:
                continue
            if added >= k:
                break
            if _segment_clear(points[i], points[j], obstacles, r):
                adj[i].add(j)
                adj[j].add(i)
                added += 1
    return Roadmap(points, adj)


def tensor_product_size(roadmaps):
    """Vertex count of the implicit composite roadmap = ``prod_i |V_i|``."""
    size = 1
    for rm in roadmaps:
        size *= len(rm)
    return size


# --------------------------------------------------------------------------- #
# the direction oracle  O_d  (the heart of dRRT)
# --------------------------------------------------------------------------- #
def direction_oracle(roadmaps, node, target_points):
    """Return the composite neighbour of ``node`` best aligned with ``target``.

    For each robot ``i`` with current vertex ``node[i]`` and desired heading
    ``target_points[i] - point(node[i])``, pick the roadmap edge whose direction
    maximises the cosine with that heading; *stay* (keep ``node[i]``) if no edge
    points forward (cosine ``<= 0``) or the robot is already at the target.  One
    pass over each robot's local edge set — never the exponential product.
    """
    new = []
    for i, rm in enumerate(roadmaps):
        here = rm.points[node[i]]
        gx, gy = target_points[i][0] - here[0], target_points[i][1] - here[1]
        gnorm = math.hypot(gx, gy)
        if gnorm <= 1e-12:
            new.append(node[i])
            continue
        gx, gy = gx / gnorm, gy / gnorm
        best_idx = node[i]          # stay
        best_score = 0.0            # cosine threshold to move
        for j in rm.adj[node[i]]:
            ex, ey = rm.points[j][0] - here[0], rm.points[j][1] - here[1]
            enorm = math.hypot(ex, ey)
            if enorm <= 1e-12:
                continue
            score = (ex * gx + ey * gy) / enorm
            if score > best_score:
                best_score = score
                best_idx = j
        new.append(best_idx)
    return tuple(new)


def _random_neighbour(roadmaps, node, rng):
    """Baseline expansion: each robot takes a uniformly random edge (or stays)."""
    new = []
    for i, rm in enumerate(roadmaps):
        choices = list(rm.adj[node[i]]) + [node[i]]
        new.append(rng.choice(choices))
    return tuple(new)


# --------------------------------------------------------------------------- #
# collision checking of one composite edge
# --------------------------------------------------------------------------- #
def _edge_collision_free(roadmaps, node, new, obstacles, r):
    n = len(roadmaps)
    segs = []
    for i in range(n):
        a = roadmaps[i].points[node[i]]
        b = roadmaps[i].points[new[i]]
        segs.append((a, b))
    # disc/obstacle: swept disc of each moving robot clears every obstacle.
    for i in range(n):
        a, b = segs[i]
        if a == b:
            continue
        for o in obstacles:
            if segment_point_distance(a, b, (o.x, o.y)) < o.radius + r - 1e-9:
                return False
    # disc/disc: simultaneous motion keeps every pair >= 2r apart.
    twor = 2.0 * r
    for i in range(n):
        for j in range(i + 1, n):
            if moving_min_distance(segs[i][0], segs[i][1],
                                   segs[j][0], segs[j][1]) < twor - 1e-9:
                return False
    return True


# --------------------------------------------------------------------------- #
# solution + independent verifier
# --------------------------------------------------------------------------- #
@dataclass
class Solution:
    paths: dict           # robot -> list of (x, y) waypoints (synchronised)
    tree_size: int
    iterations: int
    makespan: float       # sum of per-edge max robot displacement
    total_length: float   # sum over robots of path length

    def waypoint_count(self):
        return len(next(iter(self.paths.values())))


def solution_clearance(paths, obstacles, r):
    """Independent oracle: min disc/disc and disc/obstacle clearance of a plan.

    Returns ``(min_pair_sep, min_obstacle_gap)`` over every synchronised edge of
    ``paths`` (robot -> waypoint list, all equal length).  The gate compares
    these against ``2r`` and ``0`` rather than trusting the planner's own check.
    """
    ids = list(paths)
    horizon = len(paths[ids[0]])
    min_pair = math.inf
    min_obs = math.inf
    for t in range(horizon - 1):
        for a in range(len(ids)):
            sa0, sa1 = paths[ids[a]][t], paths[ids[a]][t + 1]
            for o in obstacles:
                if sa0 != sa1:
                    min_obs = min(min_obs,
                                  segment_point_distance(sa0, sa1, (o.x, o.y))
                                  - o.radius - r)
            for b in range(a + 1, len(ids)):
                sb0, sb1 = paths[ids[b]][t], paths[ids[b]][t + 1]
                min_pair = min(min_pair,
                               moving_min_distance(sa0, sa1, sb0, sb1))
    return min_pair, min_obs


# --------------------------------------------------------------------------- #
# the dRRT search
# --------------------------------------------------------------------------- #
def _reconstruct(roadmaps, parents, goal_node):
    chain = [goal_node]
    while parents[chain[-1]] is not None:
        chain.append(parents[chain[-1]])
    chain.reverse()
    paths = {i: [roadmaps[i].points[node[i]] for node in chain]
             for i in range(len(roadmaps))}
    makespan = 0.0
    for a in range(len(chain) - 1):
        step = max(_dist(roadmaps[i].points[chain[a][i]],
                         roadmaps[i].points[chain[a + 1][i]])
                   for i in range(len(roadmaps)))
        makespan += step
    total = sum(sum(_dist(p[t], p[t + 1]) for t in range(len(p) - 1))
                for p in paths.values())
    return paths, makespan, total


def drrt(roadmaps, obstacles, r, *, width=1.0, height=1.0, max_iters=4000,
         goal_bias=0.1, rng=None, oracle="direction"):
    """Explore the implicit composite roadmap with a discrete RRT.

    ``roadmaps`` — one :class:`Roadmap` per robot (vertex 0 = start, 1 = goal).
    ``oracle`` — ``"direction"`` (dRRT's ``O_d``) or ``"random"`` (the
    random-neighbour ablation the gate contrasts it against).  Returns a
    :class:`Solution` or ``None`` if the budget is exhausted.
    """
    rng = rng or random.Random(0)
    n = len(roadmaps)
    start_node = tuple(rm.start for rm in roadmaps)
    goal_node = tuple(rm.goal for rm in roadmaps)
    goal_points = [rm.points[rm.goal] for rm in roadmaps]

    parents = {start_node: None}
    nodes = [start_node]

    def _expand(near, target_points):
        if oracle == "random":
            return _random_neighbour(roadmaps, near, rng)
        return direction_oracle(roadmaps, near, target_points)

    def _try_add(near, new):
        if new == near or new in parents:
            return None
        if not _edge_collision_free(roadmaps, near, new, obstacles, r):
            return None
        parents[new] = near
        nodes.append(new)
        return new

    def _connect_to_target(node):
        """Greedily drive ``node`` straight at the goal via the oracle.

        This greedy connection IS the direction oracle in action — it only runs
        for ``oracle == "direction"``.  The ``"random"`` ablation has no aligned
        heading to follow, so it must reach the exact goal composite vertex by
        random-neighbour expansion alone (the contrast the gate measures)."""
        if oracle != "direction":
            return node == goal_node
        cur = node
        for _ in range(4 * max(len(rm) for rm in roadmaps)):
            if cur == goal_node:
                return True
            nxt = direction_oracle(roadmaps, cur, goal_points)
            added = _try_add(cur, nxt)
            if added is None:
                return cur == goal_node
            cur = added
        return cur == goal_node

    for it in range(1, max_iters + 1):
        if rng.random() < goal_bias:
            target = goal_points
        else:
            target = []
            for rm in roadmaps:
                while True:
                    p = (rng.uniform(0.0, width), rng.uniform(0.0, height))
                    if _point_clear(p, obstacles, r, width, height):
                        target.append(p)
                        break
        # nearest tree node to the random composite sample
        near = min(nodes, key=lambda nd: sum(
            _dist(roadmaps[i].points[nd[i]], target[i]) for i in range(n)))
        new = _expand(near, target)
        added = _try_add(near, new)
        if added is None:
            continue
        if added == goal_node or _connect_to_target(added):
            paths, makespan, total = _reconstruct(roadmaps, parents, goal_node)
            return Solution(paths, len(nodes), it, makespan, total)
    return None


# --------------------------------------------------------------------------- #
# dRRT*  — asymptotically-optimal multi-robot motion planning
#   Shome, Solovey, Dobson, Halperin & Bekris, "dRRT*: Scalable and Informed
#   Asymptotically-Optimal Multi-Robot Motion Planning" (Autonomous Robots 2020).
#
# Plain dRRT returns the FIRST path its tree reaches — arbitrarily far from
# optimal.  dRRT* keeps exploring and, crucially, maintains the explored part of
# the implicit composite roadmap as a GRAPH (not a tree): every new vertex is
# wired to *all* explored vertices it is implicitly adjacent to, and the solution
# is the SHORTEST PATH in that explored graph (Dijkstra), not a chain of parent
# pointers.  As exploration densifies the graph the shortest path descends
# monotonically toward the optimum of the (fixed) implicit roadmap — anytime and
# asymptotically optimal.  Informed sampling focuses the search once a solution
# of cost ``c`` is known (admissible per-robot lower bounds must sum below ``c``).
# --------------------------------------------------------------------------- #
def _composite_edge_weight(roadmaps, u, v):
    """Sum-of-costs weight of one composite edge: total robot travel."""
    return sum(_dist(roadmaps[i].points[u[i]], roadmaps[i].points[v[i]])
               for i in range(len(roadmaps)))


def _implicit_adjacent(roadmaps, u, v):
    """True iff ``u``/``v`` are adjacent in the implicit composite roadmap.

    Every robot must either stay (``u[i] == v[i]``) or traverse one of its own
    roadmap edges (``v[i] in adj[u[i]]``); they cannot all stay.
    """
    moved = False
    for i, rm in enumerate(roadmaps):
        if u[i] == v[i]:
            continue
        if v[i] not in rm.adj[u[i]]:
            return False
        moved = True
    return moved


def _dijkstra(graph, source, target):
    """Shortest path ``source -> target`` over a weighted adjacency dict."""
    import heapq

    dist = {source: 0.0}
    prev = {source: None}
    pq = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == target:
            break
        if d > dist.get(u, math.inf):
            continue
        for v, w in graph[u].items():
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if target not in dist:
        return math.inf, None
    chain = [target]
    while prev[chain[-1]] is not None:
        chain.append(prev[chain[-1]])
    chain.reverse()
    return dist[target], chain


def composite_optimum(roadmaps, obstacles, r, *, max_product=200000):
    """Ground-truth optimum: shortest path over the FULL implicit roadmap.

    Builds the entire composite graph (every collision-free implicit edge) and
    runs Dijkstra — feasible only for small instances (guarded by
    ``max_product``).  This is the independent optimum the gate certifies dRRT*
    converges to.  Returns ``(cost, path_nodes)`` or ``(inf, None)``.
    """
    if tensor_product_size(roadmaps) > max_product:
        raise ValueError("composite roadmap too large for brute optimum")
    n = len(roadmaps)
    start_node = tuple(rm.start for rm in roadmaps)
    goal_node = tuple(rm.goal for rm in roadmaps)

    # enumerate all composite vertices
    def _all_nodes():
        stack = [()]
        for rm in roadmaps:
            stack = [node + (v,) for node in stack for v in range(len(rm))]
        return stack

    verts = [nd for nd in _all_nodes()
             if all(_dist(roadmaps[i].points[nd[i]],
                          (o.x, o.y)) >= o.radius + r - 1e-12
                    for i in range(n) for o in obstacles)
             and all(_dist(roadmaps[i].points[nd[i]],
                           roadmaps[j].points[nd[j]]) >= 2 * r - 1e-9
                     for i in range(n) for j in range(i + 1, n))]
    vset = set(verts)
    graph = {v: {} for v in verts}
    for u in verts:
        # neighbours: each robot stays or steps one roadmap edge
        choices = [[u[i]] + list(roadmaps[i].adj[u[i]]) for i in range(n)]
        stack = [()]
        for opt in choices:
            stack = [c + (x,) for c in stack for x in opt]
        for v in stack:
            if v == u or v not in vset or v in graph[u]:
                continue
            if _edge_collision_free(roadmaps, u, v, obstacles, r):
                graph[u][v] = _composite_edge_weight(roadmaps, u, v)
    cost, chain = _dijkstra(graph, start_node, goal_node)
    return cost, chain


@dataclass
class StarSolution:
    paths: dict
    graph_size: int
    iterations: int
    cost: float            # sum-of-costs (total robot travel) of the best path
    first_cost: float      # cost of the first path found (dRRT-like)
    cost_history: list = field(default_factory=list)  # best cost over iters

    def waypoint_count(self):
        return len(next(iter(self.paths.values())))


def _informed_reject(roadmaps, target, c_best):
    """Admissible informed-sampling test for sum-of-costs.

    A sample can only lie on an improving solution if the sum over robots of
    (straight start->sample + straight sample->goal) stays below the incumbent
    cost — a valid lower bound, so rejecting violators never discards an
    improving path.
    """
    bound = 0.0
    for i, rm in enumerate(roadmaps):
        s = rm.points[rm.start]
        g = rm.points[rm.goal]
        bound += _dist(s, target[i]) + _dist(target[i], g)
    return bound > c_best


def drrt_star(roadmaps, obstacles, r, *, width=1.0, height=1.0, max_iters=4000,
              goal_bias=0.1, rng=None, informed=True):
    """Asymptotically-optimal dRRT*: explore, maintain a graph, extract the
    Dijkstra shortest path; refine until the budget runs out.

    Returns a :class:`StarSolution` (with the anytime ``cost_history``) or
    ``None`` if no path is found.
    """
    rng = rng or random.Random(0)
    n = len(roadmaps)
    start_node = tuple(rm.start for rm in roadmaps)
    goal_node = tuple(rm.goal for rm in roadmaps)
    goal_points = [rm.points[rm.goal] for rm in roadmaps]

    graph = {start_node: {}}
    nodes = [start_node]
    best_cost = math.inf
    first_cost = math.inf
    history = []

    def _add_vertex(v):
        """Insert ``v`` and wire it to every explored implicit neighbour."""
        if v in graph:
            return False
        graph[v] = {}
        for u in nodes:
            if _implicit_adjacent(roadmaps, u, v) and \
                    _edge_collision_free(roadmaps, u, v, obstacles, r):
                w = _composite_edge_weight(roadmaps, u, v)
                graph[u][v] = w
                graph[v][u] = w
        nodes.append(v)
        return True

    for it in range(1, max_iters + 1):
        if rng.random() < goal_bias:
            target = goal_points
        else:
            target = []
            for rm in roadmaps:
                while True:
                    p = (rng.uniform(0.0, width), rng.uniform(0.0, height))
                    if not _point_clear(p, obstacles, r, width, height):
                        continue
                    target.append(p)
                    break
            if informed and best_cost < math.inf and \
                    _informed_reject(roadmaps, target, best_cost):
                history.append(best_cost)
                continue
        near = min(nodes, key=lambda nd: sum(
            _dist(roadmaps[i].points[nd[i]], target[i]) for i in range(n)))
        new = direction_oracle(roadmaps, near, target)
        if new != near:
            _add_vertex(new)
        # greedy connect toward the goal densifies the path corridor
        cur = new
        for _ in range(4 * max(len(rm) for rm in roadmaps)):
            if cur == goal_node:
                break
            nxt = direction_oracle(roadmaps, cur, goal_points)
            if nxt == cur:
                break
            _add_vertex(nxt)
            cur = nxt
        if goal_node in graph:
            cost, chain = _dijkstra(graph, start_node, goal_node)
            if chain is not None and cost < best_cost - 1e-12:
                if first_cost == math.inf:
                    first_cost = cost
                best_cost = cost
                best_chain = chain
        history.append(best_cost)

    if best_cost == math.inf:
        return None
    paths = {i: [roadmaps[i].points[node[i]] for node in best_chain]
             for i in range(n)}
    return StarSolution(paths, len(nodes), max_iters, best_cost, first_cost,
                        history)
