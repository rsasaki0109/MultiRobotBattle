"""Token Swapping: the minimum number of adjacent swaps to sort tokens on a graph.

A reproduction of **Token Swapping**, Yamanaka, Demaine, Ito, Kawahara, Kiyomi,
Okamoto, Saitoh, Suzuki, Uchizawa, Uno, *"Swapping Labeled Tokens on Graphs"*
(FUN 2014; Theoretical Computer Science 2015), with the approximation /
hardness picture from Miltzow, Narins, Okamoto, Rote, Thomas & Uno,
*"Approximation and Hardness of Token Swapping"* (ESA 2016).

This is the **min-swap-count reconfiguration** paradigm — distinct from every
other permutation-on-a-graph solver in the zoo:

- :mod:`tswap` (Okumura & Defago), :mod:`push_and_swap`, :mod:`bibox` and
  :mod:`push_and_rotate` move tokens through **blank** vertices and minimise
  *makespan* / number of moves; a move slides one token into an empty cell.
- **Token swapping has no blanks** — *every* vertex holds a token, and the only
  operation is a **swap**: two tokens on an edge exchange places. The objective
  is the **total number of swaps**. It is the graph generalisation of "sort a
  permutation by transpositions", and that is exactly why two graph classes have
  beautiful closed forms:

  * **Path** ``P_n``: only *adjacent* transpositions are legal, so the optimum is
    the **inversion count** of the permutation (the bubble-sort lower bound is
    tight). :func:`path_inversions` / :func:`path_swaps`.
  * **Complete graph** ``K_n``: *every* transposition is legal, so the optimum is
    ``n - c`` where ``c`` is the number of cycles of the permutation (the classic
    "minimum transpositions to sort"). :func:`complete_min_swaps` /
    :func:`complete_swaps`.

General graphs are NP-complete (Miltzow et al.), so the exact solver here is an
exponential BFS over the ``n!`` token placements (:func:`optimal_swaps`) — ground
truth on small instances, and the foil that the constructive closed forms beat by
orders of magnitude on the two structured classes.

Two certificates frame the optimum:

- a **lower bound** ``ceil(D / 2)`` where ``D = sum_t dist(pos(t), target(t))``:
  one swap moves two tokens one edge each, so it lowers ``D`` by at most ``2``
  (:func:`lower_bound`); and
- the honest *negative* :func:`descent_swaps` — a naive best-improving descent
  that only takes swaps strictly reducing ``D`` **stalls** even on a path
  reversal or a ring rotation, because real progress requires *temporarily*
  pushing a token further from its target. That stall is precisely what the
  paper's non-trivial algorithms exist to defeat.

A graph is an adjacency map ``{node: iterable-of-neighbours}`` (undirected); a
token placement is a bijection ``{node: label}``. The structured solvers assume
nodes ``0..n-1`` (a path is ``0-1-..-(n-1)``; ``K_n`` is all pairs). Pure and
deterministic — no numpy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil


# --------------------------------------------------------------------------- #
# Basic mechanics
# --------------------------------------------------------------------------- #
def apply_swap(placement, u, v):
    """Return a new placement with the tokens on ``u`` and ``v`` exchanged."""
    q = dict(placement)
    q[u], q[v] = q[v], q[u]
    return q


def replay(graph, initial, swaps):
    """Apply ``swaps`` to ``initial`` in order; every swap must be a graph edge."""
    p = dict(initial)
    for u, v in swaps:
        if v not in graph[u] or u not in graph[v]:
            raise ValueError(f"swap ({u}, {v}) is not an edge")
        p = apply_swap(p, u, v)
    return p


def num_misplaced(placement, target):
    """How many vertices hold the wrong token."""
    return sum(1 for v in placement if placement[v] != target[v])


def is_solved(placement, target):
    return all(placement[v] == target[v] for v in placement)


def all_pairs_distance(graph):
    """BFS hop-distance from every vertex (unit edge weights)."""
    dist = {}
    for s in graph:
        d = {s: 0}
        dq = deque([s])
        while dq:
            x = dq.popleft()
            for y in graph[x]:
                if y not in d:
                    d[y] = d[x] + 1
                    dq.append(y)
        dist[s] = d
    return dist


def total_distance(graph, placement, target, *, dist=None):
    """``D = sum over tokens of dist(current vertex, target vertex)``."""
    if dist is None:
        dist = all_pairs_distance(graph)
    target_vertex = {target[v]: v for v in target}
    return sum(dist[v][target_vertex[placement[v]]] for v in placement)


def lower_bound(graph, initial, target, *, dist=None):
    """``ceil(D / 2)`` — a swap lowers ``D`` by at most 2, so ``opt >= this``."""
    return ceil(total_distance(graph, initial, target, dist=dist) / 2)


# --------------------------------------------------------------------------- #
# Exact optimum: BFS over the n! token placements
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SwapSolution:
    """An optimal swap sequence and the search effort that found it."""

    num_swaps: int
    swaps: list
    states_expanded: int


def optimal_swaps(graph, initial, target, *, max_states=200_000):
    """Minimum swap sequence by breadth-first search over placements.

    Returns a :class:`SwapSolution`, or ``None`` if more than ``max_states``
    placements would be expanded (the exponential blow-up the structured
    closed forms exist to avoid).
    """
    nodes = sorted(graph)
    index = {n: i for i, n in enumerate(nodes)}
    edges = [(u, v) for u in nodes for v in graph[u] if index[u] < index[v]]

    def canon(p):
        return tuple(p[n] for n in nodes)

    start = canon(initial)
    goal = canon(target)
    if start == goal:
        return SwapSolution(0, [], 0)

    parent = {start: None}
    depth = {start: 0}
    dq = deque([start])
    expanded = 0
    while dq:
        s = dq.popleft()
        expanded += 1
        if expanded > max_states:
            return None
        for u, v in edges:
            iu, iv = index[u], index[v]
            t = list(s)
            t[iu], t[iv] = t[iv], t[iu]
            t = tuple(t)
            if t not in depth:
                depth[t] = depth[s] + 1
                parent[t] = (s, (u, v))
                if t == goal:
                    swaps = []
                    cur = t
                    while parent[cur] is not None:
                        prev, edge = parent[cur]
                        swaps.append(edge)
                        cur = prev
                    swaps.reverse()
                    return SwapSolution(depth[goal], swaps, expanded)
                dq.append(t)
    return None


# --------------------------------------------------------------------------- #
# Permutation view (nodes 0..n-1)
# --------------------------------------------------------------------------- #
def permutation(initial, target):
    """``perm[i]`` = the target vertex of the token currently on vertex ``i``.

    Sorting this permutation to the identity *is* the token-swapping instance.
    Requires nodes ``0..n-1``.
    """
    n = len(initial)
    target_vertex = {target[v]: v for v in range(n)}
    return [target_vertex[initial[i]] for i in range(n)]


def cycle_count(perm):
    """Number of cycles of a permutation (fixed points count as 1-cycles)."""
    n = len(perm)
    seen = [False] * n
    cycles = 0
    for i in range(n):
        if not seen[i]:
            cycles += 1
            j = i
            while not seen[j]:
                seen[j] = True
                j = perm[j]
    return cycles


# --------------------------------------------------------------------------- #
# Closed form on a path: optimum = inversion count
# --------------------------------------------------------------------------- #
def path_inversions(initial, target):
    """Optimum on the path ``0-1-..-(n-1)``: inversions of the permutation."""
    perm = permutation(initial, target)
    n = len(perm)
    return sum(
        1 for i in range(n) for j in range(i + 1, n) if perm[i] > perm[j]
    )


def path_swaps(initial, target):
    """Construct the optimal adjacent-swap sequence on a path (bubble sort)."""
    perm = permutation(initial, target)
    n = len(perm)
    arr = list(perm)
    swaps = []
    changed = True
    while changed:
        changed = False
        for i in range(n - 1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swaps.append((i, i + 1))
                changed = True
    return swaps


# --------------------------------------------------------------------------- #
# Closed form on the complete graph: optimum = n - cycles
# --------------------------------------------------------------------------- #
def complete_min_swaps(initial, target):
    """Optimum on ``K_n``: ``n - c`` (minimum transpositions to sort)."""
    n = len(initial)
    return n - cycle_count(permutation(initial, target))


def complete_swaps(initial, target):
    """Construct the optimal transposition sequence on ``K_n`` (selection sort)."""
    n = len(initial)
    cur = [initial[i] for i in range(n)]
    want = [target[i] for i in range(n)]
    swaps = []
    for i in range(n):
        if cur[i] != want[i]:
            j = next(k for k in range(i + 1, n) if cur[k] == want[i])
            cur[i], cur[j] = cur[j], cur[i]
            swaps.append((i, j))
    return swaps


# --------------------------------------------------------------------------- #
# Honest negative: naive best-improving descent stalls
# --------------------------------------------------------------------------- #
def descent_swaps(graph, initial, target, *, max_swaps=10_000, dist=None):
    """Greedy descent that only takes swaps strictly reducing ``D``.

    Returns ``(swaps, solved)``. It **stalls** (``solved=False``) whenever
    progress would require temporarily increasing some token's distance — e.g. a
    path reversal or a ring rotation — which is exactly why the paper needs
    non-trivial algorithms. Kept as a documented negative, not a solver.
    """
    if dist is None:
        dist = all_pairs_distance(graph)
    nodes = sorted(graph)
    index = {n: i for i, n in enumerate(nodes)}
    edges = [(u, v) for u in nodes for v in graph[u] if index[u] < index[v]]
    p = dict(initial)
    swaps = []
    for _ in range(max_swaps):
        cur = total_distance(graph, p, target, dist=dist)
        if cur == 0:
            return swaps, True
        best = None
        best_d = cur
        for u, v in edges:
            d = total_distance(graph, apply_swap(p, u, v), target, dist=dist)
            if d < best_d:
                best_d = d
                best = (u, v)
        if best is None:
            return swaps, False  # stalled in a local minimum with D > 0
        p = apply_swap(p, *best)
        swaps.append(best)
    return swaps, False


# --------------------------------------------------------------------------- #
# Graph builders (convenience)
# --------------------------------------------------------------------------- #
def make_path_graph(n):
    g = {i: set() for i in range(n)}
    for i in range(n - 1):
        g[i].add(i + 1)
        g[i + 1].add(i)
    return g


def make_cycle_graph(n):
    g = {i: set() for i in range(n)}
    for i in range(n):
        g[i].add((i + 1) % n)
        g[i].add((i - 1) % n)
    return g


def make_complete_graph(n):
    return {i: {j for j in range(n) if j != i} for i in range(n)}


def make_grid_graph(rows, cols):
    g = {}
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            nb = set()
            if r > 0:
                nb.add((r - 1) * cols + c)
            if r < rows - 1:
                nb.add((r + 1) * cols + c)
            if c > 0:
                nb.add(r * cols + c - 1)
            if c < cols - 1:
                nb.add(r * cols + c + 1)
            g[v] = nb
    return g
