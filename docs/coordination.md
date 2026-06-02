# Coordination Layer (`mrn_coord`)

<p align="center">
  <img src="media/gazebo_coord_demo.gif" alt="Three robots funnel through a doorway via Conflict-Based Search then assemble a formation in a 3D Gazebo world, their 360-degree LiDAR tracing the walls" width="640">
</p>

<p align="center">
  <em>Driven by the real algorithms in the 3D Gazebo world: CBS plans the collision-free doorway crossing, then the consensus controller assembles the formation, each robot sweeping a 360° LiDAR. Rendered fully offscreen on the GPU; regenerate with <code>python3 scripts/record_gazebo_coord_gif.py</code>.</em>
</p>

`mrn_coord` is the **coordination / navigation** half of the project — the
counterpart to the cooperative-localization stack. Where localization answers
*where are we*, coordination answers *how do we move and what do we do
together*. It follows the same project pattern as the rest of the repo: pure,
ROS-free algorithm cores that are unit-tested in CI, with thin ROS/CLI wiring
layered on top.

Planned scope, built one module at a time:

1. **MAPF** (`mrn_coord.mapf`) — multi-agent path finding: collision-free
   planning on a shared grid. **Landed.**
2. **Formation** (`mrn_coord.formation`) — decentralized formation control that
   reuses the V2V relative-pose constraints already exchanged by the
   localization stack. **Landed.**
3. **Coverage** (`mrn_coord.coverage`) — cooperative exploration and task
   allocation (frontier detection + auction/Hungarian assignment). **Landed.**

## MAPF — Multi-Agent Path Finding

Given a shared grid, obstacles, and each agent's start and goal, MAPF finds
paths that never put two agents in the same cell at the same time and never let
them swap across an edge. The pieces compose bottom-up.

### The grid (`grid.py`)

`GridWorld(width, height, blocked)` is a 4-connected grid with blocked cells.
Movement is one cell per timestep in a cardinal direction or a **wait** in
place, so `neighbors(cell)` always includes the cell itself. `manhattan` is the
admissible heuristic.

### Low level: space-time A* (`space_time_astar.py`)

`plan_path(grid, start, goal, vertex_constraints, edge_constraints)` plans one
agent over `(cell, time)` states, minimizing arrival time. It honors the two
constraint types the high-level solvers branch on:

- **vertex** `(cell, time)` — may not occupy `cell` at `time`.
- **edge** `(frm, to, time)` — may not move `frm -> to` arriving at `time`
  (this is how swaps are forbidden).

The goal test requires the agent to be at the goal *and* past the last time the
goal is vertex-constrained, so a returned path can be safely held at the goal
forever (an agent waits at its goal after arrival). It returns a list of cells
indexed by timestep, or `None` if no path exists within a finite time horizon.

### Low level alternative: SIPP (`sipp.py`)

`plan_sipp(grid, start, goal, vertex_constraints, edge_constraints)` is a
**drop-in** for `plan_path` with the same signature, the same constraint
vocabulary, and the same minimal-time path — but a much smaller search space.
Time-expanded A* keeps one state per `(cell, time)`, so an agent forced to wait
out a long reservation re-expands a near-identical state every tick. **Safe
Interval Path Planning** (Phillips & Likhachev 2011) instead partitions each
cell's timeline into *safe intervals* — maximal runs of collision-free
timesteps — and keeps one state per `(cell, interval)`, letting the agent wait
*anywhere* in an interval for free. A chokepoint reserved for 200 ticks costs
SIPP one state, not 200 (`benchmarks/comparison.md` sweeps this: A* expansions
grow with the wait, SIPP's stay flat). Pass it as the `low_level` of
`prioritized_planning`, or run `mrn_mapf_bench --solver prioritized_sipp`; it
finds equal-cost solutions. Edge (swap) constraints are handled by skipping the
single forbidden arrival time into a successor interval.

### Conflicts (`conflicts.py`)

`detect_first_conflict(paths)` returns the earliest **vertex** conflict (same
cell, same time) or **edge** conflict (a swap) between any pair of paths, or
`None`. Paths may differ in length; `cell_at` clamps past the end so an agent
that has reached its goal is treated as staying there.

### High level: Conflict-Based Search (`cbs.py`)

`cbs(grid, agents)` is the optimal (sum-of-costs) solver. It searches a binary
*constraint tree* best-first by cost: each node holds the current per-agent
constraints and paths; on the first conflict it branches into two children that
each add one constraint to one of the two agents and replan only that agent.
The first conflict-free node popped is optimal. Returns a `Solution` or `None`
(infeasible, or the expansion budget is exhausted).

#### Validated against the reference libMultiRobotPlanning

"Optimal" is a strong word, so we hold it to the canonical reference:
`scripts/compare_mapf_libmrp.py` solves identical instances with our `cbs.py`
and with Wolfgang Hönig's [`libMultiRobotPlanning`](https://github.com/whoenig/libMultiRobotPlanning)
C++ `cbs`, then checks they agree. Both run the same discrete model (4-connected
grid, wait actions, unit cost, vertex + edge-swap conflicts, no
`--disappear-at-goal`) and minimize **sum-of-costs** — whose optimum is a single
number, so a correct optimal solver must reproduce the reference's value
*exactly*, with no tolerance. The checked-in numbers (and why `makespan` is
reported but not gated — many solutions share the optimal cost) live in
[`benchmarks/mapf_libmrp.md`](../benchmarks/mapf_libmrp.md).

The reference is an *optional* dependency built from source — the core build and
test suite never touch it (the equivalence test skips cleanly when the binary is
absent). To run the check locally (needs `cmake`, a C++ compiler, Boost
program-options, and yaml-cpp):

```bash
git clone https://github.com/whoenig/libMultiRobotPlanning.git /tmp/libMultiRobotPlanning
cmake -S /tmp/libMultiRobotPlanning -B /tmp/libMultiRobotPlanning/build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/libMultiRobotPlanning/build --target cbs ecbs
export LIBMRP_CBS=/tmp/libMultiRobotPlanning/build/cbs
export LIBMRP_ECBS=/tmp/libMultiRobotPlanning/build/ecbs
python3 scripts/compare_mapf_libmrp.py --check       # gated equivalence contract
python3 scripts/compare_mapf_libmrp.py --write        # refresh benchmarks/mapf_libmrp.md
```

The `mapf-libmrp-equivalence` CI job does exactly this on every push, so CBS
computing the true optimum is a guarded contract, not a claim.

### High level: Enhanced CBS (`ecbs.py`)

`ecbs(grid, agents, w=1.5)` is the **bounded-suboptimal** solver: it returns
collision-free paths whose sum-of-costs is at most `w` times the optimum, and
in exchange expands far fewer constraint-tree nodes than CBS — so it keeps
solving as the team grows past where CBS's tree explodes. The mechanism is
**focal search** at both levels. The low level
(`_focal_low_level`) is a single-agent A* that, among all paths within `w` of
the cheapest, picks the one with the fewest conflicts against the other agents'
current paths, and also reports `f_min`, a lower bound on that agent's
constrained optimum. The high level orders OPEN by `LB(N) = Σ f_min` and expands
from the FOCAL set — nodes whose actual cost is `≤ w·min LB` — choosing the one
with the fewest total conflicts. Popping a conflict-free FOCAL node gives a
solution with `cost ≤ w·min LB ≤ w·optimal`. With `w=1` it reduces to optimal
CBS. `benchmarks/comparison.md` sweeps team size: CBS's expansions blow up and
it starts exhausting its budget while ECBS stays in a handful of nodes for a
few-percent cost premium. Run `mrn_mapf_demo --solver ecbs` or
`mrn_mapf_bench --solver ecbs -w 1.3`.

#### Validated against the reference libMultiRobotPlanning

The `w·optimal` guarantee is the whole point of ECBS, so we hold it to the
reference's `ecbs` exactly as we held CBS to its `cbs`.
`scripts/compare_ecbs_libmrp.py` solves the same instances with our `ecbs.py` and
libMultiRobotPlanning's `ecbs` at the same `w`, takes the optimum from our CBS
(itself pinned to the reference `cbs`, see above), and checks the **bound**:
`cost ≤ w · optimal` for both solvers. The two need not return the *same* cost —
focal-search tie-breaking differs — so, unlike the optimal-CBS contract, equality
is not gated; the ratios (in [`benchmarks/ecbs_libmrp.md`](../benchmarks/ecbs_libmrp.md))
show the suboptimality actually taken, which sits at or below the ceiling. Build
the reference as above (the `ecbs` target ships in the same `cmake --build`), set
`LIBMRP_ECBS`, and the `mapf-libmrp-equivalence` CI job runs it on every push —
so the suboptimality *bound*, like the optimum itself, is a guarded contract.

### High level: LaCAM (`lacam.py`)

`lacam(grid, agents)` takes a different tack from the CBS family: instead of
searching a constraint tree and replanning single agents, it searches the
**configuration space** — a node is the joint position of *all* agents — with
**PIBT** as the successor generator (one collision-free joint move in
near-linear time). PIBT alone is greedy and incomplete, so LaCAM hangs a tree of
**lazy constraints** off each configuration: low-level nodes pin successive
agents to successive candidate cells, each yielding a PIBT successor under those
pins. Because the constraints eventually enumerate *every* successor and the
configuration space is finite, LaCAM is **complete** — it finds a solution
whenever one exists. It is satisficing (any valid collision-free solution, not
cost-optimal); in random tests it solves every instance CBS solves, and it
matches the optimum on the bundled example.

Completeness is cheap; *scaling* is the hard part, and it lives entirely in the
order successors are generated. LaCAM dives greedily — the unconstrained PIBT
successor is explored first, so the DFS spine *is* a PIBT rollout and the lazy
constraints are only the backtracking fallback. With a **static** per-config
priority the spine was the weak deterministic PIBT, which livelocks and drops
into the lazy-constraint enumeration; that branches-explodes (every agent × every
neighbor) and times out. On a 16–30-agent open-grid battery the old order solved
only **0.667** of instances even at 200k iterations, ~100× slower than now — and
some it could not solve at *3M* iterations: complete only in theory. The spine
now runs the **strong** PIBT, the same one `pibt_solve` (in `lifelong/`) uses:
off-goal agents *accumulate* priority, and a stall (summed distance-to-goal
failing to reach a new low) bumps the deterministic escape `salt` — reseeded each
time a stuck configuration is re-expanded, since `explored` forbids the revisits
that let `pibt_solve`'s oscillating walk recover. The constraint enumeration is
untouched, so completeness still holds; only the successor order changes. That
battery now solves **180/180**, every solution valid, in seconds — the
`lacam_scaling_convergence` gate pins it, where the toy completeness test (2–4
agents on 4×4–6×6) never reached the regime the claim lived in. Run
`mrn_mapf_demo --solver lacam` or `mrn_mapf_bench --solver lacam`.

The default is satisficing — it returns the first valid solution, which runs
~1.13× the optimal sum-of-costs on small instances and ~1.4–1.6× the lower bound
at scale. `lacam(grid, agents, optimize=True)` runs the **anytime** variant
(LaCAM\*): it keeps searching past the first solution, tracking the best cost to
each configuration (`g`) and *rewiring* a config's parent when a cheaper route
appears, pruning any node whose `g` + admissible remaining distance cannot beat
the incumbent. On small instances this reaches the **true optimum** — it matches
CBS's sum-of-costs agent-for-agent on 120/120 CBS-solvable cases (the
`lacam_optimality` gate). Measured honestly, it does **not** scale as a cost
optimizer: on 16–30-agent grids the configuration space is astronomical, the
lower bound barely prunes, and a 200k-iteration budget (~10 s/instance) returns
the *same* cost as the first dive. **For cost at scale, use LNS** (below), which
drives those instances to ~1.13× the lower bound in a fraction of the time.

`lacam_ltm(grid, agents, rounds=...)` is a Python reproduction of *A Lightweight
Traffic Map for Efficient Anytime LaCAM\** (arXiv:2603.07891, C++-only upstream)
that attacks exactly that scaling wall. Plain `optimize` re-walks the same
congested corridors every dive; LaCAM\*+LTM builds a **lightweight traffic map**
during the search — a directed-edge weight accumulating the agent moves PIBT
actually commits — then between bounded runs normalizes those counts into
`[0, 10]`, recomputes each agent's guidance distance on the congestion-weighted
graph (so dives route *around* busy edges), and restarts. The admissible
heuristic is untouched, so each round keeps its cost guarantees; only the
guidance changes. Measured at **equal total budget** (plain `optimize` given
`rounds × budget` in one run vs LTM spending the same across restarts), LTM cuts
aggregate sum-of-costs **1527 → 1264** over a 6-instance 10×10/20 + 12×12/30
battery and wins **every** instance, where plain `optimize` returns the first
dive's cost no matter the budget — a 14–32 % cut, from ~1.4–1.8× down to
~1.21–1.23× the lower bound. This is a faithful *subset* of the paper
(committed-move accumulation only; the blocked-action and wait terms are
omitted) and is deterministic. The `lacam_ltm_vs_optimize` gate pins the win;
it trips if the traffic-map guidance regresses. LNS (below) still owns the
prioritized-replanning route to cost at scale; LTM is the *search-side* answer,
keeping LaCAM's single-shot configuration search but spending extra budget on
re-guided restarts instead of wasting it on a stalled optimize.

### High level: MAPF-LNS (`lns.py`)

`mapf_lns(grid, agents, iterations=...)` is *anytime*: rather than searching for
a good solution from scratch, it takes any feasible one (prioritized planning,
falling back to complete LaCAM) and repeatedly **destroys** a small
neighborhood — rips out a handful of agents' paths — then **repairs** it by
replanning just those agents around everyone else's frozen paths, keeping the
repair whenever it doesn't raise the sum-of-costs. Each round is cheap (a few
agents, not all), the cost decreases monotonically, and you stop on a budget —
so a rough initial solution is polished toward the optimum, on teams far beyond
CBS's reach. Two destroy heuristics are mixed at random each round: a **random**
agent set, and a **worst** set built from the most-delayed agent plus the agents
whose paths cross it. Repair is collision-free by construction, so every
accepted solution stays valid; `benchmarks/comparison.md` shows it closing most
of the gap to the CBS optimum. The "on teams far beyond CBS's reach" half of that
claim is now guarded too: the `lns_scaling_improvement` gate runs LNS on a
16–20-agent open-grid battery and pins the aggregate destroy-repair gain — total
sum-of-costs falling from ~1.23× to ~1.14× the lower bound (the unit tests only
reach 5×5 / 4 agents, where CBS itself is cheap). This is the cost optimizer at
scale, where LaCAM\* (above) stops improving. Run `mrn_mapf_demo --solver lns` or
`mrn_mapf_bench --solver lns`.

**Adaptive (BALANCE) — a faithful port, and an honest negative result.**
`mapf_lns(..., adaptive=True)` replaces the fixed 50/50 coin and fixed
neighborhood size with the bi-level Thompson-Sampling bandit of BALANCE (Phan
et al., AAAI 2024): a top bandit learns which of three destroy heuristics
(random / worst / a new **map** heuristic targeting congested high-degree
vertices) pays off, and a per-heuristic bottom bandit learns the size from
`{2,4,8,16,32}`, rewarded by realized cost improvement. BALANCE reports ≥50%
cost gains — but on a specialized SIPP repair, structured warehouse maps, and
thousands of iterations. Ported onto *this* repo's prioritized-A\* repair and
measured honestly (open grids 8×8/16 … 12×12/30 and obstacle-dense 16×16 …
20×20, budgets to 1000 iterations), **the bandit does not beat the fixed
ensemble — it loses ~2%.** The mechanism works (it learns and shifts away from
weak arms), but the repo's `worst` heuristic is already strong and the bandit's
early exploration is never recovered at these scales; an open grid has no
high-degree junctions for the `map` arm to exploit. The default stays
`adaptive=False` and is byte-for-byte unchanged. The negative result is itself
guarded: the `lns_adaptive_vs_fixed` gate pins aggregate adaptive SOC ≥ fixed
SOC (1695 vs 1665 over 8 instances), so the claim can't be silently overstated,
and a future change that *did* make adaptive win would trip the gate and force
the claim to be re-pinned.

### High level: prioritized planning (`prioritized.py`)

`prioritized_planning(grid, agents, order)` is the fast, **incomplete**
alternative. It plans agents in priority order; each treats higher-priority
paths as moving obstacles (reserving their cells/times, blocking their settled
goals, and forbidding swaps against their moves). Cheap and often good enough,
but a bad order can leave a later agent with no path even when one exists — so
it can return `None` on solvable instances, unlike CBS.

### Solution helpers (`solution.py`)

`Solution(paths, cost)` plus `sum_of_costs`, `makespan`, `pad_paths` (hold the
goal to a common horizon), and `render_ascii(grid, paths, t)` for CLI/test
visualization.

### Try it

```bash
ros2 run mrn_coord mrn_mapf_demo                 # CBS on two built-in scenarios
ros2 run mrn_coord mrn_mapf_demo --solver ecbs   # bounded-suboptimal (w=1.5)
ros2 run mrn_coord mrn_mapf_demo --solver prioritized
```

The demo solves a crossing and a swap/reorder scenario and prints the
collision-free paths as an ASCII timeline — the runnable counterpart to the
unit tests.

### MovingAI benchmarks

`movingai.py` loads the standard
[MovingAI MAPF benchmark format](https://movingai.com/benchmarks/mapf.html)
(`.map` / `.scen`) so the solvers can be evaluated on the maps and scenarios the
MAPF community uses — drop in any downloaded pair. `load_map` →
`GridWorld`, `load_scen` → start/goal tasks, and `run_mapf_benchmark(grid,
tasks, solver=...)` reports solved / makespan / sum-of-costs. A tiny example is
bundled (`mrn_coord/benchmarks/example.{map,scen}`).

```bash
ros2 run mrn_coord mrn_mapf_bench                       # bundled example (CBS)
ros2 run mrn_coord mrn_mapf_bench my.map my.scen -n 8   # first 8 agents
ros2 run mrn_coord mrn_mapf_bench --solver ecbs -w 1.3  # bounded-suboptimal
ros2 run mrn_coord mrn_mapf_bench --solver lacam        # complete, satisficing
ros2 run mrn_coord mrn_mapf_bench --solver lns          # anytime, destroy & repair
ros2 run mrn_coord mrn_mapf_bench --solver pbs          # priority-ordering search
ros2 run mrn_coord mrn_mapf_bench --solver prioritized
```

CBS is optimal but scales to small teams; for many agents use **ECBS**
(`--solver ecbs`, bounded-suboptimal — much further reach for a small cost
premium), **LaCAM** (`--solver lacam`, complete and satisficing — solves large
teams when the search trees blow up), **MAPF-LNS** (`--solver lns`, anytime —
polishes a feasible solution toward the optimum), **PBS** (`--solver pbs`,
priority-ordering search — suboptimal, but reorders past the head-on deadlocks
fixed-order prioritized planning hits; see RHCR below), or the prioritized solver
(fast, incomplete).

### Lifelong / online MAPF (`lifelong/`)

CBS and prioritized planning solve a **one-shot** instance: a fixed set of
start→goal pairs, solved once, done when everyone arrives. Real fleets
(warehouse robots) never stop — a robot that reaches its goal is immediately
given the next task, so the team must plan *while moving*, indefinitely. That is
**lifelong (online) MAPF**, and its figure of merit is **throughput** (tasks
completed per timestep), not makespan.

`run_lifelong(grid, starts, stream, max_steps=...)` runs it. Tasks come from a
deterministic, endless `TaskStream` (round-robin over a pool of endpoints);
`make_warehouse(rows, cols)` builds a shelf-and-aisle grid with its endpoint
stations. The per-timestep move is computed by **PIBT** (Priority Inheritance
with Backtracking): agents step along an obstacle-aware distance gradient toward
their goals in priority order, and when a high-priority agent wants a cell held
by a lower one it *pushes* it — recursively, backtracking to its next-best cell
if the push fails. PIBT yields a collision-free configuration every step (no
vertex sharing, no swaps) and sidesteps the deadlock plain reservation planning
hits when a forced-to-wait agent sits in a cell another already claimed;
priorities rise the longer a task goes unfinished, so nothing starves. Pure and
deterministic, so the throughput is reproducible and CI-gated.

<p align="center">
  <img src="media/warehouse_demo.gif" alt="A fleet of twelve autonomous mobile robots streams endless pickup/dropoff tasks through a shelf-and-aisle warehouse, never colliding, while a running counter shows the tasks served and the throughput per timestep" width="640">
</p>

<p align="center">
  <em>A warehouse AMR fleet: twelve robots take an endless stream of pick/drop tasks, stepped collision-free by PIBT with cost-aware allocation, and the counter tracks throughput (tasks/step) — the metric a real fleet is judged on. Deterministic; regenerate with <code>python3 scripts/make_warehouse_gif.py</code>.</em>
</p>

The same code scales straight to a **fleet system** — `--preset fleet` packs
**100 AMRs** onto a six-by-nine shelf floor (108 stations). Every step is still
one PIBT collision-free configuration; throughput climbs past **25 tasks/step**:

<p align="center">
  <img src="media/fleet_demo.gif" alt="A hundred autonomous mobile robots swarm a large shelf-and-aisle warehouse on a lifelong-MAPF schedule, collision-free via PIBT, the counter showing over twenty-five tasks served per timestep" width="720">
</p>

<p align="center">
  <em>Fleet system at scale: 100 AMRs, lifelong MAPF, every move a PIBT collision-free configuration. Regenerate with <code>python3 scripts/make_warehouse_gif.py --preset fleet</code>.</em>
</p>

#### Validated against the reference pypibt

"Collision-free, by PIBT" is the load-bearing claim under every frame above, so we
hold it to the canonical reference: Keisuke Okumura's own
[`pypibt`](https://github.com/Kei18/pypibt) (the paper author's implementation).
The contract is judged by the *reference's own* code — we feed our PIBT output
through `pypibt`'s `get_neighbors` + `validate_mapf_solution` — and gates the
guarantee our code actually makes, **not** an exact path: PIBT's completeness
theorem relies on a *random* tie-break, which `pypibt` has (`rng.shuffle`) and our
`_Pibt` deliberately does not (deterministic ties keep the demos bit-reproducible).

- **collision-free** — for every instance and *every timestep*, including the full
  lifelong warehouse run, our configuration has zero vertex collisions, zero edge
  (swap) collisions, and only step-or-wait transitions. This is **gated**.
- **converged** — the honest cost of the deterministic tie-break: as a one-shot
  fixed-goal solver `_Pibt` can livelock in a symmetric standoff the reference's
  random tie-break escapes, so it is not *complete* the way `pypibt` is. We report
  the rate (~0.7 across the suite) rather than hide it.

  But we can recover it **without** giving up determinism. `pibt_solve` adds a
  livelock escape: when the team's summed distance-to-goal stalls, it bumps a
  per-step *salt* that deterministically scrambles equal-distance candidate ties
  until the symmetry breaks — the random tie-break's effect, reproduced with zero
  randomness (a pure-arithmetic hash, no `PYTHONHASHSEED` dependence). The stall is
  measured against the *running-minimum* distance, not the previous step: a
  livelock oscillates the summed distance up and down, so a step-to-step test
  resets on every transient dip and the escape silently disengages — that subtlety
  is exactly what stranded the last ~1%. Beating the best-ever is the honest
  progress signal, and with it open-grid convergence climbs from **~0.63 to 1.0**
  while every step stays collision-free, gated by the `pibt_escape_convergence`
  benchmark case (600/600 converge and collision-free; bare PIBT clears 356/600)
  and `test_pibt_escape`. The escape is
  `salt=0`-off in the lifelong/RHCR engines, so it changes nothing there — every
  throughput baseline is untouched — it is opt-in for the one-shot solver where
  completeness, not bit-identical demos, is what matters.

  In the **lifelong** regime that livelock is *bounded* — but only under a
  precondition we now make explicit and test (it used to be an unverified aside
  that "no standoff is permanent"). Across ~3000 adversarial seeds — densely
  packed warehouses, random starts, random task streams over **distinct**
  endpoints (one station per cell, the realistic regime) — the worst stall is **8
  steps**: goals changing on arrival keep churning the priority order, so no
  cluster sits forever. `test_lifelong.test_liveness_bounded_under_distinct_goals`
  gates this (`longest_stall < 15`), so a change that introduces a real livelock
  fails the build. The precondition is that no two agents are ever assigned the
  **same** goal cell at once: funnel several agents onto one contested cell (an
  out-of-contract duplicate-goal stream) and the deterministic engine *does*
  deadlock permanently — the agent already on the cell idles and squats, PIBT's
  push has nowhere to shove it, and the corner cluster never resolves (zero tasks
  ever complete). `test_duplicate_goals_break_liveness` pins that boundary as a
  tripwire. `LifelongResult.longest_stall()` (the longest zero-completion window)
  is the measure behind both.
- **makespan** — where ours converges, its length tracks the reference's within a
  bound (a bound, not equality). Numbers in
  [`benchmarks/pibt_pypibt.md`](../benchmarks/pibt_pypibt.md).

The reference is an *optional*, pure-Python dependency — the core build and test
suite never touch it (the equivalence test skips cleanly when it is absent):

```bash
python3 -m venv /tmp/pypibt-venv && . /tmp/pypibt-venv/bin/activate
pip install --upgrade pip numpy pytest
git clone https://github.com/Kei18/pypibt.git /tmp/pypibt
pip install /tmp/pypibt
python3 scripts/compare_pibt_pypibt.py --check       # gated equivalence contract
python3 scripts/compare_pibt_pypibt.py --write        # refresh benchmarks/pibt_pypibt.md
```

The `pibt-pypibt-equivalence` CI job does exactly this on every push, so the
collision-free guarantee under the warehouse/fleet demos stays a checked contract.

```bash
ros2 run mrn_coord mrn_lifelong_demo                       # 6 robots, prints throughput + frames
ros2 run mrn_coord mrn_lifelong_demo --agents 8 --steps 200
ros2 run mrn_coord mrn_lifelong_demo --allocator auction   # cost-aware assignment
```

`scripts/compare_planners.py` tabulates throughput vs. team size
([`benchmarks/comparison.md`](../benchmarks/comparison.md)): it climbs with the
fleet until aisle congestion lengthens service times — the warehouse-capacity
trade-off lifelong MAPF exists to study.

#### Task allocation (`lifelong/allocation.py`)

PIBT decides *how* robots move; **which task** each freed robot gets is a
separate lever, set by `run_lifelong(..., allocator=...)`. The default
`"stream"` is round-robin — deal out the next task in a fixed cycle, ignoring
geometry — which routinely sends a robot clear across the warehouse past a
closer one. Two cost-aware allocators instead keep a pool of open tasks and
match free robots to them by obstacle-aware travel distance:

- **`hungarian`** — the optimal solution to the linear assignment problem
  (Kuhn-Munkres with potentials, `O(n³)`): minimum total travel.
- **`auction`** — a regret-based market auction: each round the unassigned robot
  with the most to lose (largest gap between its best and second-best remaining
  task) bids first and claims its best task. Decentralized, fast, near-optimal.

Sending the *nearest* free robot to each task shortens every trip; in the
bundled benchmark (`benchmarks/comparison.md`) cost-aware allocation roughly
**doubles throughput** and halves service time over round-robin. The optimal
one-shot Hungarian match and the cheaper auction are close — and over the
lifelong horizon the auction's round-by-round greediness can even edge ahead,
since one-shot optimality is not long-run optimality.

The lead widens with scale, and the benchmark gate now pins it at fleet size.
The 40-AMR `mapf_fleet_*` cases (`scripts/benchmark_gate.py`, a 4×6 warehouse)
clear **~10.8 tasks/step** under either cost-aware allocator versus **1.65** for
round-robin — a ~6.5× gap. CI checks the exact task count of all three, and a
`test_lifelong` invariant requires the cost-aware allocators to keep clearing
well over 3× round-robin, so a change that quietly neutralizes the allocator
fails the build even if the per-case baselines are also nudged to match.

#### Rolling-Horizon Collision Resolution — RHCR (`lifelong/rhcr.py`)

PIBT steps the team one greedy, collision-free move at a time. **RHCR** (Li,
Tinka, Kiesel, Durham, Kumar & Koenig, *Lifelong MAPF in Large-Scale
Warehouses*, AAAI 2021) is the *planning* alternative — `run_rhcr(...)`
decomposes the endless run into a sequence of **Windowed MAPF** instances: every
`replan_period` (`h`) steps it re-plans full paths toward the current goals, the
windowed solver resolves collisions only within the next `window` (`w ≥ h`)
timesteps and ignores everything beyond, and the team commits the next `h` steps
before repeating. Bounding resolution to a window is what keeps each solve cheap;
the lookahead is what lets it sidestep traps a one-step stepper walks into.

The windowed solver is pluggable. The default is **PBS** (Priority-Based Search,
Ma, Harabor, Stuckey, Li & Koenig, AAAI 2019; `mrn_coord.mapf.pbs`) — a new
solver in its own right, two-level like CBS but branching on *priority orderings*
instead of constraints, with prioritized planning at the low level. PBS resolves
the head-on, order-sensitive deadlocks that fixed-order prioritized planning
cannot: where one priority order parks an agent on another's only corridor and
fails, PBS reorders and finds the plan (`test_pbs`). Also available: fixed-order
prioritized planning (`"pp"`) and a PIBT rollout (`"pibt"`). PBS and PP fall back
to a PIBT rollout for any window they cannot fully resolve, so **a run is always
collision-free and live** regardless of solver.

What the gate actually pins — and what it honestly shows — is that **on this
cramped, single-aisle warehouse, greedy PIBT wins**, and RHCR is the more
interesting object for *where* it loses:

- **The commit horizon `h` trades throughput for bounded compute.** A robot that
  finishes mid-window idles until the next replanning boundary, so throughput is
  non-increasing in `h` (gated by a `test_rhcr` invariant). At `h = 1` with the
  PIBT-rollout window, RHCR degenerates *exactly* to the one-step PIBT engine —
  it reproduces `run_lifelong` metric-for-metric for every allocator
  (`test_rhcr`), the framework's sanity anchor. As `h` grows the bounded-replan
  cost shows: the 40-AMR fleet case clears **9.73 tasks/step** at `h = 2` versus
  the PIBT engine's 10.85.
- **Windowed PBS does not scale to extreme density.** In the 1-wide aisles at
  fleet size (40 AMRs over 48 endpoints, ~83% occupancy) PBS's priority search
  explodes — tens of seconds per run — and largely defers to the PIBT fallback
  anyway. This is consistent with the literature (PBS is a moderate-density
  solver; the field uses PIBT/LaCAM at extreme density). So the gate pins PBS and
  PP only on the small warehouse (`mapf_rhcr`, `mapf_rhcr_pp`,
  `mapf_rhcr_hungarian`), and exercises the *framework* at fleet scale with the
  fast PIBT-rollout window (`mapf_rhcr_fleet`).
- **…but lookahead wins once the aisles open up — and that is the paper's
  regime.** Widen the warehouse aisles (`aisle=2`) and the congestion that lets
  a greedy stepper win relaxes: with the same immediate reassignment (`h = 1`, to
  isolate the lookahead from the commit-horizon penalty), windowed PBS clears
  **strictly more** tasks than one-step PIBT — **327 vs 310** on the gated
  `mapf_rhcr_open` case — and PBS is fast again (no congestion explosion: ~1 s vs
  the tight map's tens of seconds). A `test_rhcr` invariant pins this crossover
  (`RHCR-PBS > PIBT` on the open map), so the planning advantage cannot silently
  regress.

The honest takeaway is a **crossover**, and the gate records both sides of it: in
a corridor with no room to be clever, greedy immediate-reassignment PIBT wins; as
the map opens up, planning a few steps ahead (RHCR + PBS) pulls ahead — which is
exactly the regime RHCR was designed for, so reproducing the *win*, not just the
mechanism, is what tells us the implementation is faithful. RHCR's standing value
is also bounded, predictable planning time (the `h` knob), independent of which
side of the crossover a given map sits on.

```bash
ros2 run mrn_coord mrn_lifelong_demo --engine rhcr                       # PBS window, default w=8/h=4
ros2 run mrn_coord mrn_lifelong_demo --engine rhcr --solver pp --replan 2
ros2 run mrn_coord mrn_mapf_demo --solver pbs                            # PBS as a standalone MAPF solver
```

### ROS node

`mrn_mapf_planner` is a thin ROS wrapper around the MAPF core. It reads a
scenario (grid size, obstacles, per-agent start/goal) from parameters, solves it
once with CBS or prioritized planning, and publishes one `nav_msgs/Path` per
agent on `mapf/path/<id>` with a latched (transient-local) QoS so RViz and late
subscribers receive it. The node holds no algorithm logic — planning and
grid-to-world conversion live in the pure, CI-tested
`mrn_coord.mapf.ros_conversion`. In a live system the agent start cells would
come from the cooperative-localization estimate; as parameters they keep the
node self-contained and launch-smoke-testable.

```bash
ros2 launch mrn_coord mapf_planner.launch.py   # the doorway scenario, 3 agents
ros2 topic echo /mapf/path/a_1                  # ids are sanitized to valid tokens
```

(Agent ids that would form an invalid ROS topic token — e.g. the digit `"1"` —
are prefixed, so agent `1` publishes on `mapf/path/a_1`.)

### Path follower (closing planning → world)

A MAPF plan is a path; `pure_pursuit` (in `path_follower.py`, pure and
CI-tested) turns a path plus the robot's current pose into a unicycle command
`(v, omega)` — the non-holonomic-compatible controller the simulator wants.
`mrn_path_follower` wraps it: per agent it subscribes to a `nav_msgs/Path` and a
`geometry_msgs/PoseStamped` and publishes `geometry_msgs/Twist`.

Paired with `mrn_sim_world` (pose in, `cmd_vel` out), it closes planning →
world. The launch `mapf_through_sim.launch.py` (in `mrn_sim`) wires planner →
follower → world on matching agent ids and grid; verified end-to-end, all three
robots track their CBS paths through the doorway and arrive within ~0.3 m of
their goals:

```bash
ros2 launch mrn_sim mapf_through_sim.launch.py use_rviz:=true
```

## Local collision avoidance — ORCA

MAPF plans *globally and discretely* before motion. **ORCA** (`orca.py`) is its
continuous, reactive complement: each robot, every tick, picks the velocity
closest to where it wants to go that is still provably collision-free for a
short time horizon — assuming every neighbour reasons the same way. That mutual
assumption (Optimal *Reciprocal* Collision Avoidance — van den Berg, Guy, Lin &
Manocha, 2011) is what removes the oscillation and jitter you get from naively
summing pairwise repulsion.

Each neighbour forbids a half-plane of velocities; the admissible set is the
intersection of those half-planes and the max-speed disc, and the best velocity
in it is a small 2-D linear program (`_linear_program2`, with the RVO2
distance-minimising fallback `_linear_program3` for jointly-infeasible crowds).
Moving robots share the avoidance equally (half each); static circular obstacles
get the full responsibility and a shorter horizon.

```python
from mrn_coord.orca import orca_velocity
v = orca_velocity(position, velocity, preferred_velocity,
                  neighbors=[(pos, vel, radius), ...],
                  obstacles=[(x, y, radius), ...],
                  radius=0.25, max_speed=1.5, time_horizon=2.5)
```

`orca_velocity` is pure and the unit tests pin its guarantee: off-axis agents
slip past each other, and even a perfectly symmetric four-way crossing — where
ORCA cannot break the tie to *converge* (a documented property; a real
controller adds a small perturbation) — still never collides. The simulator
exposes it as a benchmark policy, `mrn_sim.benchmark.orca_policy` (A* plan +
carrot + ORCA), with a tiny per-robot tie-break so it converges in practice. On
the bundled scenarios it reaches all goals collision-free and noticeably faster
and tighter than the repulsion baseline (e.g. `crossing`: makespan 12.3 s vs
18.5 s); both are guarded by the CI benchmark gate. Run the comparison:

```bash
ros2 run mrn_sim mrn_sim_bench crossing --policy orca
ros2 run mrn_sim mrn_sim_bench crossing --policy navigate
```

## Formation — Decentralized Formation Control

This module is the clearest reuse of the localization stack's output: the
cooperative graph already exchanges relative-pose constraints between agents,
and a displacement-based formation controller needs exactly that — the relative
position of each neighbor. Nothing here needs a global frame.

### The control law (`control.py`)

Each agent runs the classic displacement-based consensus law

```
u_i = gain * sum_{j in N(i)} ( r_ij - r*_ij )
```

where `r_ij = p_j - p_i` is the **measured** relative position of neighbor `j`
(what a V2V `RelativePoseConstraint` carries) and `r*_ij` is the **desired**
relative offset from the `FormationSpec`. The command depends only on relative
measurements to neighbors, so the controller is fully decentralized.
`formation_error` is the RMS of `||r_ij - r*_ij||` over the edges — zero exactly
when the shape is achieved, and invariant to a global translation (only
relative offsets are observable from relative measurements).

### The shape (`spec.py`)

`FormationSpec` holds per-agent offsets in an abstract formation frame, used
only through `desired_relative(i, j) = c_j - c_i`. Builders: `line_formation`
(evenly spaced along an axis) and `polygon_formation` (a regular polygon — an
equilateral triangle for three agents).

### Behavior

On a connected graph the law drives the agents into the desired shape. With no
leader the formation centroid is invariant (it converges in place). A `leader`
agent is commanded zero and moves on its own; the rest anchor their shape to it.
Note that tracking a constant-velocity leader leaves a **bounded steady-state
lag** — the expected behavior of a proportional controller following a ramp —
rather than zero error.

### Try it

```bash
ros2 run mrn_coord mrn_formation_demo                 # converge to a triangle
ros2 run mrn_coord mrn_formation_demo --leader 1      # anchor the shape to agent 1
```

The demo pulls three scattered agents into an equilateral triangle and prints
the formation error decaying toward zero.

### ROS node

`mrn_formation_controller` wraps the control law: it subscribes to each agent's
pose on `formation/pose/<id>` (`geometry_msgs/PoseStamped`) and, on a timer,
publishes a `geometry_msgs/Twist` velocity command on `formation/cmd_vel/<id>`
computed from the relative positions of its neighbors. The spec offsets and
edge list come from parameters; the control law is the same pure
`mrn_coord.formation.control` used in the tests.

```bash
ros2 launch mrn_coord formation_controller.launch.py   # waits for pose inputs
```

## Coverage — Cooperative Exploration & Task Allocation

Given a partially-explored map and a team of robots, decide *who explores
where*. Two stages: find the candidate targets, then assign them.

### Occupancy & frontiers (`occupancy.py`, `frontier.py`)

`OccupancyGrid` is a three-state grid — `UNKNOWN`, `FREE`, `OCCUPIED` —
buildable from text rows (`.` free, `#` occupied, `?` unknown). A **frontier
cell** is a free cell adjacent to an unknown cell: the boundary of explored
space, and where moving gains new information. `frontier_cells` lists them and
`cluster_frontiers` groups 4-connected frontiers into clusters, each with a
representative (the medoid, so the target sits inside the frontier).

### Allocation (`allocation.py`)

Travel cost is the shortest distance through known-free space
(`bfs_free_distances`, 4-connected). Two strategies assign frontier targets to
robots:

- **`greedy_auction`** — repeatedly commit the globally cheapest
  `(robot, frontier)` pair. Fast, simple, not always optimal.
- **`hungarian_assignment` / `min_cost_assignment`** — the optimal
  minimum-total-cost assignment (Kuhn–Munkres), handling rectangular cost
  matrices by transposing so rows ≤ cols. The implementation is cross-checked
  against brute-force optimal assignment in the tests.

`allocate_frontiers(grid, robot_positions, frontier_targets, method=...)` ties
them together: BFS cost from each robot to each target, then an assignment;
unreachable pairs are dropped.

### Try it

```bash
ros2 run mrn_coord mrn_coverage_demo                  # optimal (Hungarian)
ros2 run mrn_coord mrn_coverage_demo --method greedy
```

The demo builds a small map with two unknown pockets, detects and clusters the
frontiers, allocates them to two robots by travel cost, and prints the map with
each robot (`R`) and its assigned frontier target (`F`).

### ROS node

`mrn_coverage_allocator` wraps the allocator: it reads the occupancy grid (text
rows) and robot cells from parameters, detects and clusters frontiers, allocates
them, and publishes each robot's assigned frontier as a
`geometry_msgs/PointStamped` goal on `coverage/goal/<id>` (latched). Frontier
detection, clustering, and allocation are the same pure coverage core used in
the tests.

```bash
ros2 launch mrn_coord coverage_allocator.launch.py     # publishes goals
ros2 topic echo /coverage/goal/a_1
```

All three coordination modules now have both a CLI demo and a thin ROS node;
agent ids are sanitized into valid topic tokens (e.g. `1` → `a_1`).

### Driving to the goals (closing coverage → world)

`mrn_goal_follower` drives robots to their allocated frontiers: per agent it
subscribes to a `geometry_msgs/PointStamped` goal (`coverage/goal/<id>`) and the
robot's pose and steers there with the same pure-pursuit core (a single-point
path). Paired with `mrn_sim_world`, `mapf_through_sim`'s sibling
`coverage_through_sim.launch.py` (in `mrn_sim`) runs allocator → follower →
world; verified end-to-end, each robot drives to within ~0.3 m of its assigned
frontier. (This executes one allocation; iterative re-mapping as frontiers are
reached is a larger loop left for later.)

```bash
ros2 launch mrn_sim coverage_through_sim.launch.py use_rviz:=true
```

## Running the loop in ROS

The nodes above publish and subscribe, but to actually *move* something you need
a plant. `mrn_agent_sim` is a minimal single-integrator simulator: it publishes
each agent's pose on `formation/pose/<id>`, integrates the `formation/cmd_vel/<id>`
commands it receives, and publishes a `visualization_msgs/MarkerArray` on
`coordination/markers` for RViz. The integration step is the pure, CI-tested
`mrn_coord.kinematics.euler_step`.

Run it with the formation controller to close the loop entirely inside ROS:

```bash
ros2 launch mrn_coord formation_closed_loop.launch.py             # headless
ros2 launch mrn_coord formation_closed_loop.launch.py use_rviz:=true
```

The sim publishes poses, the controller answers with velocity commands, the sim
integrates them, and the three agents converge into the commanded triangle —
verified end-to-end (the converged relative offsets match the spec). This is the
stand-in plant; in a real system the poses would come from the cooperative
localization estimate instead of `mrn_agent_sim`.

## Connecting to a localization estimate

The coordination nodes act on a plain `geometry_msgs/PoseStamped` per agent
(e.g. `formation/pose/<id>`). `mrn_pose_bridge` adapts a localization estimate —
`mrn_msgs/AgentState` or `CooperativePose` — into that, so the coordination
layer can run on a live cooperative-localization estimate (from the companion
[`multirobot-localization`](https://github.com/rsasaki0109/multirobot-localization)
repo, or from `mrn_sim_world`, both of which publish `AgentState`) instead of a
simulated plant. The coupling is one-way (estimate → coordination).

## Swarm flocking

Beyond small-team coordination, `mrn_coord.flocking` scales to a swarm.
`flock_velocities` is a pure, reactive Boids step — each agent steers from only
its local neighbors via the three classic rules (separation, alignment,
cohesion) — and runs over tens to hundreds of agents.

<p align="center">
  <img src="media/swarm_demo.gif" alt="Seventy agents flock in a bounded box under separation, alignment, and cohesion" width="640">
</p>

The animation above is driven by the real `flock_velocities` rules (70 agents,
seeded, deterministic; regenerate with `python3 scripts/make_swarm_gif.py`). It
shows the same simulation foundation that runs a handful of robots scaling up to
emergent swarm behavior — separation keeps them apart, alignment turns them into
a coherent flow, cohesion holds the group together.
