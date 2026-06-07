# Plan — multirobot-navigation

Scope: **multi-robot simulation, navigation, and coordination**. A deterministic
2D world plus the planning/control/swarm algorithms that move robots through it,
all as pure, CI-tested cores with thin ROS/CLI wiring.

Cooperative **localization** is out of scope here — it lives in the companion
repo [`multirobot-localization`](https://github.com/rsasaki0109/multirobot-localization)
(rosbag-centric, real-data benchmarks). The two meet only at the message
contract (`mrn_msgs/AgentState`, `RelativePoseConstraint`): this repo's
simulator emits them; that repo consumes them.

## Packages

- `mrn_msgs` — message contracts (the interface to the localization consumer).
- `mrn_sim` — deterministic 2D world (unicycle kinematics, obstacles, collision,
  V2V/GNSS/range sensors), point-to-point navigation, and the swarm driver; a
  thin `mrn_sim_world` ROS node.
- `mrn_coord` — coordination: MAPF (CBS / prioritized), formation control,
  coverage (frontier + greedy/Hungarian), swarm flocking.
- `mrn_gazebo` — optional Gazebo (`gz sim`) adapter (requires Gazebo; not in CI).

## Done

- [x] `mrn_sim` 2D world core: unicycle kinematics, collision-aware `step`,
  sensor models, proximity queries — pure, CI-tested; ROS node emitting
  `AgentState` / `RelativePoseConstraint` (stamped) and accepting `cmd_vel`.
- [x] MAPF: space-time A*, conflict detection, Conflict-Based Search (optimal),
  prioritized planning; `mrn_mapf_demo` + `mrn_mapf_planner` node + pure-pursuit
  path follower; `mapf_through_sim` closed loop.
- [x] Formation control: displacement-based consensus over relative
  measurements; `mrn_formation_demo` + controller node + `mrn_agent_sim`
  closed loop.
- [x] Coverage: occupancy grid, frontier detection/clustering, BFS cost, greedy
  / Hungarian allocation (cross-checked vs brute force); `mrn_coverage_demo` +
  allocator + goal-follower nodes; `coverage_through_sim`.
- [x] Swarm flocking: separation / alignment / cohesion + obstacle avoidance +
  migration + predator evasion (one or many) + leader following; deterministic
  swarm-in-world driver + multi-phase mission; CI-verified.
- [x] Navigation: occupancy grid + grid A* + pure pursuit; reciprocal
  multi-robot collision avoidance; replanning around dynamic obstacles.
- [x] Optional Gazebo adapter: validated diff-drive world, `ros_gz_bridge`,
  pose→`AgentState`, multi-robot spawn + swarm controller (real machine; not CI).
- [x] CI: build + `colcon test` over all packages + coordination CLI demos.
- [x] **MAPF algorithm zoo** (`mrn_coord.mapf`): 45+ algorithms faithfully
  reproduced from their papers in pure Python, each *benchmark-gated* in
  `scripts/benchmark_gate.py` (full gate **101/101**). CBS family (CBSH, ECBS,
  EECBS, FECBS, ICBS/bypass, MA-CBS, disjoint, BCP, rectangle/corridor/mutex
  symmetry), optimal joint-space search (M\*, rM\*, EPEA\*, ICTS, Standley
  OD/ID), declarative (MDD-SAT), constructive (Push-and-Rotate/Swap, TSWAP,
  Bibox, flow, DDM), suboptimal/anytime (ECBS/BCBS, MAPF-LNS/LNS2, WHCA\*),
  LaCAM/PIBT line, assignment (CBS-TA, CBM/TAPF, flow), lifelong (RHCR, Token
  Passing, TPTS, online-LNS), execution (switchable-ADG, k-robust, TPG),
  **humanoid footstep planning (Hornung et al.) + multi-humanoid footstep MAPF
  + ZMP-preview-control walking pattern generation (Kajita et al.) + Capture
  Point push recovery (Pratt et al.) + DCM walking control (Englsberger et
  al.) + trajectory-free constrained-QP MPC walking (Wieber) + automatic
  footstep placement MPC (Herdt et al.) + closed-loop walking stabilizer by
  LIPM tracking (Kajita et al.) + ankle/hip/step push-recovery decision
  surfaces (Stephens) + N-step capturability analysis (Koolen et al.) +
  whole-body resolved momentum control (Kajita et al.)**, continuous-space
  multi-robot motion planning (**discrete RRT / dRRT over an implicit
  tensor-product roadmap, Solovey, Salzman & Halperin; asymptotically-optimal
  dRRT\* with Dijkstra-over-the-explored-graph + informed sampling, Shome et
  al.**), kinodynamic multi-robot motion planning (**Kinodynamic CBS / K-CBS:
  Dubins-car robots, a kinodynamic-RRT low level + space–time constraint tubes,
  Kottinger et al.**), path–velocity decomposition (**the classic coordination
  diagram: fix paths, schedule speed by A-star over the coordination space, Kant
  & Zucker / O'Donnell & Lozano-Pérez**), decentralized position-space collision
  avoidance (**Buffered Voronoi Cells, Zhou et al.**; **Control Barrier Function
  safety certificates, Wang/Ames/Egerstedt**), and
  the low
  levels (space-time A\*,
  SIPP/SIPPS, Multi-Label A\*). Each documented algorithm-by-algorithm with its
  honest gated result in `docs/coordination.md`.
- [x] `scripts/animate_mapf.py` — render any solver's solution (or a side-by-side
  gallery) as a GIF; the comparison visual now leads the README.
- [x] Repository published (public) with a simplified description and
  discoverability topics (`mapf`, `multi-agent-pathfinding`, `pibt`, `cbs`, …).

## Next ideas

- [x] reusable benchmark environment (`mrn_sim.benchmark`): `Scenario`
  (YAML/dict) + `run_scenario(scenario, policy) -> BenchmarkResult` with
  standard metrics (success, makespan, path length, min clearance, min
  inter-robot distance, collisions); a scenario library (`mrn_sim/scenarios/`),
  a baseline `navigate_policy`, and a `mrn_sim_bench` CLI. External
  planners/controllers plug in as a `policy(world) -> commands` callable.
- [x] standard MAPF benchmarks (MovingAI `.map`/`.scen` loader +
  `run_mapf_benchmark` + `mrn_mapf_bench` CLI; bundled example, CBS /
  prioritized). Drop in downloaded benchmark sets to compare solve-rate /
  makespan.
- [x] scenario-driven CI benchmark gate: `scripts/benchmark_gate.py` runs the
  bundled scenarios + MovingAI example and regresses their metrics against
  `benchmarks/expected_metrics/` (like the localization repo's). CI-enforced.
- [x] ORCA reciprocal local collision avoidance (`mrn_coord.orca`, RVO2-style
  2-D LP); `orca_policy` benchmark policy + `mrn_sim_bench --policy orca`;
  guarded by the gate. Resolves the repulsion baseline's oscillation/deadlock.
- Solve-rate / runtime comparison tables on full MovingAI sets (the loader is
  ready; needs the downloaded data).
- Continuous-space / kinodynamic planning beyond the grid; ORCA tie-break /
  priority schemes for the perfect-symmetry case.
- Real-robot bring-up (separate effort; the localization repo is rosbag-first).

## Growth & outreach (now that the repo is public)

The MAPF algorithm zoo is the repo's rare asset; the work now is to make its
value land in **30 seconds** and to be found. Leverage order:

- [ ] **Distribution** (highest immediate leverage; topics already set): post the
  `mapf_gallery.gif` comparison to r/robotics, Hacker News (*Show HN*), and X
  with a one-line hook ("45 MAPF algorithms, faithfully reproduced and
  benchmarked"); open a PR adding the repo to *Awesome-MAPF* / *Awesome-Robotics*.
- [x] **README refresh**: leads with the MAPF zoo — the `mapf_gallery.gif` hero,
  a 5-line `pip install` + solve quickstart, and a representative comparison
  table (algorithm | paper | one-line idea | gated result) linking the full
  paper-by-paper catalogue in `docs/coordination.md`.
- [x] **Pip-installable, ROS-free MAPF core**: root `pyproject.toml` packages
  `mrn_coord` / `mrn_coord.mapf` / `mrn_coord.lifelong` as the `mapf-zoo`
  distribution — `pip install` and solve without ROS / colcon, zero required
  deps (numpy/scipy gated behind the `[bcp]` extra via a lazy import). Verified
  in a clean venv with ROS *and* numpy unreachable; `docs/pypi.md` is the long
  description. Still TODO: publish to PyPI; ship Jupyter notebooks.
- [x] **Browser demo (Pyodide)**: [`docs/demo/`](docs/demo/) runs the real
  pure-Python solvers in the browser — pick an instance + solver, watch the
  collision-free paths animate on a canvas, zero backend. `index.html` unpacks
  the `mapf-zoo` wheel onto Pyodide's `sys.path`; `bridge.py` is the JSON glue;
  four instances (crossing / swap / doorway / ring) × seven solvers. Verified
  headlessly under the same Pyodide build via a node harness (full
  `bridge.solve` matrix — 27 solving combos + the deliberately-skipped
  `ring × M*` blow-up). Pages-ready (serve `docs/`); GitHub Pages not enabled
  here. Still TODO: per-family GIFs, more presets.
- [ ] **More visuals**: per-family GIFs (rectangle/corridor symmetry, lifelong
  warehouse throughput, execution-layer delay recovery) generated by
  `animate_mapf.py` / the existing `make_*_gif.py`.

## More MAPF reproductions (the engine of the collection)

Keep the cadence: reproduce a paper faithfully → measure honestly → gate
WIN/LOSS/equivalence → record pitfalls. Read algorithms from the source PDF via
the Read tool's `pages=` param (WebFetch can't parse compressed PDFs).

- [ ] Next paper batches (continue the faithful-reproduction + honest-gating loop;
  pick distinct paradigms not already covered — verify against existing solvers
  first to avoid near-duplicates, as the Lazy CBS investigation showed).
- [ ] Deferred deep build — **Lazy CBS**'s genuine lazy-clause-generation
  CDCL + core-guided OLL engine on tiny instances (the only part not already
  subsumed by CBSH-WDG / Standley-ID / MDD-SAT / BCP; recorded as
  subsumed-for-now in the dev notes).
- [ ] Full **MovingAI** solve-rate / runtime tables across the whole solver suite
  (the `.map`/`.scen` loader is ready; needs the downloaded data + a comparison
  harness and a results page).

## Rules

- Pure algorithm cores are ROS-free and unit-tested; ROS nodes are thin shells
  smoke-tested via launch.
- Demos are synthetic, deterministic, and reproducible; GIFs are driven by the
  real algorithms.
- State honestly what is verified vs. pending (e.g. Gazebo multi-robot runs on a
  real machine, not in the CI sandbox).
