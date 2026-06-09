# Plan — MultiRobotBattle

Scope: **multi-robot swarm battle** — two or more flocking armies that seek,
maneuver, and fight — standing on a deterministic 2D world and the full
planning / control / coordination stack that moves robots through it (including a
50-paper MAPF / motion-planning **algorithm zoo**). Everything is pure, CI-tested
algorithm cores with thin ROS / CLI wiring; every demo is deterministic and
driven by the real algorithms.

**The throughline.** The battle is not a toy renderer bolted on top — it is built
from the same primitives as the rest of the repo (decentralized Boids flocking
today; the MAPF planners, formation control, and coordination diagrams next), so
each combat behaviour is a *real algorithm*, measured the same way the solvers
are. The roadmap below is mostly about closing that loop: making the fight a
living showcase of the zoo rather than a thing beside it.

Cooperative **localization** stays out of scope — it lives in the companion repo
[`multirobot-localization`](https://github.com/rsasaki0109/multirobot-localization)
(rosbag-centric, real-data benchmarks). The two meet only at the message
contract (`mrn_msgs/AgentState`, `RelativePoseConstraint`): this repo's
simulator emits them; that repo consumes them.

## Packages

- `mrn_msgs` — message contracts (the interface to the localization consumer).
- `mrn_sim` — deterministic 2D world (unicycle kinematics, obstacles, collision,
  V2V/GNSS/range sensors), point-to-point navigation, and the swarm driver; a
  thin `mrn_sim_world` ROS node.
- `mrn_coord` — coordination: **swarm battle** (`battle.py`, `battle_morale.py`,
  `battle_charge.py`, `battle_maneuver.py`, `battle_assignment.py`,
  `battle_policy/`), swarm flocking, MAPF (CBS / prioritized + the 50-paper zoo in
  `mrn_coord.mapf`), formation control, coverage (frontier + greedy/Hungarian).
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
- [x] **Swarm battle** (`mrn_coord.battle`): two flocking armies (red vs blue)
  fight to the last robot — each robot flocks with its team, advances on its
  nearest enemy, and deals continuous damage in range; per-attacker damage makes
  **focus fire emergent**. Pure Python, deterministic, built on
  `mrn_coord.flocking`; the project's headline GIF
  (`scripts/make_battle_gif.py`), unit-tested (`test_battle.py`). Optional
  wounded-retreat (`retreat_frac`), off by default so battles stay decisive. The
  same engine drives several **kinds** of fight (`battle_scenario`): N-army
  **free-for-all** (`make_free_for_all`), **unit classes** (scout/soldier/tank/
  sniper via `CLASSES` + `make_company`, so quality-vs-quantity & combined arms
  emerge), and **terrain** (circular `obstacles`); `make_battle_gallery_gif.py`
  renders a 2×2 gallery of duel / free-for-all / quality-vs-quantity / chokepoint.
- [x] **Allied multi-army total war** — four teams on two alliances
  (`RED`+`GREEN` vs `BLUE`+`YELLOW` via `BattleConfig.alliances`); allied teams
  never fire on each other but still flock only with their own colour.
  `make_allied_armies` deploys eight echelons (infantry / tank / sniper wings ×
  two fronts). Headline scenario **`grand_alliance`**: **576 bots** on a
  140×72 competition arena with `total_war_terrain` (walls, elevation, chicanes);
  **`grand_alliance_lite`** (128 bots) for the browser demo. Tested in
  `test_battle_alliance.py`; `grand_alliance_lite` pinned in `battle_gate`.
- [x] **Morale / rout** (`mrn_coord.battle_morale`) — when a team's surviving
  fraction falls below `morale_rout_frac`, its bots flee toward the spawn flank
  and are removed off-field instead of stalling in wounded-retreat draws.
  `BattleConfig(morale=True)`; headline scenario `morale_duel` (6 tanks vs 18
  scouts); GIF `morale_rout.gif`; browser demo + `battle_gate` pin.
- [x] **ORCA / BVC charge** (`mrn_coord.battle_charge`) — MAPF-zoo collision
  avoidance as a post-steering movement filter (`charge_by_team`: `orca` /
  `bvc` / `none`). Headline scenario `orca_charge_duel`; comparison GIF
  `charge_layers.gif` (greedy vs ORCA vs BVC); `battle_gate` pins
  `orca_charge_vs_greedy` / `bvc_charge_vs_greedy`.
- [x] **Hero GIF presentation pass** — README `battle.gif` is tuned for embed
  readability and combat intensity: canvas aspect matches the arena exactly
  (820 px wide, HUD overlaid *inside* field bounds so title / casualty bars are
  never clipped); `grand_alliance` opens with front ranks within rifle range
  (≈90 tracers tick 0, ~26 KIA by tick 20) via closer wedge deployment,
  per-class DPS scaling, aggressive `count_aware` stance, and faster
  `fire_interval`; `make_battle_gif.py` defaults stride 2 / 520 ticks / 18 fps.
- [x] Navigation: occupancy grid + grid A* + pure pursuit; reciprocal
  multi-robot collision avoidance; replanning around dynamic obstacles.
- [x] Optional Gazebo adapter: validated diff-drive world, `ros_gz_bridge`,
  pose→`AgentState`, multi-robot spawn + swarm controller (real machine; not CI).
- [x] CI: build + `colcon test` over all packages + coordination CLI demos.
- [x] **MAPF algorithm zoo** (`mrn_coord.mapf`): 45+ algorithms faithfully
  reproduced from their papers in pure Python, each *benchmark-gated* in
  `scripts/benchmark_gate.py` (full gate **102/102**). CBS family (CBSH, ECBS,
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
  safety certificates, Wang/Ames/Egerstedt**), graph reconfiguration by
  minimum swap count (**Token Swapping, Yamanaka et al.**: no-blank adjacent
  token exchange, exact `n!` BFS + path=inversions / `K_n`=`n−cycles` closed
  forms), and
  the low
  levels (space-time A\*,
  SIPP/SIPPS, Multi-Label A\*). Each documented algorithm-by-algorithm with its
  honest gated result in `docs/coordination.md`.
- [x] `scripts/animate_mapf.py` — render any solver's solution (or a side-by-side
  gallery) as a GIF; the comparison visual now leads the README.
- [x] Repository published (public) with a simplified description and
  discoverability topics (`mapf`, `multi-agent-pathfinding`, `pibt`, `cbs`, …).

## Battle roadmap — where the fight goes next

The battle today is the honest minimum: decentralized flocking + advance-to-
contact + continuous in-range damage, with unit classes, N-army free-for-all,
terrain, allied fronts, morale/rout, and MAPF-zoo charge layers. It already
produces **emergent focus fire** (per-attacker damage means a locally-outnumbered
robot melts) and a clean quality-vs-quantity result (5 tanks beat 16 scouts). The
headline **`grand_alliance`** GIF now reads as total war from frame one — both
coalitions trade tracers in the centre strip immediately, casualty bars climb
inside the arena bounds, and wedge echelons push until one alliance wins. The
roadmap deepens the **combat model**, adds a **tactics / AI layer**, and — the
whole point of this repo — **drives the battle with the MAPF zoo's own planners**
so the fight becomes a showcase of the algorithms rather than a thing beside them.
Same cadence as the solvers: build → measure (win-rate / invariants) → gate →
record the honest result.

### Recently landed (combat + presentation)

| Feature | Module / scenario | GIF / gate |
| --- | --- | --- |
| Allied 576-bot total war | `grand_alliance`, `make_allied_armies` | `battle.gif` |
| Morale / rout | `battle_morale`, `morale_duel` | `morale_rout.gif`, `battle_gate` |
| ORCA / BVC charge | `battle_charge`, `orca_charge_duel` | `charge_layers.gif`, `battle_gate` |
| Hero embed polish | `make_battle_gif.py` in-bounds HUD | README @ 820 px |
| Opening-barrage tuning | per-class DPS scale + wedge deploy | ~90 shots tick 0 |

**Still open on the headline scenario:** full annihilation within the GIF window
(576-bot melees often time out on survivor count — ~130+ KIA in 400 ticks but
hundreds remain; raising lethality further vs. keeping combined-arms readability
is the trade-off). Next presentation pass: shorter "decisive wipe" variant or a
`grand_alliance_decisive` seed sweep for a cleaner victory banner.

### Combat model depth
- [x] **Line of sight & cover** — obstacles (and optionally other bodies) block
  fire: a shot lands only if the segment to the target is clear; partial cover
  scales damage down. Reuses the segment/obstacle geometry already in the sim.
- [x] **Projectiles & ballistics** — discrete shots with travel time and accuracy
  that falls off with range (vs today's instant hitscan): real misses, friendly
  fire, and splash / area effects for an "artillery" class.
  **Artillery barrage** (`artillery_barrage`) — splash rounds + ``artillery`` unit
  class — landed with ``artillery_barrage.gif``, browser demo, and ``battle_gate`` pin.
  **Fog × artillery** (`fog_artillery`) — scouts spot under fog, indirect splash —
  landed with ``fog_artillery.gif``, browser demo, and ``battle_gate`` pin.
- [x] **Morale / rout** — collapsing teams flee off-field and are removed instead
  of stalling in wounded-retreat draws (`morale_duel`, ``morale_rout.gif``,
  ``battle_gate`` pin). Module: ``mrn_coord.battle_morale`` — tracks per-team
  strength fraction, overrides steering for routed bots, removes survivors past
  the arena margin. Config: ``morale``, ``morale_rout_frac``,
  ``morale_rout_speed``, ``morale_exit_margin``.
- [ ] **Typed damage & armor** — a small rock-paper-scissors (e.g. AP vs shield)
  so composition matters beyond raw dps; plus cooldown / reload / finite ammo so
  positioning and timing count.
- [ ] **Support roles** — a medic that heals nearby allies, shields / buffs, so
  non-damage units earn their place.

### Tactics & team AI
- [x] **Formations** — line / wedge / screen / square via the existing
  `mrn_coord.formation` displacement consensus: tanks screen the front, snipers
  hold the back, the army advances as a shape instead of a blob.
- [ ] **Smarter target selection** — focus the weakest-in-range or highest-threat
  enemy (not just the nearest), and **kite** with ranged units (fire while backing
  out of melee). Both are local policies, still no central commander required.
  *(partial: ``CountAwarePolicy`` in ``mrn_coord.battle_policy`` — focus fire +
  sniper kite; ``TransformerPolicy`` distills the teacher — Phase B landed.)*
- [ ] **Per-team strategy** — a selectable commander policy (aggressive /
  defensive / flank / turtle) as a small utility or finite-state AI; *strategy-vs-
  strategy* matchups become the interesting experiment.
  *(partial: ``count_aware:aggressive`` / ``defensive`` / ``balanced`` stances;
  ``auto`` adapts to ally/enemy counts TeamHOI-style; ``tactics_by_team`` +
  ``scripts/battle_gate.py`` for strategy-vs-strategy win-rates.)*
- [x] **Win conditions beyond annihilation** — king-of-the-hill and domination
  objective modules over the same engine, with ``battle_gate`` pins, browser demo,
  and headline GIFs (``objective_duel.gif``, ``mapf_total_war.gif``).
  **Base assault** (`base_assault`) — hold the enemy HQ — landed with ``base_assault.gif``,
  browser demo, and ``battle_gate`` pin.
  **Escort** (`escort`) — push the payload to the enemy HQ — landed with ``escort.gif``,
  browser demo, and ``battle_gate`` pin.
- [x] **Fog of war** — units only sense enemies within range (reuse the `mrn_sim`
  V2V / range sensor models), so scouting, ambush, and surprise become real.
  **Fog ambush** (`fog_ambush`) — limited sensing + scout-led contact — landed with
  ``fog_ambush.gif``, browser demo, and ``battle_gate`` pin.

### Drive the battle with the MAPF zoo (the synergy)
The repo's rare asset is 50 faithfully-reproduced, benchmark-gated planners. The
battle is exactly where they should *do work*:
- [ ] **Planned maneuver** — route squads to objectives with grid A\* / CBS /
  PIBT instead of greedy pursuit, so armies flank around terrain intelligently and
  move collision-free; use the coordination diagram / velocity scheduling to push
  a column through a chokepoint without jamming.
  *(partial: ``mrn_coord.battle_maneuver`` — ``maneuver=astar|prioritized|cbs|pibt``,
  per-team via ``maneuver_by_team``; headline matchups in ``battle_gate``.)*
- [ ] **Optimal target assignment** — Hungarian / CBS-TA to assign shooters to
  targets (who engages whom) for a measurably better volley than greedy-nearest.
  *(partial: ``mrn_coord.battle_assignment`` — ``hungarian`` combat matching and
  ``cbs_ta`` Murty-on-BFS path-aware matching; ``battle_gate`` chokepoint
  ``cbs_ta`` vs ``hungarian``; full joint CBS-TA path search still TODO for
  small-team maneuver coupling.)*
- [x] **Collision-free charges** — ORCA / Buffered Voronoi Cells as a post-
  steering filter on the chokepoint (`orca_charge_duel`, ``charge_layers.gif``,
  ``battle_gate`` pins ``orca_charge_vs_greedy`` / ``bvc_charge_vs_greedy``).
  Module: ``mrn_coord.battle_charge`` — ``apply_charge`` dispatches to
  ``mrn_coord.orca.orca_velocity`` or ``mrn_coord.mapf.bvc.step_bvc`` per team;
  ``charge_headline_duel`` runs greedy vs ORCA vs BVC side-by-side for the GIF.
- [ ] **Switchable-ADG execution** — when maneuver is pre-planned and someone is
  delayed, gate execution on the coordination graph.
- [x] **The headline demo** — swap *only* the movement layer (greedy ↔ A\* ↔ CBS)
  and show the win-rate difference. That single experiment turns the zoo from a
  museum into the battle's brain, and is the strongest story this repo can tell.
  *(``maneuver_headline_duel`` + ``scripts/make_maneuver_gif.py`` →
  ``docs/media/maneuver_layers.gif`` 2×2 grid; ``battle_gate`` pins chokepoint
  A* / prioritized vs greedy at ~67% red win-rate; CBS shown in GIF but skipped
  in CI gate — joint replanning is too slow.)*

### Scale, balance & evaluation
- [ ] **Balance harness + `battle_gate`** — run K seeds of each matchup, report
  win-rate / ELO, and tune `CLASSES` toward ~50/50 so no class is strictly
  dominant. Gate it like the MAPF suite: regress win-rates and invariants
  (collision-free where claimed, decisive outcome, bot-count conserved) against
  `benchmarks/expected_metrics/`, so balance is *measured*, not vibes.
  *(partial: ``scripts/battle_gate.py`` + ``benchmarks/expected_metrics/battle_gate.json``
  — decisive rates + red win-rates for count_aware/transformer vs nearest.)*
- [ ] **Tournaments** — round-robin / bracket between compositions or strategies,
  an ELO ladder, and a small results page.
- [ ] **Scale to hundreds** — spatial hashing for the O(n²) neighbour / nearest-
  enemy queries so battles of 100s of robots stay fast *and* deterministic.
  *(partial: ``mrn_coord.spatial_hash`` + ``make_grand_army`` / ``kingdom``
  scenario — 80 vs 80 line clash on a 100×56 field; ``make_kingdom_gif.py``.
  **576-bot ``grand_alliance``** runs with spatial hash enabled
  (``spatial_min_bots=24``); sim ~5 min at full fidelity, GIF subsamples with
  ``frame_stride``.)*
- [ ] **Auto-balance / find-the-meta** — hill-climb / CMA-ES over class stats or
  steering weights against a fixed opponent to discover the strongest composition
  (optional; a pure-Python optimizer, no heavyweight deps).

### Presentation & reach
- [ ] **Battle in the browser** — extend the Pyodide demo (`docs/demo/`): pick two
  armies, press go, watch them fight on the canvas. The battle is pure-Python and
  already runs under Pyodide.
  *(partial: ``docs/demo/battle.html`` + ``battle_bridge.py`` — hill / domination /
  MAPF total-war dual panel, allied fronts, kingdom lite, duel / chokepoint /
  maneuver / MAPF stack, **morale_duel**, **orca_charge_duel**; ``pages.yaml``
  deploys ``docs/`` on push to ``main``.)*
- [ ] **Battle in the sim / Gazebo** — drive the real diff-drive robots as
  combatants through the existing `mrn_sim` / `mrn_gazebo` wiring (closed-loop,
  real-machine — not CI).
- [x] **Hero GIF readability & intensity** — the README lead visual must read at
  embed size and feel like a fight from frame one:
  - **Layout** — ``make_battle_gif.py`` canvas = arena aspect ratio; ``xlim`` /
    ``ylim`` locked to ``(0, width) × (0, height)``; title + western/eastern
    casualty bars + KIA counter drawn as transparent overlays *inside* the field
    (``pad_inches=0``); output width pinned to 820 px to match README embed.
  - **Scenario tuning** — ``grand_alliance`` deploys eight wedge echelons with
    ~1 m front gap so both alliances are in rifle range on tick 0; per-class DPS
    scaled in-scenario (``cfg.dps`` does not override ``CLASSES`` stats);
    ``count_aware:aggressive``, ``w_pursue=3.35``, ``w_retreat=0.85``,
    ``fire_interval=0.028``, tighter spacing (``dense=1.52``). Measured opening:
    ~90 tracers tick 0, ~26 KIA tick 20, ~74 KIA tick 76 (stride 2).
  - **Render defaults** — ``make_battle_gif.py``: stride 2, max-ticks 520, 18 fps
    for a snappier loop without losing the central brawl.
- [ ] **More GIFs + a one-line hook** — each new feature gets a GIF driven by the
  real engine, plus a short "how it works" writeup for sharing.
  *(partial: ``maneuver_duel.gif`` + ``mapf_stack_duel.gif`` in README;
  ``make_mapf_stack_gif.py`` for assignment+maneuver stack; **``morale_rout.gif``**
  + **``charge_layers.gif``** for the latest combat-model landings;
  ``docs/media/README.md`` catalogues all generators.)*

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

The **swarm battle** is the 5-second hook; the **MAPF zoo** is the rare technical
asset underneath. The work now is to land both in **30 seconds** and to be found.
Leverage order:

- [ ] **Distribution** (highest immediate leverage; topics already set): lead with
  the `battle.gif` / `maneuver_layers.gif` eye-catcher (r/robotics, Hacker News
  *Show HN*, X) and the one-line hook ("swarm robot battles, driven by a 50-paper
  MAPF zoo"), then land the substance — the benchmarked solver collection; open a
  PR adding the repo to *Awesome-MAPF* / *Awesome-Robotics*.
  *(draft posts + links in ``docs/distribution/README.md``; hero GIF now opens
  with immediate tracer barrage and in-bounds HUD for social embeds.)*
- [x] **README refresh**: leads with the MAPF zoo — the `mapf_gallery.gif` hero,
  a 5-line `pip install` + solve quickstart, and a representative comparison
  table (algorithm | paper | one-line idea | gated result) linking the full
  paper-by-paper catalogue in `docs/coordination.md`.
- [x] **Pip-installable, ROS-free MAPF core**: root `pyproject.toml` packages
  `mrn_coord` / `mrn_coord.mapf` / `mrn_coord.lifelong` as the `mapf-zoo`
  distribution — `pip install` and solve without ROS / colcon, zero required
  deps (numpy/scipy gated behind the `[bcp]` extra via a lazy import). Verified
  in a clean venv with ROS *and* numpy unreachable; `docs/pypi.md` is the long
  description. PyPI publish is **not planned** — install from git or the checked-in
  wheel under `docs/demo/`. Still TODO: Jupyter notebooks.
- [x] **Browser demo (Pyodide)**: [`docs/demo/`](docs/demo/) runs the real
  pure-Python solvers in the browser — pick an instance + solver, watch the
  collision-free paths animate on a canvas, zero backend. `index.html` unpacks
  the `mapf-zoo` wheel onto Pyodide's `sys.path`; `bridge.py` is the JSON glue;
  four instances (crossing / swap / doorway / ring) × seven solvers. Verified
  headlessly under the same Pyodide build via a node harness (full
  `bridge.solve` matrix — 27 solving combos + the deliberately-skipped
  `ring × M*` blow-up). Live on GitHub Pages via ``.github/workflows/pages.yaml``
  → [rsasaki0109.github.io/MultiRobotBattle](https://rsasaki0109.github.io/MultiRobotBattle/).
  Still TODO: per-family GIFs, more presets.
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
