# multirobot-navigation

<p align="center">
  <img src="docs/media/gazebo_demo.gif" alt="Three robots cross a 3D Gazebo arena of cylindrical obstacles via the repo's A* grid planning, pure pursuit, and reciprocal avoidance, each sweeping a 360-degree LiDAR whose returns trace the obstacles" width="720">
</p>

<p align="center">
  <em>Three robots cross a 3D Gazebo arena — A* planning, pure pursuit, and reciprocal avoidance, all driven by the real <code>mrn_coord</code>/<code>mrn_sim</code> code, each sweeping a 360° LiDAR. Rendered fully offscreen on the GPU; regenerate with <code>scripts/record_gazebo_gif.py</code>.</em>
</p>

[![build-jazzy](https://github.com/rsasaki0109/multirobot-navigation/actions/workflows/build_jazzy.yaml/badge.svg)](https://github.com/rsasaki0109/multirobot-navigation/actions/workflows/build_jazzy.yaml)
[![docs](https://github.com/rsasaki0109/multirobot-navigation/actions/workflows/docs.yaml/badge.svg)](https://github.com/rsasaki0109/multirobot-navigation/actions/workflows/docs.yaml)

ROS 2-native **multi-robot simulation, navigation, and coordination** — a
deterministic, pure-core, CI-tested stack for developing and benchmarking
multi-robot motion algorithms without hardware.

> **Localization lives in a companion repo:** cooperative multi-agent
> localization (rosbag-centric, real-data benchmarks on UTIAS MR.CLAM / KITTI)
> is [**multirobot-localization**](https://github.com/rsasaki0109/multirobot-localization).
> This repo answers *how the robots move*; that one answers *where they are*.
> They meet at the message contract (`mrn_msgs/AgentState`,
> `RelativePoseConstraint`): the simulator here emits them, that repo consumes
> them.

## What It Is

A 2D, deterministic multi-robot world plus the coordination and navigation that
moves robots through it. Every layer is a pure, ROS-free algorithm core
unit-tested in CI, with thin ROS/CLI wiring on top.

- **Simulation** (`mrn_sim`) — a deterministic 2D world: unicycle kinematics,
  circular obstacles with collision, and V2V / GNSS / range-bearing sensor
  models. It emits the localization message contract and accepts `cmd_vel`, so
  it is the plant the rest of the stack drives. An optional **Gazebo** adapter
  (`mrn_gazebo`) runs the same contract on a 3D physics world.
- **Navigation** (`mrn_sim.navigate`, `mrn_sim.kinodynamic`) — point-to-point
  navigation: occupancy grid from the obstacles, grid A* planning, pure-pursuit
  following, with **reciprocal multi-robot collision avoidance** and
  **replanning around dynamic obstacles** — plus a continuous-space
  **Hybrid A\*** kinodynamic planner (bounded turning radius, Dubins curves +
  analytic expansion) for smooth, feasibly-followable paths, **DWA** /
  **MPC (iLQR)** optimizing local controllers for accel-limited tracking, a
  **Control Barrier Function** QP safety filter for provable collision-free
  steering, and a **certified body-true safety shield** whose braking speed cap
  keeps the robot body — not a look-ahead point — collision-free under the accel
  limit, even against moving obstacles, and **reciprocally** for several shielded
  robots in adversarial mutual pursuit with no shared coordination
  (`scripts/certify_shield.py`).
- **Coordination** (`mrn_coord`) — multi-agent path finding (optimal
  Conflict-Based Search, **bounded-suboptimal ECBS**, **complete satisficing
  LaCAM**, and **anytime MAPF-LNS** that scale further, prioritized planning,
  all over a **space-time A\*** or drop-in **SIPP** safe-interval low level,
  plus **lifelong / online MAPF** stepped by
  **PIBT** with **auction / Hungarian** task allocation for warehouse-style
  endless-task throughput), **ORCA** reciprocal
  local collision avoidance, decentralized formation control, cooperative
  coverage (frontier + greedy/Hungarian allocation), and swarm flocking (Boids:
  separation / alignment / cohesion + obstacle avoidance + migration + predator
  evasion + leader following).
- **Benchmark environment** (`mrn_sim.benchmark`) — plug your own multi-robot
  policy into a `Scenario` and get comparable metrics (success, makespan, path
  length, clearance, inter-robot distance, collisions). Five baseline policies
  ship for comparison — grid A* + pursuit, **Hybrid A\*** kinodynamic, **DWA**
  local control, **MPC** (iLQR receding-horizon optimization, space-time
  avoidance), and **ORCA** — and
  and an **end-to-end MAPF executor** (`mrn_sim.mapf_exec`) runs a discrete
  grid plan in the continuous world — exposing where the discrete guarantee
  breaks down and bridging it with a Temporal-Plan-Graph schedule — while a
  **bodied-AMR executor** (`mrn_sim.amr_footprint`) replays the same plan as a
  rectangular differential-drive robot, surfacing the turning cost and the aisle
  width below which the footprint overlaps where the point plan called it safe —
  `scripts/compare_planners.py` tabulates them all across the bundled scenarios
  ([`benchmarks/comparison.md`](benchmarks/comparison.md)). `ros2 run mrn_sim
  mrn_sim_bench crossing` runs a bundled scenario with a baseline policy. MAPF
  also loads the standard **MovingAI** `.map`/`.scen` format
  (`ros2 run mrn_coord mrn_mapf_bench`), so the planners can be evaluated on the
  community benchmark suite.

## Architecture

```
            ┌──────────── mrn_sim — the deterministic world ───────────┐
            │  unicycle kinematics · obstacles · collision · sensors    │
   cmd_vel ─▶  navigate (A* + pursuit + avoidance) / swarm / coordination│
            │  ─▶ AgentState · RelativePoseConstraint · ground truth     │
            └───────────────────────────────┬───────────────────────────┘
                                             │ message contract (mrn_msgs)
                          ▼                                ▼
              localization consumer                  Gazebo (mrn_gazebo)
              (multirobot-localization repo)         3D physics, same contract
```

The layers connect only through message contracts and pure interfaces, so each
is testable and replaceable in isolation. The simulator emits exactly what a
cooperative-localization consumer (the companion repo) ingests.

## Demos

All animations are driven by the real algorithms (no hand-drawn paths) and are
deterministic; regenerate with the matching `scripts/make_*_gif.py`.

**Coordination** — MAPF (Conflict-Based Search / prioritized), formation
control, frontier coverage. Each has a CLI demo (`mrn_mapf_demo`,
`mrn_formation_demo`, `mrn_coverage_demo`) and a thin ROS node; the top GIF
shows CBS + formation.

**Simulation & swarm** — the `mrn_sim` 2D world and Boids flocking, the same
foundation from a handful of robots to a swarm:

<p align="center">
  <img src="docs/media/sim_demo.gif" alt="Robots roam a 2D world with obstacles, exchanging V2V links" width="410">
  <img src="docs/media/swarm_demo.gif" alt="Seventy agents flock via separation, alignment, and cohesion" width="410">
</p>

Flocking *through* the collision-aware world — migrate to a goal, flee a
predator, and a multi-phase mission (regroup → migrate → evade → reach):

<p align="center">
  <img src="docs/media/swarm_sim_demo.gif" alt="A flock migrating to a goal around obstacles" width="270">
  <img src="docs/media/predator_demo.gif" alt="A flock fleeing a pursuing predator" width="270">
  <img src="docs/media/mission_demo.gif" alt="A swarm carrying out a multi-phase mission" width="270">
</p>

**Navigation** — grid A* plan + pure-pursuit follow to a goal, with reciprocal
multi-robot avoidance and replanning around a moving obstacle:

<p align="center">
  <img src="docs/media/nav_demo.gif" alt="Robots planning A* paths around obstacles to their goals" width="270">
  <img src="docs/media/recip_nav_demo.gif" alt="Robots navigating to crossing goals while avoiding each other" width="270">
  <img src="docs/media/replan_demo.gif" alt="A robot replanning around a moving obstacle" width="270">
</p>

**ORCA** — Optimal Reciprocal Collision Avoidance: two crowds walk straight at
each other and pass through, collision-free, each picking the velocity closest
to its goal that stays provably safe (`mrn_coord.orca`, regenerate with
`scripts/make_orca_gif.py`). Our port is checked against the reference RVO2
library — same scenarios, same velocity to ~1e-5
([`benchmarks/orca_rvo2.md`](benchmarks/orca_rvo2.md)):

<p align="center">
  <img src="docs/media/orca_demo.gif" alt="Two crowds of agents walk into each other and pass through collision-free via ORCA reciprocal avoidance" width="560">
</p>

**Warehouse AMR fleet** — lifelong (online) MAPF: a fleet of autonomous mobile
robots takes an endless stream of pickup/dropoff tasks through a shelf-and-aisle
warehouse, stepped collision-free by **PIBT** with cost-aware task allocation.
The running counter tracks **throughput** (tasks served per timestep) — the
metric a real fleet is judged on (`mrn_coord.lifelong`, regenerate with
`scripts/make_warehouse_gif.py`):

<p align="center">
  <img src="docs/media/warehouse_demo.gif" alt="Twelve autonomous mobile robots stream endless pickup/dropoff tasks through a shelf-and-aisle warehouse, collision-free via PIBT, while a counter shows the throughput per timestep" width="560">
</p>

The same engine scales to a full **fleet system** — **100 AMRs** working a
six-by-nine shelf floor, every per-timestep move still the collision-free PIBT
configuration, the counter climbing past **25 tasks/step**
(`scripts/make_warehouse_gif.py --preset fleet`):

<p align="center">
  <img src="docs/media/fleet_demo.gif" alt="A hundred autonomous mobile robots swarm a large shelf-and-aisle warehouse floor on a lifelong-MAPF schedule, collision-free via PIBT, the counter showing over twenty-five tasks served per timestep" width="640">
</p>

**3D physics — Gazebo** — the demo at the top of this README runs in the
`mrn_gazebo` (`gz sim`, Harmonic) **3D** world: three robots cross the obstacle
arena under the repo's own A\* grid planning + pure-pursuit + reciprocal
avoidance, driven over `cmd_vel`. Each carries a **360° LiDAR** whose live returns
are overlaid on the render, so you can watch the lasers trace the obstacles and
the other robots. It is rendered and recorded **fully offscreen on the GPU** (no
GUI, no desktop window) by `scripts/record_gazebo_gif.py` — the 3D counterpart to
the deterministic 2D demos, driven by the same algorithms.

The same offscreen seam runs the other layers in 3D too — **ORCA** crowds passing
through each other, **Boids** swarming past obstacles (with the flock's LiDAR
point cloud), **CBS + formation** funneling through a doorway, and a **warehouse
AMR fleet** working a lifelong-MAPF schedule around the racking — each driven by
the matching `mrn_coord` algorithm and sharing one recording harness
(`scripts/_gz_record.py`, `scripts/record_gazebo_{orca,swarm,coord,warehouse}_gif.py`):

<p align="center">
  <img src="docs/media/gazebo_orca_demo.gif" alt="Two streams of robots in a 3D Gazebo world pass through each other collision-free via ORCA" width="265">
  <img src="docs/media/gazebo_swarm_demo.gif" alt="Twelve robots flock through a 3D Gazebo arena past obstacles via Boids rules, their LiDAR returns drawn as a point cloud" width="265">
  <img src="docs/media/gazebo_coord_demo.gif" alt="Three robots funnel through a doorway via Conflict-Based Search then assemble a formation in 3D Gazebo, their LiDAR tracing the wall" width="265">
  <img src="docs/media/gazebo_warehouse_demo.gif" alt="Six autonomous mobile robots work a 3D Gazebo shelf-and-aisle warehouse on a lifelong-MAPF schedule, their 360-degree LiDAR tracing the racking" width="265">
</p>

## Packages

| Package | Role |
| --- | --- |
| `mrn_msgs` | message contracts (agent state, V2V relative-pose constraints, …) — the interface to the localization consumer |
| `mrn_sim` | deterministic 2D world (kinematics, obstacles, sensors), point-to-point navigation, and the swarm driver; a `mrn_sim_world` ROS node |
| `mrn_coord` | coordination: MAPF (CBS / prioritized), formation control, coverage, swarm flocking — pure cores, CLI demos, and thin ROS nodes |
| `mrn_gazebo` | optional Gazebo (`gz sim`) adapter: bridges model poses into `AgentState` so a 3D physics world can be the plant (requires Gazebo; not in CI) |

## Quick Start

Build with ROS 2 Jazzy:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Run the coordination CLI demos (pure, no ROS daemon):

```bash
ros2 run mrn_coord mrn_mapf_demo            # Conflict-Based Search
ros2 run mrn_coord mrn_formation_demo       # formation control
ros2 run mrn_coord mrn_coverage_demo        # frontier allocation
```

Drive a planned path through the simulator, or close a formation loop in ROS:

```bash
ros2 launch mrn_sim mapf_through_sim.launch.py use_rviz:=true
ros2 launch mrn_coord formation_closed_loop.launch.py use_rviz:=true
```

Regenerate any demo GIF: `python3 scripts/make_<name>_gif.py`.

See [docs/simulation.md](docs/simulation.md), [docs/coordination.md](docs/coordination.md),
and [docs/gazebo.md](docs/gazebo.md).

## Continuous Integration

Every push builds the workspace and runs `colcon test` over all packages (the
pure algorithm cores), then exercises the coordination CLI demos end-to-end. A
final **benchmark gate** (`scripts/benchmark_gate.py`) runs the bundled
scenarios and the MovingAI MAPF example and compares their metrics against
checked-in expectations in `benchmarks/expected_metrics/`, so a regression that
drops a goal, introduces a collision, or worsens a makespan / sum-of-costs fails
the build — the benchmarks are a guarded contract, not decoration.

Three further jobs check our implementations against the **reference libraries**
they reproduce, each built from source so the core build never depends on it: our
ORCA against [RVO2](https://github.com/snape/RVO2) (same velocity to ~1e-5,
[`benchmarks/orca_rvo2.md`](benchmarks/orca_rvo2.md)); our MAPF search against
[libMultiRobotPlanning](https://github.com/whoenig/libMultiRobotPlanning) — CBS
reproducing its identical optimal sum-of-costs
([`benchmarks/mapf_libmrp.md`](benchmarks/mapf_libmrp.md)) and ECBS honoring the
same `w·optimal` suboptimality bound
([`benchmarks/ecbs_libmrp.md`](benchmarks/ecbs_libmrp.md)); and the **PIBT** core
that steps the warehouse/fleet demos against the paper author's own
[`pypibt`](https://github.com/Kei18/pypibt) — every configuration we emit judged
collision-free by the *reference's own* validator, over the full lifelong run
([`benchmarks/pibt_pypibt.md`](benchmarks/pibt_pypibt.md)). "Faithful port",
"optimal solver", "bounded-suboptimal", and "collision-free PIBT" are measured
contracts, not claims.

## License

Apache-2.0.
