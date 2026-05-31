# Simulation (`mrn_sim`)

<p align="center">
  <img src="media/sim_demo.gif" alt="Three robots roam a bounded 2D world with circular obstacles under unicycle kinematics, exchanging V2V links when in range" width="640">
</p>

<p align="center">
  <em>Driven by the real <code>mrn_sim</code> step (unicycle kinematics + collision); the controller in the demo is a deterministic potential-field waypoint seeker. Regenerate with <code>python3 scripts/make_sim_gif.py</code>.</em>
</p>

`mrn_sim` is the foundation for simulation-based multirobot work: a **true world
model** with real robot states, kinematics, obstacles, and sensor models. Like
the rest of the project it is pure and ROS-free at the core, unit-tested in CI.

The point of a true world model is that it is a **plug-in test-bed**: the
coordination layer's velocity commands drive the robots, and the (noisy) sensor
messages it emits are exactly what a cooperative-localization consumer ingests.

## Benchmark environment (`benchmark.py`)

The most reusable face of `mrn_sim`: a small environment others can plug their
own multi-robot algorithm into and get comparable numbers.

- **`Scenario`** — a declarative spec (world size, obstacles, robots, goals),
  built in Python or loaded from YAML (`load_scenario`). A library lives in
  `mrn_sim/scenarios/` (`around_obstacle`, `crossing`, `doorway`).
- **a policy** — any callable `policy(world) -> {robot_id: (v, omega)}` (your
  planner/controller). Nothing else about it is assumed.
- **`run_scenario(scenario, policy)`** — runs the closed loop on the
  collision-aware world and returns a `BenchmarkResult` with standard,
  reproducible metrics: success, makespan, path length, min obstacle clearance,
  **min inter-robot distance** (collision-freeness), and a collision count.

```bash
ros2 run mrn_sim mrn_sim_bench crossing      # built-in scenario + default policy
ros2 run mrn_sim mrn_sim_bench path/to/your_scenario.yaml
```

```python
from mrn_sim.benchmark import Scenario, run_scenario
def my_policy(world):
    return {rid: (1.0, 0.0) for rid in world.robots}   # your algorithm here
result = run_scenario(Scenario.from_dict(spec), my_policy)
print(result.as_dict())
```

The built-in `navigate_policy` (A* + pursuit + reciprocal avoidance) is a
turnkey baseline; on the bundled scenarios it solves all goals collision-free
(e.g. `crossing`: 3/3 reached, min inter-robot distance ≈ 1.4 m, 0 collisions).
Because it is pure and deterministic, a benchmark result is reproducible and
CI-checkable.

### Regression gate

`scripts/benchmark_gate.py` makes that reproducibility a guarded contract. It
runs the bundled scenarios (with `navigate_policy`) and the MovingAI example
(CBS / prioritized), then compares every metric against the checked-in
expectations in `benchmarks/expected_metrics/` — discrete metrics (success,
collisions, goals reached, makespan steps, sum-of-costs) exactly, floats within
a small tolerance.

```bash
python3 scripts/benchmark_gate.py            # check — exits non-zero on a regression
python3 scripts/benchmark_gate.py --update   # rewrite the expectations after an intended change
```

CI runs the gate on every push, so a change that drops a goal, introduces a
collision, or worsens a makespan / sum-of-costs fails the build. When a change
*intentionally* moves the numbers, regenerate with `--update` and commit the
new expectations alongside it.

## Kinematics (`kinematics.py`)

A pose is `(x, y, theta)`. `unicycle_step(pose, v, omega, dt)` advances a
differential-drive robot, integrating with the midpoint heading so a
simultaneous turn-and-drive is clean and pure straight-line motion falls out
when `omega == 0`. `normalize_angle` wraps to `(-pi, pi]`.

## World (`world.py`)

`World` holds the **true** state of every `Robot` (pose + radius), a list of
circular `Obstacle`s, and rectangular bounds. `step(world, commands, dt)`
advances each robot by its `(v, omega)` command and is **collision-aware**: a
proposed pose that leaves the bounds or enters an obstacle is rejected (the
robot holds its position for that step, though an in-place turn still applies).
It returns a new `World` — deterministic, no hidden state.

## Sensors (`sensors.py`)

Geometric sensor models over the true state, producing the measurements the
localization stack consumes. The geometry is pure and noiseless (so it is
exactly testable); `add_gaussian_noise(value, sigma, rng)` layers reproducible
noise on top using a caller-supplied `random.Random`.

- `range_bearing(observer, target_xy)` — range and body-frame bearing (a
  range-bearing radio, like the UWB constraint source).
- `relative_pose_body(observer, target)` — the full SE(2) relative pose of
  another robot in the observer's body frame (what a V2V
  `RelativePoseConstraint` carries).
- `gnss_observation(pose)` — the absolute `(x, y)` of a robot (a GNSS fix).

## ROS node

`mrn_sim_world` wraps the world as a ROS node. It holds the `World`, integrates
the per-robot `geometry_msgs/Twist` commands it receives (`v = linear.x`,
`omega = angular.z`), and publishes:

- `/<id>/mrn/agent_state` (`mrn_msgs/AgentState`) — the per-agent estimate
  (true pose + reproducible GNSS-like noise), which the localization stack and
  `mrn_pose_bridge` already consume;
- `/<id>/ground_truth/pose` (`geometry_msgs/PoseStamped`) — noiseless truth;
- `sim/markers` (`visualization_msgs/MarkerArray`) — robots, obstacles, and
  in-range V2V links for RViz.

```bash
ros2 launch mrn_sim sim_world.launch.py                 # robots stand still
ros2 launch mrn_sim sim_world.launch.py use_rviz:=true  # watch it in RViz
ros2 topic pub /robot_1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 1.0}}"
```

It also emits **V2V constraints**: for each in-range directed pair it publishes
a `mrn_msgs/RelativePoseConstraint` on `/<id>/mrn/relative_constraints`, built
from the noisy relative-pose observation (`relative_pose_observation`) with a
covariance from the configured sigmas and `source_type = SOURCE_FAKE_GROUND_TRUTH`
(honest: derived from the sim's truth, not a real sensor). These feed the
cooperative-localization graph directly, and are verified to pass
`constraint_gate` — the project's "a constraint source is correct iff its output
passes the gate" rule — in the test suite.

Because it *emits* the localization messages (estimate + V2V constraints) and
*accepts* velocity commands, it is the plant that closes the loop. The
world/proximity/sensor math is the pure, CI-tested core; this node is the thin
shell (agent ids are sanitized into valid topic tokens).

Note the velocity contract: `mrn_sim` is a **unicycle** (`v`, `omega`), whereas
the existing formation controller emits a **holonomic** velocity vector. Closing
a coordination loop *through this world* therefore wants a unicycle-aware
controller (e.g. a path follower for the MAPF planner) — the next step below.
The holonomic loop is already demonstrated with `mrn_coord`'s `mrn_agent_sim`.

## Feeding a localization consumer

The world emits `AgentState` (true pose + reproducible GNSS-like noise; a
`degraded_agents` parameter simulates a GNSS outage with a large position sigma
and `STATUS_DEGRADED`) and, for in-range pairs, V2V `RelativePoseConstraint`
(see "V2V constraints" above). The agent-state messages are stamped with a TTL
so downstream freshness gates accept them — a simulator must emit valid,
non-expired messages.

These are exactly the messages a **cooperative-localization consumer** ingests.
That consumer is the companion repo
[`multirobot-localization`](https://github.com/rsasaki0109/multirobot-localization)
(rosbag-centric, real-data benchmarks); point it at the sim's topics — or record
a bag of them — to run cooperative localization on simulated data. This repo's
scope ends at emitting the contract.

## Swarm in the world

<p align="center">
  <img src="media/swarm_sim_demo.gif" alt="A flock of differential-drive robots flowing around circular obstacles in a bounded world" width="640">
</p>

`mrn_sim.swarm.flock_in_world` flocks a swarm *through this world*: it combines
`mrn_coord`'s Boids (`flock_velocities`), obstacle avoidance
(`obstacle_avoidance`), and `velocity_to_unicycle` with the collision-aware
`world.step`, advancing all robots one deterministic tick. It is the **verifiable
twin of the Gazebo swarm** — the same control loop (Boids → unicycle command →
world step), but pure and deterministic, so CI can assert end-to-end properties:
the run is reproducible, every robot stays in bounds and never enters an
obstacle, and the flock actually moves. Pass a `goal` (the `goal_seek` migration
term) and the flock travels there as a group, flowing around the obstacles —
verified by the flock centroid closing most of the distance to the goal.
`scripts/make_swarm_sim_gif.py` renders the migrating flock (above); regenerate
with `python3 scripts/make_swarm_sim_gif.py`.

Pass a `predator` `(x, y)` and the flock flees it (`predator_evasion`): a strong,
ranged outward push. `scripts/make_predator_gif.py` renders a pursuer chasing
the flock's centroid while the robots scatter away from it and around the
obstacles — verified deterministically (the flock's mean distance from the
predator grows while it stays in bounds).

<p align="center">
  <img src="media/predator_demo.gif" alt="A flock of robots fleeing a pursuing predator while avoiding obstacles" width="640">
</p>

The terms compose into a small **mission** (`scripts/make_mission_gif.py`):
scattered robots regroup, migrate through a sequence of waypoints across the
obstacle field, scatter when a predator lunges in, then recover and reach the
final goal. A `leader` index (followers steer to that robot) and multiple
`predators` are also supported. The mission is verified deterministically — the
flock centroid completes the waypoints and reaches the final goal.

<p align="center">
  <img src="media/mission_demo.gif" alt="A swarm carrying out a multi-phase mission: regroup, migrate via waypoints, evade a predator, reach the goal" width="640">
</p>

## Point-to-point navigation

`mrn_sim.navigate` is the classic single-robot navigation pipeline, assembled
from pieces already in the repo: `occupancy_from_world` discretizes the world's
circular obstacles into an inflated occupancy grid, `plan_world_path` plans a
shortest path on it with the MAPF grid A* (`plan_path` with no constraints) and
returns world waypoints, and the pure-pursuit follower drives the unicycle robot
along them through the collision-aware `world.step`. Plan → follow → arrive,
around the obstacles. Verified deterministically (the robot reaches its goal and
never enters an obstacle); `scripts/make_nav_gif.py` shows several robots each
navigating to its own goal.

<p align="center">
  <img src="media/nav_demo.gif" alt="Four robots planning A* paths around obstacles and following them to their goals" width="640">
</p>

`navigate_step` adds **reciprocal collision avoidance** for multi-robot
navigation: each robot is pulled toward the carrot on its own path (public
`carrot_point`) while being pushed away from obstacles *and from the other
robots* (`mutual_avoidance`), the combined velocity realized as a unicycle
command. So independent navigators heading to crossing goals sidestep one
another instead of passing through — verified deterministically (all reach their
goals and never come within two robot radii). Reactive avoidance is collision-
free but not deadlock-free, so a symmetric all-cross chokepoint can still stall;
`scripts/make_recip_nav_gif.py` shows lanes plus counter-flow robots weaving past
each other and the obstacles.

<p align="center">
  <img src="media/recip_nav_demo.gif" alt="Robots navigating to crossing goals while avoiding each other and the obstacles" width="640">
</p>

Navigation also handles **dynamic obstacles by replanning**: `path_blocked`
checks whether a moving obstacle has invalidated the current path, and if so the
robot replans from its current pose against the obstacles' current positions
(`plan_world_path`). `scripts/make_replan_gif.py` shows a robot whose straight
path is cut off by a sliding obstacle — it detects the block and routes around
it to the goal (verified: it reaches the goal, replans at least once, and never
enters the obstacle).

<p align="center">
  <img src="media/replan_demo.gif" alt="A robot replanning its path around a moving obstacle to reach its goal" width="640">
</p>

## Roadmap

This is the world core, its ROS node, the localization integration, and a
verifiable swarm. Planned next:

- a unicycle path-follower so a MAPF plan can be driven through this world,
  closing world → localization → planning → world in ROS;
- swarm-scale runs (tens to hundreds of agents) for emergent behaviors;
- an optional Gazebo (`gz sim`) adapter for full 3D physics, kept out of the
  pure/CI core.
