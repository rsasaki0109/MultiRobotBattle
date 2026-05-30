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

The point of a true world model is that **both halves of the project plug into
the same world**: the localization stack consumes the (noisy) sensor
measurements it emits, and the coordination layer's velocity commands drive the
robots — so one deterministic world can close the whole loop
(world → sensors → localization → coordination → commands → world).

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

Because it *emits* the localization messages and *accepts* velocity commands,
it is the plant that closes the loop. The world/proximity/sensor math is the
pure, CI-tested core; this node is the thin shell (agent ids are sanitized into
valid topic tokens).

Note the velocity contract: `mrn_sim` is a **unicycle** (`v`, `omega`), whereas
the existing formation controller emits a **holonomic** velocity vector. Closing
a coordination loop *through this world* therefore wants a unicycle-aware
controller (e.g. a path follower for the MAPF planner) — the next step below.
The holonomic loop is already demonstrated with `mrn_coord`'s `mrn_agent_sim`.

## Roadmap

This is the world core plus its ROS node. Planned next:

- a unicycle path-follower so a MAPF plan can be driven through this world,
  closing world → localization → planning → world in ROS;
- emitting `mrn_msgs/RelativePoseConstraint` for in-range pairs (V2V), feeding
  the cooperative-localization graph directly;
- swarm-scale runs (tens to hundreds of agents) for emergent behaviors;
- an optional Gazebo (`gz sim`) adapter for full 3D physics, kept out of the
  pure/CI core.
