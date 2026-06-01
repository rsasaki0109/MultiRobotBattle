# Gazebo Adapter (`mrn_gazebo`) — optional

`mrn_gazebo` is the 3D, physics-backed counterpart to [`mrn_sim`](simulation.md):
it lets the rest of the stack treat a **Gazebo (`gz sim`) world as the plant**.
Where `mrn_sim` is a deterministic in-house 2D world, this runs the robots in
Gazebo physics — at the cost of an external dependency.

> **Optional and not run in CI.** It requires Gazebo (`gz sim`, tested with
> Harmonic / Sim 8) and `ros_gz` (`ros_gz_sim`, `ros_gz_bridge`). The pure
> message builder is unit-tested; the world, bridge, and launch are exercised
> manually. The core localization/coordination packages do not depend on it.

## How it fits

```
Gazebo (gz sim)                ros_gz_bridge              mrn_gazebo            localization
 model physics + DiffDrive  ─▶  /model/<id>/pose      ─▶  GzPoseAdapter   ─▶  /<id>/mrn/agent_state
 /model/<id>/cmd_vel        ◀─  (Twist ROS->gz)       ◀─  (controllers)        cooperative_pose ...
```

The adapter (`mrn_gz_pose_adapter`) subscribes to the bridged model pose and
republishes it as `mrn_msgs/AgentState` — exactly the contract `mrn_sim` emits —
so everything downstream (the coordination nodes, or a cooperative-localization
consumer such as the companion `multirobot-localization` repo) works unchanged.
This is the same seam as `mrn_pose_bridge`, just sourced from Gazebo. The emitted
`AgentState` is stamped with a TTL so freshness gates accept it.

## Pieces

- `worlds/multirobot.sdf` — a ground plane, an obstacle, and one differential-
  drive vehicle (`robot_1`) with a `PosePublisher` (publishes `/model/robot_1/pose`)
  and a `DiffDrive` plugin (subscribes `/model/robot_1/cmd_vel`). Validated with
  `gz sdf -k` and loads headless. Duplicate the `robot_1` model block (unique
  name/topics) for more robots.
- `config/gz_bridge.yaml` — `ros_gz_bridge` config bridging the pose (gz→ROS)
  and `cmd_vel` (ROS→gz). Add a pair per robot.
- `mrn_gazebo/gz_agent_state.py` — pure `build_agent_state` (CI-tested).
- `mrn_gazebo/gz_pose_adapter_node.py` — the thin adapter node.
- `launch/gz_world.launch.py` — gz server + bridge + adapter.

## Run it

```bash
# headless server (no GUI):
ros2 launch mrn_gazebo gz_world.launch.py gz_args:="-r -s /path/to/multirobot.sdf"
# or with the GUI (default gz_args runs the world):
ros2 launch mrn_gazebo gz_world.launch.py

# the adapter now publishes /robot_1/mrn/agent_state; drive the robot with:
ros2 topic pub /model/robot_1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5}}"
```

From here the localization and coordination stacks attach exactly as they do to
`mrn_sim` — e.g. run the relative-anchor graph on the bridged `AgentState`, or a
path follower publishing `cmd_vel`.

## Multi-robot swarm

`swarm.launch.py` spawns `num_robots` differential-drive vehicles on a circle
(each with a unique name → unique `/model/<name>/{pose,cmd_vel}` topics),
bridges every robot's pose and `cmd_vel`, and runs `mrn_gz_swarm_controller`.
The controller subscribes to all poses, runs the Boids
`mrn_coord.flocking.flock_velocities` over their positions, and converts each
desired holonomic velocity into a differential-drive command with
`velocity_to_unicycle` — so the same swarm rules that drive the 2D demo flock a
Gazebo multi-robot world.

```bash
ros2 launch mrn_gazebo swarm.launch.py num_robots:=8          # GUI
ros2 launch mrn_gazebo swarm.launch.py num_robots:=6 headless:=true
```

**Verification status (honest):** the controller's math
(`flock_velocities`, `velocity_to_unicycle`) is unit-tested in CI; spawning a
vehicle and driving it via `cmd_vel` (and the single-robot pose→`AgentState`
adapter) are verified end-to-end headless. The *full N-robot flocking run* is
provided and runs on a normal machine (with the GUI or a healthy DDS), but was
**not cleanly verified in the CI-less sandbox** used during development: DDS
discovery across the many gz/bridge/spawn/controller processes there is
unreliable (only a subset of pose topics get discovered), which degrades the
flock. Disabling shared-memory transport (`FASTDDS_BUILTIN_TRANSPORTS=UDPv4`)
helps but did not fully resolve it in that environment.

## Demo video (`record_gazebo_gif.py`)

The 3D demo GIF in the README is recorded from a dedicated world,
`worlds/multirobot_demo.sdf`: three robots cross an arena of cylindrical
obstacles, driven over `cmd_vel` by the repo's **own navigation stack** — A\*
grid planning + pure-pursuit + reciprocal avoidance (`mrn_sim.navigate`
primitives), closed over Gazebo's reported poses. Each robot moves via the
kinematic `VelocityControl` system (commanded body twist → motion, stable and
slip-free — steadier than tuning a `DiffDrive` chassis for a demo).

Each robot also carries a 360° `gpu_lidar`; the script bridges every scan
(`sensor_msgs/LaserScan`), projects the returns back onto the rendered frame
through the fixed camera's known projection, and draws them as colored rays +
hit points — so the GIF shows the lasers sweeping the obstacles and the other
robots, like a real robotics simulator.

Recording is **fully offscreen**: a static isometric camera sensor renders on
the GPU (forced onto the NVIDIA EGL vendor) and publishes frames on `/rec_camera`;
the harness bridges them to ROS (`ros_gz_image`, over CycloneDDS to dodge the
shared-memory transport flakiness) and encodes a GIF. No GUI or desktop window is
ever opened. The shared harness `scripts/_gz_record.py` orchestrates the gz
server, the bridges, the control loop, the optional LiDAR overlay, and the
encode, and tears every process down on exit; each demo script is just a thin
controller plus a `Scenario` (world, camera, controller, LiDAR options).

```bash
# ROS 2 Jazzy sourced; needs gz sim (Harmonic), ros_gz, a GPU with EGL, rclpy:
python3 scripts/record_gazebo_gif.py
python3 scripts/record_gazebo_gif.py --duration 16 --fps 15 --width 720
```

Companion scripts record the other layers the same way, each from its own world
and driven by the matching `mrn_coord` algorithm (the 3D counterparts of the 2D
`make_*_gif.py` demos):

- `record_gazebo_orca_gif.py` (`worlds/orca_demo.sdf`) — two streams of holonomic
  robots on near-head-on lanes pass *through* each other via `mrn_coord.orca`.
- `record_gazebo_swarm_gif.py` (`worlds/swarm_demo.sdf`) — twelve robots flock
  past obstacles via `mrn_coord.flocking` Boids + a migration pull, with the
  flock's LiDAR returns overlaid as a point cloud.
- `record_gazebo_coord_gif.py` (`worlds/coord_demo.sdf`) — three robots funnel
  through a doorway on a Conflict-Based-Search schedule, then assemble a
  formation; the CBS + formation trajectory is precomputed by `mrn_coord` and
  replayed by tracking each robot's waypoint, LiDAR tracing the wall.
- `record_gazebo_warehouse_gif.py` (`worlds/warehouse_demo.sdf`) — six
  autonomous mobile robots work a shelf-and-aisle warehouse on a lifelong-MAPF
  (PIBT) schedule precomputed by `mrn_coord.lifelong`, replayed by waypoint
  tracking, their 360° LiDAR tracing the racking. Unlike the others its world
  and bridge are **generated** from the same `make_warehouse` grid the algorithm
  runs on (a 2×3-block warehouse is too much SDF to hand-write), so the scene and
  the plan can never drift apart.

The swarm, coordination, and warehouse robots also carry a `gpu_lidar`, so all
the companions can overlay scans too — `Scenario.use_lidar` (with `lidar_rays` /
`lidar_step`) toggles it in the shared harness.

Being wall-clock-paced 3D, the result is **not bit-for-bit deterministic**
(unlike the 2D `make_*_gif.py` demos) — it is media-generation only and, like the
rest of `mrn_gazebo`, not part of CI.

## Scope

This is a working seam, not a full benchmark. Multi-robot spawning, sensors
(LiDAR/IMU), and a faithful obstacle layout matching the localization grids are
natural extensions; the contract (`AgentState` in, `cmd_vel` out) is already the
same as the in-house simulator, so they slot in without touching the other
layers.
