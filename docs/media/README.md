# Demo Media

The README/docs GIFs. All are **synthetic, deterministic, driven by the real
algorithms** — regenerate any of them with the matching script:

| GIF | Script | Shows |
| --- | --- | --- |
| `coordination_demo.gif` | `make_coordination_gif.py` | CBS doorway crossing → formation |
| `sim_demo.gif` | `make_sim_gif.py` | the 2D world: robots, obstacles, V2V links |
| `swarm_demo.gif` | `make_swarm_gif.py` | Boids flocking (70 agents) |
| `swarm_sim_demo.gif` | `make_swarm_sim_gif.py` | flock migrating to a goal through obstacles |
| `predator_demo.gif` | `make_predator_gif.py` | flock fleeing a pursuing predator |
| `mission_demo.gif` | `make_mission_gif.py` | multi-phase mission (regroup→migrate→evade→reach) |
| `nav_demo.gif` | `make_nav_gif.py` | point-to-point A* + pure-pursuit navigation |
| `recip_nav_demo.gif` | `make_recip_nav_gif.py` | multi-robot navigation with reciprocal avoidance |
| `replan_demo.gif` | `make_replan_gif.py` | replanning around a moving obstacle |
| `orca_demo.gif` | `make_orca_gif.py` | ORCA reciprocal avoidance: two crowds pass through each other |
| `gazebo_demo.gif` | `record_gazebo_gif.py` | **3D Gazebo**: three robots cross an obstacle arena via A\* + pure-pursuit + reciprocal avoidance, with live 360° LiDAR overlaid |
| `gazebo_orca_demo.gif` | `record_gazebo_orca_gif.py` | **3D Gazebo**: two robot streams pass through each other collision-free via ORCA |
| `gazebo_swarm_demo.gif` | `record_gazebo_swarm_gif.py` | **3D Gazebo**: twelve robots flock past obstacles via Boids, with their LiDAR point cloud |
| `gazebo_coord_demo.gif` | `record_gazebo_coord_gif.py` | **3D Gazebo**: three robots funnel through a doorway via CBS then form up, LiDAR tracing the wall |

```bash
python3 scripts/make_<name>_gif.py     # writes docs/media/<name>_demo.gif
```

The `make_*_gif.py` generators need only `matplotlib` + `Pillow` (and the
`mrn_sim` / `mrn_coord` packages on the path, which the scripts add
automatically). No ROS or running stack required.

The `record_gazebo_*_gif.py` scripts are the exception: they drive the **real
Gazebo** worlds (sharing one harness, `scripts/_gz_record.py`) and so need
`gz sim` (Harmonic), `ros_gz` (`ros_gz_bridge` / `ros_gz_image`), a GPU with EGL,
and `rclpy` — run them with ROS 2 Jazzy sourced. They render fully offscreen (no
GUI / desktop window). Being wall-clock-paced 3D, they are not bit-for-bit
deterministic; they are media-generation only and not part of CI.
