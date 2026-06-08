# Demo Media

The README/docs GIFs. All are **synthetic, deterministic, driven by the real
algorithms** — regenerate any of them with the matching script:

| GIF | Script | Shows |
| --- | --- | --- |
| `battle.gif` | `make_battle_gif.py` | Total war — tracer rounds, walls, elevation, 576 bots |
| `battle_gallery.gif` | `make_battle_gallery_gif.py` | 2×2 grid — duel uses real ballistics, RoboMaster chassis |
| `objective_triple.gif` | `make_objective_triple_gif.py` | hill · domination · CTF — three objective modes, RoboMaster chassis |
| `objective_duel.gif` | `make_objective_gif.py` | hill vs domination — zone hold progress, RoboMaster chassis |
| `ctf_duel.gif` | `make_ctf_gif.py` | capture the flag — centre pickup, score at home base, RoboMaster chassis |
| `ctf_mapf.gif` | `make_ctf_mapf_gif.py` | CTF × MAPF — Hungarian+greedy vs CBS-TA+prioritized, RoboMaster chassis |
| `maneuver_layers.gif` | `make_maneuver_gif.py` | 2×2 headline — greedy / A* / prioritized / CBS red vs greedy blue |
| `maneuver_duel.gif` | *(legacy)* | prioritized MAPF red vs greedy blue — two-panel chokepoint |
| `mapf_stack_duel.gif` | `make_mapf_stack_gif.py` | Hungarian+greedy vs CBS-TA+prioritized — wall terrain |
| `mapf_total_war.gif` | `make_mapf_total_war_gif.py` | MAPF stack on 18 vs 18 KOTH — RoboMaster chassis side-by-side |
| `kingdom_clash.gif` | `make_kingdom_gif.py` | 80 vs 80 — rectangular berms, elevation pads, RoboMaster chassis |
| `sim_demo.gif` | `make_sim_gif.py` | the 2D world: robots, obstacles, V2V links |
| `swarm_demo.gif` | `make_swarm_gif.py` | Boids flocking (70 agents) |
| `swarm_sim_demo.gif` | `make_swarm_sim_gif.py` | flock migrating to a goal through obstacles |
| `predator_demo.gif` | `make_predator_gif.py` | flock fleeing a pursuing predator |
| `mission_demo.gif` | `make_mission_gif.py` | multi-phase mission (regroup→migrate→evade→reach) |
| `nav_demo.gif` | `make_nav_gif.py` | point-to-point A* + pure-pursuit navigation |
| `recip_nav_demo.gif` | `make_recip_nav_gif.py` | multi-robot navigation with reciprocal avoidance |
| `replan_demo.gif` | `make_replan_gif.py` | replanning around a moving obstacle |
| `orca_demo.gif` | `make_orca_gif.py` | ORCA reciprocal avoidance: two crowds pass through each other |
| `warehouse_demo.gif` | `make_warehouse_gif.py` | warehouse AMR fleet: lifelong MAPF / PIBT, twelve robots streaming endless pick/drop tasks, with live throughput |
| `fleet_demo.gif` | `make_warehouse_gif.py --preset fleet` | fleet system at scale: 100 AMRs on a 6×9 shelf floor, lifelong MAPF / PIBT, throughput past 25 tasks/step |
| `gazebo_demo.gif` | `record_gazebo_gif.py` | **3D Gazebo**: three robots cross an obstacle arena via A\* + pure-pursuit + reciprocal avoidance, with live 360° LiDAR overlaid |
| `gazebo_orca_demo.gif` | `record_gazebo_orca_gif.py` | **3D Gazebo**: two robot streams pass through each other collision-free via ORCA |
| `gazebo_swarm_demo.gif` | `record_gazebo_swarm_gif.py` | **3D Gazebo**: twelve robots flock past obstacles via Boids, with their LiDAR point cloud |
| `gazebo_coord_demo.gif` | `record_gazebo_coord_gif.py` | **3D Gazebo**: three robots funnel through a doorway via CBS then form up, LiDAR tracing the wall |
| `gazebo_warehouse_demo.gif` | `record_gazebo_warehouse_gif.py` | **3D Gazebo**: six AMRs work a shelf-and-aisle warehouse on a lifelong-MAPF (PIBT) schedule, LiDAR tracing the racking |

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
