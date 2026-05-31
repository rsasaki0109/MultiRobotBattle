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

```bash
python3 scripts/make_<name>_gif.py     # writes docs/media/<name>_demo.gif
```

The generators need only `matplotlib` + `Pillow` (and the `mrn_sim` / `mrn_coord`
packages on the path, which the scripts add automatically). No ROS or running
stack required.
