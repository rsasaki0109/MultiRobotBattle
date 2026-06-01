# Our ORCA vs. the reference RVO2

Our `mrn_coord.orca` is a pure-Python port of the reference RVO2 (van den Berg, Guy, Lin & Manocha) agent-agent collision-avoidance core. This report turns *"ported faithfully"* into a measured contract: the same agents-only scenarios run through both our code and the reference C++ library (`Python-RVO2`, imported as `rvo2`), with identical parameters (radius 0.25 m, max speed 1.5 m/s, time horizon 2.0 s, dt 0.1 s). Regenerate with `python3 scripts/compare_orca_rvo2.py --write` inside a venv that has `rvo2` built (see `docs/simulation.md`).

- **`max_vel_dev`** — open-loop: both implementations are fed the *same* state every tick (the reference rollout) and we compare the velocity each returns. This isolates the function; a faithful port agrees to ~machine precision wherever the half-planes are jointly feasible.
- **`max_traj_div`** — closed-loop: each runs as its own independent simulation and we measure trajectory drift. Reported, not gated: a near-symmetric head-on pass is ill-conditioned, so two faithful implementations can transiently differ in along-track phase while still passing on the same side and reaching the same goals.
- **`gap_*`** — worst surface-to-surface clearance each reaches (negative = bodies overlap, the over-constrained LP3 regime). The **gated** safety contract is that these two agree — an outcome that is invariant to which way a symmetric tie breaks.

| scenario | N | max_vel_dev | max_traj_div | gap_rvo2 (m) | gap_ours (m) |
| :-- | :-: | --: | --: | --: | --: |
| head_on | 2 | 0.000001 | 0.7718 | 0.0026 | 0.0042 |
| crossing | 4 | 0.000000 | 0.0000 | -0.0000 | 0.0000 |
| circle8 | 8 | 0.000000 | 0.0000 | 0.0000 | 0.0000 |
| random10 | 10 | 0.000020 | 0.0000 | -0.0414 | -0.0414 |

Gate (`--check`): `max_vel_dev` < 0.001 (the port-fidelity claim) and `|gap_ours - gap_rvo2|` < 0.02 (safety parity). Velocity agreement is at the 1e-5 level across the suite — the port reproduces the reference linear program, not merely its qualitative behaviour. Static obstacles are out of scope: RVO2 models them as line-segment polygons, while our `obstacles=` argument is a different object (a full-responsibility disc), so comparing them would compare two models rather than a port.

