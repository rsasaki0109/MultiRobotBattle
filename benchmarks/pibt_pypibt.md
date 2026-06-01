# Our PIBT vs. the reference `pypibt`

Our `mrn_coord.lifelong._Pibt` advances every warehouse/fleet timestep with PIBT (Priority Inheritance with Backtracking, Okumura et al. 2022). This report turns *"it's PIBT"* into a measured contract: identical scenarios run through both our `_Pibt` core and the paper author's own reference, [`pypibt`](https://github.com/Kei18/pypibt) — and our output is judged by the **reference's own** code. Regenerate with `python3 scripts/compare_pibt_pypibt.py --write` inside a venv that has `pypibt` installed (see `docs/coordination.md`).

PIBT's completeness theorem relies on a **random** tie-break (`pypibt` uses `rng.shuffle`); our `_Pibt` breaks ties **deterministically** so the demos stay bit-reproducible. That trade is the whole story here:

- **`collision_free`** — the invariant our code guarantees and the only thing the fleet demo needs. For every instance and **every timestep** (including the real `run_lifelong` warehouse run), the configuration `_Pibt` emits has zero vertex collisions, zero edge (swap) collisions, and only step-or-wait transitions — checked with `pypibt`'s own `get_neighbors` + `validate_mapf_solution`. **This is the gated claim.**
- **`converged`** — the honest cost of the deterministic tie-break: as a one-shot fixed-goal solver, `_Pibt` can livelock in a symmetric standoff that the reference's random tie-break escapes, so it is not *complete* the way `pypibt` is. We report the rate rather than hide it; it is irrelevant to lifelong throughput, where goals change on arrival.
- **`ratio`** — makespan `ours / reference` on the instances where ours converges. A bound, not equality (same algorithm, different tie-break).

| scenario | grid | N | collision-free | converged | mean ratio | max ratio |
| :-- | :-: | :-: | :-: | :-: | --: | --: |
| open_8x8_8 | 8x8 | 8 | yes | 9/10 | 1.016 | 1.143 |
| open_8x8_16 | 8x8 | 16 | yes | 7/10 | 1.231 | 1.875 |
| open_10x10_20 | 10x10 | 20 | yes | 7/10 | 0.969 | 1.071 |
| open_12x12_30 | 12x12 | 30 | yes | 7/10 | 0.922 | 1.111 |
| open_16x16_50 | 16x16 | 50 | yes | 8/10 | 1.011 | 1.200 |
| open_20x12_60 | 20x12 | 60 | yes | 5/10 | 1.074 | 1.333 |
| warehouse_lifelong | 19x13 | 40 | yes | n/a | — | — |

Across the one-shot suite our deterministic solver converged on 43/60 instances; **every** configuration on **every** instance — converged or not, plus the full lifelong warehouse run — is collision-free under the reference's own checks.

Gate (`--check`): `collision_free` holds on every scenario (the load-bearing claim); the makespan ratio stays within 2.0 per instance and 1.4 on the convergent mean; and convergence does not collapse below 40% (a regression backstop, not a completeness claim — deterministic PIBT is knowingly incomplete).

