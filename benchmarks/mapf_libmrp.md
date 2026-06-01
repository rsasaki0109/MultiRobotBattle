# Our CBS vs. the reference libMultiRobotPlanning

Our `mrn_coord.mapf.cbs` is a from-scratch implementation of Conflict-Based Search (Sharon, Stern, Felner & Sturtevant). This report turns *"it finds the optimal solution"* into a measured contract: the same instances are solved by our code and by the canonical reference implementation, Wolfgang Hönig's `libMultiRobotPlanning` C++ `cbs` (`github.com/whoenig/libMultiRobotPlanning`). Regenerate with `python3 scripts/compare_mapf_libmrp.py --write` once the `cbs` binary is built and on `LIBMRP_CBS`/`PATH` (see `docs/coordination.md`).

Both solvers run the **same discrete model** — 4-connected grid, wait actions, unit edge cost, vertex + edge(swap) conflicts, agents that stay on their goal (we never pass `--disappear-at-goal`). Both minimize **sum-of-costs**.

- **`sum_of_costs`** — *gated*. The objective both solvers minimize; its optimum is a single number, so a correct optimal solver must reproduce the reference's value exactly (and agree on solvability). There is no tolerance — a mismatch is a real defect.
- **`makespan`** — *reported, not gated*. Many solutions share the optimal sum-of-costs; which one a tie-break returns first decides the makespan, so a difference here is an artifact, not a correctness gap.

| scenario | N | solved (ours/lib) | sum_of_costs (ours/lib) | makespan (ours/lib) |
| :-- | :-: | :-: | :-: | :-: |
| swap2 | 2 | True / True | 6 / 6 | 4 / 4 |
| doorway | 2 | True / True | 11 / 11 | 7 / 7 |
| crossing4 | 4 | True / True | 21 / 21 | 6 / 6 |
| blocks3 | 3 | True / True | 30 / 30 | 10 / 10 |
| random4 | 4 | True / True | 22 / 22 | 7 / 7 |
| movingai_example | 3 | True / True | 42 / 42 | 14 / 14 |

Gate (`--check`): identical `sum_of_costs` on every solved instance and matching solvability. Across the suite our CBS reproduces the reference's optimal cost exactly — the implementation computes the same optimum, not merely a feasible solution. The discrete model is shared by construction (the comparison would otherwise compare two different problems, not two solvers), so this isolates the search itself.

