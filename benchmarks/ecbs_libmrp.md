# Our ECBS vs. the reference libMultiRobotPlanning

Our `mrn_coord.mapf.ecbs` is bounded-suboptimal Enhanced CBS (Barer, Sharon, Stern & Felner): with suboptimality factor `w` it returns a solution whose sum-of-costs is at most `w` times the optimum. This report turns that guarantee into a measured contract against the reference implementation, Wolfgang Hönig's `libMultiRobotPlanning` `ecbs` — same discrete model, same `w`. The optimum is our CBS cost, which [`benchmarks/mapf_libmrp.md`](mapf_libmrp.md) already proved equal to the reference `cbs`. Regenerate with `python3 scripts/compare_ecbs_libmrp.py --write` (see `docs/coordination.md`).

`w = 1.5`. The gated property is `cost <= w · optimal` for **both** solvers (`ratio <= w`). The two need not agree on cost — focal-search tie-breaking differs — so, unlike the optimal CBS contract, equality is not gated; the ratios show the suboptimality actually taken.

| scenario | N | optimal | ours (ratio) | lib (ratio) |
| :-- | :-: | --: | --: | --: |
| swap2 | 2 | 6 | 6 (1.000) | 6 (1.000) |
| doorway | 2 | 11 | 12 (1.091) | 12 (1.091) |
| crossing4 | 4 | 21 | 21 (1.000) | 21 (1.000) |
| blocks3 | 3 | 30 | 30 (1.000) | 30 (1.000) |
| random4 | 4 | 22 | 22 (1.000) | 22 (1.000) |
| movingai_example | 3 | 42 | 42 (1.000) | 42 (1.000) |

Gate (`--check`): every `ratio <= 1.5`, on both implementations, on every solvable instance. The bound holds across the suite — and the ratios sit at or below it, so ECBS is taking only the slack it needs. This is the bounded-suboptimal sibling of the exact-optimal CBS contract.

