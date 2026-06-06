# mapf-zoo

**45+ Multi-Agent Path Finding (MAPF) algorithms, faithfully reproduced from
their papers and benchmark-gated — in pure, ROS-free Python.**

Most MAPF code online is one algorithm per repo, in C++, wired to a build
system. This is the whole family in one importable package: solve an instance
and compare paradigms in five lines, no ROS and no compiler.

Every solver is reproduced from its source paper and **benchmark-gated** — a CI
gate pins each one's honest result (a WIN, a LOSS, or an equivalence against a
reference solver), so the claims are measured, not asserted.

```bash
pip install mapf-zoo          # pure Python, zero required dependencies
pip install "mapf-zoo[bcp]"   # + numpy/scipy for the LP-based solver
```

```python
from mrn_coord.mapf import GridWorld, cbs, render_ascii, pad_paths

# A 5x5 grid; two agents whose straight-line paths cross at the centre.
grid = GridWorld(5, 5)
agents = {"1": ((0, 2), (4, 2)), "2": ((2, 0), (2, 4))}

sol = cbs(grid, agents)            # optimal, sum-of-costs Conflict-Based Search
print(sol.cost, sol.makespan)      # -> 9 5  (collision-free, optimal)

padded = pad_paths(sol.paths)
print(render_ascii(grid, padded, t=2))   # ASCII snapshot at timestep 2
```

Swap `cbs` for any of the others — they share the `(grid, agents)` interface:

```python
from mrn_coord.mapf import ecbs, lacam, mapf_lns, pbs, prioritized_planning

ecbs(grid, agents, w=1.5)     # bounded-suboptimal (cost <= w * optimal)
lacam(grid, agents)           # complete satisficing, scales to large teams
mapf_lns(grid, agents)        # anytime large-neighborhood search
```

## What's in the zoo

| Family | Solvers |
| --- | --- |
| **CBS family** | CBS, CBSH (CG/DG/WDG), ICBS bypass, MA-CBS, disjoint splitting, ECBS, BCBS, EECBS, FECBS, highway, k-robust, CBS-TA, CCBS (continuous-time) |
| **Symmetry reasoning** | rectangle, corridor, mutex propagation |
| **Optimal joint-space** | M\*, recursive M\*, EPEA\*, ICTS, Standley OD / ID |
| **Declarative / LP** | MDD-SAT, branch-and-cut-and-price (BCP) |
| **Constructive** | Push-and-Swap, Push-and-Rotate, Bibox, TSWAP, flow (anonymous), DDM |
| **Suboptimal / anytime** | MAPF-LNS, MAPF-LNS2, WHCA\*, PIBT, LaCAM, LaCAM\*+LTM |
| **Assignment / teams** | CBS-TA, CBM/TAPF, anonymous flow |
| **Lifelong** | RHCR, Token Passing, TPTS, online-LNS, auction / Hungarian allocation |
| **Execution layer** | switchable-ADG, k-robust, TPG schedules |
| **Low levels** | space-time A\*, SIPP, SIPPS, Multi-Label A\* |

Each is documented algorithm-by-algorithm — with its paper and its gated result
— in [`docs/coordination.md`](https://github.com/rsasaki0109/multirobot-navigation/blob/main/docs/coordination.md).

## Why "faithfully reproduced + gated" matters

A reproduction is only worth as much as its evidence. For every solver the gate
asserts a concrete, measured claim, e.g.:

- CBSH expands **~13× fewer** high-level nodes than CBS for the *same* optimum.
- EPEA\* generates **~58× fewer** nodes than fully-expanding joint A\*.
- The rectangle barrier collapses a same-direction-crossing blowup **~20×**.
- FECBS beats ECBS **~9×** on dense, tight-`w` instances (and only there — the
  gate records where each method *loses*, too).

These are not cherry-picked headlines; they are the numbers CI refuses to
regress.

## Standard benchmarks

The package loads the community **MovingAI** `.map` / `.scen` format, so you can
evaluate any solver on the standard benchmark suite:

```python
from mrn_coord.mapf.movingai import load_map, load_scen, run_mapf_benchmark
```

## License

Apache-2.0. Part of
[multirobot-navigation](https://github.com/rsasaki0109/multirobot-navigation) —
a ROS 2-native multi-robot simulation, navigation, and coordination stack. This
package is its ROS-free coordination core, carved out so the algorithms are
usable without ROS or colcon.
