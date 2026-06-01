#!/usr/bin/env python3
"""Benchmark comparison tables: planners/controllers and MAPF solvers, side by side.

Runs every navigation policy on every bundled scenario, and every MAPF solver on
the MovingAI example, then writes a Markdown report with two comparison tables to
``benchmarks/comparison.md`` (and prints it). Everything is pure and
deterministic, so the report is reproducible — regenerate it after a change and
diff it, or read the checked-in copy.

    python3 scripts/compare_planners.py            # write + print the report
    python3 scripts/compare_planners.py --check     # fail if the report is stale

The numbers are produced by the same ``mrn_sim.benchmark`` / MovingAI machinery
the CI regression gate (``scripts/benchmark_gate.py``) checks, so a row here that
changes is a row the gate guards.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir))
sys.path.insert(0, os.path.join(_REPO, "mrn_sim"))
sys.path.insert(0, os.path.join(_REPO, "mrn_coord"))

_REPORT = os.path.join(_REPO, "benchmarks", "comparison.md")
_SCENARIOS = ["around_obstacle", "crossing", "doorway"]

# label -> how to build the policy from a scenario
_POLICIES = [
    ("grid A* + pursuit", "navigate"),
    ("Hybrid A* (kinodynamic)", "kinodynamic"),
    ("grid A* + DWA", "dwa"),
    ("Hybrid A* + DWA", "dwa_kino"),
    ("grid A* + MPC (iLQR)", "mpc"),
    ("grid A* + MPC + CBF", "mpc_cbf"),
    ("grid A* + ORCA", "orca"),
]


def _build(policy: str):
    from mrn_sim.benchmark import (
        dwa_policy,
        kinodynamic_policy,
        mpc_policy,
        navigate_policy,
        orca_policy,
    )
    return {
        "navigate": navigate_policy,
        "kinodynamic": kinodynamic_policy,
        "dwa": dwa_policy,
        "dwa_kino": lambda s: dwa_policy(s, planner="kino"),
        "mpc": mpc_policy,
        "mpc_cbf": lambda s: mpc_policy(s, safety="cbf"),
        "orca": orca_policy,
    }[policy]


def _nav_rows():
    from mrn_sim.benchmark import load_scenario, run_scenario

    rows = []
    for sname in _SCENARIOS:
        scenario = load_scenario(
            os.path.join(_REPO, "mrn_sim", "scenarios", sname + ".yaml"))
        for label, policy in _POLICIES:
            r = run_scenario(scenario, _build(policy)(scenario),
                             dt=0.1, max_steps=600)
            rows.append((sname, label, r))
    return rows


def _mapf_rows():
    from mrn_coord.mapf.movingai import load_map, load_scen, run_mapf_benchmark

    bench = os.path.join(_REPO, "mrn_coord", "benchmarks")
    grid = load_map(os.path.join(bench, "example.map"))
    tasks = load_scen(os.path.join(bench, "example.scen"))
    rows = []
    for solver in ("cbs", "ecbs", "lacam", "lns", "prioritized",
                   "prioritized_sipp"):
        res = run_mapf_benchmark(grid, tasks, solver=solver, max_expansions=50_000)
        rows.append((solver, res))
    return rows


def _sipp_rows():
    """Low-level planner expansions: time-expanded A* vs SIPP, vs. wait length.

    SIPP's advantage is forced *waiting*: a state per waited timestep for A*, a
    single ``(cell, interval)`` state for SIPP. To isolate it, an agent must
    cross a one-cell chokepoint that a parked obstacle reserves for the first
    ``N`` ticks, forcing an ``N``-tick wait. We sweep ``N`` and report the states
    each planner expands — A* grows with ``N``, SIPP stays flat.
    """
    from mrn_coord.mapf.grid import GridWorld
    from mrn_coord.mapf.sipp import plan_sipp
    from mrn_coord.mapf.space_time_astar import plan_path

    grid = GridWorld(8, 1)            # a straight corridor (0,0)..(7,0)
    start, goal = (0, 0), (7, 0)
    rows = []
    for n in (10, 40, 160, 640):
        # Park an obstacle on (1,0) for ticks 1..N: the agent must wait it out.
        vertex = frozenset(((1, 0), t) for t in range(1, n + 1))
        astar_stats: dict = {}
        sipp_stats: dict = {}
        a = plan_path(grid, start, goal, vertex, stats=astar_stats,
                      max_time=n + 50)
        b = plan_sipp(grid, start, goal, vertex, stats=sipp_stats,
                      max_time=n + 50)
        assert a is not None and b is not None
        assert len(a) == len(b)       # identical minimal-time path
        rows.append((n, len(a) - 1, astar_stats["expansions"],
                     sipp_stats["expansions"]))
    return rows


def _ecbs_scaling_rows():
    """Optimal CBS vs. bounded-suboptimal ECBS as the team grows.

    For each team size, ``K`` seeded random instances are drawn on a fixed
    pillared arena and solved by CBS (optimal) and ECBS (``w = 1.3``) under a
    shared expansion budget. We report how many each solves within the budget,
    the mean high-level expansions over the instances *both* solve, and the mean
    ECBS/CBS cost ratio. As agents pack in, CBS's constraint tree explodes and it
    starts blowing the budget; ECBS resolves the same instances in a handful of
    nodes for a few-percent cost premium. Deterministic (fixed seeds).
    """
    import random

    from mrn_coord.mapf.cbs import cbs
    from mrn_coord.mapf.ecbs import ecbs
    from mrn_coord.mapf.grid import GridWorld

    width, height = 6, 5
    pillars = frozenset({(2, 2), (3, 2)})
    budget = 800
    k = 25
    weight = 1.3

    rows = []
    grid = GridWorld(width, height, pillars)
    free = [(x, y) for x in range(width) for y in range(height)
            if grid.is_free((x, y))]
    for n in (3, 4, 6, 8, 10):
        rng = random.Random(1000 + n)
        cbs_solved = ecbs_solved = paired = 0
        cbs_exp = ecbs_exp = 0
        ratio = 0.0
        for _ in range(k):
            pts = rng.sample(free, 2 * n)
            agents = {str(i): (pts[2 * i], pts[2 * i + 1]) for i in range(n)}
            cstats, estats = {}, {}
            c = cbs(grid, agents, max_expansions=budget, stats=cstats)
            e = ecbs(grid, agents, w=weight, max_expansions=budget, stats=estats)
            cbs_solved += c is not None
            ecbs_solved += e is not None
            if c is not None and e is not None:
                cbs_exp += cstats["expansions"]
                ecbs_exp += estats["expansions"]
                ratio += (e.cost / c.cost) if c.cost else 1.0
                paired += 1
        rows.append((n, k, cbs_solved, ecbs_solved,
                     cbs_exp / paired if paired else 0.0,
                     ecbs_exp / paired if paired else 0.0,
                     ratio / paired if paired else 1.0))
    return rows, weight, budget, k


def _lns_rows():
    """MAPF-LNS anytime improvement: initial -> LNS-final vs. CBS-optimal.

    Seeded random instances per team size on the fixed pillared arena (those CBS
    can still solve, for the optimal reference). Reports the mean initial cost
    (the feasible solution LNS starts from), the mean cost after a fixed LNS
    budget, and the mean CBS optimum — showing LNS closes most of the gap.
    Deterministic.
    """
    import random

    from mrn_coord.mapf.cbs import cbs
    from mrn_coord.mapf.grid import GridWorld
    from mrn_coord.mapf.lns import mapf_lns

    width, height = 6, 5
    pillars = frozenset({(2, 2), (3, 2)})
    grid = GridWorld(width, height, pillars)
    free = [(x, y) for x in range(width) for y in range(height)
            if grid.is_free((x, y))]
    iters = 60
    k = 20
    rows = []
    for n in (4, 6, 8):
        rng = random.Random(2000 + n)
        init_sum = final_sum = opt_sum = 0.0
        paired = 0
        for _ in range(k):
            pts = rng.sample(free, 2 * n)
            agents = {str(i): (pts[2 * i], pts[2 * i + 1]) for i in range(n)}
            opt = cbs(grid, agents, max_expansions=2000)
            if opt is None:
                continue
            stats = {}
            mapf_lns(grid, agents, iterations=iters, seed=1, stats=stats)
            init_sum += stats["initial_cost"]
            final_sum += stats["final_cost"]
            opt_sum += opt.cost
            paired += 1
        if paired:
            rows.append((n, paired, init_sum / paired, final_sum / paired,
                         opt_sum / paired))
    return rows, iters


def _mapf_exec_rows():
    """Execute the SAME discrete MAPF plan in the continuous world three ways."""
    from mrn_coord.mapf import GridWorld
    from mrn_sim.mapf_exec import execute_mapf_plan

    grid = GridWorld(7, 7)
    agents = {"0": ((0, 3), (6, 3)), "1": ((6, 3), (0, 3)),
              "2": ((3, 0), (3, 6)), "3": ((3, 6), (3, 0))}
    rows = []
    for controller in ("pursuit", "tpg", "dwa"):
        res = execute_mapf_plan(grid, agents, solver="lacam",
                                controller=controller)
        rows.append((controller, res))
    return rows


def _lifelong_rows():
    from mrn_coord.lifelong import TaskStream, make_warehouse, run_lifelong

    grid, endpoints = make_warehouse(rows=2, cols=3)
    rows = []
    for n in (2, 4, 6, 8):
        n = min(n, len(endpoints))
        starts = {f"r{i}": endpoints[i] for i in range(n)}
        res = run_lifelong(grid, starts, TaskStream(list(endpoints)),
                           max_steps=150)
        rows.append((n, grid, res))
    return rows


def _allocator_rows():
    """Task allocation: round-robin vs cost-aware auction / Hungarian."""
    from mrn_coord.lifelong import TaskStream, make_warehouse, run_lifelong

    grid, endpoints = make_warehouse(rows=2, cols=3)
    n = min(6, len(endpoints))
    rows = []
    for allocator in ("stream", "auction", "hungarian"):
        starts = {f"r{i}": endpoints[i] for i in range(n)}
        res = run_lifelong(grid, starts, TaskStream(list(endpoints)),
                           max_steps=150, allocator=allocator)
        rows.append((allocator, res))
    return n, rows


def _fmt(report_lines):
    return "\n".join(report_lines) + "\n"


def build_report() -> str:
    lines = [
        "# Benchmark comparison",
        "",
        "Generated by `scripts/compare_planners.py` (pure + deterministic; "
        "regenerate after changes). The same metrics are regression-gated by "
        "`scripts/benchmark_gate.py`.",
        "",
        "## Navigation policies",
        "",
        "Each navigation policy run on every bundled scenario "
        "(`mrn_sim/scenarios/`). `success` = all goals reached; `makespan` = "
        "time to the last arrival; `clear` = min obstacle-surface clearance (m); "
        "`min dist` = min inter-robot centre distance (m); `coll` = colliding "
        "steps.",
        "",
        "| scenario | policy | success | steps | makespan (s) | path len (m) | "
        "clear (m) | min dist (m) | coll |",
        "| --- | --- | :-: | --: | --: | --: | --: | --: | :-: |",
    ]
    for sname, label, r in _nav_rows():
        d = r.as_dict()
        min_dist = "—" if r.goals_total < 2 else f"{d['min_robot_distance']:.2f}"
        lines.append(
            f"| {sname} | {label} | {'✓' if d['success'] else '✗'} | "
            f"{d['steps']} | {d['makespan_sec']:.1f} | "
            f"{d['total_path_length']:.1f} | {d['min_obstacle_clearance']:.2f} | "
            f"{min_dist} | {d['collisions']} |"
        )

    lines += [
        "",
        "**Reading it.** Grid A\\* plans fast but axis-aligned; the kinodynamic "
        "Hybrid A\\* trades a little planning time for smooth, bounded-curvature "
        "paths (often fewer steps and more clearance to follow). DWA decides "
        "speed and avoidance by forward-simulating accel-limited velocities, so "
        "it tracks tighter and reacts to the other robots as moving obstacles, "
        "at a higher per-tick compute cost. MPC (iLQR) goes further still: it "
        "*optimizes* a whole receding-horizon trajectory each tick — smooth and "
        "far-sighted (often the shortest makespan) — predicting the other robots "
        "along their paths to avoid them in space-time. Its safety layer is a "
        "hard brake by default; **MPC + CBF** swaps that for a control-barrier "
        "QP that returns the nearest *forward-invariant-safe* command, so it "
        "steers around the conflict instead of braking — keeping more inter-robot "
        "clearance (see the doorway row) while staying collision-free.",
        "",
        "## MAPF solvers (MovingAI example)",
        "",
        "Conflict-Based Search (optimal sum-of-costs) vs. prioritized planning "
        "(fast, incomplete) on `mrn_coord/benchmarks/example.{map,scen}`.",
        "",
        "| solver | solved | makespan | sum of costs |",
        "| --- | :-: | --: | --: |",
    ]
    for solver, res in _mapf_rows():
        solved = res.get("solved", False)
        mk = res.get("makespan", "—")
        soc = res.get("sum_of_costs", "—")
        lines.append(
            f"| {solver} | {'✓' if solved else '✗'} | {mk} | {soc} |")

    lines += [
        "",
        "`ecbs` is bounded-suboptimal (cost ≤ `w`·optimal, here `w=1.5`); on this "
        "small example it happens to match the optimum. Its payoff shows as the "
        "team grows (next table). `lacam` is a *complete* satisficing search over "
        "whole configurations (PIBT successors + lazy constraints): not "
        "cost-optimal in general — though it also matches the optimum here — but "
        "it keeps finding solutions for large teams where the search-tree "
        "solvers blow up. `lns` is anytime large-neighborhood search: it polishes "
        "a feasible solution toward the optimum by destroy-and-repair (next "
        "tables). "
        "`prioritized` and `prioritized_sipp` are the same high-level planner "
        "with two interchangeable low-level planners — time-expanded A\\* and "
        "**SIPP** (safe-interval). They find equal-cost solutions; SIPP just "
        "reaches them while expanding far fewer states (next table).",
        "",
        "## Low-level planner: SIPP vs. time-expanded A\\*",
        "",
        "Single-agent query where a parked obstacle reserves a one-cell "
        "chokepoint for the first `N` ticks, forcing an `N`-tick wait. Both "
        "planners return the same minimal-time path, but time-expanded A\\* "
        "expands roughly one `(cell, time)` state per waited tick while SIPP "
        "collapses the wait into a single `(cell, interval)` state — so its "
        "expansions stay flat as `N` grows. On open instances with little "
        "waiting the two are comparable; this is where SIPP pays off.",
        "",
        "| wait N | path length | A\\* expansions | SIPP expansions |",
        "| --: | --: | --: | --: |",
    ]
    for n, plen, astar_exp, sipp_exp in _sipp_rows():
        lines.append(f"| {n} | {plen} | {astar_exp} | {sipp_exp} |")

    scaling, ecbs_w, ecbs_budget, ecbs_k = _ecbs_scaling_rows()
    lines += [
        "",
        "## Scaling: optimal CBS vs. bounded-suboptimal ECBS",
        "",
        f"{ecbs_k} seeded random instances per team size on a fixed pillared "
        f"arena, each solved by CBS (optimal) and ECBS (`w={ecbs_w}`) under a "
        f"shared {ecbs_budget}-node expansion budget. `solved` counts instances "
        "finished within the budget; `mean exp` is the average high-level nodes "
        "expanded over the instances *both* solve; `cost ratio` is mean "
        "ECBS/CBS sum-of-costs. As agents pack in, CBS's constraint tree blows "
        "up and it starts exhausting the budget, while ECBS stays in a handful "
        "of nodes for a few-percent cost premium — well inside its "
        f"`w={ecbs_w}` guarantee.",
        "",
        "| agents | CBS solved | ECBS solved | CBS mean exp | ECBS mean exp | "
        "cost ratio |",
        "| --: | :-: | :-: | --: | --: | --: |",
    ]
    for n, k, cbs_s, ecbs_s, cbs_e, ecbs_e, ratio in scaling:
        lines.append(
            f"| {n} | {cbs_s}/{k} | {ecbs_s}/{k} | {cbs_e:.0f} | {ecbs_e:.0f} | "
            f"{ratio:.3f} |")

    lns_rows, lns_iters = _lns_rows()
    lines += [
        "",
        "## Anytime improvement: MAPF-LNS",
        "",
        f"Seeded random instances per team size on the same arena, each given a "
        f"feasible starting solution and then {lns_iters} rounds of "
        "destroy-and-repair by LNS. `initial` is the mean cost of the starting "
        "solution (prioritized planning, or complete LaCAM where that fails), "
        "`LNS` the mean cost after the budget, and `optimal` the mean CBS "
        "sum-of-costs. LNS polishes the rough initial solution down toward the "
        "optimum — replanning only a few agents per round, so it keeps working "
        "at team sizes the optimal search cannot reach.",
        "",
        "| agents | initial cost | LNS cost | optimal (CBS) |",
        "| --: | --: | --: | --: |",
    ]
    for n, paired, init_c, final_c, opt_c in lns_rows:
        lines.append(
            f"| {n} | {init_c:.1f} | {final_c:.1f} | {opt_c:.1f} |")

    lines += [
        "",
        "## Lifelong MAPF throughput (PIBT)",
        "",
        "Online/lifelong MAPF on a shelf-and-aisle warehouse "
        "(`mrn_coord.lifelong.make_warehouse`): tasks never run out, so the "
        "metric is **throughput** (tasks completed per timestep), not makespan. "
        "Stepped collision-free by PIBT over 150 ticks; service time is the mean "
        "ticks from a task's assignment to its completion.",
        "",
        "| agents | grid | completed | throughput (tasks/step) | "
        "avg service (steps) | max wait |",
        "| --: | --- | --: | --: | --: | --: |",
    ]
    for n, grid, res in _lifelong_rows():
        d = res.as_dict()
        lines.append(
            f"| {n} | {grid.width}×{grid.height} | {d['completed']} | "
            f"{d['throughput']:.3f} | {d['avg_service_time']:.1f} | "
            f"{d['max_wait']} |"
        )
    lines += [
        "",
        "Throughput rises with the team size until aisle congestion starts to "
        "lengthen service times — the warehouse-capacity trade-off lifelong MAPF "
        "exists to study.",
        "",
    ]

    alloc_n, alloc_rows = _allocator_rows()
    lines += [
        "## Task allocation: round-robin vs. auction / Hungarian",
        "",
        f"Same warehouse and {alloc_n} robots over 150 ticks, varying only *how* "
        "a freed robot is handed its next task. `stream` is geometry-blind "
        "round-robin (the next task in a fixed cycle); `auction` and `hungarian` "
        "instead match free robots to the pool of open tasks by obstacle-aware "
        "travel distance (a regret-based market auction, and the optimal "
        "linear-assignment solution). Sending the *nearest* free robot shortens "
        "every trip, so far more tasks finish in the same time.",
        "",
        "| allocator | completed | throughput (tasks/step) | "
        "avg service (steps) | max wait |",
        "| --- | --: | --: | --: | --: |",
    ]
    for allocator, res in alloc_rows:
        d = res.as_dict()
        lines.append(
            f"| {allocator} | {d['completed']} | {d['throughput']:.3f} | "
            f"{d['avg_service_time']:.1f} | {d['max_wait']} |"
        )
    lines += [
        "",
        "Cost-aware allocation roughly doubles throughput and halves service "
        "time here. The optimal one-shot matching (Hungarian) and the cheaper "
        "regret auction are close; over the lifelong horizon the auction's "
        "round-by-round greediness can even edge ahead, since one-shot optimality "
        "is not the same as long-run optimality.",
        "",
        "## MAPF plan execution: plan vs. reality",
        "",
        "The *same* LaCAM plan for a 4-way crossing, executed in the continuous "
        "world three ways. The plan is collision-free on the grid in discrete "
        "time, but the robots are discs with unicycle kinematics. `pursuit` "
        "follows the spatial route but ignores the *schedule*, so the discs reach "
        "the shared centre together and collide. `tpg` gates each robot with a "
        "**Temporal Plan Graph** — enter your next cell only once its previous "
        "occupant has left — so the discrete coordination transfers exactly: "
        "collision-free, at the cost of makespan stretch while robots wait. `dwa` "
        "keeps the route but reacts to the others as moving obstacles. "
        "`coll` = robot-robot overlap steps; `cont./disc.` = continuous vs. grid "
        "makespan; `dev` = furthest a robot strayed from its planned line (m).",
        "",
        "| execution | success | coll | disc. makespan | cont. steps | dev (m) |",
        "| --- | :-: | :-: | --: | --: | --: |",
    ]
    for controller, res in _mapf_exec_rows():
        d = res.as_dict()
        lines.append(
            f"| {controller} | {'✓' if d['success'] else '✗'} | "
            f"{d['robot_collisions']} | {d['discrete_makespan']} | "
            f"{d['continuous_steps']} | {d['max_path_deviation']:.2f} |")
    lines += [
        "",
        "The discrete guarantee is necessary but not sufficient: it takes either "
        "a schedule-aware executor (TPG) or a reactive controller (DWA) to keep "
        "it collision-free once the robots are real.",
        "",
    ]
    return _fmt(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if the checked-in report is stale")
    args = parser.parse_args()

    report = build_report()
    if args.check:
        if not os.path.exists(_REPORT):
            print(f"FAIL: {_REPORT} does not exist; run compare_planners.py")
            return 1
        with open(_REPORT, "r", encoding="utf-8") as fh:
            current = fh.read()
        if current != report:
            print(f"FAIL: {_REPORT} is stale; rerun scripts/compare_planners.py")
            return 1
        print(f"ok: {_REPORT} is up to date")
        return 0

    os.makedirs(os.path.dirname(_REPORT), exist_ok=True)
    with open(_REPORT, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)
    print(f"wrote {_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
