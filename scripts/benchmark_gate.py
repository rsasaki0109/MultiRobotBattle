#!/usr/bin/env python3
"""Scenario-driven benchmark regression gate.

Runs the bundled benchmarks (deterministic) and compares their metrics against
checked-in expectations in ``benchmarks/expected_metrics/``. Exits non-zero on
any regression, so CI fails if a change degrades planning/control/allocation —
the benchmarks become a guarded contract, not decoration.

    python3 scripts/benchmark_gate.py            # check (CI gate)
    python3 scripts/benchmark_gate.py --update   # rewrite the expectations

Pure and deterministic: no ROS daemon, no external data. Discrete metrics
(success, collisions, makespan steps, solved, sum-of-costs) are compared
exactly; floats within a small tolerance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir))
sys.path.insert(0, os.path.join(_REPO, "mrn_sim"))
sys.path.insert(0, os.path.join(_REPO, "mrn_coord"))

_EXPECTED_DIR = os.path.join(_REPO, "benchmarks", "expected_metrics")
_FLOAT_TOL = 0.05


def _run_sim_scenario(name: str, policy: str = "navigate") -> dict:
    from mrn_sim.benchmark import (
        dwa_policy,
        kinodynamic_policy,
        load_scenario,
        mpc_policy,
        navigate_policy,
        orca_policy,
        run_scenario,
    )

    builders = {
        "navigate": navigate_policy,
        "orca": orca_policy,
        "kinodynamic": kinodynamic_policy,
        "dwa": dwa_policy,                                  # grid plan + DWA
        "dwa_kino": lambda s: dwa_policy(s, planner="kino"),  # kino plan + DWA
        "mpc": mpc_policy,                                  # grid plan + iLQR MPC
        "mpc_cbf": lambda s: mpc_policy(s, safety="cbf"),   # MPC + CBF safety QP
        "mpc_shield": lambda s: mpc_policy(s, safety="shield"),  # MPC + certified shield
    }
    scenario = load_scenario(os.path.join(_REPO, "mrn_sim", "scenarios", name + ".yaml"))
    result = run_scenario(scenario, builders[policy](scenario), dt=0.1, max_steps=600)
    out = result.as_dict()
    out["policy"] = policy
    return out


def _run_mapf_example(solver: str, **kwargs) -> dict:
    from mrn_coord.mapf.movingai import load_map, load_scen, run_mapf_benchmark

    bench = os.path.join(_REPO, "mrn_coord", "benchmarks")
    grid = load_map(os.path.join(bench, "example.map"))
    tasks = load_scen(os.path.join(bench, "example.scen"))
    res = run_mapf_benchmark(grid, tasks, solver=solver, max_expansions=50_000,
                             **kwargs)
    res["case"] = "mapf_example_" + solver
    return res


def _run_mapf_exec(controller: str) -> dict:
    from mrn_coord.mapf import GridWorld
    from mrn_sim.mapf_exec import execute_mapf_plan

    grid = GridWorld(7, 7)
    agents = {"0": ((0, 3), (6, 3)), "1": ((6, 3), (0, 3)),
              "2": ((3, 0), (3, 6)), "3": ((3, 6), (3, 0))}
    res = execute_mapf_plan(grid, agents, solver="lacam", controller=controller)
    out = res.as_dict()
    out["case"] = "mapf_exec_" + controller
    return out


def _run_shield_certify() -> dict:
    # Adversarial certification of the runtime safety shield: a nominal command
    # engineered to crash (steer at the nearest obstacle, full speed) on
    # randomized fields. The contract is that the certified body-true shield
    # never lets the robot body cross an obstacle boundary, while the same attack
    # collides every time unshielded.
    sys.path.insert(0, os.path.join(_REPO, "scripts"))
    from certify_shield import certify

    return certify(seed=0, trials=400, steps=200)


def _run_shield_reciprocal() -> dict:
    # Reciprocal certification: several shielded robots in adversarial mutual
    # pursuit, each treating the others as moving obstacles. The contract is that
    # all-shielded never collide, while the same pursuit unshielded always does.
    sys.path.insert(0, os.path.join(_REPO, "scripts"))
    from certify_shield import certify_reciprocal

    return certify_reciprocal(seed=0, trials=150, n_robots=4)


def _run_lifelong(agents: int = 6, steps: int = 120, allocator: str = "stream",
                  rows: int = 2, cols: int = 3, case: str | None = None) -> dict:
    from mrn_coord.lifelong import TaskStream, make_warehouse, run_lifelong

    grid, endpoints = make_warehouse(rows=rows, cols=cols)
    starts = {f"r{i}": endpoints[i] for i in range(min(agents, len(endpoints)))}
    res = run_lifelong(grid, starts, TaskStream(list(endpoints)),
                       max_steps=steps, allocator=allocator)
    out = res.as_dict()
    if case is None:
        suffix = "" if allocator == "stream" else "_" + allocator
        case = "mapf_lifelong" + suffix
    out["case"] = case
    return out


def _run_pibt_convergence() -> dict:
    # Deterministic livelock escape for the PIBT core: plain deterministic PIBT
    # livelocks on a chunk of random instances (the price of a reproducible
    # tie-break vs the completeness theorem's random one). A stall-triggered,
    # per-step scramble of equal-distance candidate ties breaks the symmetry with
    # zero randomness. The contract pins the *gap*: on a fixed 600-instance
    # open-grid battery the escape converges on *every* instance and stays
    # collision-free throughout, while bare deterministic PIBT livelocks on a
    # measurable slice (``bare_converged < instances``). Measuring the stall
    # against the running-minimum distance (not the previous step) is what closes
    # the last ~1% the step-to-step detector left on the table — an oscillation
    # can no longer fool the escape into disengaging.
    import random

    from mrn_coord.lifelong import pibt_solve
    from mrn_coord.mapf import GridWorld

    def _collision_free(cfgs) -> bool:
        for prev, cur in zip(cfgs, cfgs[1:]):
            if len({tuple(c) for c in cur}) != len(cur):
                return False
            for i in range(len(cur)):
                for j in range(i + 1, len(cur)):
                    if cur[i] == prev[j] and cur[j] == prev[i]:
                        return False
        return True

    converged = collision_free = bare_converged = instances = 0
    for w, h, n in ((8, 8, 16), (10, 10, 20), (12, 12, 30)):
        grid = GridWorld(w, h)
        for seed in range(200):
            rng = random.Random(seed)
            cells = [(x, y) for x in range(w) for y in range(h)]
            starts, goals = rng.sample(cells, n), rng.sample(cells, n)
            cfgs, ok = pibt_solve(grid, starts, goals, escape=True)
            instances += 1
            converged += int(ok)
            collision_free += int(_collision_free(cfgs))
            bare_converged += int(pibt_solve(grid, starts, goals, escape=False)[1])
    return {"case": "pibt_escape_convergence", "instances": instances,
            "converged": converged, "collision_free": collision_free,
            "bare_converged": bare_converged}


def _run_lacam_convergence() -> dict:
    # LaCAM's documented selling point is *scaling* — complete, and fast on teams
    # CBS cannot touch. That lives entirely in the order successors are generated:
    # its greedy DFS spine is a PIBT rollout, and a *static* per-config priority
    # made that the weak deterministic PIBT, which livelocked and dropped into the
    # exponential lazy-constraint fallback — solving only ~0.667 of this open-grid
    # battery even at 200k iterations, ~100x slower. The spine now runs the strong
    # PIBT (accumulating priority + the deterministic escape salt reseeded per
    # re-expansion). The contract: it solves *every* instance, each solution valid
    # (collision-free, every agent on its goal), inside a bounded budget. The toy
    # completeness test never reached this regime; this pins the scaling claim.
    import random

    from mrn_coord.mapf import GridWorld, lacam
    from mrn_coord.mapf.conflicts import detect_first_conflict
    from mrn_coord.mapf.solution import pad_paths

    def _valid(grid, agents, sol) -> bool:
        if sol is None or detect_first_conflict(pad_paths(sol.paths)) is not None:
            return False
        return all(sol.paths[a][0] == agents[a][0] and sol.paths[a][-1] == agents[a][1]
                   for a in agents)

    solved = valid = instances = 0
    for w, h, n in ((8, 8, 16), (10, 10, 20), (12, 12, 30)):
        grid = GridWorld(w, h)
        cells = [(x, y) for x in range(w) for y in range(h)]
        for seed in range(60):
            rng = random.Random(seed)
            starts, goals = rng.sample(cells, n), rng.sample(cells, n)
            agents = {i: (starts[i], goals[i]) for i in range(n)}
            sol = lacam(grid, agents, max_iterations=200_000)
            instances += 1
            solved += int(sol is not None)
            valid += int(_valid(grid, agents, sol))
    return {"case": "lacam_scaling_convergence", "instances": instances,
            "solved": solved, "valid": valid}


def _run_lacam_optimality() -> dict:
    # LaCAM's anytime mode (optimize=True / LaCAM*) keeps searching past the first
    # solution -- g-tracking + parent rewiring + lower-bound pruning -- and on small
    # instances reaches the true optimum. The contract checks it agent-for-agent
    # against CBS (which minimizes sum-of-costs): on this fixed battery it must match
    # the optimum on *every* CBS-solvable instance, and never undercut it (a
    # cheaper-than-CBS result would mean the cost accounting is broken). Pins the
    # win; the docstring records the boundary (it does not scale -- LNS owns cost at
    # scale), and `test_optimize_never_costs_more_than_satisficing` guards the rest.
    import random

    from mrn_coord.mapf import GridWorld, cbs, lacam
    from mrn_coord.mapf.conflicts import detect_first_conflict
    from mrn_coord.mapf.solution import pad_paths

    def _soc(paths) -> int:
        total = 0
        for p in paths.values():
            goal, arrival = p[-1], 0
            for t, c in enumerate(p):
                if c != goal:
                    arrival = t + 1
            total += arrival
        return total

    def _valid(grid, agents, sol) -> bool:
        if sol is None or detect_first_conflict(pad_paths(sol.paths)) is not None:
            return False
        return all(sol.paths[a][0] == agents[a][0] and sol.paths[a][-1] == agents[a][1]
                   for a in agents)

    rng = random.Random(1)
    checked = optimal = valid = below = 0
    for _ in range(120):
        w = h = rng.randint(4, 6)
        grid = GridWorld(w, h)
        free = [(x, y) for x in range(w) for y in range(h) if grid.is_free((x, y))]
        n = rng.randint(2, 4)
        if len(free) < 2 * n:
            continue
        pts = rng.sample(free, 2 * n)
        agents = {str(i): (pts[2 * i], pts[2 * i + 1]) for i in range(n)}
        opt = cbs(grid, agents, max_expansions=20_000)
        if opt is None:
            continue
        sol = lacam(grid, agents, optimize=True, max_iterations=200_000)
        checked += 1
        valid += int(_valid(grid, agents, sol))
        optimal += int(_soc(sol.paths) == _soc(opt.paths))
        below += int(_soc(sol.paths) < _soc(opt.paths))
    return {"case": "lacam_optimality", "checked": checked, "valid": valid,
            "optimal": optimal, "below_cbs": below}


def _run_lacam_ltm_vs_optimize() -> dict:
    # LaCAM*+LTM (lacam_ltm) is a Python reproduction of "A Lightweight Traffic
    # Map for Efficient Anytime LaCAM*" (arXiv:2603.07891, C++-only upstream).
    # It targets exactly the boundary `lacam_optimality`'s docstring records:
    # plain optimize=True stalls at scale -- on these 16-30-agent grids it returns
    # the *same* cost as the first dive even given the full budget, because the
    # config space is too large for the lower bound to prune. LTM breaks that by
    # building a congestion-weighted traffic map from committed PIBT moves and
    # re-guiding each restart's dive around the busy edges. This gate pins the
    # reproduction WIN at EQUAL total budget (optimize gets rounds*budget in one
    # run; LTM spends the same across `rounds` restarts) so the gain is the
    # mechanism, not extra iterations: aggregate LTM sum-of-costs must stay below
    # plain optimize's, every instance must improve, and every solution must be
    # valid. If the traffic-map guidance regresses, sum_ltm rises and it trips.
    import random

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.conflicts import detect_first_conflict
    from mrn_coord.mapf.lacam import lacam, lacam_ltm
    from mrn_coord.mapf.solution import pad_paths

    def _soc(sol) -> int:
        total = 0
        for p in sol.paths.values():
            goal, arrival = p[-1], 0
            for t, c in enumerate(p):
                if c != goal:
                    arrival = t + 1
            total += arrival
        return total

    def _valid(grid, agents, sol) -> bool:
        if sol is None or detect_first_conflict(pad_paths(sol.paths)) is not None:
            return False
        return all(sol.paths[a][0] == agents[a][0]
                   and sol.paths[a][-1] == agents[a][1] for a in agents)

    rounds, budget = 4, 6000
    instances = sum_opt = sum_ltm = ltm_wins = valid_all = 0
    for w, h, n, seeds in ((10, 10, 20, range(4)), (12, 12, 30, range(2))):
        grid = GridWorld(w, h)
        cells = [(x, y) for x in range(w) for y in range(h)]
        for seed in seeds:
            rng = random.Random(seed * 1000 + w * 7 + n)
            starts, goals = rng.sample(cells, n), rng.sample(cells, n)
            agents = {i: (starts[i], goals[i]) for i in range(n)}
            opt = lacam(grid, agents, optimize=True,
                        max_iterations=rounds * budget)   # equal total budget
            ltm = lacam_ltm(grid, agents, rounds=rounds, max_iterations=budget,
                            optimize=True)
            instances += 1
            sum_opt += _soc(opt)
            sum_ltm += _soc(ltm)
            ltm_wins += int(_soc(ltm) < _soc(opt))
            valid_all += int(_valid(grid, agents, opt)
                             and _valid(grid, agents, ltm))
    return {"case": "lacam_ltm_vs_optimize", "instances": instances,
            "sum_opt": sum_opt, "sum_ltm": sum_ltm, "ltm_wins": ltm_wins,
            "valid_all": valid_all, "ltm_beats_opt": sum_ltm < sum_opt}


def _run_lns_scaling_improvement() -> dict:
    # LNS's selling point is anytime cost improvement "on team sizes far beyond
    # CBS's reach" -- but the only other LNS gate pins a single 3-agent example's
    # final cost, never the *improvement* and never at scale. (The unit tests cover
    # <=5x5 / <=5 agents only.) Same gap-pattern LaCAM's scaling claim had: a true
    # claim with no guard. This battery is the regime the claim lives in -- 16-20
    # agents on open grids -- and pins the aggregate destroy-repair gain: total SOC
    # must fall from `sum_initial` to `sum_final` (here ~1.23x -> ~1.14x the
    # lower bound), improving most instances. If repair silently stops accepting or
    # the worst-neighborhood heuristic degrades, sum_final rises and the gate trips.
    import random
    from collections import deque

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.lns import mapf_lns

    def _lb(grid, starts, goals, n):
        total = 0
        for i in range(n):
            dist = {goals[i]: 0}
            q = deque([goals[i]])
            while q:
                c = q.popleft()
                for nb in grid.neighbors(c):
                    if nb not in dist:
                        dist[nb] = dist[c] + 1
                        q.append(nb)
            total += dist.get(starts[i], 0)
        return total

    instances = improved = sum_initial = sum_final = sum_lower_bound = 0
    for w, h, n in ((8, 8, 16), (10, 10, 20)):
        grid = GridWorld(w, h)
        cells = [(x, y) for x in range(w) for y in range(h)]
        for seed in range(8):
            rng = random.Random(seed)
            starts, goals = rng.sample(cells, n), rng.sample(cells, n)
            agents = {i: (starts[i], goals[i]) for i in range(n)}
            stats: dict = {}
            sol = mapf_lns(grid, agents, iterations=100, seed=0, stats=stats)
            if sol is None:
                continue
            instances += 1
            improved += int(stats["final_cost"] < stats["initial_cost"])
            sum_initial += stats["initial_cost"]
            sum_final += stats["final_cost"]
            sum_lower_bound += _lb(grid, starts, goals, n)
    return {"case": "lns_scaling_improvement", "instances": instances,
            "improved": improved, "sum_initial": sum_initial,
            "sum_final": sum_final, "sum_lower_bound": sum_lower_bound}


def _run_lns_adaptive_vs_fixed() -> dict:
    # `mapf_lns(adaptive=True)` is a faithful port of BALANCE (Phan et al., AAAI
    # 2024): a bi-level Thompson-Sampling bandit that learns the destroy
    # heuristic and neighborhood size online, where vanilla LNS uses a fixed
    # 50/50 coin and a fixed size. BALANCE reports >=50% cost gains -- but on a
    # specialized SIPP repair, structured warehouse maps, and thousands of
    # iterations. Ported onto THIS repo's prioritized-A* repair and measured
    # honestly across open grids and budgets up to 1000 iters, the bandit does
    # NOT beat the fixed ensemble: it loses ~2% because the repo's `worst`
    # heuristic is already strong and the bandit's early exploration cost is
    # never recovered at these scales. This gate PINS that negative result --
    # aggregate adaptive SOC stays >= fixed SOC -- so nobody later overstates
    # "we have adaptive LNS, it's better", and so a future change that *did*
    # make adaptive win would trip the gate and force the claim to be re-pinned.
    import random

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.lns import mapf_lns

    instances = sum_initial = sum_fixed = sum_adaptive = 0
    fixed_wins = adaptive_wins = ties = 0
    for w, h, n in ((10, 10, 20), (12, 12, 30)):
        grid = GridWorld(w, h)
        cells = [(x, y) for x in range(w) for y in range(h)]
        for seed in range(4):
            rng = random.Random(seed * 1000 + w * 7 + n)
            starts, goals = rng.sample(cells, n), rng.sample(cells, n)
            agents = {i: (starts[i], goals[i]) for i in range(n)}
            sf: dict = {}
            sa: dict = {}
            mapf_lns(grid, agents, iterations=80, seed=0, stats=sf,
                     adaptive=False)
            mapf_lns(grid, agents, iterations=80, seed=0, stats=sa,
                     adaptive=True)
            instances += 1
            sum_initial += sf["initial_cost"]
            sum_fixed += sf["final_cost"]
            sum_adaptive += sa["final_cost"]
            if sa["final_cost"] < sf["final_cost"]:
                adaptive_wins += 1
            elif sa["final_cost"] > sf["final_cost"]:
                fixed_wins += 1
            else:
                ties += 1
    return {"case": "lns_adaptive_vs_fixed", "instances": instances,
            "sum_initial": sum_initial, "sum_fixed": sum_fixed,
            "sum_adaptive": sum_adaptive, "fixed_wins": fixed_wins,
            "adaptive_wins": adaptive_wins, "ties": ties,
            "adaptive_not_better": sum_adaptive >= sum_fixed}


def _run_rhcr(agents: int = 6, steps: int = 120, allocator: str = "stream",
              rows: int = 2, cols: int = 3, aisle: int = 1, window: int = 8,
              replan_period: int = 4, solver: str = "pbs",
              case: str = "mapf_rhcr") -> dict:
    from mrn_coord.lifelong import TaskStream, make_warehouse, run_rhcr

    grid, endpoints = make_warehouse(rows=rows, cols=cols, aisle=aisle)
    starts = {f"r{i}": endpoints[i] for i in range(min(agents, len(endpoints)))}
    res = run_rhcr(grid, starts, TaskStream(list(endpoints)), max_steps=steps,
                   window=window, replan_period=replan_period, solver=solver,
                   allocator=allocator)
    out = res.as_dict()
    out["case"] = case
    return out


# (case name, producer) — each returns a flat metrics dict.
SUITE = [
    ("sim_around_obstacle", lambda: _run_sim_scenario("around_obstacle")),
    ("sim_crossing", lambda: _run_sim_scenario("crossing")),
    ("sim_doorway", lambda: _run_sim_scenario("doorway")),
    ("sim_crossing_orca", lambda: _run_sim_scenario("crossing", "orca")),
    ("sim_doorway_orca", lambda: _run_sim_scenario("doorway", "orca")),
    # continuous-space Hybrid A* planner (kinodynamic)
    ("sim_around_obstacle_kino", lambda: _run_sim_scenario("around_obstacle", "kinodynamic")),
    ("sim_crossing_kino", lambda: _run_sim_scenario("crossing", "kinodynamic")),
    ("sim_doorway_kino", lambda: _run_sim_scenario("doorway", "kinodynamic")),
    # DWA local controller (grid plan + dynamic-window tracking)
    ("sim_around_obstacle_dwa", lambda: _run_sim_scenario("around_obstacle", "dwa")),
    ("sim_doorway_dwa", lambda: _run_sim_scenario("doorway", "dwa")),
    # MPC local controller (grid plan + iLQR receding-horizon optimization)
    ("sim_around_obstacle_mpc", lambda: _run_sim_scenario("around_obstacle", "mpc")),
    ("sim_crossing_mpc", lambda: _run_sim_scenario("crossing", "mpc")),
    ("sim_doorway_mpc", lambda: _run_sim_scenario("doorway", "mpc")),
    # MPC with a control-barrier-function QP safety filter (steer, don't brake)
    ("sim_crossing_mpc_cbf", lambda: _run_sim_scenario("crossing", "mpc_cbf")),
    ("sim_doorway_mpc_cbf", lambda: _run_sim_scenario("doorway", "mpc_cbf")),
    # certified body-true safety shield (steer/brake QP) vs an adversary
    ("sim_crossing_mpc_shield", lambda: _run_sim_scenario("crossing", "mpc_shield")),
    ("sim_doorway_mpc_shield", lambda: _run_sim_scenario("doorway", "mpc_shield")),
    ("shield_certify", _run_shield_certify),
    ("shield_certify_reciprocal", _run_shield_reciprocal),
    ("mapf_example_cbs", lambda: _run_mapf_example("cbs")),
    # bounded-suboptimal ECBS (cost <= w * optimal)
    ("mapf_example_ecbs", lambda: _run_mapf_example("ecbs", weight=1.5)),
    # complete satisficing LaCAM (configuration-space search via PIBT)
    ("mapf_example_lacam", lambda: _run_mapf_example("lacam")),
    # anytime large-neighborhood search (destroy & repair)
    ("mapf_example_lns", lambda: _run_mapf_example("lns")),
    # priority-ordering search (PBS): suboptimal but reorders past deadlocks
    ("mapf_example_pbs", lambda: _run_mapf_example("pbs")),
    ("mapf_example_prioritized", lambda: _run_mapf_example("prioritized")),
    # same prioritized planner, safe-interval (SIPP) low level
    ("mapf_example_prioritized_sipp", lambda: _run_mapf_example("prioritized_sipp")),
    # lifelong / online MAPF throughput (PIBT), with each task allocator
    ("mapf_lifelong", _run_lifelong),
    ("mapf_lifelong_auction", lambda: _run_lifelong(allocator="auction")),
    ("mapf_lifelong_hungarian", lambda: _run_lifelong(allocator="hungarian")),
    # fleet scale (40 AMRs in a 4x6 warehouse, the README hero): throughput is
    # the metric that actually matters, and the contract is that the cost-aware
    # allocators keep their large lead over geometry-blind round-robin. A change
    # that silently degrades fleet throughput (or neutralizes the allocator)
    # fails here, where the small 6-AMR cases above are too easy to show it.
    ("mapf_fleet_stream",
     lambda: _run_lifelong(agents=40, steps=60, rows=4, cols=6,
                           case="mapf_fleet_stream")),
    ("mapf_fleet_hungarian",
     lambda: _run_lifelong(agents=40, steps=60, rows=4, cols=6,
                           allocator="hungarian", case="mapf_fleet_hungarian")),
    ("mapf_fleet_auction",
     lambda: _run_lifelong(agents=40, steps=60, rows=4, cols=6,
                           allocator="auction", case="mapf_fleet_auction")),
    # RHCR (Rolling-Horizon Collision Resolution, Li et al. 2021): lifelong MAPF
    # by windowed *planning* (commit h steps, resolve conflicts w steps ahead)
    # instead of one-step PIBT. Pins the PBS and PP windowed solvers on the small
    # warehouse, plus the framework at fleet scale (where the planning solver
    # yields to a PIBT rollout in the cramped aisles — see docs/coordination.md).
    ("mapf_rhcr", lambda: _run_rhcr(case="mapf_rhcr")),
    ("mapf_rhcr_pp", lambda: _run_rhcr(solver="pp", case="mapf_rhcr_pp")),
    ("mapf_rhcr_hungarian",
     lambda: _run_rhcr(allocator="hungarian", case="mapf_rhcr_hungarian")),
    ("mapf_rhcr_fleet",
     lambda: _run_rhcr(agents=40, steps=60, rows=4, cols=6, window=10,
                       replan_period=2, solver="pibt", allocator="hungarian",
                       case="mapf_rhcr_fleet")),
    # the other side of the crossover: widen the aisles (aisle=2) and the
    # congestion that lets greedy PIBT win on the cramped map relaxes, so RHCR's
    # windowed PBS lookahead clears *more* tasks than PIBT (327 vs 310 here, a
    # contract gated alongside in test_rhcr). This is the paper's regime — RHCR
    # winning on a reasonable map — and where PBS is also fast again.
    ("mapf_rhcr_open",
     lambda: _run_rhcr(agents=16, steps=80, rows=3, cols=4, aisle=2, window=10,
                       replan_period=1, solver="pbs", allocator="hungarian",
                       case="mapf_rhcr_open")),
    # deterministic livelock escape recovers PIBT convergence (no randomness)
    ("pibt_escape_convergence", _run_pibt_convergence),
    # strong-PIBT spine makes LaCAM's documented scaling actually deliver
    ("lacam_scaling_convergence", _run_lacam_convergence),
    # LaCAM* anytime mode reaches the CBS optimum on small instances
    ("lacam_optimality", _run_lacam_optimality),
    ("lacam_ltm_vs_optimize", _run_lacam_ltm_vs_optimize),
    # LNS destroy-repair lowers sum-of-costs at scale (anytime improvement)
    ("lns_scaling_improvement", _run_lns_scaling_improvement),
    ("lns_adaptive_vs_fixed", _run_lns_adaptive_vs_fixed),
    # executing a discrete MAPF plan in the continuous world (plan vs reality)
    ("mapf_exec_tpg", lambda: _run_mapf_exec("tpg")),
    ("mapf_exec_dwa", lambda: _run_mapf_exec("dwa")),
    # free-running pursuit under the certified shield: fails safe (collision-free
    # but deadlocks at the symmetric merge) where bare pursuit collides
    ("mapf_exec_shield", lambda: _run_mapf_exec("shield")),
]


def _compare(expected: dict, actual: dict) -> list:
    """Return a list of human-readable mismatch strings (empty == pass)."""
    diffs = []
    for key, exp in expected.items():
        act = actual.get(key, "<missing>")
        if isinstance(exp, bool) or isinstance(exp, int) or isinstance(exp, str):
            if act != exp:
                diffs.append(f"{key}: expected {exp!r}, got {act!r}")
        elif isinstance(exp, float):
            if not isinstance(act, (int, float)) or abs(act - exp) > _FLOAT_TOL:
                diffs.append(f"{key}: expected {exp} ± {_FLOAT_TOL}, got {act}")
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true",
                        help="rewrite the expected metrics from current results")
    args = parser.parse_args()
    os.makedirs(_EXPECTED_DIR, exist_ok=True)

    failures = 0
    for case, run in SUITE:
        actual = run()
        path = os.path.join(_EXPECTED_DIR, case + ".json")
        if args.update:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(actual, fh, indent=2, sort_keys=True)
            print(f"updated {case}")
            continue
        if not os.path.exists(path):
            print(f"FAIL {case}: no expected metrics ({path})")
            failures += 1
            continue
        with open(path, "r", encoding="utf-8") as fh:
            expected = json.load(fh)
        diffs = _compare(expected, actual)
        if diffs:
            failures += 1
            print(f"FAIL {case}:")
            for d in diffs:
                print(f"    {d}")
        else:
            print(f"ok   {case}")

    if args.update:
        return 0
    print(f"\n{len(SUITE) - failures}/{len(SUITE)} benchmark cases within expectation")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
