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


def _run_mapf_lns2() -> dict:
    # MAPF-LNS2 (lns2.py) is a Python reproduction of Li, Chen, Harabor, Stuckey &
    # Koenig's "MAPF-LNS2: Fast Repairing for MAPF via Large Neighborhood Search"
    # (AAAI 2022). Where the optimizer mapf_lns starts FEASIBLE and polishes
    # sum-of-costs (every repair collision-free by construction), MAPF-LNS2 solves
    # the harder PRIOR problem -- finding a feasible solution at all -- by starting
    # from each agent's individual shortest path (collision-ridden) and MINIMIZING
    # THE NUMBER OF COLLISIONS with LNS until it hits zero. Its low level is
    # collision-MINIMIZING (other paths are soft penalties, not walls), so it makes
    # progress on tangles that have no collision-free completion yet.
    #
    # The gate pins two regimes plus the soundness of the collision count:
    # (1) REPAIR -- a battery of feasible 8x8/9-agent instances, each started from
    #     a colliding shortest-path solution (aggregate initial_collisions 26),
    #     driven to zero (repair_feasible == instances, every solution truly
    #     collision-free);
    # (2) SCALE -- dense 6x6/12-agent instances on which plain CBS exhausts a
    #     2000-node budget (scale_cbs_busts == 3) yet MAPF-LNS2 still repairs to a
    #     feasible solution (scale_feasible == 3) from an aggregate 36 collisions;
    # (3) SOUNDNESS -- the stats["feasible"] flag (final count == 0) agrees with an
    #     independent detect_first_conflict on EVERY instance (counts_match_cf),
    #     so the feasibility guarantee rests on the exact global count.
    # It is honest about being anytime/incomplete: the claim is "drives these to
    # zero within budget", not "complete".
    import random

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.cbs import cbs
    from mrn_coord.mapf.conflicts import detect_first_conflict
    from mrn_coord.mapf.lns2 import mapf_lns2

    def inst(n, w, h, seed):
        rng = random.Random(seed)
        free = [(x, y) for x in range(w) for y in range(h)]
        cells = rng.sample(free, 2 * n)
        return (GridWorld(w, h),
                {i: (cells[i], cells[n + i]) for i in range(n)})

    counts_match_cf = all_cf = True
    repair_inst = repair_feasible = repair_initial = 0
    for seed in (0, 2, 3, 4, 5, 6):
        grid, agents = inst(9, 8, 8, seed)
        s: dict = {}
        sol = mapf_lns2(grid, agents, iterations=600, neighborhood_size=8,
                        seed=1, stats=s)
        repair_inst += 1
        repair_initial += s["initial_collisions"]
        repair_feasible += int(s["feasible"])
        cf = detect_first_conflict(sol.paths) is None
        counts_match_cf = counts_match_cf and (cf == s["feasible"])
        all_cf = all_cf and cf

    scale_inst = scale_feasible = scale_cbs_busts = scale_initial = 0
    for seed in (0, 1, 2):
        grid, agents = inst(14, 6, 6, seed)
        base = cbs(grid, agents, max_expansions=2000)
        s = {}
        sol = mapf_lns2(grid, agents, iterations=600, neighborhood_size=10,
                        seed=1, stats=s)
        scale_inst += 1
        scale_initial += s["initial_collisions"]
        scale_feasible += int(s["feasible"])
        scale_cbs_busts += int(base is None)
        cf = detect_first_conflict(sol.paths) is None
        counts_match_cf = counts_match_cf and (cf == s["feasible"])
        all_cf = all_cf and cf

    return {"case": "mapf_lns2",
            "repair_instances": repair_inst,
            "repair_initial_collisions": repair_initial,
            "repair_feasible": repair_feasible,
            "scale_instances": scale_inst,
            "scale_initial_collisions": scale_initial,
            "scale_feasible": scale_feasible,
            "scale_cbs_busts": scale_cbs_busts,
            "counts_match_cf": counts_match_cf,
            "all_collision_free": all_cf,
            "drives_to_zero": repair_feasible == repair_inst
            and scale_feasible == scale_inst}


def _run_cbsh_vs_cbs() -> dict:
    # CBSH (cbsh) is a Python reproduction of Li et al.'s "Improved Heuristics
    # for MAPF with Conflict-Based Search" (IJCAI 2019) plus ICBS conflict
    # prioritization. It returns the SAME optimal sum-of-costs as plain cbs but
    # expands far fewer high-level nodes by adding an admissible CG/DG/WDG
    # heuristic and splitting cardinal conflicts first. This gate pins that win
    # on a battery where plain CBS's constraint tree actually blows up -- an
    # obstacle-dense 6x6 with 6 agents is where the gap is largest. It locks:
    # (1) optimality -- cbsh's cost equals cbs's on every instance (opt_match);
    # (2) the expansion counts per variant, monotone wdg <= dg <= cg <= cbs in
    # aggregate (the whole point: stronger heuristic, fewer nodes). If a change
    # weakens the heuristic or breaks classification, the cbsh sums rise toward
    # cbs and the gate trips; if it ever breaks optimality, opt_match goes False.
    import random

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.cbs import cbs
    from mrn_coord.mapf.cbsh import cbsh

    def _instance(w, h, n, seed, obstacle):
        rng = random.Random(seed)
        blocked = {(x, y) for x in range(w) for y in range(h)
                   if rng.random() < obstacle}
        free = [(x, y) for x in range(w) for y in range(h)
                if (x, y) not in blocked]
        rng.shuffle(free)
        starts = free[:n]
        rng.shuffle(free)
        goals = free[:n]
        grid = GridWorld(w, h, frozenset(blocked))
        return grid, {i: (starts[i], goals[i]) for i in range(n)}

    instances = exp_cbs = exp_cg = exp_dg = exp_wdg = opt_match = 0
    # First 10 seeds of two configs: an open 7x7/8 and a conflict-heavy 6x6/6
    # at 12% obstacles (no cherry-picking -- plain `range(10)`).
    for w, h, n, obstacle in ((7, 7, 8, 0.0), (6, 6, 6, 0.12)):
        for seed in range(10):
            grid, agents = _instance(w, h, n, seed, obstacle)
            s: dict = {}
            base = cbs(grid, agents, stats=s, max_expansions=20000)
            if base is None:
                continue
            instances += 1
            exp_cbs += s["expansions"]
            costs = {"cbs": base.cost}
            for mode, acc in (("cg", "cg"), ("dg", "dg"), ("wdg", "wdg")):
                sh: dict = {}
                sol = cbsh(grid, agents, heuristic=mode, stats=sh,
                           max_expansions=20000)
                costs[mode] = None if sol is None else sol.cost
                if mode == "cg":
                    exp_cg += sh["expansions"]
                elif mode == "dg":
                    exp_dg += sh["expansions"]
                else:
                    exp_wdg += sh["expansions"]
            opt_match += int(all(c == base.cost for c in costs.values()))
    return {"case": "cbsh_vs_cbs", "instances": instances,
            "exp_cbs": exp_cbs, "exp_cg": exp_cg, "exp_dg": exp_dg,
            "exp_wdg": exp_wdg, "opt_match": opt_match,
            "monotone": exp_wdg <= exp_dg <= exp_cg <= exp_cbs}


def _run_eecbs_vs_ecbs() -> dict:
    # EECBS (eecbs) is a Python reproduction of Li, Ruml & Koenig's "EECBS: A
    # Bounded-Suboptimal Search for MAPF" (AAAI 2021). It keeps ECBS's focal,
    # conflict-avoiding low level but replaces ECBS's loose lower bound (sum of
    # the agents' individual optima) with CBSH's admissible WDG heuristic, and
    # drives the high level with Explicit Estimation Search. At a near-optimal
    # suboptimality factor (here w = 1.02) the tighter bound certifies the
    # solution after far fewer high-level expansions. This gate pins that win
    # against the existing ecbs at the SAME w, on the same battery cbsh uses.
    #
    # It locks the expansion counts per variant. Two structural invariants make
    # the win unambiguous and attributable: (1) eecbs(heuristic=None) -- the EES
    # skeleton with h = 0 -- expands EXACTLY as many nodes as ecbs (none==ecbs),
    # so EES alone changes nothing; (2) adding the heuristic is monotone,
    # wdg <= dg <= cg <= none, i.e. the entire reduction comes from the
    # admissible bound. The cost bound (<= w * optimal) is verified in the unit
    # tests, where the cbs optimum is available; the gate stays on the cheap
    # performance contract. If a change weakens the heuristic the eecbs sums
    # rise back toward ecbs and the gate trips.
    import random

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.ecbs import ecbs
    from mrn_coord.mapf.eecbs import eecbs

    def _instance(w, h, n, seed, obstacle):
        rng = random.Random(seed)
        blocked = {(x, y) for x in range(w) for y in range(h)
                   if rng.random() < obstacle}
        free = [(x, y) for x in range(w) for y in range(h)
                if (x, y) not in blocked]
        rng.shuffle(free)
        starts = free[:n]
        goals = free[n:2 * n]
        grid = GridWorld(w, h, frozenset(blocked))
        return grid, {i: (starts[i], goals[i]) for i in range(n)}

    weight = 1.02
    instances = exp_ecbs = exp_none = exp_cg = exp_dg = exp_wdg = 0
    wdg_le_ecbs = 0
    # First 12 seeds of two configs: a conflict-heavy 8x8/7 at 12% obstacles and
    # an open 7x7/8 (no cherry-picking -- plain `range(12)`).
    for w, h, n, obstacle in ((8, 8, 7, 0.12), (7, 7, 8, 0.0)):
        for seed in range(12):
            grid, agents = _instance(w, h, n, seed, obstacle)
            se: dict = {}
            base = ecbs(grid, agents, w=weight, stats=se, max_expansions=20000)
            if base is None:
                continue
            row = {}
            ok = True
            for mode in (None, "cg", "dg", "wdg"):
                sh: dict = {}
                sol = eecbs(grid, agents, w=weight, heuristic=mode, stats=sh,
                            max_expansions=20000)
                if sol is None:
                    ok = False
                    break
                row[mode] = sh["expansions"]
            if not ok:
                continue
            instances += 1
            exp_ecbs += se["expansions"]
            exp_none += row[None]
            exp_cg += row["cg"]
            exp_dg += row["dg"]
            exp_wdg += row["wdg"]
            wdg_le_ecbs += int(row["wdg"] <= se["expansions"])
    return {"case": "eecbs_vs_ecbs", "instances": instances,
            "exp_ecbs": exp_ecbs, "exp_none": exp_none, "exp_cg": exp_cg,
            "exp_dg": exp_dg, "exp_wdg": exp_wdg,
            "none_eq_ecbs": exp_none == exp_ecbs,
            "wdg_le_ecbs": wdg_le_ecbs,
            "monotone": exp_wdg <= exp_dg <= exp_cg <= exp_none}


def _run_icts_vs_cbs() -> dict:
    # ICTS (icts) is a Python reproduction of Sharon, Stern, Goldenberg & Felner's
    # "The increasing cost tree search for optimal multi-agent pathfinding" (AIJ
    # 2013). It is an optimal paradigm ORTHOGONAL to CBS: it branches on per-agent
    # COSTS (the Increasing Cost Tree) rather than on constraints, and tests each
    # cost vector by searching the cross-product of the agents' MDDs for a
    # conflict-free joint path. It returns the SAME optimal sum-of-costs as plain
    # cbs (opt_match). The win it does claim -- and what this gate pins -- is its
    # signature accelerator: PAIRWISE PRUNING. Before each expensive k-agent joint
    # search, every pair of agents is checked in isolation (the same 2-agent MDD
    # dependency test cbsh's WDG uses); a node with any dependent pair is hopeless
    # and skipped. With pruning the joint searches drop sharply versus the
    # ablation icts(prune=None), which searches every node (joint_searches_none ==
    # nodes). Both settings find the identical optimum -- pruning changes only the
    # work. If pruning ever regresses, joint_searches_pairwise rises toward
    # joint_searches_none and the gate trips.
    import random

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.cbs import cbs
    from mrn_coord.mapf.icts import icts

    def _instance(w, h, n, seed, obstacle):
        rng = random.Random(seed)
        blocked = {(x, y) for x in range(w) for y in range(h)
                   if rng.random() < obstacle}
        free = [(x, y) for x in range(w) for y in range(h)
                if (x, y) not in blocked]
        rng.shuffle(free)
        starts = free[:n]
        goals = free[n:2 * n]
        grid = GridWorld(w, h, frozenset(blocked))
        return grid, {i: (starts[i], goals[i]) for i in range(n)}

    instances = opt_match = nodes = 0
    js_pairwise = js_none = pruned = 0
    # First 12 seeds of three small configs. ICTS's joint search is exponential
    # in the agent count, so the battery stays few-but-coupled (4-5 agents):
    # exactly the regime where ICTS is meant to compete (no cherry-picking --
    # plain `range(12)`).
    for w, h, n, obstacle in ((6, 6, 5, 0.12), (7, 7, 5, 0.0), (6, 6, 4, 0.15)):
        for seed in range(12):
            grid, agents = _instance(w, h, n, seed, obstacle)
            base = cbs(grid, agents, max_expansions=20000)
            if base is None:
                continue
            sp: dict = {}
            sol = icts(grid, agents, prune="pairwise", stats=sp,
                       max_nodes=20000)
            sn: dict = {}
            soln = icts(grid, agents, prune=None, stats=sn, max_nodes=20000)
            if sol is None or soln is None:
                continue
            instances += 1
            opt_match += int(sol.cost == base.cost and soln.cost == base.cost)
            nodes += sp["nodes"]
            js_pairwise += sp["joint_searches"]
            js_none += sn["joint_searches"]
            pruned += sp["pruned"]
    return {"case": "icts_vs_cbs", "instances": instances,
            "opt_match": opt_match, "nodes": nodes,
            "joint_searches_pairwise": js_pairwise,
            "joint_searches_none": js_none, "pruned": pruned,
            "pruning_helps": js_pairwise < js_none}


def _run_rectangle_symmetry() -> dict:
    # Rectangle symmetry reasoning (cbsh's rectangle=True, rectangle.py) is a
    # Python reproduction of Li, Harabor, Stuckey, Felner & Koenig's
    # "Symmetry-Breaking Constraints for Grid-Based MAPF" (AAAI 2019). When two
    # agents cross the same open rectangular region in the same direction, every
    # pair of their optimal paths collides; plain CBS/CBSH must enumerate an
    # exponential number of symmetric one-cell resolutions before escaping. A
    # *barrier* constraint blocks a whole exit border at once and breaks the
    # symmetry in a single split, provably preserving the optimum.
    #
    # Random instances almost never contain a (phase-locked, same-direction)
    # rectangle symmetry, so -- as the paper does on structured maps -- this gate
    # uses explicit crossing scenarios: agents whose starts share an
    # anti-diagonal (x+y = const, which phase-locks them) heading up-and-right
    # into a shared open rectangle. It compares cbsh with rectangle reasoning ON
    # vs OFF (everything else, including the WDG heuristic, identical, so the
    # delta is purely the barrier split), and pins: (1) optimality -- the
    # rectangle cost equals both plain-cbsh's and CBS's on every scenario
    # (opt_match); (2) the collapse -- barrier splits cut the aggregate
    # high-level expansions 298 -> 15 (~20x). If a barrier ever dropped a
    # solution, opt_match would fall; if rectangle detection regressed, the
    # rectangle expansions would rise back toward the OFF baseline.
    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.cbs import cbs
    from mrn_coord.mapf.cbsh import cbsh
    from mrn_coord.mapf.conflicts import detect_first_conflict

    # (name, width, height, {agent: (start, goal)}). Each is a same-direction
    # anti-diagonal crossing that forms a rectangle symmetry.
    scenarios = [
        ("cross6", 6, 6, {0: ((2, 0), (4, 5)), 1: ((1, 1), (4, 2))}),
        ("cross7c", 7, 7, {0: ((1, 1), (5, 6)), 1: ((0, 2), (6, 3))}),
        ("cross7b", 7, 7, {0: ((2, 0), (6, 6)), 1: ((0, 2), (6, 4))}),
        ("cross7a", 7, 7, {0: ((2, 0), (4, 6)), 1: ((0, 2), (5, 6))}),
    ]
    scn = opt_match = rectangles = exp_off = exp_on = 0
    all_valid = True
    for _, w, h, agents in scenarios:
        grid = GridWorld(w, h)
        son: dict = {}
        on = cbsh(grid, agents, heuristic="wdg", rectangle=True, stats=son,
                  max_expansions=20000)
        soff: dict = {}
        off = cbsh(grid, agents, heuristic="wdg", rectangle=False, stats=soff,
                   max_expansions=20000)
        base = cbs(grid, agents, max_expansions=20000)
        scn += 1
        rectangles += son["rectangles"]
        exp_on += son["expansions"]
        exp_off += soff["expansions"]
        opt_match += int(on.cost == off.cost == base.cost)
        all_valid = all_valid and detect_first_conflict(on.paths) is None
    return {"case": "rectangle_symmetry", "scenarios": scn,
            "opt_match": opt_match, "rectangles": rectangles,
            "exp_off": exp_off, "exp_on": exp_on,
            "all_valid": all_valid, "reduces": exp_on < exp_off}


def _run_corridor_symmetry() -> dict:
    # Corridor symmetry reasoning (cbsh's corridor=True, corridor.py) is a Python
    # reproduction of Li, Harabor, Stuckey, Felner & Koenig's "New Techniques for
    # Pairwise Symmetry Breaking in Multi-Agent Path Finding" (ICAPS 2020) -- the
    # third leg of the symmetry-breaking trilogy alongside rectangle.py and
    # mutex.py. When two agents traverse the same one-wide passage in OPPOSITE
    # directions, the head-on conflict can be shifted one cell at a time, so plain
    # CBS/CBSH branches a chain whose length grows with the corridor. A *range*
    # constraint -- forbidding an agent from the shared entry opening across a
    # whole band of timesteps -- holds it outside until the other has cleared,
    # collapsing the chain to a single split.
    #
    # The reasoning is exhaustive (so optimality-preserving) only when the
    # corridor is the SOLE route between its two sides, so it fires there and
    # falls back to a plain split when a bypass exists. This gate uses hand-built
    # forced corridors of increasing length and pins: (1) optimality -- the
    # corridor cost equals plain-cbsh's and CBS's on every scenario (opt_match);
    # (2) the collapse -- one range split per corridor (corridors == scenarios)
    # cuts aggregate expansions from a length-growing 52 to a constant 8; (3)
    # honesty -- on corridors that DO have a bypass the reasoning declines
    # (bypass_corridors == 0) yet still returns the optimum (bypass_opt_match).
    # If a range split ever dropped a solution, opt_match would fall; if detection
    # regressed, exp_on would climb back toward exp_off.
    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.cbs import cbs
    from mrn_coord.mapf.cbsh import cbsh
    from mrn_coord.mapf.conflicts import detect_first_conflict

    def forced(L):
        # 1-wide corridor (row y=1, x=1..L); the y=0/y=2 strips beside it are
        # walled, so the corridor is the only crossing. Agents swap sides.
        blocked = {(x, 0) for x in range(1, L + 1)} | \
                  {(x, 2) for x in range(1, L + 1)}
        return (GridWorld(L + 2, 3, blocked=frozenset(blocked)),
                {0: ((0, 0), (L + 1, 2)), 1: ((L + 1, 0), (0, 2))})

    def bypass(L):
        # Same corridor but the y=0 strip is OPEN -> a detour exists, so corridor
        # reasoning must decline and fall back.
        blocked = {(x, 2) for x in range(1, L + 1)}
        return (GridWorld(L + 2, 3, blocked=frozenset(blocked)),
                {0: ((0, 1), (L + 1, 1)), 1: ((L + 1, 1), (0, 1))})

    scn = opt_match = corridors = exp_off = exp_on = 0
    all_valid = True
    for L in (2, 3, 4, 5):
        grid, agents = forced(L)
        son: dict = {}
        on = cbsh(grid, agents, heuristic="wdg", corridor=True, stats=son,
                  max_expansions=20000)
        soff: dict = {}
        off = cbsh(grid, agents, heuristic="wdg", corridor=False, stats=soff,
                   max_expansions=20000)
        base = cbs(grid, agents, max_expansions=20000)
        scn += 1
        corridors += son["corridors"]
        exp_on += son["expansions"]
        exp_off += soff["expansions"]
        opt_match += int(on.cost == off.cost == base.cost)
        all_valid = all_valid and detect_first_conflict(on.paths) is None

    bypass_scn = bypass_corridors = bypass_opt_match = 0
    for L in (3, 4):
        grid, agents = bypass(L)
        s: dict = {}
        on = cbsh(grid, agents, heuristic="wdg", corridor=True, stats=s,
                  max_expansions=20000)
        base = cbs(grid, agents, max_expansions=20000)
        bypass_scn += 1
        bypass_corridors += s["corridors"]
        bypass_opt_match += int(on.cost == base.cost)
        all_valid = all_valid and detect_first_conflict(on.paths) is None

    return {"case": "corridor_symmetry", "scenarios": scn,
            "opt_match": opt_match, "corridors": corridors,
            "exp_off": exp_off, "exp_on": exp_on,
            "bypass_scenarios": bypass_scn,
            "bypass_corridors": bypass_corridors,
            "bypass_opt_match": bypass_opt_match,
            "all_valid": all_valid, "reduces": exp_on < exp_off}


def _run_mutex_cardinal_detection() -> dict:
    # Mutex propagation (mutex.py) is a Python reproduction of Zhang, Li, Surynek,
    # Koenig & Kumar's "Multi-Agent Path Finding with Mutex Propagation" (ICAPS
    # 2020). It propagates mutexes over a pair of MDDs to decide, in polynomial
    # time, whether two agents can be reached conflict-free -- and from that
    # classifies cardinal conflicts (pre-goal PC / after-goal AC / not-cardinal
    # NC) and synthesizes symmetry-breaking constraints automatically, generalizing
    # the hand-designed rectangle reasoning of rectangle.py.
    #
    # This gate pins the verified DETECTOR (not a brancher -- the paper's full
    # constraint-generation loop is impractically slow in pure Python; see
    # mutex.py). On a fixed battery of 2-agent MDD pairs it locks:
    # (1) THE CORRECTNESS GUARANTEE (the paper's Theorem 2): classify_conflict
    #     returns NC iff a conflict-free pair of optimal paths exists -- exactly
    #     what mdd.are_dependent computes directly -- so disagreements == 0;
    # (2) the classification distribution (PC / AC / NC counts);
    # (3) GENERALITY: hidden_cardinals counts cardinal pairs that the width-based
    #     test (cbsh's: a level where both MDDs are pinned to the same cell)
    #     MISSES but mutex catches -- the rectangle/corridor-type dependencies
    #     mutex was built for;
    # (4) every PC pair yields non-empty disjunctive constraint sets.
    # If propagation regressed, disagreements would go non-zero (the gate trips).
    import random

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.mdd import are_dependent, build_mdd
    from mrn_coord.mapf.mutex import classify_conflict, pc_constraints
    from mrn_coord.mapf.space_time_astar import plan_path

    def _width_cardinal(mi, mj) -> bool:
        # cbsh-style cardinal: a level where both agents are pinned (width 1) to
        # the same cell -- the only cardinals the width test alone can see.
        for t in range(max(mi.cost, mj.cost) + 1):
            if (mi.width(t) == 1 and mj.width(t) == 1
                    and mi.cells(t) == mj.cells(t)):
                return True
        return False

    pairs = disagreements = pc = ac = nc = hidden = pc_empty = 0
    # First 2500 seeds; each draws a 2-agent instance on a small open grid (no
    # cherry-picking -- plain `range(2500)`).
    for seed in range(2500):
        rng = random.Random(seed)
        w, h = rng.choice([(5, 5), (6, 5), (5, 6), (6, 6)])
        free = [(x, y) for x in range(w) for y in range(h)]
        rng.shuffle(free)
        sa, ga, sb, gb = free[:4]
        grid = GridWorld(w, h)
        pa = plan_path(grid, sa, ga)
        pb = plan_path(grid, sb, gb)
        if pa is None or pb is None:
            continue
        ca, cb = len(pa) - 1, len(pb) - 1
        if ca > cb:  # classify_conflict requires cost_i <= cost_j
            sa, ga, sb, gb, ca, cb = sb, gb, sa, ga, cb, ca
        mi = build_mdd(grid, sa, ga, ca)
        mj = build_mdd(grid, sb, gb, cb)
        if mi is None or mj is None:
            continue
        cls = classify_conflict(grid, mi, mj)
        dep = are_dependent(grid, mi, mj, sa, sb)
        pairs += 1
        if (cls == "NC") != (not dep):
            disagreements += 1
        if cls == "PC":
            pc += 1
        elif cls == "AC":
            ac += 1
        else:
            nc += 1
        if cls in ("PC", "AC"):
            if not _width_cardinal(mi, mj):
                hidden += 1
            if cls == "PC":
                ci, cj = pc_constraints(grid, mi, mj)
                if not (ci and cj):
                    pc_empty += 1
    return {"case": "mutex_cardinal_detection", "pairs": pairs,
            "disagreements": disagreements, "theorem2_holds": disagreements == 0,
            "pc": pc, "ac": ac, "nc": nc, "hidden_cardinals": hidden,
            "pc_constraints_empty": pc_empty}


def _run_disjoint_vs_standard() -> dict:
    # Disjoint splitting (cbs's disjoint=True) is a Python reproduction of Li,
    # Harabor, Stuckey, Ma & Koenig's "Disjoint Splitting for Multi-Agent Path
    # Finding with Conflict-Based Search" (ICAPS 2019). Standard CBS resolves a
    # vertex conflict (a1, a2, v, t) by forbidding v-at-t to a1 in one child and
    # to a2 in the other; the two subtrees OVERLAP -- every solution in which
    # neither agent sits on v at t satisfies both, so it is re-searched twice.
    # Disjoint splitting picks ONE agent ai and branches on ai-IS-at-(v,t)
    # (a positive constraint, which by vertex exclusivity also pins every other
    # agent OFF v at t) versus ai-is-NOT. Those children PARTITION the solution
    # space, so nothing is searched twice. It returns the SAME optimal
    # sum-of-costs as standard CBS (opt_match) -- the win it claims, and what
    # this gate pins, is FEWER high-level expansions, the saving growing with
    # congestion (where the redundant subtrees are largest).
    #
    # The positive half rides on plan_path's positive_vertex/positive_edge
    # support (verified separately: a must-occupy (v,t) path equals the path
    # found by forbidding every OTHER cell at t). Edge conflicts keep the
    # standard split (the positive-edge derivation for all other agents is
    # finicky and swaps are rare); mixing is still sound and optimal because
    # each individual split covers the whole solution space.
    #
    # Battery: congested few-but-coupled instances (where disjoint is meant to
    # help), first 12 seeds of three configs, no cherry-picking (plain
    # range(12)). Pins: (1) opt_match == instances (disjoint never drops the
    # optimum -- if a positive constraint ever lost a solution this falls);
    # (2) all_valid (every disjoint solution is conflict-free); (3) the
    # collapse: exp_disjoint < exp_standard. If disjoint splitting regressed,
    # exp_disjoint would rise back toward (or past) exp_standard and the gate
    # trips.
    import random

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.cbs import cbs
    from mrn_coord.mapf.conflicts import detect_first_conflict

    def _instance(w, h, n, seed, obstacle):
        rng = random.Random(seed)
        blocked = {(x, y) for x in range(w) for y in range(h)
                   if rng.random() < obstacle}
        free = [(x, y) for x in range(w) for y in range(h)
                if (x, y) not in blocked]
        rng.shuffle(free)
        grid = GridWorld(w, h, frozenset(blocked))
        return grid, {i: (free[i], free[n + i]) for i in range(n)}

    instances = opt_match = exp_standard = exp_disjoint = 0
    all_valid = True
    for w, h, n, obstacle in ((6, 6, 7, 0.05), (7, 7, 8, 0.05), (8, 8, 8, 0.05)):
        for seed in range(12):
            grid, agents = _instance(w, h, n, seed, obstacle)
            ss: dict = {}
            std = cbs(grid, agents, disjoint=False, stats=ss,
                      max_expansions=60000)
            sd: dict = {}
            dis = cbs(grid, agents, disjoint=True, stats=sd,
                      max_expansions=60000)
            if std is None or dis is None:
                continue
            instances += 1
            opt_match += int(std.cost == dis.cost)
            exp_standard += ss["expansions"]
            exp_disjoint += sd["expansions"]
            all_valid = all_valid and detect_first_conflict(dis.paths) is None
    return {"case": "disjoint_vs_standard", "instances": instances,
            "opt_match": opt_match, "exp_standard": exp_standard,
            "exp_disjoint": exp_disjoint, "all_valid": all_valid,
            "reduces": exp_disjoint < exp_standard}


def _run_ccbs_continuous_time() -> dict:
    # CCBS (ccbs.py) is a Python reproduction of Andreychuk, Yakovlev, Atzmon &
    # Stern's "Multi-Agent Pathfinding with Continuous Time" (IJCAI 2019 / AIJ
    # 2022). Classical CBS runs on a discrete clock: unit-timestep moves,
    # same-cell / swap conflicts, whole-timestep yields. CCBS drops the clock --
    # it plans over CONTINUOUS time on an 8-connected geometric roadmap (diagonal
    # moves take sqrt(2), an irrational duration the discrete model cannot even
    # represent), with each agent a DISK of radius r that collides whenever two
    # centres come within 2r at any real instant -- including mid-edge, where two
    # crossing paths share no vertex and no edge. Its low level is continuous-time
    # SIPP (yield by the minimal REAL duration, not a whole tick); its high level
    # branches on continuous unsafe intervals of starting the colliding action.
    #
    # This gate pins three things, against an independent geometric oracle
    # (min_separation, which solves the quadratic distance on every linear
    # segment -- not the planner's own conflict check):
    # (1) SOUNDNESS / the whole point: every CCBS solution keeps all pairs >= 2r
    #     apart (collision_free == solved). On a 3-agent battery (the regime where
    #     CCBS's continuous search converges; it is known to be expensive), first
    #     10 seeds of two configs, no cherry-picking (plain range(10)).
    # (2) IT CATCHES WHAT DISCRETE MISSES: baseline_collided counts instances
    #     whose uncoordinated 8-connected shortest paths geometrically collide
    #     (centres < 2r) -- conflicts a vertex/edge model is blind to -- which
    #     CCBS then resolves (all collision-free).
    # (3) THE CONTINUOUS SIGNATURE: four explicit crossing scenarios (two agents'
    #     paths cross mid-square, sharing no vertex/edge) are each resolved with a
    #     fractional real wait to exactly the 2r clearance; pinned by their summed
    #     cost. Every uncoordinated baseline there collides (centres meet at 0).
    # If continuous collision detection or interval resolution regressed, a CCBS
    # solution would dip below 2r and collision_free would fall (the gate trips).
    import random

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.ccbs import ccbs, min_separation, shortest_trajectory

    radius = 0.4
    thr = 2 * radius

    def _instance(w, h, n, seed):
        rng = random.Random(seed)
        free = [(x, y) for x in range(w) for y in range(h)]
        rng.shuffle(free)
        return GridWorld(w, h), {i: (free[i], free[n + i]) for i in range(n)}

    def _all_clear(trajs):
        ids = list(trajs)
        return all(min_separation(trajs[ids[i]], trajs[ids[j]]) >= thr - 1e-6
                   for i in range(len(ids)) for j in range(i + 1, len(ids)))

    def _any_collide(trajs):
        ids = list(trajs)
        return any(min_separation(trajs[ids[i]], trajs[ids[j]]) < thr - 1e-6
                   for i in range(len(ids)) for j in range(i + 1, len(ids)))

    instances = solved = collision_free = baseline_collided = 0
    cost_sum = 0.0
    for w, h, n in ((5, 5, 3), (6, 6, 3)):
        for seed in range(10):
            grid, agents = _instance(w, h, n, seed)
            sol = ccbs(grid, agents, radius=radius, max_expansions=20000)
            instances += 1
            if sol is None:
                continue
            solved += 1
            cost_sum += sol.cost
            if _all_clear(sol.trajectories):
                collision_free += 1
            base = {k: shortest_trajectory(grid, s, g)
                    for k, (s, g) in agents.items()}
            if _any_collide(base):
                baseline_collided += 1

    # Explicit mid-square crossings: no shared vertex/edge, yet the disks meet.
    crossings = [
        (5, 5, {0: ((0, 0), (2, 2)), 1: ((2, 0), (0, 2))}),
        (7, 7, {0: ((0, 0), (3, 3)), 1: ((3, 0), (0, 3))}),
        (5, 5, {0: ((0, 2), (4, 2)), 1: ((2, 0), (2, 4))}),
        (6, 6, {0: ((0, 0), (3, 3)), 1: ((3, 0), (0, 3)), 2: ((0, 3), (3, 0))}),
    ]
    x_scn = x_clear = x_base_collide = 0
    x_cost_sum = 0.0
    for w, h, agents in crossings:
        grid = GridWorld(w, h)
        sol = ccbs(grid, agents, radius=radius, max_expansions=20000)
        x_scn += 1
        if sol is None:
            continue
        x_cost_sum += sol.cost
        if _all_clear(sol.trajectories):
            x_clear += 1
        base = {k: shortest_trajectory(grid, s, g)
                for k, (s, g) in agents.items()}
        if _any_collide(base):
            x_base_collide += 1

    return {"case": "ccbs_continuous_time", "instances": instances,
            "solved": solved, "collision_free": collision_free,
            "baseline_collided": baseline_collided,
            "cost_sum": round(cost_sum, 3),
            "crossings": x_scn, "crossings_clear": x_clear,
            "crossings_base_collide": x_base_collide,
            "crossings_cost_sum": round(x_cost_sum, 3),
            "sound": collision_free == solved}


def _run_flow_anonymous_makespan() -> dict:
    # flow.py is a Python reproduction of Yu & LaValle's "Multi-agent Path
    # Planning and Network Flow" / "Optimal Multi-Robot Path Planning on Graphs"
    # (AAAI 2013). When the targets are INTERCHANGEABLE (the anonymous problem),
    # minimum-MAKESPAN collision-free routing is solvable in polynomial time by
    # reduction to integer MAX FLOW on a time-expanded network -- a paradigm with
    # no search tree and no priorities. Vertex collisions are blocked by an in/out
    # cap-1 split per cell-time; head-on swaps by a shared cap-1 move gadget;
    # feasibility is monotone in the horizon T, so a binary search finds the
    # minimum makespan and the optimum is SELF-CERTIFIED (flow == n at T,
    # flow < n at T-1).
    #
    # This gate pins, on a random battery (n in {2,3}, small grids, plain
    # range(10) of four configs -- no cherry-picking):
    # (1) OPTIMALITY, self-certified: certified == solved (the horizon one below
    #     is provably infeasible). [Cross-checked against a brute-force joint-BFS
    #     anonymous optimum offline: 0 makespan mismatches / 120 instances.]
    # (2) VALIDITY: every extracted flow decomposes into collision-free per-agent
    #     paths (collision_free == solved), a perfect start->goal matching.
    # (3) THE RELAXATION BITES: the anonymous makespan never exceeds the labeled
    #     CBS makespan (anon_le_cbs == both_solved) and is STRICTLY smaller on a
    #     majority (strictly_cheaper) -- interchangeable targets are genuinely
    #     cheaper. Plus a corridor showcase where the labeled swap is IMPOSSIBLE
    #     (cbs returns None on all three) yet the anonymous routing is trivial
    #     (anon_solved == corridors).
    # If the swap gadget or vertex split regressed, collision_free would fall; if
    # the binary search lost optimality, certified would fall (the gate trips).
    import random

    from mrn_coord.mapf import GridWorld, cbs, makespan
    from mrn_coord.mapf.conflicts import detect_first_conflict
    from mrn_coord.mapf.flow import anonymous_makespan

    def _instance(w, h, n, seed):
        rng = random.Random(seed)
        free = [(x, y) for x in range(w) for y in range(h)]
        rng.shuffle(free)
        return GridWorld(w, h), free[:n], free[n:2 * n]

    instances = solved = certified = collision_free = 0
    both = anon_le_cbs = strictly_cheaper = makespan_sum = 0
    for w, h, n in ((4, 4, 3), (5, 5, 3), (4, 4, 2), (5, 4, 3)):
        for seed in range(10):
            grid, starts, goals = _instance(w, h, n, seed)
            st: dict = {}
            res = anonymous_makespan(grid, starts, goals, stats=st)
            instances += 1
            if res is None:
                continue
            solved += 1
            certified += int(st["certified"])
            makespan_sum += st["makespan"]
            paths = {i: p for i, p in enumerate(res[0])}
            collision_free += int(detect_first_conflict(paths) is None)
            lab = cbs(grid, {i: (starts[i], goals[i]) for i in range(n)},
                      max_expansions=20000)
            if lab is not None:
                both += 1
                cm = makespan(lab.paths)
                anon_le_cbs += int(st["makespan"] <= cm)
                strictly_cheaper += int(st["makespan"] < cm)

    # Corridor swaps: labeled-impossible (1-wide), anonymous-trivial.
    corridors = [
        (3, 1, [(0, 0), (2, 0)], [(2, 0), (0, 0)]),
        (4, 1, [(0, 0), (3, 0)], [(3, 0), (0, 0)]),
        (5, 1, [(0, 0), (4, 0)], [(4, 0), (0, 0)]),
    ]
    cor = cor_anon = cor_cbs_none = 0
    for w, h, starts, goals in corridors:
        grid = GridWorld(w, h)
        cor += 1
        cor_anon += int(anonymous_makespan(grid, starts, goals) is not None)
        lab = cbs(grid, {i: (starts[i], goals[i]) for i in range(len(starts))},
                  max_expansions=5000)
        cor_cbs_none += int(lab is None)

    return {"case": "flow_anonymous_makespan", "instances": instances,
            "solved": solved, "certified": certified,
            "collision_free": collision_free, "makespan_sum": makespan_sum,
            "both_solved": both, "anon_le_cbs": anon_le_cbs,
            "strictly_cheaper": strictly_cheaper, "corridors": cor,
            "anon_solved": cor_anon, "labeled_cbs_none": cor_cbs_none,
            "optimal": certified == solved}


def _run_tswap_anonymous() -> dict:
    # tswap.py is a Python reproduction of Offline TSWAP -- Okumura & Defago,
    # "Solving Simultaneous Target Assignment and Path Planning Efficiently with
    # Time-Independent Execution" (ICAPS 2022; AIJ 2023). Like flow it solves the
    # ANONYMOUS (interchangeable-target) problem, but from the opposite corner of
    # the trade-off: instead of flow's makespan-OPTIMAL but heavy max-flow over a
    # time-expanded network, TSWAP is CONSTRUCTIVE -- it takes an arbitrary
    # initial assignment and repeats one-timestep planning with TARGET SWAPPING
    # until every agent is on a target. Collision-free BY CONSTRUCTION (an agent
    # moves only into a cell empty at its turn, vacating its own), complete BY A
    # POTENTIAL ARGUMENT (a swap when blocked by a settled agent, a target
    # rotation when the "wants" pointers close a cycle -- each strictly drops the
    # potential). The anonymous analogue of push_and_rotate's constructive stance.
    #
    # This gate pins:
    # (1) CONSTRUCTIVE COMPLETENESS+VALIDITY on a random battery (no cherry-pick):
    #     every instance is solved, collision-free, and ends on the goal SET
    #     (solved == cf == covers == instances).
    # (2) SOUND SUB-OPTIMALITY vs the flow optimum: TSWAP's makespan is NEVER
    #     below flow's optimal (never_below_optimal == both_solved -- it cannot
    #     beat the optimum) yet MATCHES it on a good fraction (matches_optimal),
    #     i.e. near-optimal, not optimal.
    # (3) ASSIGNMENT-INDEPENDENCE: handed a deliberately reversed (bad) initial
    #     assignment it still solves every instance collision-free and covers the
    #     goals (repair_solved == repair_cf_cover == repair_instances) -- the
    #     completeness does not rely on the initial matching.
    # (4) THE TWO MECHANISMS FIRE, isolated: a corridor where an agent must pass
    #     agents sitting on their own targets triggers exactly the target SWAP
    #     (swap_showcase_swaps, no rotation); a head-on corridor triggers the
    #     target ROTATION (rotation_showcase_rotations, no swap).
    # (5) SCALE: 40 agents on a 12x12 grid (where flow's time-expanded network is
    #     costly) is solved collision-free and covered in a blink.
    # If the live-occupancy move broke, cf would fall; if swap/rotation regressed,
    # the showcases (or completeness on bad assignments) would trip.
    import random

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.conflicts import detect_first_conflict
    from mrn_coord.mapf.flow import anonymous_makespan
    from mrn_coord.mapf.tswap import tswap

    def _instance(w, h, n, seed):
        rng = random.Random(seed)
        free = [(x, y) for x in range(w) for y in range(h)]
        rng.shuffle(free)
        return GridWorld(w, h), free[:n], free[n:2 * n]

    instances = solved = cf = covers = 0
    both = never_below = matches = 0
    for w, h, n in ((6, 6, 3), (6, 6, 4), (5, 5, 3)):
        for seed in range(10):
            grid, starts, goals = _instance(w, h, n, seed)
            st: dict = {}
            paths = tswap(grid, starts, goals, stats=st)
            instances += 1
            if paths is None:
                continue
            solved += 1
            cf += int(detect_first_conflict(paths) is None)
            covers += int(sorted(p[-1] for p in paths.values()) == sorted(goals))
            opt = anonymous_makespan(grid, starts, goals)
            if opt is not None:
                both += 1
                never_below += int(st["makespan"] >= opt[1])
                matches += int(st["makespan"] == opt[1])

    # (3) Assignment-independence: hand each instance a reversed (bad) matching.
    repair_instances = repair_solved = repair_cf_cover = 0
    for seed in range(10):
        grid, starts, goals = _instance(6, 6, 4, seed)
        st = {}
        paths = tswap(grid, starts, goals,
                      assignment=list(reversed(range(4))), stats=st)
        repair_instances += 1
        if paths is None:
            continue
        repair_solved += 1
        repair_cf_cover += int(
            detect_first_conflict(paths) is None
            and sorted(p[-1] for p in paths.values()) == sorted(goals))

    # (4) The two mechanisms in isolation, on 1-wide corridors.
    swap_grid = GridWorld(5, 1)
    sw: dict = {}
    tswap(swap_grid, [(0, 0), (2, 0), (3, 0)], [(4, 0), (2, 0), (3, 0)],
          assignment=[0, 1, 2], stats=sw)
    rot_grid = GridWorld(5, 1)
    ro: dict = {}
    tswap(rot_grid, [(0, 0), (4, 0)], [(4, 0), (0, 0)],
          assignment=[0, 1], stats=ro)

    # (5) Scale where flow's time-expanded network is costly.
    rng = random.Random(0)
    free = [(x, y) for x in range(12) for y in range(12)]
    rng.shuffle(free)
    sc: dict = {}
    sc_paths = tswap(GridWorld(12, 12), free[:40], free[40:80], stats=sc)
    scale_ok = (sc_paths is not None
                and detect_first_conflict(sc_paths) is None
                and sorted(p[-1] for p in sc_paths.values()) == sorted(free[40:80]))

    return {"case": "tswap_anonymous",
            "instances": instances, "solved": solved,
            "collision_free": cf, "covers_goals": covers,
            "both_solved": both, "never_below_optimal": never_below,
            "matches_optimal": matches,
            "repair_instances": repair_instances,
            "repair_solved": repair_solved,
            "repair_cf_cover": repair_cf_cover,
            "swap_showcase_swaps": sw["swaps"],
            "swap_showcase_rotations": sw["rotations"],
            "rotation_showcase_rotations": ro["rotations"],
            "rotation_showcase_swaps": ro["swaps"],
            "scale_agents": 40, "scale_solved": bool(scale_ok),
            "complete_valid": (solved == instances == cf == covers),
            "sound_suboptimal": (never_below == both and matches < both)}


def _run_push_and_rotate() -> dict:
    # push_and_rotate.py is a Python reproduction of the movement-primitive family
    # -- Luna & Bekris's "Push and Swap" (IJCAI 2011) and de Wilde, ter Mors &
    # Witteveen's "Push and Rotate" (JAIR 2014). It is a CONSTRUCTIVE solver, not
    # a search: it manipulates the configuration with reversible primitives (push
    # an agent toward its goal shoving blockers aside; swap two agents by rotating
    # them around a degree->=3 hub; rotate a cyclic component) until everyone is
    # placed. Because every primitive only steps one agent to an adjacent EMPTY
    # cell, any returned plan is collision-free and ends with all agents on goal
    # BY CONSTRUCTION -- so it trades optimality for a guarantee, exactly where
    # optimal CBS blows up on crowded maps.
    #
    # This gate pins two regimes:
    # (1) COMPLETE-WITH-SLACK + VALID + SUBOPTIMAL. On a moderate battery (three
    #     configs, plain range(10)) every instance CBS proves solvable is also
    #     solved by the primitives (complete_match == cbs_solved), every returned
    #     plan is collision-free and on-goal (valid == pnr_solved), and the cost
    #     is far above CBS's optimum (pnr_cost >> cbs_cost) -- the price of the
    #     guarantee (the single-mover serialisation makes it loose on purpose).
    # (2) SOLVES WHERE SEARCH CANNOT. On crowded 8x8 instances (18 agents) CBS
    #     exhausts a 800-node budget every time (cbs_timeout == 6), while the
    #     primitives place all agents in polynomial time (timeout_pnr_solved == 6,
    #     all valid).
    # Honest scope (see docs): this is the primitive core with a deterministic
    # priority-order sweep -- complete when the map has ample empty space, but the
    # near-fully-packed regime (1-3 empty cells, cyclic dependencies) needs Push-
    # and-Rotate's full rotate + subproblem machinery, which is NOT reproduced; on
    # such packed instances it solves only a fraction. The gate therefore pins
    # completeness only where the primitive core is complete, plus the
    # search-blowup advantage, plus the absolute validity guarantee.
    import random

    from mrn_coord.mapf import GridWorld, cbs
    from mrn_coord.mapf.conflicts import detect_first_conflict
    from mrn_coord.mapf.push_and_rotate import push_and_rotate

    def _instance(w, h, n, seed):
        rng = random.Random(seed)
        free = [(x, y) for x in range(w) for y in range(h)]
        rng.shuffle(free)
        return GridWorld(w, h), {i: (free[i], free[n + i]) for i in range(n)}

    def _valid(sol, agents):
        return (detect_first_conflict(sol.paths) is None
                and all(sol.paths[k][-1] == g for k, (s, g) in agents.items()))

    instances = cbs_solved = pnr_solved = complete_match = valid = 0
    pnr_cost = cbs_cost = 0
    for w, h, n in ((4, 4, 4), (5, 5, 5), (6, 6, 6)):
        for seed in range(10):
            grid, agents = _instance(w, h, n, seed)
            base = cbs(grid, agents, max_expansions=20000)
            sol = push_and_rotate(grid, agents)
            instances += 1
            if sol is not None:
                pnr_solved += 1
                valid += int(_valid(sol, agents))
            if base is not None:
                cbs_solved += 1
                complete_match += int(sol is not None)
                if sol is not None:
                    pnr_cost += sol.cost
                    cbs_cost += base.cost

    timeout_instances = cbs_timeout = timeout_pnr_solved = timeout_valid = 0
    for seed in range(6):
        grid, agents = _instance(8, 8, 18, seed)
        base = cbs(grid, agents, max_expansions=800)
        sol = push_and_rotate(grid, agents)
        timeout_instances += 1
        if base is None:
            cbs_timeout += 1
        if sol is not None:
            timeout_pnr_solved += 1
            timeout_valid += int(_valid(sol, agents))

    # (3) DENSE PACKED RECTANGLES -- the near-packed gap, now closed for >=2 empty
    # cells. A fully packed grid is the 15-puzzle regime: the greedy push/swap
    # primitives stall at once (no slack to shove blockers), so Push-and-Rotate
    # dispatches it with a CONSTRUCTIVE row/column reduction (place the rectangle
    # top row by top row, then peel the strip's columns, finishing a 2x3 corner
    # exactly) rather than a search -- search is what CBS does, and exactly what
    # blows up here. Instances are packed formations (the target leaves its empty
    # cells in a bottom band) scrambled by a random walk from the goal, so each is
    # solvable BY CONSTRUCTION; the reduction's moves only ever step one agent into
    # an adjacent empty cell, so every plan is collision-free and on-goal by
    # construction too. This pins: every instance solved (complete_packed) and
    # valid (packed_all_valid), and that optimal CBS busts a 300-node budget on
    # every one of them (packed_cbs_busts == packed_instances) -- the constructive
    # method wins precisely where search cannot. The single-blank case (exactly one
    # empty cell, the tightest 15-puzzle regime) is gated separately in (4) below.
    def _packed(w, h, blanks, seed):
        rng = random.Random(seed * 131 + w * 7 + h * 3 + blanks)
        grid = GridWorld(w, h)
        cells = [(x, y) for y in range(h) for x in range(w)]
        goal = cells[:len(cells) - blanks]
        n = len(goal)
        pos = {i: goal[i] for i in range(n)}
        occ = {goal[i]: i for i in range(n)}
        empt = set(cells) - set(goal)
        nb = lambda c: [d for d in ((c[0] + 1, c[1]), (c[0] - 1, c[1]),
                                    (c[0], c[1] + 1), (c[0], c[1] - 1))
                        if grid.is_free(d)]
        for _ in range(30 * n):
            e = rng.choice(sorted(empt))
            cand = [c for c in nb(e) if c in occ]
            if not cand:
                continue
            c = rng.choice(cand)
            a = occ.pop(c)
            occ[e] = a
            pos[a] = e
            empt.discard(e)
            empt.add(c)
        return grid, {i: (pos[i], goal[i]) for i in range(n)}

    packed_instances = packed_solved = packed_valid = packed_cbs_busts = 0
    for w, h, blanks in ((4, 4, 2), (4, 4, 3), (5, 5, 2), (5, 5, 3)):
        for seed in range(6):
            grid, agents = _packed(w, h, blanks, seed)
            sol = push_and_rotate(grid, agents)
            packed_instances += 1
            if sol is not None:
                packed_solved += 1
                packed_valid += int(_valid(sol, agents))
            if cbs(grid, agents, max_expansions=300) is None:
                packed_cbs_busts += 1

    # (4) SINGLE-BLANK PACKED -- the tightest sub-case, exactly one empty cell, i.e.
    # the (W*H - 1)-puzzle proper. Here the row/column reduction can paint the lone
    # blank into a corner (with two empties the spare slack escapes; with one it
    # does not), so Push-and-Rotate places each tile -- or last-two pair -- with an
    # exact BFS over the whole unsolved region that tracks ONLY the agents being
    # placed; every other tile is an anonymous filler, so the state is tiny and the
    # exhaustive search can never strand the blank. This closes the gap the >=2-cell
    # regime left open. Pins: every single-blank instance solved (complete_single_
    # blank) and valid, and CBS busts a 300-node budget on every one of them.
    unit_instances = unit_solved = unit_valid = unit_cbs_busts = 0
    for w, h in ((4, 4), (5, 5), (6, 6)):
        for seed in range(6):
            grid, agents = _packed(w, h, 1, seed)
            sol = push_and_rotate(grid, agents)
            unit_instances += 1
            if sol is not None:
                unit_solved += 1
                unit_valid += int(_valid(sol, agents))
            if cbs(grid, agents, max_expansions=300) is None:
                unit_cbs_busts += 1

    return {"case": "push_and_rotate", "instances": instances,
            "cbs_solved": cbs_solved, "pnr_solved": pnr_solved,
            "complete_match": complete_match, "valid": valid,
            "pnr_cost": pnr_cost, "cbs_cost": cbs_cost,
            "complete_with_slack": complete_match == cbs_solved,
            "always_valid": valid == pnr_solved,
            "suboptimal": pnr_cost > cbs_cost,
            "timeout_instances": timeout_instances, "cbs_timeout": cbs_timeout,
            "timeout_pnr_solved": timeout_pnr_solved,
            "timeout_valid": timeout_valid,
            "packed_instances": packed_instances, "packed_solved": packed_solved,
            "packed_valid": packed_valid, "packed_cbs_busts": packed_cbs_busts,
            "complete_packed": packed_solved == packed_instances,
            "packed_all_valid": packed_valid == packed_instances,
            "packed_beats_search": packed_cbs_busts == packed_instances,
            "unit_instances": unit_instances, "unit_solved": unit_solved,
            "unit_valid": unit_valid, "unit_cbs_busts": unit_cbs_busts,
            "complete_single_blank": unit_solved == unit_instances,
            "single_blank_all_valid": unit_valid == unit_instances,
            "single_blank_beats_search": unit_cbs_busts == unit_instances}


def _run_mstar_subdimensional() -> dict:
    # M* (mstar) is a Python reproduction of Wagner & Choset's "M*" /
    # "Subdimensional expansion for multirobot path planning" (IROS 2011 / AIJ
    # 2015). It is an OPTIMAL (sum-of-costs) paradigm distinct from CBS: it plans
    # in the JOINT configuration space but keeps the search dimension low by
    # pinning each agent to its individual optimal policy until a collision
    # couples it, at which point only the colliding agents branch over their full
    # moves (the collision set). It returns the SAME optimal sum-of-costs as plain
    # cbs (opt_match), checked here both on small random maps (breadth) and on a
    # constructed family (mechanism).
    #
    # The signature property this gate pins: the search couples ONLY the agents
    # that actually interact. The constructed family is one isolated head-on swap
    # (agents 0,1) plus `nby` bystanders, each alone in its own walled lane with a
    # unique straight path -- no bystander can ever collide. So M*'s collision set
    # must stay {0,1} (peak_collision_set == 2, well below the team size) and its
    # expansion count must NOT grow with the bystanders (mstar_expansions is the
    # same 33 for every instance). A fully coupled joint A* (joint_astar) -- the
    # straw man with no decoupling -- expands strictly more, and MORE as the team
    # grows (joint_search_grows_with_team), because it re-explores the interleaved
    # forced moves M* collapses. If decoupling ever regresses, peak_collision_set
    # rises toward the team size and mstar_expansions stops being constant.
    import random

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.cbs import cbs
    from mrn_coord.mapf.conflicts import detect_first_conflict
    from mrn_coord.mapf.mstar import joint_astar, mstar
    from mrn_coord.mapf.solution import sum_of_costs

    def _valid(sol, agents):
        return (detect_first_conflict(sol.paths) is None
                and all(sol.paths[a][-1] == agents[a][1] for a in agents))

    def _rand(w, h, n, seed, obs):
        rng = random.Random(seed)
        blocked = {(x, y) for x in range(w) for y in range(h)
                   if rng.random() < obs}
        free = [(x, y) for x in range(w) for y in range(h)
                if (x, y) not in blocked]
        rng.shuffle(free)
        return (GridWorld(w, h, frozenset(blocked)),
                {i: (free[i], free[n + i]) for i in range(n)})

    rand_inst = rand_opt = rand_valid = 0
    for (w, h, n, obs) in ((5, 5, 3, 0.0), (5, 5, 3, 0.12), (6, 6, 3, 0.1)):
        for seed in range(10):
            grid, ag = _rand(w, h, n, seed, obs)
            base = cbs(grid, ag, max_expansions=20000)
            if base is None:
                continue
            sol = mstar(grid, ag, max_expansions=50000)
            if sol is None:
                continue
            rand_inst += 1
            rand_opt += int(sum_of_costs(sol.paths) == base.cost)
            rand_valid += int(_valid(sol, ag))

    def _isolated(nby, seed):
        # corridor row y=0 (agents 0,1 swap through the single pocket at (2,1));
        # `nby` bystander lanes at y=2,4,... each walled off from the rest.
        L, pk = 5, 2
        W, H = L + 1, 2 + 2 * nby
        blocked = set()
        for x in range(W):
            if (x, 1) != (pk, 1):
                blocked.add((x, 1))
        for k in range(nby):
            sep = 2 + 2 * k + 1
            if sep < H:
                for x in range(W):
                    blocked.add((x, sep))
        grid = GridWorld(W, H, frozenset(blocked))
        ag = {0: ((0, 0), (L, 0)), 1: ((L, 0), (0, 0))}
        rng = random.Random(seed)
        for k in range(nby):
            ly = 2 + 2 * k
            ag[2 + k] = (((0, ly), (W - 1, ly)) if rng.random() < 0.5
                         else ((W - 1, ly), (0, ly)))
        return grid, ag

    con_inst = con_opt = con_valid = 0
    peak_cs = 0
    mstar_sizes = set()
    joint_by_nby = {}
    for nby in (2, 3, 4, 5):
        jt = 0
        for seed in range(6):
            grid, ag = _isolated(nby, seed)
            base = cbs(grid, ag, max_expansions=20000)
            sm: dict = {}
            sol = mstar(grid, ag, stats=sm, max_expansions=50000)
            sj: dict = {}
            jol = joint_astar(grid, ag, stats=sj, max_expansions=200000)
            if base is None or sol is None or jol is None:
                continue
            con_inst += 1
            con_opt += int(sum_of_costs(sol.paths) == base.cost)
            con_valid += int(_valid(sol, ag))
            peak_cs = max(peak_cs, sm["max_collision_set"])
            mstar_sizes.add(sm["expansions"])
            jt += sj["expansions"]
        joint_by_nby[nby] = jt

    joints = [joint_by_nby[k] for k in (2, 3, 4, 5)]
    grows = all(joints[i] < joints[i + 1] for i in range(len(joints) - 1))

    return {
        "case": "mstar_subdimensional",
        "rand_instances": rand_inst,
        "rand_opt_match": rand_opt,
        "rand_valid": rand_valid,
        "con_instances": con_inst,
        "con_opt_match": con_opt,
        "con_valid": con_valid,
        "peak_collision_set": peak_cs,
        "mstar_expansions": sorted(mstar_sizes)[-1],
        "joint_expansions_smallest_team": joint_by_nby[2],
        "joint_expansions_largest_team": joint_by_nby[5],
        "optimal_matches_cbs": (rand_opt == rand_inst and con_opt == con_inst),
        "all_collision_free": (rand_valid == rand_inst and con_valid == con_inst),
        "couples_only_the_pair": peak_cs == 2,
        "mstar_search_size_constant": len(mstar_sizes) == 1,
        "joint_search_grows_with_team": grows,
    }


def _run_standley_id_od() -> dict:
    # Standley's "Finding Optimal Solutions to Cooperative Pathfinding Problems"
    # (AAAI 2010), reproduced in standley.py: two attacks on the b**n joint
    # branching. OPERATOR DECOMPOSITION (od_astar) assigns a move to one agent at
    # a time -- effective branching b, not b**n -- so it GENERATES a small
    # fraction of the successors a fully coupled joint A* (mstar.joint_astar)
    # does, and the gap widens with the team (od_advantage_grows_with_team).
    # INDEPENDENCE DETECTION (independence_detection) plans agents separately and
    # merges only colliding groups, so on the isolated-swap family it only ever
    # solves the 2-agent pair jointly (peak_group == 2) while every bystander
    # stays its own group (num_groups grows with the team). Both return CBS's
    # exact optimum (od_matches_cbs / id_matches_cbs), checked on random maps and
    # the constructed family. If OD's decomposition regresses, od_generated rises
    # toward joint_generated; if ID stops decoupling, peak_group rises toward the
    # team size.
    import random

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.cbs import cbs
    from mrn_coord.mapf.conflicts import detect_first_conflict
    from mrn_coord.mapf.mstar import joint_astar
    from mrn_coord.mapf.solution import sum_of_costs
    from mrn_coord.mapf.standley import independence_detection, od_astar

    def _valid(sol, ag):
        return (detect_first_conflict(sol.paths) is None
                and all(sol.paths[a][-1] == ag[a][1] for a in ag))

    def _rand(w, h, n, seed, obs):
        rng = random.Random(seed)
        blocked = {(x, y) for x in range(w) for y in range(h)
                   if rng.random() < obs}
        free = [(x, y) for x in range(w) for y in range(h)
                if (x, y) not in blocked]
        rng.shuffle(free)
        return (GridWorld(w, h, frozenset(blocked)),
                {i: (free[i], free[n + i]) for i in range(n)})

    def _isolated(nby, seed):
        L, pk = 5, 2
        W, H = L + 1, 2 + 2 * nby
        blocked = set()
        for x in range(W):
            if (x, 1) != (pk, 1):
                blocked.add((x, 1))
        for k in range(nby):
            sep = 2 + 2 * k + 1
            if sep < H:
                for x in range(W):
                    blocked.add((x, sep))
        grid = GridWorld(W, H, frozenset(blocked))
        ag = {0: ((0, 0), (L, 0)), 1: ((L, 0), (0, 0))}
        rng = random.Random(seed)
        for k in range(nby):
            ly = 2 + 2 * k
            ag[2 + k] = (((0, ly), (W - 1, ly)) if rng.random() < 0.5
                         else ((W - 1, ly), (0, ly)))
        return grid, ag

    rand_inst = od_opt = id_opt = od_val = id_val = 0
    for (w, h, n, obs) in ((5, 5, 3, 0.0), (5, 5, 4, 0.1), (6, 6, 3, 0.12)):
        for seed in range(8):
            grid, ag = _rand(w, h, n, seed, obs)
            base = cbs(grid, ag, max_expansions=20000)
            if base is None:
                continue
            od = od_astar(grid, ag, max_expansions=100000)
            idsol = independence_detection(grid, ag, max_expansions=100000)
            if od is None or idsol is None:
                continue
            rand_inst += 1
            od_opt += int(od.cost == base.cost
                          and sum_of_costs(od.paths) == base.cost)
            id_opt += int(idsol.cost == base.cost
                          and sum_of_costs(idsol.paths) == base.cost)
            od_val += int(_valid(od, ag))
            id_val += int(_valid(idsol, ag))

    od_gen = {}
    joint_gen = {}
    for n in (3, 4):
        og = jg = 0
        for seed in range(8):
            grid, ag = _rand(6, 6, n, seed, 0.05)
            if cbs(grid, ag, max_expansions=20000) is None:
                continue
            so = {}
            od_astar(grid, ag, stats=so, max_expansions=100000)
            sj = {}
            joint_astar(grid, ag, stats=sj, max_expansions=300000)
            og += so["generated"]
            jg += sj["generated"]
        od_gen[n] = og
        joint_gen[n] = jg

    con_inst = id_con_opt = 0
    peak_group = 0
    groups_largest = 0
    for nby in (2, 3, 4, 5):
        for seed in range(6):
            grid, ag = _isolated(nby, seed)
            base = cbs(grid, ag, max_expansions=20000)
            st = {}
            sol = independence_detection(grid, ag, stats=st,
                                         max_expansions=100000)
            if base is None or sol is None:
                continue
            con_inst += 1
            id_con_opt += int(sol.cost == base.cost
                              and sum_of_costs(sol.paths) == base.cost
                              and detect_first_conflict(sol.paths) is None)
            peak_group = max(peak_group, st["max_group"])
            if nby == 5:
                groups_largest = st["num_groups"]

    ratio3 = joint_gen[3] / max(1, od_gen[3])
    ratio4 = joint_gen[4] / max(1, od_gen[4])
    return {
        "case": "standley_id_od",
        "rand_instances": rand_inst,
        "od_opt_match": od_opt,
        "id_opt_match": id_opt,
        "od_valid": od_val,
        "id_valid": id_val,
        "od_generated_n3": od_gen[3],
        "od_generated_n4": od_gen[4],
        "joint_generated_n3": joint_gen[3],
        "joint_generated_n4": joint_gen[4],
        "con_instances": con_inst,
        "id_con_opt_match": id_con_opt,
        "peak_group": peak_group,
        "num_groups_largest_team": groups_largest,
        "od_matches_cbs": od_opt == rand_inst,
        "id_matches_cbs": id_opt == rand_inst and id_con_opt == con_inst,
        "all_valid": od_val == rand_inst and id_val == rand_inst,
        "od_branching_below_joint": (od_gen[3] < joint_gen[3]
                                     and od_gen[4] < joint_gen[4]),
        "od_advantage_grows_with_team": ratio4 > ratio3,
        "id_couples_only_the_pair": peak_group == 2,
        "id_groups_one_per_independent_agent": groups_largest == 6,
    }


def _run_satmdd_makespan() -> dict:
    # MDD-SAT (satmdd.py) reproduces Surynek et al.'s SAT encoding of MAPF (ECAI
    # / IJCAI 2016) -- the DECLARATIVE paradigm: encode "is there a collision-free
    # plan of makespan mu?" as CNF over per-agent MDD cells, solve with a SAT
    # solver, sweep mu up from the trivial lower bound. The first satisfiable mu
    # is the optimal LABELED makespan, SELF-CERTIFIED because every smaller mu was
    # proved UNSAT (unsat_below counts them; certified holds for all). This gate
    # pins: every plan is collision-free and on-goal (all_valid); the reported
    # makespan equals the realised one (ms_matches_stat); it is <= the makespan of
    # CBS's sum-of-costs-optimal plan (makespan_optimal_le_cbs -- a different
    # objective); and it is always >= flow's ANONYMOUS makespan
    # (labeled_ge_anonymous), strictly more when labels force a detour
    # (label_can_cost_more). The constructed pocket-corridor swap makes that
    # vivid: anonymous makespan 0, labeled makespan 5, certified by two UNSAT
    # rounds below it.
    import random

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.cbs import cbs
    from mrn_coord.mapf.conflicts import detect_first_conflict
    from mrn_coord.mapf.flow import anonymous_makespan
    from mrn_coord.mapf.satmdd import satmdd
    from mrn_coord.mapf.solution import makespan as mkspan

    def _rand(w, h, n, seed, obs):
        rng = random.Random(seed)
        blocked = {(x, y) for x in range(w) for y in range(h)
                   if rng.random() < obs}
        free = [(x, y) for x in range(w) for y in range(h)
                if (x, y) not in blocked]
        rng.shuffle(free)
        return (GridWorld(w, h, frozenset(blocked)),
                {i: (free[i], free[n + i]) for i in range(n)})

    inst = solved = valid = certified = ms_match = le_cbs = 0
    both_flow = ge_flow = gt_flow = 0
    for (w, h, n, obs) in ((4, 4, 2, 0.0), (5, 5, 2, 0.1), (4, 4, 3, 0.0),
                           (5, 5, 3, 0.08), (5, 5, 4, 0.0), (6, 6, 3, 0.05)):
        for seed in range(6):
            grid, ag = _rand(w, h, n, seed, obs)
            base = cbs(grid, ag, max_expansions=20000)
            if base is None:
                continue
            inst += 1
            st: dict = {}
            sol = satmdd(grid, ag, stats=st)
            if sol is None:
                continue
            solved += 1
            valid += int(detect_first_conflict(sol.paths) is None
                         and all(sol.paths[a][0] == ag[a][0]
                                 and sol.paths[a][-1] == ag[a][1] for a in ag))
            certified += int(st["certified"])
            ms_match += int(mkspan(sol.paths) == st["makespan"])
            le_cbs += int(st["makespan"] <= mkspan(base.paths))
            fr = anonymous_makespan(grid, [ag[a][0] for a in ag],
                                    [ag[a][1] for a in ag])
            if fr is not None:
                both_flow += 1
                ge_flow += int(st["makespan"] >= fr[1])
                gt_flow += int(st["makespan"] > fr[1])

    grid = GridWorld(4, 2, frozenset({(0, 1), (2, 1), (3, 1)}))
    pag = {0: ((0, 0), (3, 0)), 1: ((3, 0), (0, 0))}
    pst: dict = {}
    psol = satmdd(grid, pag, stats=pst)
    pfr = anonymous_makespan(grid, [(0, 0), (3, 0)], [(3, 0), (0, 0)])
    pocket_valid = (psol is not None
                    and detect_first_conflict(psol.paths) is None)

    return {
        "case": "satmdd_makespan",
        "instances": inst,
        "solved": solved,
        "valid": valid,
        "certified": certified,
        "ms_matches_stat": ms_match,
        "sat_le_cbs_makespan": le_cbs,
        "both_flow": both_flow,
        "sat_ge_flow": ge_flow,
        "sat_gt_flow_strict": gt_flow,
        "pocket_makespan": pst["makespan"],
        "pocket_unsat_below": pst["unsat_below"],
        "pocket_certified": pst["certified"],
        "pocket_anon_makespan": pfr[1],
        "all_solved": solved == inst,
        "all_valid": valid == inst,
        "all_certified": certified == inst,
        "makespan_optimal_le_cbs": le_cbs == inst,
        "labeled_ge_anonymous": ge_flow == both_flow,
        "label_can_cost_more": (gt_flow > 0 and pocket_valid
                                and pst["makespan"] > pfr[1]),
    }


def _run_bcp_branch_price() -> dict:
    # Branch-and-cut-and-price (bcp.py), a Python reproduction of Lam, Le Bodic,
    # Harabor & Stuckey's "Branch-and-Cut-and-Price for Multi-Agent Path Finding"
    # (IJCAI 2019). This is the OPTIMIZATION paradigm: where CBS / M* / Standley
    # SEARCH, MDD-SAT DECIDES, and flow ROUTES, BCP solves the path-based
    # (Dantzig-Wolfe / set-partitioning) LINEAR PROGRAM and certifies optimality
    # by LP DUALITY. Paths enter the LP only when a reduced-cost shortest path
    # PRICES them in (column generation); vertex/edge conflict rows enter only
    # when the LP solution violates one (lazy CUTting); branching on a fractional
    # agent-vertex-time usage closes the integrality gap. It returns the SAME
    # optimal sum-of-costs as plain cbs.
    #
    # Two things this gate pins. First, CORRECTNESS + CERTIFICATION on an open
    # battery: every instance matches cbs's optimum, is collision-free, and the
    # ROOT LP objective is a valid lower bound (lp_bound <= cost) -- the LP
    # optimal VALUE is unique, so this certificate is environment-independent.
    # On these sparse open maps branch-and-price solves every instance at the
    # ROOT (rand_root_integral == rand_instances): the LP + pricing + lazy cuts
    # alone are integral, no branching needed -- the paradigm's signature.
    # Second, the MECHANISM on a constructed case where the LP genuinely is NOT
    # integral: a head-on swap in a one-wide corridor with a single pocket. There
    # the root LP relaxes to 7 but the integer optimum is 8 (corridor_lp_bound <
    # corridor_optimum) -- a real integrality gap -- and branch-and-price closes
    # it (corridor_nodes > 1) to the certified optimum 8 == cbs, driven by lazily
    # separated conflict cuts (corridor_cuts) over priced-in columns
    # (corridor_columns). If pricing/cut/branch ever regress, the root stops
    # being integral on the open maps, or the corridor gap fails to close to cbs.
    import random

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.bcp import bcp
    from mrn_coord.mapf.cbs import cbs
    from mrn_coord.mapf.conflicts import detect_first_conflict
    from mrn_coord.mapf.solution import sum_of_costs

    def _rand(w, h, n, seed):
        rng = random.Random(seed)
        free = [(x, y) for x in range(w) for y in range(h)]
        rng.shuffle(free)
        return GridWorld(w, h), {i: (free[i], free[n + i]) for i in range(n)}

    inst = opt = val = cert = rootint = branched = 0
    for (w, h, n) in ((4, 4, 2), (5, 5, 3), (5, 5, 4)):
        for seed in range(8):
            grid, ag = _rand(w, h, n, seed)
            base = cbs(grid, ag, max_expansions=20000)
            if base is None:
                continue
            sm: dict = {}
            sol = bcp(grid, ag, stats=sm)
            if sol is None:
                continue
            inst += 1
            cost = sum_of_costs(sol.paths)
            opt += int(cost == base.cost)
            val += int(detect_first_conflict(sol.paths) is None
                       and all(sol.paths[a][-1] == ag[a][1] for a in ag))
            cert += int(sm["lp_bound"] is not None
                        and sm["lp_bound"] <= cost + 1e-6)
            rootint += int(sm["root_integral"])
            branched += int(not sm["root_integral"])

    # constructed integrality-gap demonstrator: head-on swap, one-wide corridor
    # (row y=0, length 3) with a single pocket at (1, 1) to step aside into.
    blocked = {(x, 1) for x in range(4) if x != 1}
    grid = GridWorld(4, 2, frozenset(blocked))
    ag = {0: ((0, 0), (3, 0)), 1: ((3, 0), (0, 0))}
    base = cbs(grid, ag)
    cm: dict = {}
    csol = bcp(grid, ag, stats=cm)
    corr_cost = sum_of_costs(csol.paths)
    corr_lb = int(round(cm["lp_bound"]))
    corr_cf = detect_first_conflict(csol.paths) is None

    return {
        "case": "bcp_branch_price",
        "rand_instances": inst,
        "rand_opt_match": opt,
        "rand_valid": val,
        "rand_certified": cert,
        "rand_root_integral": rootint,
        "rand_branched": branched,
        "corridor_lp_bound": corr_lb,
        "corridor_optimum": corr_cost,
        "corridor_cbs": base.cost,
        "corridor_nodes": cm["nodes"],
        "corridor_cuts": cm["cuts"],
        "corridor_columns": cm["columns"],
        "corridor_cf": int(corr_cf),
        "optimal_matches_cbs": (opt == inst and corr_cost == base.cost),
        "all_collision_free": (val == inst and corr_cf),
        "lp_bound_certifies_optimum": cert == inst,
        "price_and_cut_solve_root": rootint == inst,
        "branching_closes_integrality_gap": (corr_lb < corr_cost
                                             and cm["nodes"] > 1),
        "lazy_cuts_and_priced_columns": (cm["cuts"] > 0 and cm["columns"] > 2),
    }


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


def _run_token_passing() -> dict:
    # Token Passing (token_passing.py), a reproduction of Ma, Li, Kumar & Koenig's
    # "Lifelong Multi-Agent Path Finding for Online Pickup and Delivery Tasks"
    # (AAAI 2017). The THIRD lifelong engine and a third paradigm: run_lifelong
    # steps PIBT (a greedy one-step rule), run_rhcr solves a WINDOWED batch every
    # few ticks, and Token Passing commits FULL space-time paths into a shared
    # token of reservations -- so the team is collision-free BY CONSTRUCTION (no
    # per-step rule, no fallback rollout). Agents update the token one at a time,
    # each planning a path that avoids every other agent's reserved cells/swaps.
    #
    # Two things this gate pins. First, the WELL-FORMED regime (a roomy warehouse,
    # aisle=2, with parking homes disjoint from the task endpoints so a resting
    # agent never blocks a task cell): TP is live (no stall, win_tp_blocked == 0)
    # and matches the throughput of both PIBT and RHCR task-for-task
    # (matches_baselines) -- its contract is collision-free + complete +
    # competitive, not higher throughput. Second, the defining INVARIANT and its
    # SCOPE: TP is collision-free on BOTH maps (collision_free_by_construction),
    # but on a cramped map (aisle=1) where the well-formed property fails, its
    # reservation planning STALLS -- agents get blocked (cr_tp_blocked > 0) and
    # complete far fewer tasks than greedy PIBT (reservation_stalls_when_cramped),
    # the documented reason RHCR falls back to PIBT in narrow aisles. If the
    # reservation logic regresses, collision_free_by_construction breaks; if the
    # paradigm regresses, TP stops matching the baselines on the well-formed map.
    from mrn_coord.lifelong import (TaskStream, make_warehouse, run_lifelong,
                                    run_rhcr, run_token_passing)
    from mrn_coord.mapf.conflicts import detect_first_conflict

    def _collision_free(res) -> bool:
        paths: dict = {}
        for snap in res.history:
            for a, c in snap.items():
                paths.setdefault(a, []).append(c)
        return detect_first_conflict(paths) is None

    def _scenario(rows, cols, aisle, agents):
        grid, eps = make_warehouse(rows=rows, cols=cols, aisle=aisle)
        n = min(agents, len(eps) // 2)
        homes = {f"r{i}": eps[i] for i in range(n)}     # parking endpoints
        tasks = eps[n:]                                  # disjoint task endpoints
        return grid, homes, tasks

    # well-formed regime: roomy aisle=2 warehouse.
    grid, homes, tasks = _scenario(3, 4, 2, 8)
    win_pibt = run_lifelong(grid, dict(homes), TaskStream(list(tasks)),
                            max_steps=120, allocator="hungarian")
    win_rhcr = run_rhcr(grid, dict(homes), TaskStream(list(tasks)), max_steps=120,
                        window=10, replan_period=2, solver="pbs",
                        allocator="hungarian")
    win_tp = run_token_passing(grid, dict(homes), TaskStream(list(tasks)),
                               max_steps=120, allocator="hungarian", homes=homes,
                               keep_history=True)
    win_cf = _collision_free(win_tp)

    # cramped regime: aisle=1 -- the well-formed property fails, TP stalls.
    cgrid, chomes, ctasks = _scenario(2, 3, 1, 5)
    cr_pibt = run_lifelong(cgrid, dict(chomes), TaskStream(list(ctasks)),
                           max_steps=40, allocator="hungarian")
    cr_tp = run_token_passing(cgrid, dict(chomes), TaskStream(list(ctasks)),
                              max_steps=40, allocator="hungarian", homes=chomes,
                              horizon=12, keep_history=True)
    cr_cf = _collision_free(cr_tp)

    return {
        "case": "mapf_token_passing",
        "win_tp_completed": win_tp.completed,
        "win_pibt_completed": win_pibt.completed,
        "win_rhcr_completed": win_rhcr.completed,
        "win_tp_blocked": win_tp.blocked,
        "win_tp_longest_stall": win_tp.longest_stall(),
        "win_tp_collision_free": int(win_cf),
        "cr_tp_completed": cr_tp.completed,
        "cr_pibt_completed": cr_pibt.completed,
        "cr_tp_blocked": cr_tp.blocked,
        "cr_tp_collision_free": int(cr_cf),
        "collision_free_by_construction": win_cf and cr_cf,
        "matches_baselines_when_well_formed": (
            win_tp.completed == win_pibt.completed == win_rhcr.completed),
        "live_when_well_formed": (win_tp.blocked == 0
                                  and win_tp.longest_stall() <= 8),
        "reservation_stalls_when_cramped": (cr_tp.blocked > 0
                                            and cr_tp.completed < cr_pibt.completed),
    }


def _run_tpts() -> dict:
    # Token Passing with Task Swaps (token_passing_swaps.py), a reproduction of
    # Algorithm 2 of Ma, Li, Kumar & Koenig's "Lifelong Multi-Agent Path Finding
    # for Online Pickup and Delivery Tasks" (AAAI 2017) -- the paper's improvement
    # over plain Token Passing. Every task is now a real pickup->delivery pair; a
    # task is swappable only while ASSIGNED (en route to pickup), never once
    # EXECUTING (the package is in hand). The defining rule: a freshly-free agent
    # may STEAL an assigned task from a holder that is farther from the pickup, so
    # tasks migrate to better-placed robots instead of being frozen to whoever
    # grabbed them first. Motion is the same shared-token reservation scheme as TP
    # -- collision-free BY CONSTRUCTION -- and ``swaps=False`` recovers plain
    # two-leg TP, so a single run pair isolates exactly what the swap rule buys.
    #
    # Two things this gate pins. (1) A CONSTRUCTED forced-swap instance on an open
    # grid: r1 collects a task under it and frees up next to T0's pickup while r0,
    # the only other free agent, is still trudging toward the farther T0 it was
    # handed -- so TPTS fires EXACTLY ONE steal (r1 takes T0, r0 grabs the near
    # T1). That single swap drops average service 5.33->4.00 and max wait 10->6;
    # plain TP (swaps off) fires zero and pays the longer trips. (2) A realistic
    # well-formed warehouse batch where swaps fire a few times and shorten service
    # without ever losing a delivery. Collision-free holds with swaps on AND off,
    # on BOTH maps. If the steal logic regresses, the swap count or the service
    # win breaks; if the reservation logic regresses, collision-free breaks.
    from mrn_coord.lifelong import make_warehouse
    from mrn_coord.lifelong.token_passing_swaps import PickupDelivery, run_tpts
    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.conflicts import detect_first_conflict

    def _cf(res) -> bool:
        paths: dict = {}
        for snap in res.history:
            for a, c in snap.items():
                paths.setdefault(a, []).append(c)
        return detect_first_conflict(paths) is None

    # (1) constructed forced-swap instance on an open 12x3 grid (no obstacles, so
    # planning is fast and never stalls).
    grid = GridWorld(12, 3, blocked=frozenset())
    homes = {"r0": (0, 1), "r1": (10, 1)}

    def ctor_tasks():
        return [
            PickupDelivery((10, 1), (6, 1)),   # r1 collects at once, frees at (6,1)
            PickupDelivery((7, 1), (7, 2)),    # T0: handed to r0, then stolen by r1
            PickupDelivery((1, 1), (1, 2)),    # T1: r0 takes it once freed
        ]

    coff = run_tpts(grid, dict(homes), ctor_tasks(), swaps=False, max_steps=40,
                    homes=homes, keep_history=True)
    con = run_tpts(grid, dict(homes), ctor_tasks(), swaps=True, max_steps=40,
                   homes=homes, keep_history=True)

    # (2) realistic well-formed warehouse: roomy aisle=2, parking homes disjoint
    # from the pickup/delivery endpoints, a 10-task batch.
    wgrid, eps = make_warehouse(rows=3, cols=4, aisle=2)
    whomes = {f"r{i}": eps[i] for i in range(4)}
    wpool = eps[4:]
    wpairs = [(wpool[i], wpool[i + 1]) for i in range(0, len(wpool) - 1, 2)]

    def wh_tasks():
        return [PickupDelivery(p, d) for p, d in wpairs]

    woff = run_tpts(wgrid, dict(whomes), wh_tasks(), swaps=False, max_steps=60,
                    homes=whomes, keep_history=True)
    won = run_tpts(wgrid, dict(whomes), wh_tasks(), swaps=True, max_steps=60,
                   homes=whomes, keep_history=True)

    cf_all = _cf(coff) and _cf(con) and _cf(woff) and _cf(won)
    return {
        "case": "mapf_tpts",
        "ctor_off_completed": coff.completed,
        "ctor_on_completed": con.completed,
        "ctor_off_avg_service": round(coff.avg_service_time, 3),
        "ctor_on_avg_service": round(con.avg_service_time, 3),
        "ctor_off_max_wait": coff.max_wait,
        "ctor_on_max_wait": con.max_wait,
        "ctor_off_swaps": coff.swaps_fired,
        "ctor_on_swaps": con.swaps_fired,
        "wh_off_completed": woff.completed,
        "wh_on_completed": won.completed,
        "wh_off_avg_service": round(woff.avg_service_time, 3),
        "wh_on_avg_service": round(won.avg_service_time, 3),
        "wh_off_swaps": woff.swaps_fired,
        "wh_on_swaps": won.swaps_fired,
        "collision_free_by_construction": cf_all,
        "swaps_fire_only_when_enabled": (
            coff.swaps_fired == 0 and woff.swaps_fired == 0
            and con.swaps_fired > 0 and won.swaps_fired > 0),
        "forced_swap_is_single": con.swaps_fired == 1,
        "swap_improves_service": (
            con.avg_service_time < coff.avg_service_time
            and won.avg_service_time < woff.avg_service_time),
        "swap_lowers_max_wait": con.max_wait < coff.max_wait,
        "delivers_all_either_way": (
            con.completed == coff.completed and won.completed == woff.completed),
    }


def _run_online_lns() -> dict:
    # Online LNS for lifelong MAPF (online_lns.py). RHCR solves a fresh Windowed
    # MAPF instance from scratch every boundary -- the CENTRAL strategy, replan
    # EVERY agent. Online LNS instead keeps the team's committed paths and only
    # REPAIRS what must change (agents that just finished a task) plus a small
    # Large-Neighborhood destroy set, each repaired agent replanning around the
    # others' frozen paths -- collision-free by construction, like one-shot
    # mapf_lns. A single `mode` flag selects between the two, so one run pair
    # isolates exactly what reusing the previous plan buys.
    #
    # This gate pins the trade honestly on both sides. (1) WELL-FORMED moderate
    # density (roomy aisle=2, 6 agents): online LNS serves EXACTLY as many tasks
    # as CENTRAL (74 == 74) while replanning far fewer agents per boundary
    # (131 vs 162, no rejected boundaries) -- the anytime/incremental win, since
    # on an open map re-planning the unchanged agents buys CENTRAL no throughput.
    # (2) HIGH density (10 agents, tight aisles of motion): minimal repair can no
    # longer keep up -- LNS boundaries get rejected (repair finds no path for some
    # agent, so the prior collision-free plan is kept) and throughput collapses
    # (48 vs 139), the regime where CENTRAL's full replan earns its cost. Motion
    # is collision-free by construction in BOTH modes on BOTH maps. If the repair
    # logic regresses, collision-free breaks; if the reuse logic regresses, the
    # replan saving disappears.
    from mrn_coord.lifelong import TaskStream, make_warehouse, run_online_lns
    from mrn_coord.mapf.conflicts import detect_first_conflict

    def _cf(res) -> bool:
        paths: dict = {}
        for snap in res.history:
            for a, c in snap.items():
                paths.setdefault(a, []).append(c)
        return detect_first_conflict(paths) is None

    def _pair(aisle, n, steps, period, k):
        grid, eps = make_warehouse(rows=3, cols=4, aisle=aisle)
        starts = {f"r{i}": eps[i] for i in range(n)}
        out = {}
        for mode in ("central", "lns"):
            st: dict = {}
            res = run_online_lns(grid, dict(starts), TaskStream(list(eps)),
                                 mode=mode, max_steps=steps, replan_period=period,
                                 neighborhood=k, allocator="hungarian",
                                 keep_history=True, stats=st)
            out[mode] = (res, st)
        return out

    # (1) well-formed, moderate density -> equal throughput, fewer replans.
    win = _pair(aisle=2, n=6, steps=80, period=3, k=4)
    wc, wcs = win["central"]
    wl, wls = win["lns"]
    # (2) high density -> minimal repair rejects, CENTRAL's full replan wins.
    td = _pair(aisle=2, n=10, steps=100, period=5, k=2)
    tc, tcs = td["central"]
    tl, tls = td["lns"]

    cf_all = _cf(wc) and _cf(wl) and _cf(tc) and _cf(tl)
    return {
        "case": "mapf_online_lns",
        "win_central_completed": wc.completed,
        "win_lns_completed": wl.completed,
        "win_central_replans": wcs["replans"],
        "win_lns_replans": wls["replans"],
        "win_lns_rejected": wls["rejected"],
        "td_central_completed": tc.completed,
        "td_lns_completed": tl.completed,
        "td_central_replans": tcs["replans"],
        "td_lns_replans": tls["replans"],
        "td_lns_rejected": tls["rejected"],
        "collision_free_by_construction": cf_all,
        "lns_matches_throughput_with_fewer_replans": (
            wl.completed == wc.completed
            and wls["replans"] < wcs["replans"]
            and wls["rejected"] == 0),
        "central_wins_at_high_density": (
            tc.completed > tl.completed and tls["rejected"] > 0),
        "lns_never_does_more_work": (
            wls["replans"] <= wcs["replans"] and tls["replans"] <= tcs["replans"]),
    }


def _run_switchable_adg() -> dict:
    # Switchable Action Dependency Graph (mrn_sim/switchable_adg.py), a
    # reproduction of Berndt, Palmieri et al.'s "Receding-Horizon Re-ordering of
    # Multi-Agent Execution Schedules" (IROS 2020 / T-RO 2024). A MAPF plan is
    # collision-free only on schedule; the ADG (Hoenig et al.) records the passing
    # ORDER at every shared cell as precedence edges and executes while merely
    # respecting them -- collision-free whatever the timing, deadlock-free because
    # the graph is acyclic. But a FIXED order stalls everyone behind a delayed
    # first-mover. The Switchable ADG makes each passing-order edge reversible:
    # flip it so a ready robot goes first, PROVIDED the flip keeps the graph
    # acyclic (a single reachability query) -- recovering throughput with the same
    # collision-free AND deadlock-free guarantees.
    #
    # Three things this gate pins. (1) WIN: on a plus crossing, the robot scheduled
    # to cross the centre first is given a short path and then delayed; fixed order
    # makes the long-haul robot wait it out (makespan 15), the switchable ADG flips
    # the single crossing edge so the long robot goes first (makespan 10, one
    # switch). (2) NO-OP: delay the *second*-crossing (long) robot instead and
    # flipping cannot help -- the switchable run fires zero switches and matches
    # fixed exactly, so re-ordering never fires gratuitously. (3) DEADLOCK SAFETY:
    # in a head-on single-file corridor every passing-order reversal would close a
    # cycle, so the acyclicity guard refuses them all (zero switches) and the run
    # still finishes deadlock-free on the fixed order. Collision-free holds on every
    # run. If the reorder logic regresses, the WIN makespan or switch count breaks;
    # if the acyclicity guard regresses, the corridor deadlocks.
    from mrn_coord.mapf import GridWorld, cbs
    from mrn_sim.switchable_adg import (build_adg, schedule_is_collision_free,
                                        simulate)

    def _plus(n):
        mid = n // 2
        free = set()
        for x in range(n):
            free.add((x, mid))
        for y in range(n):
            free.add((mid, y))
        blocked = {(x, y) for x in range(n) for y in range(n)} - free
        return GridWorld(n, n, blocked=frozenset(blocked)), mid

    def _corridor(L, ax):
        free = set((x, 1) for x in range(L))
        free.add((ax, 2))
        blocked = {(x, y) for x in range(L) for y in range(3)} - free
        return GridWorld(L, 3, blocked=frozenset(blocked))

    def _pair(paths, delay):
        cf_cells, cf_edges = build_adg(paths)
        fix = simulate(cf_cells, cf_edges, delay, switchable=False, keep_history=True)
        sw_cells, sw_edges = build_adg(paths)
        sw = simulate(sw_cells, sw_edges, delay, switchable=True, keep_history=True)
        return fix, sw

    # plus crossing: r_block crosses the centre first (short path), r_main second.
    pg, mid = _plus(9)
    cross = cbs(pg, {"r_main": ((0, mid), (8, mid)),
                     "r_block": ((mid, mid - 1), (mid, mid + 1))})
    wf, ws = _pair(cross.paths, {"r_block": 8})       # delay the first-mover -> WIN
    nf, ns = _pair(cross.paths, {"r_main": 8})        # delay the second-mover -> no-op

    # head-on corridor with one passing bay: every reversal would deadlock.
    cg = _corridor(7, 3)
    head = cbs(cg, {"r0": ((0, 1), (6, 1)), "r1": ((6, 1), (0, 1))})
    cf, cs = _pair(head.paths, {"r0": 8})

    runs = [wf, ws, nf, ns, cf, cs]
    cf_all = all(schedule_is_collision_free(r.history) for r in runs)
    finished_all = all(r.finished for r in runs)
    deadlock_any = any(r.deadlock for r in runs)
    return {
        "case": "mapf_switchable_adg",
        "win_fix_makespan": wf.makespan,
        "win_sw_makespan": ws.makespan,
        "win_sw_switches": ws.switches,
        "noop_fix_makespan": nf.makespan,
        "noop_sw_makespan": ns.makespan,
        "noop_sw_switches": ns.switches,
        "cor_sw_switches": cs.switches,
        "cor_fix_finished": int(cf.finished),
        "cor_sw_finished": int(cs.finished),
        "collision_free_by_construction": cf_all,
        "deadlock_free_always": finished_all and not deadlock_any,
        "switch_helps_when_first_mover_delayed": (
            ws.makespan < wf.makespan and ws.switches > 0),
        "switch_is_noop_when_it_cannot_help": (
            ns.makespan == nf.makespan and ns.switches == 0),
        "unsafe_reversals_refused": (
            cs.switches == 0 and cs.finished and not cs.deadlock),
    }


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
    # Token Passing (Ma et al. 2017): lifelong MAPF by a shared reservation token
    # -- collision-free by construction; matches PIBT/RHCR on a well-formed map,
    # stalls on cramped maps where the well-formed property fails
    ("mapf_token_passing", _run_token_passing),
    ("mapf_tpts", _run_tpts),
    ("mapf_online_lns", _run_online_lns),
    ("mapf_switchable_adg", _run_switchable_adg),
    # deterministic livelock escape recovers PIBT convergence (no randomness)
    ("pibt_escape_convergence", _run_pibt_convergence),
    # strong-PIBT spine makes LaCAM's documented scaling actually deliver
    ("lacam_scaling_convergence", _run_lacam_convergence),
    # LaCAM* anytime mode reaches the CBS optimum on small instances
    ("lacam_optimality", _run_lacam_optimality),
    ("lacam_ltm_vs_optimize", _run_lacam_ltm_vs_optimize),
    # CBSH improved heuristics (CG/DG/WDG) + cardinal prioritization: same
    # optimum as CBS, far fewer high-level expansions
    ("cbsh_vs_cbs", _run_cbsh_vs_cbs),
    # EECBS: admissible WDG bound + EES cut bounded-suboptimal expansions vs ECBS
    ("eecbs_vs_ecbs", _run_eecbs_vs_ecbs),
    # ICTS: cost-tree optimal search (same optimum as CBS); pairwise pruning
    # cuts the k-agent joint searches
    ("icts_vs_cbs", _run_icts_vs_cbs),
    # Rectangle symmetry: barrier splits collapse the symmetric blowup CBS/CBSH
    # suffer on open same-direction crossings (same optimum, ~20x fewer nodes)
    # MAPF-LNS2: collision-minimizing LNS repairs a colliding start to a feasible
    # (collision-free) solution -- finds feasibility where CBS busts its budget
    ("mapf_lns2", _run_mapf_lns2),
    ("rectangle_symmetry", _run_rectangle_symmetry),
    # Corridor symmetry: a range split collapses the cell-by-cell chain CBS/CBSH
    # walk on opposite-direction one-wide crossings (same optimum, length-growing
    # 52 -> constant 8 expansions; declines when a bypass exists)
    ("corridor_symmetry", _run_corridor_symmetry),
    # Mutex propagation: a verified cardinal-conflict detector (Theorem 2 holds;
    # catches cardinals the width test misses)
    ("mutex_cardinal_detection", _run_mutex_cardinal_detection),
    # Disjoint splitting: positive/negative on one agent partitions the solution
    # space CBS's two-negative split overlaps (same optimum, fewer expansions)
    ("disjoint_vs_standard", _run_disjoint_vs_standard),
    # CCBS: continuous-time CBS with disk agents -- geometrically collision-free
    # where the discrete vertex/edge model is blind (mid-edge crossings)
    ("ccbs_continuous_time", _run_ccbs_continuous_time),
    # Network-flow: anonymous makespan-optimal MAPF via integer max-flow on a
    # time-expanded graph (polynomial, self-certified optimum)
    ("flow_anonymous_makespan", _run_flow_anonymous_makespan),
    # TSWAP: constructive complete ANONYMOUS MAPF by target swapping -- the fast,
    # sub-optimal counterpart of flow's optimal max-flow; collision-free by
    # construction, complete by a potential argument, near-optimal at scale
    ("tswap_anonymous", _run_tswap_anonymous),
    # Push and Swap/Rotate: constructive primitive-based solver -- complete with
    # slack, valid by construction, solves crowded maps where CBS blows up
    ("push_and_rotate", _run_push_and_rotate),
    # M*: subdimensional expansion -- same optimum as CBS, couples only the agents
    # that interact (collision set stays small; expansions flat as the team grows)
    ("mstar_subdimensional", _run_mstar_subdimensional),
    # Standley OD + ID: operator decomposition cuts joint branching b**n -> b;
    # independence detection solves only the colliding groups (same optimum as CBS)
    ("standley_id_od", _run_standley_id_od),
    # MDD-SAT: declarative makespan-optimal MAPF -- encode to CNF, sweep makespan,
    # self-certified by UNSAT below the optimum (labeled makespan >= flow's anon)
    ("satmdd_makespan", _run_satmdd_makespan),
    # BCP: branch-and-cut-and-price -- the LP/duality paradigm. Path-formulation
    # LP solved by column generation (pricing) + lazy conflict cuts + branching;
    # same optimum as CBS, certified by the LP lower bound (gap zero)
    ("bcp_branch_price", _run_bcp_branch_price),
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
