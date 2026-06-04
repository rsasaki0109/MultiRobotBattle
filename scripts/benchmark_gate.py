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


def _run_bcbs() -> dict:
    # BCBS (bcbs.py) is a Python reproduction of the OTHER suboptimal CBS variant
    # in Barer, Sharon, Stern & Felner's "Suboptimal Variants of the Conflict-
    # Based Search Algorithm" (SoCS 2014) -- the one ECBS (already in the package)
    # improved on. Both run focal search at both levels; the difference is the
    # high-level focal bound. BCBS(w_high, w_low) bounds the high-level focal by
    # w_high * the best COST in OPEN and the low level by w_low, so the factors
    # MULTIPLY: cost <= w_high * w_low * optimal. ECBS instead bounds by w * the
    # best LOWER BOUND (a true bound on the optimum), giving just w -- the tighter
    # accounting that superseded BCBS. This gate keeps BCBS as a faithful, honest
    # contrast (it reuses ECBS's focal low level unchanged; only the high-level
    # bound differs).
    #
    # Pins: (1) OPTIMAL -- BCBS(1,1) equals cbs on a battery (optimal_at_1_1).
    # (2) PRODUCT BOUND -- BCBS(1.5,1.5) cost <= 1.5^2 * optimum on every instance
    # (product_bound_holds), and the independent w_high/w_low knobs (which ECBS,
    # single-w, does not have) stay within w_high*w_low*opt and collision-free.
    # (3) THE DISTINCTION, made concrete -- on a found instance (5x5, 7 agents,
    # seed 23) BCBS(1.5,1.5) and ECBS(1.5) DIVERGE: BCBS expands FEWER nodes (3 vs
    # 4) and returns a HIGHER cost (31 vs 28, optimum 24) -- the looser product
    # bound lets BCBS stop earlier at a worse but still-bounded solution, exactly
    # the trade ECBS's tight bound removes. Both stay within their own guarantees
    # (BCBS 31 <= 1.5^2*24=54; ECBS 28 <= 1.5*24=36).
    import random

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.bcbs import bcbs
    from mrn_coord.mapf.cbs import cbs
    from mrn_coord.mapf.conflicts import detect_first_conflict
    from mrn_coord.mapf.ecbs import ecbs

    def _rand(seed, w, h, n, ob):
        rng = random.Random(seed)
        blocked = {(x, y) for x in range(w) for y in range(h)
                   if rng.random() < ob}
        free = [(x, y) for x in range(w) for y in range(h)
                if (x, y) not in blocked]
        if len(free) < 2 * n:
            return None
        rng.shuffle(free)
        return GridWorld(w, h, frozenset(blocked)), \
            {i: (free[i], free[n + i]) for i in range(n)}

    # (1)+(2) battery
    inst = opt_match = bound_ok = cf = 0
    W = 1.5
    for seed in range(100):
        r = _rand(seed, 5, 5, 4, 0.1)
        if r is None:
            continue
        grid, ag = r
        base = cbs(grid, ag, max_expansions=40000)
        if base is None:
            continue
        b11 = bcbs(grid, ag, w_high=1.0, w_low=1.0, max_expansions=40000)
        bww = bcbs(grid, ag, w_high=W, w_low=W, max_expansions=40000)
        if b11 is None or bww is None:
            continue
        inst += 1
        opt_match += int(b11.cost == base.cost)
        bound_ok += int(bww.cost <= W * W * base.cost + 1e-9)
        cf += int(detect_first_conflict(bww.paths) is None)

    # independent knobs (unique to BCBS) on the showcase instance
    grid, ag = _rand(23, 5, 5, 7, 0.05)
    opt = cbs(grid, ag, max_expansions=80000).cost
    knob_ok = 0
    knob_cf = 0
    for wh, wl in ((1.0, 3.0), (3.0, 1.0), (2.0, 1.0)):
        rk = bcbs(grid, ag, w_high=wh, w_low=wl, max_expansions=40000)
        knob_ok += int(rk.cost <= wh * wl * opt + 1e-9)
        knob_cf += int(detect_first_conflict(rk.paths) is None)

    # (3) divergence showcase
    sb: dict = {}
    se: dict = {}
    bshow = bcbs(grid, ag, w_high=W, w_low=W, max_expansions=40000, stats=sb)
    eshow = ecbs(grid, ag, w=W, max_expansions=40000, stats=se)

    return {
        "case": "mapf_bcbs",
        "battery_instances": inst,
        "battery_opt_match": opt_match,
        "battery_bound_ok": bound_ok,
        "battery_collision_free": cf,
        "knob_configs": 3,
        "knob_bound_ok": knob_ok,
        "knob_collision_free": knob_cf,
        "showcase_optimum": opt,
        "showcase_bcbs_cost": bshow.cost,
        "showcase_bcbs_exp": sb["expansions"],
        "showcase_ecbs_cost": eshow.cost,
        "showcase_ecbs_exp": se["expansions"],
        "optimal_at_1_1": opt_match == inst and inst > 0,
        "product_bound_holds": (bound_ok == inst and knob_ok == 3
                                and cf == inst),
        "bcbs_fewer_exp_higher_cost": (sb["expansions"] < se["expansions"]
                                       and bshow.cost > eshow.cost),
        "both_within_own_bounds": (bshow.cost <= W * W * opt
                                   and eshow.cost <= W * opt),
    }


def _run_highway() -> dict:
    # highway.py reproduces Cohen, Uras & Koenig's "Feasibility Study: Using
    # Highways for Bounded-Suboptimal Multi-Agent Path Finding" (SoCS 2015): a
    # highway is a set of DIRECTED edges marking a preferred flow direction, layered
    # on top of ECBS. ECBS already searches within a suboptimality factor w; a
    # highway steers it -- among the many w-bounded paths an agent could take --
    # toward the ones that flow WITH the highway, so when everyone follows a
    # consistent circulation the head-on/crossing conflicts largely vanish before
    # the high level branches on them, and ECBS expands far fewer nodes for the SAME
    # cost guarantee.
    #
    # The mechanism is a tiny, bound-PRESERVING change to ECBS's low level. ECBS's
    # FOCAL sublist (the w-bounded nodes) is ranked by a secondary heuristic; plain
    # ECBS ranks it by "fewest conflicts". The highway heuristic appends one key:
    # among equal-conflict paths, prefer fewest OFF-highway moves. Only the FOCAL
    # ORDERING changes; OPEN -- the admissible lower bound that certifies w -- is
    # untouched, so cost <= w * optimal still holds, and with no highway the
    # secondary key is constant and the search is byte-for-byte plain ECBS.
    #
    # The canonical map is a TWO-LANE corridor (2 rows) with traffic both ways: a
    # "keep to one side" highway (even rows ->, odd rows <-) gives each direction
    # its own lane, so the right- and left-goers never meet head-on. The gate pins:
    # (1) OFF == plain ECBS. With an empty highway, ecbs_highway matches ecbs in
    #     both expansions and cost on every instance (off_byte_identical) -- the
    #     feature is genuinely opt-in.
    # (2) FEWER EXPANSIONS, BOUND + CF PRESERVED. Across the two-lane family the
    #     highway cuts total high-level expansions about 2x (89 -> 42), is NEVER
    #     worse on any instance (lose == 0), and every highway solution stays within
    #     w * optimal and collision-free.
    # (3) SHOWCASE. On the 2x5 corridor at w=1.5 the highway cuts expansions 12 -> 3
    #     (4x) at the SAME cost (20 == 20 == optimum) -- a free win.
    # (4) HONEST: A HIGHWAY IS ADVICE, NOT FREE. On the 2x6 corridor at w=2.0 the
    #     highway's lane discipline RAISES cost (28 -> 30) -- still within the bound
    #     -- the price of advice that does not perfectly match the instance, exactly
    #     the "feasibility study" caveat.
    from mrn_coord.mapf import GridWorld, cbs
    from mrn_coord.mapf.conflicts import detect_first_conflict
    from mrn_coord.mapf.ecbs import ecbs
    from mrn_coord.mapf.highway import ecbs_highway, keep_side_highway

    def _two_lane(width):
        g = GridWorld(width, 2)
        agents = {0: ((0, 0), (width - 1, 0)), 1: ((0, 1), (width - 1, 1)),
                  2: ((width - 1, 0), (0, 0)), 3: ((width - 1, 1), (0, 1))}
        return g, agents

    widths = (4, 5, 6, 7)
    ws = (1.5, 2.0, 3.0)

    # (1) OFF == plain ECBS (empty highway).
    equiv_instances = equiv_match = 0
    for width in widths:
        g, ag = _two_lane(width)
        for w in ws:
            s1: dict = {}
            s2: dict = {}
            a = ecbs(g, ag, w=w, stats=s1)
            b = ecbs_highway(g, ag, w=w, highways=frozenset(), stats=s2)
            equiv_instances += 1
            if (a is not None and b is not None
                    and s1["expansions"] == s2["expansions"]
                    and a.cost == b.cost):
                equiv_match += 1

    # (2) FEWER EXPANSIONS, BOUND + CF preserved.
    inst = off_exp = on_exp = win = tie = lose = 0
    bound_ok = cf = cost_off = cost_on = 0
    for width in widths:
        g, ag = _two_lane(width)
        hwy = keep_side_highway(g, axis="x")
        opt = cbs(g, ag, max_expansions=20000)
        for w in ws:
            so: dict = {}
            sh: dict = {}
            off = ecbs(g, ag, w=w, stats=so)
            on = ecbs_highway(g, ag, w=w, highways=hwy, stats=sh)
            inst += 1
            off_exp += so["expansions"]
            on_exp += sh["expansions"]
            cost_off += off.cost
            cost_on += on.cost
            if on.cost <= w * opt.cost + 1e-9:
                bound_ok += 1
            if detect_first_conflict(on.paths) is None:
                cf += 1
            if sh["expansions"] < so["expansions"]:
                win += 1
            elif sh["expansions"] == so["expansions"]:
                tie += 1
            else:
                lose += 1

    # (3) SHOWCASE -- 2x5 corridor at w=1.5.
    g5, ag5 = _two_lane(5)
    h5 = keep_side_highway(g5, axis="x")
    so5: dict = {}
    sh5: dict = {}
    off5 = ecbs(g5, ag5, w=1.5, stats=so5)
    on5 = ecbs_highway(g5, ag5, w=1.5, highways=h5, stats=sh5)
    opt5 = cbs(g5, ag5, max_expansions=20000)

    # (4) HONEST -- 2x6 corridor at w=2.0, highway raises cost (within bound).
    g6, ag6 = _two_lane(6)
    h6 = keep_side_highway(g6, axis="x")
    so6: dict = {}
    sh6: dict = {}
    off6 = ecbs(g6, ag6, w=2.0, stats=so6)
    on6 = ecbs_highway(g6, ag6, w=2.0, highways=h6, stats=sh6)
    opt6 = cbs(g6, ag6, max_expansions=20000)

    return {
        "case": "mapf_highway",
        "equiv_instances": equiv_instances, "equiv_match": equiv_match,
        "off_byte_identical": equiv_match == equiv_instances,
        "battery_instances": inst,
        "battery_off_exp": off_exp, "battery_on_exp": on_exp,
        "battery_win": win, "battery_tie": tie, "battery_lose": lose,
        "battery_bound_ok": bound_ok, "battery_cf": cf,
        "battery_cost_off": cost_off, "battery_cost_on": cost_on,
        "highway_cuts_expansions": on_exp < off_exp,
        "highway_never_worse_exp": lose == 0,
        "bound_preserved": bound_ok == inst,
        "all_collision_free": cf == inst,
        "showcase_off_exp": so5["expansions"], "showcase_on_exp": sh5["expansions"],
        "showcase_off_cost": off5.cost, "showcase_on_cost": on5.cost,
        "showcase_opt": opt5.cost, "showcase_edges": len(h5),
        "showcase_fewer_exp": sh5["expansions"] < so5["expansions"],
        "showcase_equal_cost": on5.cost == off5.cost == opt5.cost,
        "honest_off_cost": off6.cost, "honest_on_cost": on6.cost,
        "honest_opt": opt6.cost,
        "honest_highway_costs_more": on6.cost > off6.cost,
        "honest_within_bound": on6.cost <= 2.0 * opt6.cost + 1e-9,
    }


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


def _run_macbs() -> dict:
    # Meta-Agent CBS (macbs.py) is a Python reproduction of Sharon, Stern, Felner
    # & Sturtevant's "Conflict-based search for optimal multi-agent pathfinding"
    # (AAAI 2012; AIJ 2015, the meta-agent section). Plain CBS is fully DECOUPLED
    # (plan each agent alone, branch one constraint per conflict) -- great when
    # agents barely interact, but on a tight bottleneck two agents collide over
    # and over and CBS re-splits the same conflict deep into the tree (an
    # exponential blow-up). MA-CBS interpolates toward a COUPLED search with one
    # knob, the conflict bound B: when two meta-agents have conflicted more than B
    # times, instead of splitting it MERGES them into one meta-agent solved by a
    # coupled (joint, time-expanded) low level. B = inf never merges (== standard
    # CBS); B = 0 merges on first conflict (collapses toward a single joint search).
    # EVERY B yields the same optimal sum-of-costs; what changes is WHERE the work
    # happens -- a bottleneck that explodes the CBS tree is absorbed into one
    # coupled solve. (Cousin of mstar's group-merging, but merging by conflict
    # FREQUENCY rather than a single collision.)
    #
    # This gate pins:
    # (1) OPTIMALITY for every B: on a random battery, macbs(B) matches the cbs
    #     optimum and is collision-free for B in {inf, 2, 1, 0} (opt_all_B ==
    #     instances), and B = inf performs zero merges with all-singleton groups
    #     (binf_no_merge == instances) -- i.e. it IS standard CBS at B = inf.
    # (2) MERGING CUTS THE SEARCH: on a 3-agent symmetry bottleneck the high-level
    #     expansions collapse 71 (B=inf) -> 11 (B=1) -> 3 (B=0) for the SAME
    #     optimum, the conflicting agents absorbed into one coupled meta-agent
    #     (bottleneck_b0_max_group_size == 3); and a corridor swap collapses
    #     16 -> 2.
    import random

    from mrn_coord.mapf import GridWorld, cbs
    from mrn_coord.mapf.conflicts import detect_first_conflict
    from mrn_coord.mapf.macbs import macbs

    BIG = 10 ** 9

    def _inst(seed, n, w, h):
        rng = random.Random(seed)
        free = [(x, y) for x in range(w) for y in range(h)]
        cells = rng.sample(free, 2 * n)
        return GridWorld(w, h), {i: (cells[i], cells[n + i]) for i in range(n)}

    # (1) Optimality battery.
    instances = opt_all_B = cf_bad = binf_no_merge = 0
    for seed in range(40):
        for n, w, h in ((3, 5, 5), (3, 4, 4), (4, 5, 4)):
            grid, agents = _inst(seed, n, w, h)
            base = cbs(grid, agents, max_expansions=40000)
            if base is None:
                continue
            instances += 1
            ok = True
            for b in (BIG, 2, 1, 0):
                sol = macbs(grid, agents, merge_bound=b, max_expansions=40000)
                if sol is None or sol.cost != base.cost:
                    ok = False
                elif detect_first_conflict(sol.paths) is not None:
                    cf_bad += 1
            opt_all_B += int(ok)
            si: dict = {}
            macbs(grid, agents, merge_bound=BIG, max_expansions=40000, stats=si)
            binf_no_merge += int(si["merges"] == 0 and si["max_group_size"] == 1)

    # (2a) A 3-agent symmetry bottleneck (a frozen random instance).
    bgrid = GridWorld(4, 4)
    bagents = {0: ((2, 3), (1, 0)), 1: ((0, 3), (1, 1)), 2: ((3, 2), (0, 0))}
    bcbs = cbs(bgrid, bagents)
    binf: dict = {}
    sol_inf = macbs(bgrid, bagents, merge_bound=BIG, stats=binf)
    b1: dict = {}
    macbs(bgrid, bagents, merge_bound=1, stats=b1)
    b0: dict = {}
    sol_b0 = macbs(bgrid, bagents, merge_bound=0, stats=b0)

    # (2b) A corridor swap: one pocket, head-on.
    cfree = set((x, 0) for x in range(5))
    cfree.add((2, 1))
    cblocked = {(x, y) for x in range(5) for y in range(2)} - cfree
    cgrid = GridWorld(5, 2, blocked=frozenset(cblocked))
    cagents = {0: ((0, 0), (4, 0)), 1: ((4, 0), (0, 0))}
    cinf: dict = {}
    macbs(cgrid, cagents, merge_bound=BIG, stats=cinf)
    c0: dict = {}
    sol_c0 = macbs(cgrid, cagents, merge_bound=0, stats=c0)

    return {
        "case": "mapf_macbs",
        "instances": instances,
        "opt_all_B": opt_all_B,
        "cf_bad": cf_bad,
        "binf_no_merge": binf_no_merge,
        "bottleneck_cbs_cost": bcbs.cost,
        "bottleneck_cost": sol_inf.cost,
        "bottleneck_binf_expansions": binf["expansions"],
        "bottleneck_b1_expansions": b1["expansions"],
        "bottleneck_b0_expansions": b0["expansions"],
        "bottleneck_b0_cost": sol_b0.cost,
        "bottleneck_b0_max_group_size": b0["max_group_size"],
        "corridor_cost": sol_c0.cost,
        "corridor_binf_expansions": cinf["expansions"],
        "corridor_b0_expansions": c0["expansions"],
        "corridor_b0_max_group_size": c0["max_group_size"],
        "optimal_for_every_B": (opt_all_B == instances and cf_bad == 0),
        "cbs_reproduced_at_inf": (binf_no_merge == instances
                                  and sol_inf.cost == bcbs.cost),
        "merging_cuts_search": (b0["expansions"] < binf["expansions"]
                                and sol_b0.cost == bcbs.cost
                                and c0["expansions"] < cinf["expansions"]),
        "coupling_collapse": (b0["max_group_size"] == 3
                              and c0["max_group_size"] == 2),
    }


def _run_whca() -> dict:
    # Windowed Hierarchical Cooperative A* (whca.py) is a Python reproduction of
    # David Silver's "Cooperative Pathfinding" (AIIDE 2005). It layers three
    # ideas on prioritized planning:
    #   - Cooperative A* (CA*): plan agents in priority order, each reserving its
    #     space-time path so later agents avoid it (== prioritized_planning).
    #   - Hierarchical (HCA*): use the TRUE shortest-path distance to the goal on
    #     the static map (Reverse Resumable A*, RRA*) as the heuristic instead of
    #     Manhattan -- perfect on the obstacle map, so the cooperative A* stops
    #     exploring the dead ends a wall creates.
    #   - Windowed (WHCA*): cooperate only within a w-step lookahead window, then
    #     roll the window and replan with a ROTATING priority order. The window
    #     bounds the search depth (it scales) and the rotation lets a blocked
    #     agent lead next window, breaking transient deadlocks a single fixed
    #     priority order livelocks on. Collision-free by construction.
    #
    # This gate pins:
    # (1) THE ABSTRACT HEURISTIC IS EXACT: RRA*'s true distance equals a BFS from
    #     the goal on an obstacle map (rra_mismatches == 0).
    # (2) HIERARCHICAL CUTS THE SEARCH: on a map where a wall makes Manhattan
    #     badly misleading, the cooperative A* with the true-distance heuristic
    #     expands far fewer states than with Manhattan (101 vs 551).
    # (3) WINDOWED EXECUTION IS COLLISION-FREE AND REACHES EVERY GOAL across a
    #     random battery, for small and large windows alike.
    # (4) ROLLING WINDOWS BEAT FIXED PRIORITY: on a battery where plain
    #     prioritized planning (and full-horizon, non-rotating WHCA*, which is
    #     prioritized planning with the true-distance heuristic) FAIL, the
    #     windowed rolling WHCA* routes every agent collision-free -- isolating
    #     the win to the window + rotation, not the heuristic.
    import random
    from collections import deque

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.grid import manhattan
    from mrn_coord.mapf.conflicts import detect_first_conflict
    from mrn_coord.mapf.prioritized import prioritized_planning
    from mrn_coord.mapf.whca import RRAStar, _segment_search, whca_star

    # (1) RRA* (true distance) == BFS from the goal.
    def _bfs(grid, goal):
        d = {goal: 0}
        q = deque([goal])
        while q:
            c = q.popleft()
            for nb in grid.neighbors(c):
                if nb != c and nb not in d:
                    d[nb] = d[c] + 1
                    q.append(nb)
        return d

    hgrid = GridWorld(8, 8, blocked=frozenset((3, y) for y in range(0, 6)))
    bfs = _bfs(hgrid, (7, 7))
    rra = RRAStar(hgrid, (7, 7), (0, 0))
    rra_mismatches = sum(1 for c in bfs if rra.distance(c) != bfs[c])

    # (2) Hierarchical heuristic vs Manhattan on a wall the agent must go around.
    wgrid = GridWorld(11, 11, blocked=frozenset((5, y) for y in range(0, 9)))
    wstart, wgoal = (0, 0), (10, 0)
    wrra = RRAStar(wgrid, wgoal, wstart)
    st_true: dict = {}
    _segment_search(wgrid, wstart, wgoal, 0, 999, {}, set(), wrra.distance,
                    stats=st_true)
    st_man: dict = {}
    _segment_search(wgrid, wstart, wgoal, 0, 999, {}, set(),
                    lambda c: manhattan(c, wgoal), stats=st_man)

    # (3) Random battery: collision-free + reaches every goal for small/large w.
    def _inst(seed, n, w, h):
        rng = random.Random(seed)
        free = [(x, y) for x in range(w) for y in range(h)]
        cells = rng.sample(free, 2 * n)
        return GridWorld(w, h), {i: (cells[i], cells[n + i]) for i in range(n)}

    batt_inst = batt_solved = batt_cf = 0
    for win in (4, 8, 16):
        for seed in range(60):
            grid, agents = _inst(seed, 5, 8, 8)
            batt_inst += 1
            sol = whca_star(grid, agents, window=win)
            if sol is None:
                continue
            batt_solved += 1
            if detect_first_conflict(sol.paths) is None:
                batt_cf += 1

    # (4) Rolling windows beat fixed priority. Congested obstacle instances where
    # prioritized planning fails; count where windowed WHCA* solves CF and where
    # full-horizon non-rotating WHCA* ALSO fails (isolating the window+rotation).
    def _cinst(seed, n, w, h, blk):
        rng = random.Random(seed)
        free = [(x, y) for x in range(w) for y in range(h)]
        nb = int(len(free) * blk)
        blocked: set = set()
        while len(blocked) < nb:
            blocked.add(rng.choice(free))
        free2 = [c for c in free if c not in blocked]
        if len(free2) < 2 * n:
            return None
        cells = rng.sample(free2, 2 * n)
        return (GridWorld(w, h, blocked=frozenset(blocked)),
                {i: (cells[i], cells[n + i]) for i in range(n)})

    # WHCA* is incomplete, so it does not solve *every* prioritized failure --
    # what it must do is solve a non-trivial set of them, and on each one it
    # solves, the fixed-priority baselines (plain prioritized planning AND
    # full-horizon non-rotating WHCA*, i.e. prioritized planning with the
    # true-distance heuristic) must BOTH fail -- isolating the win to the rolling
    # window, not the heuristic.
    win_tested = win_prio_failed = win_whca_solved_cf = win_isolated = 0
    for seed in range(400):
        r = _cinst(seed, 6, 7, 7, 0.18)
        if r is None:
            continue
        grid, agents = r
        win_tested += 1
        if prioritized_planning(grid, agents) is not None:
            continue
        win_prio_failed += 1
        wh = whca_star(grid, agents, window=8)
        if wh is None or detect_first_conflict(wh.paths) is not None:
            continue
        win_whca_solved_cf += 1
        # prioritized already failed here; confirm full-horizon non-rotating does too
        if whca_star(grid, agents, window=400, rotate_priority=False) is None:
            win_isolated += 1

    # A frozen showcase from that battery (seed 20): 6 agents, prioritized and
    # full-horizon non-rotating both fail; windowed rolling WHCA* solves it.
    sgrid, sagents = _cinst(20, 6, 7, 7, 0.18)
    s_wh = whca_star(sgrid, sagents, window=8)

    return {
        "case": "mapf_whca",
        "rra_cells": len(bfs),
        "rra_mismatches": rra_mismatches,
        "hca_true_expansions": st_true["expansions"],
        "manhattan_expansions": st_man["expansions"],
        "battery_instances": batt_inst,
        "battery_solved": batt_solved,
        "battery_collision_free": batt_cf,
        "win_tested": win_tested,
        "win_prio_failed": win_prio_failed,
        "win_whca_solved_cf": win_whca_solved_cf,
        "win_isolated": win_isolated,
        "showcase_whca_cost": s_wh.cost if s_wh else -1,
        "showcase_prioritized_none": prioritized_planning(sgrid, sagents) is None,
        "showcase_fullhorizon_none": whca_star(sgrid, sagents, window=400,
                                               rotate_priority=False) is None,
        "showcase_cf": (s_wh is not None
                        and detect_first_conflict(s_wh.paths) is None),
        "true_distance_is_exact": rra_mismatches == 0,
        "hierarchical_heuristic_prunes":
            st_true["expansions"] < st_man["expansions"],
        "windowed_complete_and_collision_free_here":
            batt_solved == batt_inst and batt_cf == batt_inst,
        "rolling_window_beats_fixed_priority":
            (win_whca_solved_cf > 0
             and win_isolated == win_whca_solved_cf),
    }


def _run_cbs_bypass() -> dict:
    # CBS with bypassing conflicts (bypass.py) is a Python reproduction of
    # Boyarski et al.'s "Don't Split, Try to Work It Out: Bypassing Conflicts in
    # Multi-Agent Pathfinding" (ICAPS 2015), the BP component of ICBS. Standard
    # CBS always SPLITS a conflict into two constraint-tree children. BP first
    # checks whether either child is a valid BYPASS of the current node: same
    # cost AND strictly fewer conflicts. If so it ADOPTS that child's path into
    # the node (constraints unchanged) and re-examines it -- no new tree nodes.
    # The adopted path is valid (found under more constraints) and same-cost, so
    # the optimum is preserved; a cardinal conflict can never be bypassed (both
    # children gain cost), so BP shrinks the tree precisely on the non-cardinal
    # conflicts plain CBS wastefully splits.
    #
    # This gate pins:
    # (1) SAME OPTIMUM AS CBS: on a random battery, cbs_bypass matches the cbs
    #     optimum and is collision-free (opt_match == instances, cf == instances).
    # (2) BYPASS SHRINKS THE SEARCH: aggregated over the battery, bypass cuts both
    #     high-level expansions (867 -> 490) and generated tree nodes (1094 ->
    #     340), and NEVER expands more than the no-bypass ablation (worse == 0).
    # (3) A frozen showcase (seed 54, 5 agents on 6x6): same optimum 22, but
    #     expansions 17 -> 3 and generated nodes 32 -> 4 via 3 bypasses.
    import random

    from mrn_coord.mapf import GridWorld, cbs
    from mrn_coord.mapf.bypass import cbs_bypass
    from mrn_coord.mapf.conflicts import detect_first_conflict

    def _inst(seed, n, w, h):
        rng = random.Random(seed)
        free = [(x, y) for x in range(w) for y in range(h)]
        cells = rng.sample(free, 2 * n)
        return GridWorld(w, h), {i: (cells[i], cells[n + i]) for i in range(n)}

    instances = opt_match = cf = 0
    exp_off = exp_on = gen_off = gen_on = bypasses = worse = 0
    for seed in range(80):
        for n, w, h in ((3, 5, 5), (4, 5, 5), (4, 6, 6), (5, 6, 6)):
            grid, agents = _inst(seed, n, w, h)
            base = cbs(grid, agents, max_expansions=40000)
            if base is None:
                continue
            instances += 1
            so: dict = {}
            sb: dict = {}
            off = cbs_bypass(grid, agents, bypass=False, max_expansions=40000,
                             stats=so)
            on = cbs_bypass(grid, agents, bypass=True, max_expansions=40000,
                            stats=sb)
            if on is not None and on.cost == base.cost:
                opt_match += 1
            if on is not None and detect_first_conflict(on.paths) is None:
                cf += 1
            exp_off += so["expansions"]
            exp_on += sb["expansions"]
            gen_off += so["generated"]
            gen_on += sb["generated"]
            bypasses += sb["bypasses"]
            if sb["expansions"] > so["expansions"]:
                worse += 1

    # A frozen showcase where bypassing collapses the tree.
    sgrid, sagents = _inst(54, 5, 6, 6)
    sbase = cbs(sgrid, sagents)
    sso: dict = {}
    ssb: dict = {}
    cbs_bypass(sgrid, sagents, bypass=False, stats=sso)
    son = cbs_bypass(sgrid, sagents, bypass=True, stats=ssb)

    return {
        "case": "mapf_cbs_bypass",
        "instances": instances,
        "opt_match": opt_match,
        "cf": cf,
        "battery_exp_off": exp_off,
        "battery_exp_on": exp_on,
        "battery_gen_off": gen_off,
        "battery_gen_on": gen_on,
        "battery_bypasses": bypasses,
        "battery_worse": worse,
        "showcase_cbs_cost": sbase.cost,
        "showcase_bypass_cost": son.cost,
        "showcase_off_exp": sso["expansions"],
        "showcase_off_gen": sso["generated"],
        "showcase_on_exp": ssb["expansions"],
        "showcase_on_gen": ssb["generated"],
        "showcase_bypasses": ssb["bypasses"],
        "same_optimum_as_cbs": (opt_match == instances and cf == instances
                                and son.cost == sbase.cost),
        "bypass_shrinks_search": (exp_on < exp_off and gen_on < gen_off
                                  and bypasses > 0),
        "bypass_never_hurts": worse == 0,
    }


def _run_ddm() -> dict:
    # DDM (ddm.py) is a Python reproduction of Han & Yu's "DDM: Fast Near-Optimal
    # Multi-Robot Path Planning using Diversified-Path and Optimal Sub-Problem
    # Solution Database Heuristics" (RA-L 2020). DDM is a decoupled planner whose
    # two named heuristics this gate reproduces and pins:
    #   (1) the OPTIMAL SUB-PROBLEM SOLUTION DATABASE -- conflicts are resolved in
    #       small 2x3 / 3x3 windows by an *optimal* (min-makespan) collision-free
    #       joint motion, precomputed once and reused in O(1); and
    #   (2) PATH DIVERSIFICATION -- robots pick among several shortest paths the
    #       one overlapping the others least, so fewer conflicts ever form.
    # The two are wired into a database-driven online loop that is COLLISION-FREE
    # BY CONSTRUCTION (every committed step is a database-certified joint move or
    # an unconflicted advance).
    #
    # Honest scope: this reproduces the two heuristics and a database-driven
    # resolver, NOT the paper's full warehouse pipeline; like DDM it is INCOMPLETE
    # (it can livelock / hit a coupling larger than a local window and return
    # None). It is not claimed to beat prioritized planning on open random grids
    # -- the gate pins the *verified mechanisms*, not an overclaimed win:
    # database optimality, translation-invariant reuse, the canonical local
    # maneuvers, the diversification effect, and the collision-free guarantee.
    import random
    from collections import deque
    from itertools import product

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.conflicts import detect_first_conflict
    from mrn_coord.mapf.ddm import (
        LocalDatabase, _dist_field, _diversified_paths, _shortest_paths, ddm,
    )

    # (1) Database == brute-force optimal makespan over 2x3 and 3x3 sub-instances.
    def _brute(cells, starts, goals):
        order = sorted(starts)
        cs = set(cells)

        def nb(c):
            x, y = c
            return [c] + [n for n in ((x + 1, y), (x - 1, y), (x, y + 1),
                                      (x, y - 1)) if n in cs]

        s = tuple(starts[r] for r in order)
        g = tuple(goals[r] for r in order)
        n = len(order)
        seen = {s}
        q = deque([(s, 0)])
        while q:
            u, d = q.popleft()
            if u == g:
                return d
            for v in product(*[nb(u[i]) for i in range(n)]):
                if len(set(v)) != n:
                    continue
                if any(u[i] == v[j] and u[j] == v[i] and u[i] != u[j]
                       for i in range(n) for j in range(i + 1, n)):
                    continue
                if v not in seen:
                    seen.add(v)
                    q.append((v, d + 1))
        return None

    def _valid(plan, goals):
        if plan is None:
            return False
        for cfg in plan:
            if len(set(cfg.values())) != len(cfg):
                return False
        for t in range(len(plan) - 1):
            u, v = plan[t], plan[t + 1]
            for a in u:
                for b in u:
                    if a < b and u[a] == v[b] and u[b] == v[a] and u[a] != u[b]:
                        return False
        return all(plan[-1][r] == goals[r] for r in goals)

    db_match = {}
    for rw, rh, key in ((3, 2, "db23"), (3, 3, "db33")):
        cells = [(x, y) for x in range(rw) for y in range(rh)]
        db = LocalDatabase()
        rng = random.Random(7)
        ok = 0
        for _ in range(300):
            k = rng.randint(2, min(4, len(cells) // 2))
            pts = rng.sample(cells, 2 * k)
            s = {i: pts[i] for i in range(k)}
            g = {i: pts[k + i] for i in range(k)}
            pl = db.solve(cells, s, g)
            bm = _brute(cells, s, g)
            if ((pl is None and bm is None)
                    or (pl is not None and bm is not None
                        and len(pl) - 1 == bm and _valid(pl, g))):
                ok += 1
        db_match[key] = ok

    # (2) Translation-invariant reuse: a shifted copy of a solved pattern reuses
    # the cache (no fresh solve).
    db = LocalDatabase()
    cells23 = [(x, y) for x in range(3) for y in range(2)]
    db.solve(cells23, {0: (0, 0), 1: (1, 0), 2: (2, 0)},
             {0: (1, 0), 1: (2, 0), 2: (0, 0)})
    solves_after_first = db.solves
    db.solve([(x + 10, y + 10) for x, y in cells23],
             {0: (10, 10), 1: (11, 10), 2: (12, 10)},
             {0: (11, 10), 1: (12, 10), 2: (10, 10)})
    translation_reused = (db.solves == solves_after_first)

    # (3) Canonical local maneuvers in a 2x3 window: a 3-robot rotation and a
    # 2-robot end-swap -- motions a single-cell view cannot perform.
    dbc = LocalDatabase()
    rot = dbc.solve(cells23, {0: (0, 0), 1: (1, 0), 2: (2, 0)},
                    {0: (1, 0), 1: (2, 0), 2: (0, 0)})
    swp = dbc.solve(cells23, {0: (0, 0), 1: (2, 0)}, {0: (2, 0), 1: (0, 0)})
    rotation_steps = len(rot) - 1
    swap_steps = len(swp) - 1

    # (4) Diversification reduces the space-time footprint overlap.
    def _inst(seed, n, w, h):
        rng = random.Random(seed)
        free = [(x, y) for x in range(w) for y in range(h)]
        c = rng.sample(free, 2 * n)
        return GridWorld(w, h), {i: (c[i], c[n + i]) for i in range(n)}

    def _overlap(paths):
        claimed: dict = {}
        o = 0
        for p in paths.values():
            for t, c in enumerate(p):
                o += claimed.get((c, t), 0)
                claimed[(c, t)] = claimed.get((c, t), 0) + 1
        return o

    ov_on = ov_off = 0
    for seed in range(200):
        grid, ag = _inst(seed, 8, 8, 8)
        fields = {r: _dist_field(grid, ag[r][1]) for r in ag}
        d = _diversified_paths(grid, ag, fields, candidates=4)
        f = {r: _shortest_paths(grid, ag[r][0], ag[r][1], fields[r], 1)[0]
             for r in ag}
        if d is None:
            continue
        ov_on += _overlap(d)
        ov_off += _overlap(f)

    # (5) Collision-free by construction across a battery (incomplete: it solves
    # a fraction, but every returned solution is collision-free and on-goal).
    batt_inst = batt_solved = batt_cf_violations = 0
    for seed in range(300):
        grid, ag = _inst(seed, 6, 8, 8)
        batt_inst += 1
        sol = ddm(grid, ag)
        if sol is None:
            continue
        batt_solved += 1
        bad = (detect_first_conflict(sol.paths) is not None
               or not all(sol.paths[r][-1] == ag[r][1]
                          and sol.paths[r][0] == ag[r][0] for r in ag))
        if bad:
            batt_cf_violations += 1

    # A frozen showcase that fires the database and solves collision-free.
    sgrid, sag = _inst(4, 5, 5, 5)
    sst: dict = {}
    ssol = ddm(sgrid, sag, stats=sst)

    return {
        "case": "mapf_ddm",
        "db23_optimal": db_match["db23"],
        "db33_optimal": db_match["db33"],
        "translation_reused": translation_reused,
        "rotation_steps": rotation_steps,
        "swap_steps": swap_steps,
        "diversify_overlap_off": ov_off,
        "diversify_overlap_on": ov_on,
        "battery_instances": batt_inst,
        "battery_solved": batt_solved,
        "battery_cf_violations": batt_cf_violations,
        "showcase_cost": ssol.cost if ssol else -1,
        "showcase_makespan": sst.get("makespan"),
        "showcase_database_solves": sst.get("database_solves"),
        "showcase_cf": (ssol is not None
                        and detect_first_conflict(ssol.paths) is None),
        "database_optimal": (db_match["db23"] == 300 and db_match["db33"] == 300),
        "database_translation_invariant": translation_reused,
        "canonical_maneuvers": (rotation_steps == 3 and swap_steps == 4),
        "diversification_reduces_congestion": ov_on < ov_off,
        "collision_free_by_construction": batt_cf_violations == 0,
    }


def _run_epea() -> dict:
    # EPEA* (epea.py) is a Python reproduction of Goldenberg, Felner, Stern,
    # Sharon, Sturtevant, Holte & Schaeffer's "Enhanced Partial Expansion A*"
    # (JAIR 2014), with MAPF as the showcase domain. Plain A* over the joint
    # configuration space (mstar.joint_astar) generates ALL successors of every
    # expanded node -- including the high-f ones that just sit in OPEN. EPEA*
    # generates only the successors whose f equals the node's current f, using a
    # domain Operator Selection Function (OSF), and re-inserts the node with its
    # f bumped to the next achievable child f -- so high-f children are produced
    # lazily, and not at all if the search finishes first. Same optimum as CBS;
    # far fewer generated nodes than the fully-expanding joint A* it beats.
    #
    # This gate pins:
    # (1) SAME OPTIMUM AS CBS: on a random battery, epea_star matches the cbs
    #     optimum and is collision-free (opt == instances, cf == instances).
    # (2) PARTIAL EXPANSION CUTS NODE GENERATION: aggregated over the battery,
    #     EPEA* generates far fewer nodes than joint A* (150867 -> 2605, ~58x)
    #     and never more on any instance (worse == 0). It pops slightly MORE
    #     nodes (partial re-expansions) -- the honest trade EPEA* makes.
    # (3) A frozen showcase (seed 87, 3 agents on 5x5): same optimum 13, but
    #     generated nodes 6572 -> 84.
    import random

    from mrn_coord.mapf import GridWorld, cbs
    from mrn_coord.mapf.conflicts import detect_first_conflict
    from mrn_coord.mapf.epea import epea_star
    from mrn_coord.mapf.mstar import joint_astar

    def _inst(seed, n, w, h):
        rng = random.Random(seed)
        free = [(x, y) for x in range(w) for y in range(h)]
        cells = rng.sample(free, 2 * n)
        return GridWorld(w, h), {i: (cells[i], cells[n + i]) for i in range(n)}

    instances = opt = cf = worse = fewer = 0
    gen_ja = gen_ep = exp_ja = exp_ep = 0
    for seed in range(60):
        for n, w, h in ((2, 5, 5), (3, 5, 5), (3, 4, 4)):
            grid, agents = _inst(seed, n, w, h)
            base = cbs(grid, agents, max_expansions=40000)
            if base is None:
                continue
            instances += 1
            sja: dict = {}
            sep: dict = {}
            ja = joint_astar(grid, agents, stats=sja, max_expansions=200000)
            ep = epea_star(grid, agents, stats=sep, max_expansions=200000)
            if ep is not None and ep.cost == base.cost:
                opt += 1
            if ep is not None and detect_first_conflict(ep.paths) is None:
                cf += 1
            if ja is not None and ep is not None:
                gen_ja += sja["generated"]
                gen_ep += sep["generated"]
                exp_ja += sja["expansions"]
                exp_ep += sep["expansions"]
                if sep["generated"] < sja["generated"]:
                    fewer += 1
                if sep["generated"] > sja["generated"]:
                    worse += 1

    # A frozen showcase where partial expansion collapses node generation.
    sgrid, sagents = _inst(87, 3, 5, 5)
    sbase = cbs(sgrid, sagents)
    ssja: dict = {}
    ssep: dict = {}
    joint_astar(sgrid, sagents, stats=ssja)
    sep_sol = epea_star(sgrid, sagents, stats=ssep)

    return {
        "case": "mapf_epea",
        "instances": instances,
        "opt": opt,
        "cf": cf,
        "battery_gen_joint_astar": gen_ja,
        "battery_gen_epea": gen_ep,
        "battery_exp_joint_astar": exp_ja,
        "battery_exp_epea": exp_ep,
        "battery_fewer": fewer,
        "battery_worse": worse,
        "showcase_cbs_cost": sbase.cost,
        "showcase_epea_cost": sep_sol.cost,
        "showcase_gen_joint_astar": ssja["generated"],
        "showcase_gen_epea": ssep["generated"],
        "same_optimum_as_cbs": (opt == instances and cf == instances
                                and sep_sol.cost == sbase.cost),
        "partial_expansion_cuts_generation": (gen_ep < gen_ja and fewer > 0),
        "epea_never_generates_more": worse == 0,
    }


def _run_sipps() -> dict:
    # SIPPS (sipps.py) is a Python reproduction of the safe-interval low-level
    # planner behind MAPF-LNS2 (Li, Chen, Harabor, Stuckey & Koenig, "MAPF-LNS2",
    # AAAI 2022). It is to lns2's time-expanded _plan_min_collision what plan_sipp
    # is to plan_path: the SAME answer over a far smaller state space. It splits a
    # cell's timeline into safe intervals by HARD constraints (never violable) and
    # MINIMIZES the number of SOFT collisions (other agents' paths -- passable at
    # one collision each, counted even while waiting), shortest among ties.
    #
    # The gate pins:
    # (1) NO-SOFT EXACTNESS: with no soft constraints SIPPS == plan_sipp (same
    #     length, zero collisions) -- it really is SIPP plus soft accounting.
    # (2) OPTIMALITY: on a 300-instance battery (replan agent 0 against the other
    #     agents' shortest paths as soft), SIPPS's (collisions, length) equals the
    #     true time-expanded Dijkstra optimum on EVERY instance (optimal == 300,
    #     length_optimal == 300). Its corrected, wait-aware safe-interval
    #     dominance is what makes this exact.
    # (3) SAFE-INTERVAL WIN: on a 12-cell corridor with a long blocked stretch
    #     (hard, then soft) SIPPS expands 11 states where the time-expanded
    #     collision-minimizer expands 253, finding the zero-collision wait path.
    # (4) HONEST EXTRAS: aggregate battery expansions (it pops more than the
    #     showcase suggests on dense tiny grids -- worse_expansions == 2), and that
    #     being EXACT it strictly beats lns2's shipped _plan_min_collision
    #     heuristic on 20 of the 300 instances (fewer collisions).
    import heapq
    import random

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.lns import _bfs_dist_from
    from mrn_coord.mapf.lns2 import _plan_min_collision, _soft_reservations
    from mrn_coord.mapf.sipp import plan_sipp
    from mrn_coord.mapf.sipps import plan_sipps
    from mrn_coord.mapf.space_time_astar import plan_path
    import mrn_coord.mapf.lns2 as _L

    def _sh(sv, se):
        h = 0
        for k in sv:
            h = max(h, k[1])
        for k in se:
            h = max(h, k[2])
        return h

    def _count(path, sv, se):
        H = max(_sh(sv, se), len(path) - 1)
        col = 0
        for t in range(H + 1):
            cell = path[t] if t < len(path) else path[-1]
            col += sv.get((cell, t), 0)
        for t in range(1, len(path)):
            if path[t] != path[t - 1]:
                col += se.get((path[t - 1], path[t], t), 0)
        return col

    def _eff(path):
        g = path[-1]
        i = len(path) - 1
        while i > 0 and path[i - 1] == g:
            i -= 1
        return i

    def _brute(grid, s, go, sv, se):
        """True lexicographic (collisions, length) optimum, time-expanded."""
        H = _sh(sv, se)
        maxt = H + 2 * grid.width * grid.height + 5
        sc = sv.get((s, 0), 0)
        pq = [(sc, 0, s)]
        best = {(s, 0): (sc, 0)}
        bcol = blen = None
        while pq:
            col, t, cell = heapq.heappop(pq)
            cur = best.get((cell, t))
            if cur is not None and (cur[0] < col or (cur[0] == col and cur[1] < t)):
                continue
            if cell == go:
                tot = col + sum(sv.get((go, tt), 0) for tt in range(t + 1, H + 1))
                if bcol is None or tot < bcol or (tot == bcol and t < blen):
                    bcol, blen = tot, t
            if t >= maxt:
                continue
            for nc in grid.neighbors(cell):
                nt = t + 1
                add = sv.get((nc, nt), 0)
                if nc != cell:
                    add += se.get((cell, nc, nt), 0)
                ncol = col + add
                key = (nc, nt)
                cur = best.get(key)
                if cur is None or (ncol, nt) < cur:
                    best[key] = (ncol, nt)
                    heapq.heappush(pq, (ncol, nt, nc))
        return bcol, blen

    def _te(grid, s, go, hard_v, sv, se, stats):
        """Time-expanded collision-minimizer; counts (cell,time) expansions."""
        H = _sh(sv, se)
        for (_c, t) in hard_v:
            H = max(H, t)
        maxt = H + 2 * grid.width * grid.height + 5
        gh = max([tt for (c, tt) in hard_v if c == go] + [0])
        sc = sv.get((s, 0), 0)
        pq = [(sc, 0, s)]
        best = {(s, 0): sc}
        exp = 0
        while pq:
            col, t, cell = heapq.heappop(pq)
            if best.get((cell, t), 1e9) < col:
                continue
            exp += 1
            if cell == go and t >= gh:
                stats["expansions"] = exp
                return col + sum(sv.get((go, tt), 0) for tt in range(t + 1, H + 1))
            if t >= maxt:
                continue
            for nc in grid.neighbors(cell):
                nt = t + 1
                if (nc, nt) in hard_v:
                    continue
                add = sv.get((nc, nt), 0)
                if nc != cell:
                    add += se.get((cell, nc, nt), 0)
                ncol = col + add
                if best.get((nc, nt), 1e9) > ncol:
                    best[(nc, nt)] = ncol
                    heapq.heappush(pq, (ncol, nt, nc))
        stats["expansions"] = exp
        return None

    # (1) no-soft exactness vs plain SIPP
    g8 = GridWorld(8, 8)
    nosoft_mismatch = 0
    for seed in range(200):
        rng = random.Random(seed)
        free = [(x, y) for x in range(8) for y in range(8)]
        s, go = rng.sample(free, 2)
        p1 = plan_sipp(g8, s, go)
        st: dict = {}
        p2 = plan_sipps(g8, s, go, stats=st)
        if (p1 is None) != (p2 is None):
            nosoft_mismatch += 1
            continue
        if p1 is not None and (len(p1) != len(p2) or st["collisions"] != 0):
            nosoft_mismatch += 1

    # (2)+(4) optimality battery vs true brute, expansion aggregate, beats-heuristic
    n = optimal = length_optimal = beats = worse_exp = 0
    sipps_exp = te_exp = 0
    for seed in range(300):
        rng = random.Random(seed)
        free = [(x, y) for x in range(7) for y in range(7)]
        cells = rng.sample(free, 10)
        grid = GridWorld(7, 7)
        ag = {i: (cells[i], cells[5 + i]) for i in range(5)}
        paths: dict = {}
        ok = True
        for i in range(1, 5):
            p = plan_path(grid, ag[i][0], ag[i][1])
            if p is None:
                ok = False
                break
            paths[i] = p
        if not ok:
            continue
        horizon = max(len(p) for p in paths.values()) + 10
        sv, se = _soft_reservations({**paths}, set(), horizon)
        s, go = ag[0]
        bcol, blen = _brute(grid, s, go, sv, se)
        st = {}
        pst = plan_sipps(grid, s, go, soft_vertex=sv, soft_edge=se, stats=st)
        if pst is None or bcol is None:
            continue
        n += 1
        cst = _count(pst, sv, se)
        if cst == bcol:
            optimal += 1
        if cst == bcol and _eff(pst) == blen:
            length_optimal += 1
        tst: dict = {}
        _te(grid, s, go, frozenset(), sv, se, tst)
        sipps_exp += st["expansions"]
        te_exp += tst["expansions"]
        if st["expansions"] > tst["expansions"]:
            worse_exp += 1
        _L.came_from = {}
        ph = _plan_min_collision(grid, s, go, sv, se, horizon + 5,
                                 _bfs_dist_from(grid, go))
        if ph is not None and _count(ph, sv, se) > cst:
            beats += 1

    # (3) corridor showcase: a long blocked stretch, hard then soft
    corr = GridWorld(12, 1)
    cs, cg = (0, 0), (11, 0)
    hard = frozenset({((5, 0), t) for t in range(3, 40)})
    sh: dict = {}
    sp_hard = plan_sipps(corr, cs, cg, hard_vertex=hard, stats=sh)
    th: dict = {}
    _te(corr, cs, cg, hard, {}, {}, th)
    soft = {((5, 0), t): 1 for t in range(3, 40)}
    ss: dict = {}
    sp_soft = plan_sipps(corr, cs, cg, soft_vertex=soft, stats=ss)
    ts: dict = {}
    te_soft = _te(corr, cs, cg, frozenset(), soft, {}, ts)

    return {
        "case": "mapf_sipps",
        "nosoft_mismatches": nosoft_mismatch,
        "battery_instances": n,
        "battery_optimal": optimal,
        "battery_length_optimal": length_optimal,
        "battery_sipps_expansions": sipps_exp,
        "battery_te_expansions": te_exp,
        "battery_worse_expansions": worse_exp,
        "beats_existing_heuristic": beats,
        "showcase_hard_sipps_exp": sh["expansions"],
        "showcase_hard_te_exp": th["expansions"],
        "showcase_hard_collisions": sh["collisions"],
        "showcase_soft_sipps_exp": ss["expansions"],
        "showcase_soft_te_exp": ts["expansions"],
        "showcase_soft_collisions": ss["collisions"],
        "matches_sipp_without_soft": nosoft_mismatch == 0,
        "minimizes_collisions_optimally": (optimal == n and length_optimal == n
                                           and n == 300),
        "safe_interval_cuts_expansions": (sh["expansions"] < th["expansions"]
                                          and ss["expansions"] < ts["expansions"]
                                          and ss["collisions"] == 0
                                          and te_soft == 0),
    }


def _run_k_robust_cbs() -> dict:
    # k-robust CBS (k_robust.py) is a Python reproduction of Atzmon, Stern,
    # Felner, Wagner, Bartak & Zhou's "Robust Multi-Agent Path Finding" (SoCS
    # 2018 / JAIR 2020). A k-robust plan stays collision-free as long as no agent
    # is delayed by more than k timesteps -- it leaves a k-step buffer at every
    # shared cell (no two agents use the same cell within k steps). It is the
    # smallest change to cbs: detect a k-DELAY vertex conflict (same cell within k
    # steps, which also catches a swap that a delay would turn into a vertex
    # collision) and split it with single negative vertex constraints; swaps split
    # on edges as usual. Same low level and cost model, so it returns the optimal
    # (min sum-of-costs) k-robust plan.
    #
    # The gate pins:
    # (1) k=0 IS PLAIN CBS: detection and split degenerate to the standard ones,
    #     so k_robust_cbs(k=0) matches cbs's optimum on every instance.
    # (2) ROBUSTNESS: for k in {1,2} the returned plan has NO k-delay conflict
    #     (krobust_holds) AND empirically survives delaying any single agent by up
    #     to k steps (delay_survives), while the plain CBS plan VIOLATES
    #     k-robustness on many instances (cbs_violates_k -- the buffer is doing
    #     real work).
    # (3) COST OF ROBUSTNESS: cost is monotone non-decreasing in k (k0<=k1<=k2 on
    #     every instance) and strictly higher on some (cost_strict_k1).
    # (4) SHOWCASE seed 1 (3 agents, 5x5): the plain plan COLLIDES when one agent
    #     is delayed a single step; the 1-robust plan costs one more (11 -> 12)
    #     and survives that delay.
    import random

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.cbs import cbs
    from mrn_coord.mapf.conflicts import detect_first_conflict
    from mrn_coord.mapf.k_robust import detect_first_k_conflict, k_robust_cbs

    def inst(seed, n, w, h):
        rng = random.Random(seed)
        free = [(x, y) for x in range(w) for y in range(h)]
        cells = rng.sample(free, 2 * n)
        return GridWorld(w, h), {i: (cells[i], cells[n + i]) for i in range(n)}

    def collides_under_delay(paths, k):
        for a in paths:
            for d in range(1, k + 1):
                dp = dict(paths)
                dp[a] = [paths[a][0]] * d + list(paths[a])
                if detect_first_conflict(dp) is not None:
                    return True
        return False

    # (1) k=0 == cbs
    k0_inst = k0_match = 0
    for seed in range(60):
        for na, w, h in ((2, 5, 5), (3, 5, 5), (3, 4, 4)):
            grid, ag = inst(seed, na, w, h)
            base = cbs(grid, ag, max_expansions=40000)
            if base is None:
                continue
            sol = k_robust_cbs(grid, ag, k=0, max_expansions=40000)
            k0_inst += 1
            if sol is not None and sol.cost == base.cost:
                k0_match += 1

    # (2)+(3) robustness battery: solve k in {0,1,2}, check property/delay/monotone
    n = 0
    krob = {1: 0, 2: 0}
    delay_ok = {1: 0, 2: 0}
    cbs_violates = {1: 0, 2: 0}
    cost_ge = {1: 0, 2: 0}
    monotone = cost_strict_k1 = 0
    for seed in range(60):
        for na, w, h in ((3, 5, 5), (3, 6, 6)):
            grid, ag = inst(seed, na, w, h)
            s0 = k_robust_cbs(grid, ag, k=0, max_expansions=40000)
            s1 = k_robust_cbs(grid, ag, k=1, max_expansions=80000)
            s2 = k_robust_cbs(grid, ag, k=2, max_expansions=120000)
            if None in (s0, s1, s2):
                continue
            n += 1
            if s0.cost <= s1.cost <= s2.cost:
                monotone += 1
            if s0.cost < s1.cost:
                cost_strict_k1 += 1
            for k, sk in ((1, s1), (2, s2)):
                if detect_first_k_conflict(sk.paths, k) is None:
                    krob[k] += 1
                if not collides_under_delay(sk.paths, k):
                    delay_ok[k] += 1
                if sk.cost >= s0.cost:
                    cost_ge[k] += 1
                if detect_first_k_conflict(s0.paths, k) is not None:
                    cbs_violates[k] += 1

    # (4) showcase
    sgrid, sag = inst(1, 3, 5, 5)
    sc0 = cbs(sgrid, sag)
    st0: dict = {}
    st1: dict = {}
    sk0 = k_robust_cbs(sgrid, sag, k=0, stats=st0)
    sk1 = k_robust_cbs(sgrid, sag, k=1, stats=st1)

    return {
        "case": "mapf_k_robust",
        "k0_instances": k0_inst,
        "k0_match_cbs": k0_match,
        "robust_instances": n,
        "krobust_holds_k1": krob[1],
        "krobust_holds_k2": krob[2],
        "delay_survives_k1": delay_ok[1],
        "delay_survives_k2": delay_ok[2],
        "cost_ge_cbs_k1": cost_ge[1],
        "cost_ge_cbs_k2": cost_ge[2],
        "cbs_violates_k1": cbs_violates[1],
        "cbs_violates_k2": cbs_violates[2],
        "cost_monotone": monotone,
        "cost_strict_k1": cost_strict_k1,
        "showcase_cbs_cost": sc0.cost,
        "showcase_k0_cost": sk0.cost,
        "showcase_k1_cost": sk1.cost,
        "showcase_k0_exp": st0["expansions"],
        "showcase_k1_exp": st1["expansions"],
        "showcase_cbs_collides_under_delay": collides_under_delay(sc0.paths, 1),
        "showcase_k1_survives_delay": not collides_under_delay(sk1.paths, 1),
        "k0_is_plain_cbs": k0_match == k0_inst,
        "guarantees_k_robustness": (krob[1] == n and krob[2] == n
                                    and delay_ok[1] == n and delay_ok[2] == n),
        "robustness_costs_monotone": (monotone == n and cost_ge[1] == n
                                      and cost_ge[2] == n),
    }


def _run_cbm_tapf() -> dict:
    # CBM (cbm.py) is a Python reproduction of Hang Ma & Sven Koenig's "Optimal
    # Target Assignment and Path Finding for Teams of Agents" (AAMAS 2016). TAPF
    # partitions agents into TEAMS; targets within a team are interchangeable
    # (anonymous), across teams distinct (labeled). CBM marries the two paradigms:
    # the low level solves each team independently as an anonymous makespan
    # max-flow on the time-expanded grid (reusing the Yu & LaValle reduction, with
    # high-level constraints baked in as removed vertices/gadget entries), and the
    # high level is CBS over INTER-TEAM conflicts -- resolve a shared cell / swap
    # by forbidding it to one team or the other and re-flowing that team. Best
    # first on makespan, the first conflict-free node is makespan-optimal.
    #
    # The gate pins the interpolation between the two extremes we already have:
    # (1) ONE TEAM == ANONYMOUS FLOW: a single team containing everyone makes CBM
    #     degenerate to pure network flow -- its makespan equals
    #     anonymous_makespan on every instance, collision-free.
    # (2) SINGLETON TEAMS == LABELED OPTIMUM: one agent/one target per team is
    #     fully labeled MAPF; CBM's makespan equals a brute-force makespan-optimal
    #     joint BFS on every tiny instance, collision-free, every agent on ITS
    #     target, and never below the anonymous lower bound.
    # (3) INTERPOLATION: the anonymous (one-team) makespan lower-bounds the labeled
    #     (singleton) one on every instance.
    # (4) TEAMS SCALE + RESOLVE: a 5x5 / 3-agent singleton battery stays
    #     collision-free and valid, exercising cross-team conflict resolution.
    # (5) SHOWCASE seed 0 (two 2-agent teams on 5x5): makespan 5 via 3 high-level
    #     nodes, above the anonymous lower bound 4, with a within-team target
    #     interchange (an agent fills its team's *other* target).
    import itertools as _it
    import random
    from collections import deque

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.cbm import cbm
    from mrn_coord.mapf.conflicts import detect_first_conflict
    from mrn_coord.mapf.flow import anonymous_makespan

    def brute_labeled(grid, starts, goals):
        na = len(starts)
        cells = [(x, y) for x in range(grid.width) for y in range(grid.height)
                 if grid.is_free((x, y))]
        nbr = {c: grid.neighbors(c) for c in cells}
        start, goal = tuple(starts), tuple(goals)
        if start == goal:
            return 0
        seen = {start}
        q = deque([(start, 0)])
        while q:
            cfg, t = q.popleft()
            if t > 30:
                continue
            for nxt in _it.product(*[nbr[c] for c in cfg]):
                if len(set(nxt)) != na:
                    continue
                bad = False
                for i in range(na):
                    for j in range(i + 1, na):
                        if nxt[i] == cfg[j] and nxt[j] == cfg[i]:
                            bad = True
                            break
                    if bad:
                        break
                if bad:
                    continue
                if nxt == goal:
                    return t + 1
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, t + 1))
        return None

    # (1) one team == anonymous flow
    one_inst = one_match = one_cf = 0
    for seed in range(60):
        rng = random.Random(seed)
        free = [(x, y) for x in range(5) for y in range(5)]
        cells = rng.sample(free, 8)
        grid = GridWorld(5, 5)
        starts, goals = cells[:4], cells[4:]
        fmk = anonymous_makespan(grid, starts, goals)
        res = cbm(grid, [(starts, goals)])
        one_inst += 1
        if fmk is not None and res is not None and res[1] == fmk[1]:
            one_match += 1
        if res is not None and detect_first_conflict(res[0]) is None:
            one_cf += 1

    # (2)+(3) singleton == brute labeled optimum, anon lower bound, interpolation
    lab_inst = lab_opt = lab_cf = lab_valid = lab_geq = interp = 0
    for seed in range(120):
        rng = random.Random(seed)
        free = [(x, y) for x in range(4) for y in range(4)]
        cells = rng.sample(free, 4)
        grid = GridWorld(4, 4)
        starts, goals = cells[:2], cells[2:]
        bms = brute_labeled(grid, starts, goals)
        res = cbm(grid, [([starts[i]], [goals[i]]) for i in range(2)])
        if res is None or bms is None:
            continue
        lab_inst += 1
        paths, ms = res
        if ms == bms:
            lab_opt += 1
        if detect_first_conflict(paths) is None:
            lab_cf += 1
        if all(paths[(i, 0)][-1] == goals[i] for i in range(2)):
            lab_valid += 1
        fmk = anonymous_makespan(grid, starts, goals)
        if fmk is not None and ms >= fmk[1]:
            lab_geq += 1
        one = cbm(grid, [(starts, goals)])
        if one is not None and one[1] <= ms:
            interp += 1

    # (4) teams scale + resolve (5x5, 3 singleton teams)
    sc_inst = sc_cf = sc_valid = sc_resolve = 0
    for seed in range(80):
        rng = random.Random(seed)
        free = [(x, y) for x in range(5) for y in range(5)]
        cells = rng.sample(free, 6)
        grid = GridWorld(5, 5)
        starts, goals = cells[:3], cells[3:]
        st: dict = {}
        res = cbm(grid, [([starts[i]], [goals[i]]) for i in range(3)], stats=st)
        if res is None:
            continue
        sc_inst += 1
        paths, ms = res
        if detect_first_conflict(paths) is None:
            sc_cf += 1
        if all(paths[(i, 0)][-1] == goals[i] for i in range(3)):
            sc_valid += 1
        if st["expansions"] > 1:
            sc_resolve += 1

    # (5) two-team showcase
    rng = random.Random(0)
    cells = rng.sample([(x, y) for x in range(5) for y in range(5)], 8)
    sgrid = GridWorld(5, 5)
    tA = (cells[0:2], cells[2:4])
    tB = (cells[4:6], cells[6:8])
    sst: dict = {}
    sres = cbm(sgrid, [tA, tB], stats=sst)
    spaths, sms = sres
    s_cf = detect_first_conflict(spaths) is None
    s_anon = anonymous_makespan(
        sgrid, cells[0:2] + cells[4:6], cells[2:4] + cells[6:8])
    interchange = False
    valid = True
    for ti, (s, g) in enumerate((tA, tB)):
        ends = {spaths[(ti, ai)][-1] for ai in range(len(s))}
        if ends != set(g):
            valid = False
        for ai in range(len(s)):
            if spaths[(ti, ai)][-1] != g[ai]:
                interchange = True

    return {
        "case": "mapf_cbm",
        "one_team_instances": one_inst,
        "one_team_matches_flow": one_match,
        "one_team_collision_free": one_cf,
        "labeled_instances": lab_inst,
        "labeled_optimal": lab_opt,
        "labeled_collision_free": lab_cf,
        "labeled_valid": lab_valid,
        "labeled_ge_anon_lb": lab_geq,
        "interpolation_anon_le_labeled": interp,
        "scale_instances": sc_inst,
        "scale_collision_free": sc_cf,
        "scale_valid": sc_valid,
        "scale_resolves": sc_resolve,
        "showcase_makespan": sms,
        "showcase_expansions": sst["expansions"],
        "showcase_anon_lb": s_anon[1],
        "showcase_collision_free": s_cf,
        "showcase_valid": valid,
        "showcase_within_team_interchange": interchange,
        "one_team_is_anonymous_flow": (one_match == one_inst
                                       and one_cf == one_inst),
        "singleton_is_labeled_optimum": (lab_opt == lab_inst
                                         and lab_cf == lab_inst
                                         and lab_valid == lab_inst
                                         and lab_geq == lab_inst),
        "teams_resolve_collision_free": (sc_cf == sc_inst
                                         and sc_valid == sc_inst
                                         and sc_resolve > 0),
    }


def _run_cbs_ta() -> dict:
    # CBS-TA (cbs_ta.py) is a Python reproduction of Hönig, Kiesel, Tinka, Durham
    # & Ayanian's "Conflict-Based Search with Optimal Task Assignment" (ICAPS
    # 2018). Plain CBS is handed one goal per agent; CBS-TA leaves the assignment
    # open -- each agent may serve any goal from a pool -- and finds the JOINTLY
    # optimal assignment+paths. It keeps CBS's two levels but replaces the single
    # root with a FOREST of roots, one per target assignment, unfolded lazily in
    # increasing assignment-cost order by Murty's K-best algorithm over the
    # agent x target distance matrix (min-cost matching by hungarian). Only when a
    # root is expanded is the next-cheapest assignment materialized; the whole
    # forest is searched best-first by sum-of-costs, so the first conflict-free
    # node popped is optimal over BOTH assignment and paths.
    #
    # The gate pins the mechanism and the two interpolation extremes:
    # (1) MURTY EXACT: the K-best generator yields assignment costs in exactly the
    #     brute-force sorted order (0 mismatches over 300 random matrices).
    # (2) DEGENERATE == CBS: give each agent a single distinct goal and the forest
    #     has one root -- cbs_ta's sum-of-costs equals cbs on every instance,
    #     collision-free.
    # (3) JOINTLY OPTIMAL: with an anonymous pool (3 agents may serve any of 3
    #     targets) cbs_ta's cost equals a brute force that runs cbs on EVERY
    #     assignment and takes the min -- on every instance, collision-free.
    # (4) ASSIGNMENT MATTERS: on the same battery, cbs_ta strictly beats fixing the
    #     distance-cheapest (Hungarian 1-best) assignment and routing it with cbs on
    #     7 of 120 -- the cheapest matching is NOT always jointly optimal once
    #     collisions are priced in. SHOWCASE seed 8: the cheapest assignment costs 8
    #     (a forced conflict), cbs_ta swaps two agents' targets for cost 7, via 2
    #     expansions across 3 materialized roots.
    import itertools as _it
    import random

    from mrn_coord.lifelong.allocation import hungarian
    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.cbs import cbs
    from mrn_coord.mapf.cbs_ta import _bfs_dist, _murty, cbs_ta
    from mrn_coord.mapf.conflicts import detect_first_conflict

    INF = float("inf")

    # (1) Murty K-best == brute sorted assignment costs.
    def brute_kbest(cost):
        R, C = len(cost), len(cost[0])
        res = []
        for perm in _it.permutations(range(C), R):
            t = sum(cost[i][perm[i]] for i in range(R))
            if t < INF:
                res.append(t)
        res.sort()
        return res

    murty_inst = murty_mismatch = 0
    for seed in range(300):
        rng = random.Random(seed)
        R = rng.randint(2, 4)
        C = rng.randint(R, R + 2)
        cost = [[float(rng.randint(1, 9)) if rng.random() > 0.15 else INF
                 for _ in range(C)] for _ in range(R)]
        bt = brute_kbest(cost)
        if not bt:
            continue
        murty_inst += 1
        got = [t for (a, t), _ in zip(_murty(cost), range(min(len(bt), 8)))]
        if got != bt[:len(got)]:
            murty_mismatch += 1

    # (2) degenerate (single goal each) == cbs.
    deg_inst = deg_match = deg_cf = 0
    for seed in range(120):
        rng = random.Random(seed)
        cells = rng.sample([(x, y) for x in range(5) for y in range(5)], 6)
        grid = GridWorld(5, 5)
        agents = {i: (cells[i], cells[i + 3]) for i in range(3)}
        c = cbs(grid, agents)
        ct = cbs_ta(grid, {i: (cells[i], [cells[i + 3]]) for i in range(3)})
        if (c is None) != (ct is None):
            continue
        deg_inst += 1
        if c is not None and c.cost == ct.cost:
            deg_match += 1
        if ct is not None and detect_first_conflict(ct.paths) is None:
            deg_cf += 1

    # (3)+(4) anonymous pool: jointly optimal vs brute, vs fixed cheapest.
    def brute_best(grid, starts, targets, na):
        best = None
        for combo in _it.permutations(targets, na):
            sol = cbs(grid, {i: (starts[i], combo[i]) for i in range(na)})
            if sol is not None and (best is None or sol.cost < best):
                best = sol.cost
        return best

    def cheapest_fixed(grid, starts, targets, na):
        dist = [_bfs_dist(grid, starts[i]) for i in range(na)]
        cost = [[float(dist[i].get(t, INF)) for t in targets] for i in range(na)]
        a = hungarian(cost)
        if len(a) < na:
            return None
        return {i: (starts[i], targets[a[i]]) for i in range(na)}

    na = 3
    anon_inst = anon_opt = anon_cf = beats = 0
    showcase = {}
    for seed in range(120):
        rng = random.Random(seed)
        cells = rng.sample([(x, y) for x in range(5) for y in range(5)], 6)
        grid = GridWorld(5, 5)
        starts, targets = cells[:na], cells[na:na + na]
        bm = brute_best(grid, starts, targets, na)
        st: dict = {}
        sol = cbs_ta(grid, {i: (starts[i], targets) for i in range(na)}, stats=st)
        if bm is None or sol is None:
            continue
        anon_inst += 1
        if sol.cost == bm:
            anon_opt += 1
        if detect_first_conflict(sol.paths) is None:
            anon_cf += 1
        fixed = cheapest_fixed(grid, starts, targets, na)
        fixed_sol = cbs(grid, fixed) if fixed is not None else None
        if fixed_sol is not None and sol.cost < fixed_sol.cost:
            beats += 1
        if seed == 8:
            showcase = {
                "ta_cost": sol.cost,
                "fixed_cost": fixed_sol.cost if fixed_sol else -1,
                "roots": st["roots"],
                "expansions": st["expansions"],
            }

    return {
        "case": "mapf_cbs_ta",
        "murty_instances": murty_inst,
        "murty_mismatches": murty_mismatch,
        "degenerate_instances": deg_inst,
        "degenerate_match_cbs": deg_match,
        "degenerate_collision_free": deg_cf,
        "anon_instances": anon_inst,
        "anon_opt_match_brute": anon_opt,
        "anon_collision_free": anon_cf,
        "ta_beats_fixed_cheapest": beats,
        "showcase_ta_cost": showcase.get("ta_cost"),
        "showcase_fixed_cost": showcase.get("fixed_cost"),
        "showcase_roots": showcase.get("roots"),
        "showcase_expansions": showcase.get("expansions"),
        "murty_kbest_exact": murty_mismatch == 0 and murty_inst > 0,
        "degenerate_is_cbs": (deg_match == deg_inst and deg_cf == deg_inst
                              and deg_inst > 0),
        "jointly_optimal": (anon_opt == anon_inst and anon_cf == anon_inst
                            and anon_inst > 0),
        "assignment_matters": beats > 0 and showcase.get("ta_cost", 0) <
        showcase.get("fixed_cost", 0),
    }


def _run_pibt_swap() -> dict:
    # pibt_swap.py is a Python reproduction of the SWAP operation that improves
    # PIBT successor generation in LaCAM2 (Okumura, "Improving LaCAM for Scalable
    # Eventually Optimal Multi-Agent Pathfinding", IJCAI 2023). Plain PIBT builds
    # one collision-free step by priority inheritance (a high-priority agent
    # pushes whoever blocks it), but two agents that must EXCHANGE ends of a narrow
    # corridor livelock -- the pushed agent has nowhere to go but back. The swap
    # operation detects when a swap with the blocking agent is both REQUIRED (it
    # cannot get out of the way) and POSSIBLE (a degree>=2 pocket exists to step
    # aside), and resolves it: the agent reverses its candidate order (moves AWAY
    # from its goal, vacating the corridor) and PULLS the partner into the cell it
    # left. Faithful port of Okumura's reference C++ (Kei18/lacam2 planner.cpp:
    # funcPIBT / swap_possible_and_required / is_swap_required / is_swap_possible),
    # with the random tie-break replaced by a deterministic coordinate one.
    #
    # The repo already solves these livelocks differently -- pibt_solve uses a
    # deterministic ESCAPE SALT that perturbs ties until symmetry breaks; the swap
    # is the canonical alternative, and this gate pins what it buys:
    # (1) CORRIDOR SHOWCASE: a 1-wide 5-cell corridor with a single pocket and two
    #     agents that must exchange. Base PIBT (swap=False) livelocks to the budget;
    #     swap PIBT solves it collision-free in makespan 6, stepping one agent into
    #     the pocket (2,1) to let the other pass.
    # (2) BATTERY: 200 random 5x5 / 2-5 agent instances. swap PIBT is collision-free
    #     on every instance it solves (198/200; PIBT is incomplete, so 2 time out).
    # (3) SWAP DOES REAL WORK: base PIBT livelocks on 10 of the 200; the swap
    #     rescues 8 of those 10 (solves where base fails), isolating the mechanism.
    import random

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.conflicts import detect_first_conflict
    from mrn_coord.mapf.pibt_swap import pibt_swap

    # (1) corridor with a single pocket at (2,1)
    free = {(x, 0) for x in range(5)} | {(2, 1)}
    blocked = {(x, y) for x in range(5) for y in range(2) if (x, y) not in free}
    cgrid = GridWorld(5, 2, frozenset(blocked))
    cagents = {0: ((0, 0), (4, 0)), 1: ((4, 0), (0, 0))}
    base = pibt_swap(cgrid, cagents, swap=False, max_timestep=200)
    sw = pibt_swap(cgrid, cagents, swap=True, max_timestep=200)
    c_base_solved = base is not None
    c_swap_solved = sw is not None
    c_cf = c_swap_solved and detect_first_conflict(sw) is None
    c_makespan = (len(sw[0]) - 1) if c_swap_solved else -1
    c_pocket = c_swap_solved and (2, 1) in sw[0]

    # (2)+(3) battery
    total = swap_solved = swap_cf = base_livelock = rescues = 0
    for seed in range(200):
        rng = random.Random(seed)
        n = rng.randint(2, 5)
        cells = rng.sample([(x, y) for x in range(5) for y in range(5)], 2 * n)
        g = GridWorld(5, 5)
        ag = {i: (cells[i], cells[n + i]) for i in range(n)}
        total += 1
        sw = pibt_swap(g, ag, swap=True, max_timestep=500)
        if sw is not None:
            swap_solved += 1
            if detect_first_conflict(sw) is None:
                swap_cf += 1
        bs = pibt_swap(g, ag, swap=False, max_timestep=500)
        if bs is None:
            base_livelock += 1
            if sw is not None:
                rescues += 1

    return {
        "case": "mapf_pibt_swap",
        "corridor_base_solved": c_base_solved,
        "corridor_swap_solved": c_swap_solved,
        "corridor_swap_collision_free": c_cf,
        "corridor_makespan": c_makespan,
        "corridor_uses_pocket": c_pocket,
        "battery_instances": total,
        "battery_swap_solved": swap_solved,
        "battery_swap_collision_free": swap_cf,
        "battery_base_livelock": base_livelock,
        "battery_swap_rescues_base": rescues,
        "swap_resolves_corridor": (not c_base_solved) and c_swap_solved and c_cf
        and c_pocket,
        "swap_always_collision_free": swap_cf == swap_solved and swap_solved > 0,
        "swap_rescues_base_livelocks": rescues > 0,
    }


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


def _run_push_and_swap() -> dict:
    # push_and_swap.py reproduces Luna & Bekris's "Push and Swap" (IJCAI 2011) --
    # the swap-only ANCESTOR of push_and_rotate, kept as its own solver so the
    # exact completeness gap that motivated Push-and-Rotate is visible and gated.
    # It uses two primitives only -- push (advance toward goal, shoving blockers)
    # and swap (exchange two agents around a degree->=3 hub, restoring everyone
    # else) -- with NO rotate primitive and no packed-grid reduction. Internally
    # it reuses push_and_rotate's _Solver (the SAME push/swap machinery, byte for
    # byte) but runs the order-sweep with allow_rotate=False, allow_residual=False
    # and never dispatches the reduction; the contrast is therefore clean -- the
    # rotate completion is the only thing removed.
    #
    # Luna & Bekris claimed completeness for instances with >= 2 empty vertices,
    # but de Wilde et al. (JAIR 2014) showed the bare push/swap core stalls on
    # CYCLIC, slack-free regions (a packed rectangle / a full ring), where the
    # only way past a blocking agent is to rotate a whole cycle by one. This gate
    # pins exactly that:
    # (1) SLACK == push_and_rotate. On the sparse battery push_and_rotate's gate
    #     uses, push_and_swap solves every instance push_and_rotate does (agree ==
    #     instances) and every plan it returns is collision-free and on-goal --
    #     the swap-only core is already complete where there is room.
    # (2) THE GAP. On the packed formations push_and_rotate's reduction solves,
    #     push_and_swap (no rotate) solves a strictly smaller fraction; the gap is
    #     largest at one blank (the tightest 15-puzzle regime, gap == every
    #     instance) and shrinks as blanks add slack (16 -> 15 -> 4) -- a monotone
    #     curve that pins the rotate primitive's exact contribution. Crucially,
    #     wherever push_and_swap DOES solve a packed instance it is still valid by
    #     construction (packed_ps_valid == packed_ps_solved).
    # (3) SWAP FIRES (positive isolation). On a T-junction (a degree-3 hub) two
    #     agents that must exchange ends cannot be separated by push alone; the
    #     swap primitive rotates them around the hub and push_and_swap solves it
    #     collision-free.
    import random

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.conflicts import detect_first_conflict
    from mrn_coord.mapf.push_and_rotate import push_and_rotate
    from mrn_coord.mapf.push_and_swap import push_and_swap

    def _instance(w, h, n, seed):
        rng = random.Random(seed)
        free = [(x, y) for x in range(w) for y in range(h)]
        rng.shuffle(free)
        return GridWorld(w, h), {i: (free[i], free[n + i]) for i in range(n)}

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

    def _valid(sol, agents):
        return (detect_first_conflict(sol.paths) is None
                and all(sol.paths[k][-1] == g for k, (s, g) in agents.items()))

    # (1) SLACK -- equivalence with push_and_rotate on its own sparse battery.
    slack_instances = slack_ps = slack_pr = slack_agree = slack_ps_valid = 0
    for w, h, n in ((4, 4, 4), (5, 5, 5), (6, 6, 6)):
        for seed in range(10):
            grid, agents = _instance(w, h, n, seed)
            sps = push_and_swap(grid, agents)
            spr = push_and_rotate(grid, agents)
            slack_instances += 1
            if sps is not None:
                slack_ps += 1
                slack_ps_valid += int(_valid(sps, agents))
            if spr is not None:
                slack_pr += 1
            if (sps is not None) == (spr is not None):
                slack_agree += 1

    # (2) THE GAP -- packed formations, per blank count.
    gap = {}
    packed_ps_total = packed_pr_total = packed_ps_valid = 0
    for blanks in (1, 2, 3):
        pi = ps = pr = g = 0
        for w, h in ((4, 4), (5, 5)):
            for seed in range(8):
                grid, agents = _packed(w, h, blanks, seed)
                sps = push_and_swap(grid, agents)
                spr = push_and_rotate(grid, agents)
                pi += 1
                if sps is not None:
                    ps += 1
                    packed_ps_valid += int(_valid(sps, agents))
                if spr is not None:
                    pr += 1
                if spr is not None and sps is None:
                    g += 1
        gap[blanks] = (pi, ps, pr, g)
        packed_ps_total += ps
        packed_pr_total += pr

    g1, g2, g3 = gap[1], gap[2], gap[3]
    # monotone: the gap shrinks as blanks (slack) grow.
    gap_monotone = g1[3] >= g2[3] >= g3[3]
    # single blank: push_and_swap solves NONE that push_and_rotate solves.
    single_blank_total_gap = g1[3] == g1[2] and g1[1] == 0

    # (3) SWAP FIRES -- a T-junction exchange the push primitive alone cannot do.
    tgrid = GridWorld(3, 2, frozenset({(0, 0), (2, 0)}))
    tag = {0: ((0, 1), (2, 1)), 1: ((2, 1), (0, 1))}
    tstats: dict = {}
    tsol = push_and_swap(tgrid, tag, stats=tstats)
    swap_solves = tsol is not None
    swap_valid = bool(tsol is not None and _valid(tsol, tag))
    swap_moves = tstats.get("moves", -1) if tsol is not None else -1

    # showcase single-blank packed: push_and_swap fails, push_and_rotate solves.
    sgrid, sag = _packed(4, 4, 1, 0)
    show_ps = push_and_swap(sgrid, sag)
    sstats: dict = {}
    show_pr = push_and_rotate(sgrid, sag, stats=sstats)

    return {"case": "mapf_push_and_swap",
            "slack_instances": slack_instances, "slack_ps_solved": slack_ps,
            "slack_pr_solved": slack_pr, "slack_agree": slack_agree,
            "slack_ps_valid": slack_ps_valid,
            "slack_matches_rotate": slack_agree == slack_instances
                                    and slack_ps == slack_pr,
            "slack_all_valid": slack_ps_valid == slack_ps,
            "packed_b1_ps": g1[1], "packed_b1_pr": g1[2], "packed_b1_gap": g1[3],
            "packed_b2_ps": g2[1], "packed_b2_pr": g2[2], "packed_b2_gap": g2[3],
            "packed_b3_ps": g3[1], "packed_b3_pr": g3[2], "packed_b3_gap": g3[3],
            "packed_ps_solved": packed_ps_total,
            "packed_pr_solved": packed_pr_total,
            "packed_ps_valid": packed_ps_valid,
            "packed_ps_valid_when_solved": packed_ps_valid == packed_ps_total,
            "gap_exists": packed_pr_total > packed_ps_total,
            "gap_monotone_in_slack": gap_monotone,
            "single_blank_total_gap": single_blank_total_gap,
            "swap_solves": swap_solves, "swap_valid": swap_valid,
            "swap_moves": swap_moves,
            "showcase_ps_fails": show_ps is None,
            "showcase_pr_solves": show_pr is not None,
            "showcase_pr_moves": sstats.get("moves", -1)}


def _run_bibox() -> dict:
    # bibox.py reproduces Surynek's "Bibox" (ICRA 2009 / AAMAS 2014) -- a
    # CONSTRUCTIVE, polynomial-time COMPLETE solver for biconnected graphs with
    # >= 2 blanks, and a paradigm distinct from every search/primitive solver in
    # the repo: it works off an OPEN EAR DECOMPOSITION. The graph is split into a
    # basic cycle L0 plus derived ears (chains attached at both endpoints to the
    # part built so far); ears are solved in REVERSE order and locked, each filled
    # by ROTATING the cycle it forms with a return path so staged agents are
    # conveyed into its interior. The basic cycle is then closed on the theta
    # region L0 union int(L1). A BorrowBlanks goal transform first parks two blanks
    # in L0 (undone by ReturnBlanks). Every move steps one agent into an adjacent
    # empty cell, so plans are collision-free and on-goal BY CONSTRUCTION.
    #
    # What this gate pins:
    # (1) STRUCTURE. The ear decomposition is a real open decomposition: the basic
    #     cycle is a cycle, the ears cover every vertex, endpoints lie in the
    #     prefix, interiors are new, and no ear is closed (open).
    # (2) COMPLETE + SOUND (the defining property). Against a brute-force
    #     solvability oracle on small biconnected maps, Bibox solves EVERY solvable
    #     instance (complete_incomplete == 0) and NEVER "solves" an unsolvable one
    #     (complete_unsound == 0).
    # (3) VALID BY CONSTRUCTION. On a broad random battery every returned plan is
    #     collision-free (battery_cf_violations == 0) and on-goal (battery_goalfail
    #     == 0).
    # (4) COMPLETENESS CONTRAST. On packed biconnected formations (>= 2 blanks)
    #     where optimal CBS busts its expansion budget, Bibox still solves them all,
    #     valid by construction -- bibox_solved > cbs_solved.
    # (5) ROTATION EXERCISED. A multi-ear instance is solved through the ear-
    #     rotation machinery (>= 2 ears), reported with its move count.
    # (6) HONEST SCOPE. Outside its class Bibox returns None: a graph with a cut
    #     vertex (not biconnected), and an instance with fewer than two blanks.
    import random
    from collections import deque

    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.bibox import bibox, ear_decomposition
    from mrn_coord.mapf.cbs import cbs
    from mrn_coord.mapf.conflicts import detect_first_conflict

    def _adj(grid):
        free = [(x, y) for x in range(grid.width) for y in range(grid.height)
                if grid.is_free((x, y))]
        fset = set(free)
        return {c: [n for n in ((c[0] + 1, c[1]), (c[0] - 1, c[1]),
                                (c[0], c[1] + 1), (c[0], c[1] - 1)) if n in fset]
                for c in free}, free

    def _valid(sol, agents):
        return (detect_first_conflict(sol.paths) is None
                and all(sol.paths[a][-1] == agents[a][1] for a in agents))

    def _brute_solvable(grid, agents, cap=300000):
        adj, _ = _adj(grid)
        ids = sorted(agents)
        start = tuple(agents[a][0] for a in ids)
        goal = tuple(agents[a][1] for a in ids)
        if start == goal:
            return True
        seen = {start}
        q = deque([start])
        while q and len(seen) < cap:
            cfg = q.popleft()
            occ = set(cfg)
            for i, a in enumerate(ids):
                for nb in adj[cfg[i]]:
                    if nb not in occ:
                        nc = cfg[:i] + (nb,) + cfg[i + 1:]
                        if nc not in seen:
                            seen.add(nc)
                            if nc == goal:
                                return True
                            q.append(nc)
        return None                                  # unknown (hit cap)

    def _rand(grid, free, n, rng):
        s = rng.sample(free, n)
        g = rng.sample(free, n)
        return {i: (s[i], g[i]) for i in range(n)}

    # (1) STRUCTURE -- 3x3 and 4x4 open ear decompositions.
    def _struct(grid):
        adj, _ = _adj(grid)
        bc, ears = ear_decomposition(grid)
        adjset = {c: set(ns) for c, ns in adj.items()}
        is_cycle = (len(bc) >= 3
                    and all(bc[(i + 1) % len(bc)] in adjset[bc[i]]
                            for i in range(len(bc))))
        built = set(bc)
        prefix_ok = open_ok = True
        cover = set(bc)
        for e in ears:
            if e[0] not in built or e[-1] not in built:
                prefix_ok = False
            if e[0] == e[-1]:
                open_ok = False
            for iv in e[1:-1]:
                if iv in built:
                    prefix_ok = False
            built |= set(e)
            cover |= set(e)
        return bc, ears, is_cycle, prefix_ok, open_ok, cover == set(adj)

    bc3, ears3, cyc3, pre3, open3, cov3 = _struct(GridWorld(3, 3))
    bc4, ears4, cyc4, pre4, open4, cov4 = _struct(GridWorld(4, 4))

    # (2) COMPLETE + SOUND -- brute oracle on small biconnected maps.
    rng = random.Random(2026)
    complete_checks = complete_incomplete = complete_unsound = 0
    for w, h, blk in ((3, 3, ()), (2, 4, ()), (2, 3, ()), (3, 3, ((1, 1),))):
        grid = GridWorld(w, h, frozenset(blk))
        _, free = _adj(grid)
        V = len(free)
        for n in range(1, min(4, V - 1) + 1):
            for _ in range(25):
                agents = _rand(grid, free, n, rng)
                sol = bibox(grid, agents)
                bt = _brute_solvable(grid, agents)
                if bt is None:
                    continue
                complete_checks += 1
                if bt and sol is None:
                    complete_incomplete += 1
                if (not bt) and sol is not None:
                    complete_unsound += 1

    # (3) VALID BY CONSTRUCTION -- broad random battery.
    battery_instances = battery_solved = battery_cf = battery_goalfail = 0
    for w, h in ((3, 3), (2, 4), (4, 3), (4, 4), (3, 4)):
        grid = GridWorld(w, h)
        _, free = _adj(grid)
        V = len(free)
        for n in range(1, V - 1):
            for _ in range(12):
                agents = _rand(grid, free, n, rng)
                sol = bibox(grid, agents)
                battery_instances += 1
                if sol is not None:
                    battery_solved += 1
                    if detect_first_conflict(sol.paths) is not None:
                        battery_cf += 1
                    if any(sol.paths[a][-1] != agents[a][1] for a in agents):
                        battery_goalfail += 1

    # (4) COMPLETENESS CONTRAST -- packed formations vs CBS's expansion budget.
    def _packed(w, h, blanks, seed):
        rng2 = random.Random(seed * 131 + w * 7 + h * 3 + blanks)
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
            e = rng2.choice(sorted(empt))
            cand = [c for c in nb(e) if c in occ]
            if not cand:
                continue
            c = rng2.choice(cand)
            a = occ.pop(c)
            occ[e] = a
            pos[a] = e
            empt.discard(e)
            empt.add(c)
        return grid, {i: (pos[i], goal[i]) for i in range(n)}

    packed_instances = packed_bibox = packed_cbs = packed_bibox_valid = 0
    for blanks in (2, 3):
        for w, h in ((4, 4), (3, 4)):
            for seed in range(8):
                grid, agents = _packed(w, h, blanks, seed)
                packed_instances += 1
                sol = bibox(grid, agents)
                if sol is not None:
                    packed_bibox += 1
                    packed_bibox_valid += int(_valid(sol, agents))
                base = cbs(grid, agents, max_expansions=2000)
                if base is not None:
                    packed_cbs += 1

    # (5) ROTATION EXERCISED -- a multi-ear instance solved by the ear machinery.
    rgrid = GridWorld(4, 3)
    _, rfree = _adj(rgrid)
    rrng = random.Random(7)
    rot_stats: dict = {}
    rot_sol = None
    while rot_sol is None:
        rag = _rand(rgrid, rfree, 6, rrng)
        rot_stats = {}
        rot_sol = bibox(rgrid, rag, stats=rot_stats)

    # (6) HONEST SCOPE -- out of class returns None.
    cut = GridWorld(1, 4)                             # a path: articulation points
    scope_noncbc = bibox(cut, {0: ((0, 0), (0, 3)), 1: ((0, 3), (0, 0))}) is None
    tiny = GridWorld(2, 2)                            # 4 cells, 3 agents -> 1 blank
    scope_fewblank = bibox(tiny, {0: ((0, 0), (1, 1)), 1: ((1, 0), (0, 1)),
                                  2: ((0, 1), (1, 0))}) is None

    # showcase: a 2x3 theta swap solved by Bibox (suboptimal vs CBS optimum).
    sgrid = GridWorld(2, 3)
    sag = {0: ((0, 0), (0, 2)), 1: ((0, 2), (0, 0))}
    sstats: dict = {}
    ssol = bibox(sgrid, sag, stats=sstats)
    sopt = cbs(sgrid, sag, max_expansions=20000)

    return {"case": "mapf_bibox",
            "struct_3x3_basic_len": len(bc3), "struct_3x3_ears": len(ears3),
            "struct_3x3_is_cycle": cyc3, "struct_3x3_prefix_ok": pre3,
            "struct_3x3_open": open3, "struct_3x3_covers": cov3,
            "struct_4x4_ears": len(ears4), "struct_4x4_covers": cov4,
            "struct_4x4_prefix_ok": pre4, "struct_4x4_open": open4,
            "complete_checks": complete_checks,
            "complete_incomplete": complete_incomplete,
            "complete_unsound": complete_unsound,
            "is_complete": complete_incomplete == 0,
            "is_sound": complete_unsound == 0,
            "battery_instances": battery_instances,
            "battery_solved": battery_solved,
            "battery_cf_violations": battery_cf,
            "battery_goalfail": battery_goalfail,
            "battery_all_valid": battery_cf == 0 and battery_goalfail == 0,
            "packed_instances": packed_instances,
            "packed_bibox_solved": packed_bibox,
            "packed_cbs_solved": packed_cbs,
            "packed_bibox_valid": packed_bibox_valid,
            "packed_bibox_valid_when_solved": packed_bibox_valid == packed_bibox,
            "packed_bibox_beats_cbs": packed_bibox > packed_cbs,
            "rotation_ears": rot_stats.get("ears", -1),
            "rotation_basic_len": rot_stats.get("basic_cycle_len", -1),
            "rotation_moves": rot_stats.get("moves", -1),
            "rotation_multi_ear": rot_stats.get("ears", 0) >= 2,
            "rotation_cf": detect_first_conflict(rot_sol.paths) is None,
            "scope_noncbc_none": scope_noncbc,
            "scope_fewblank_none": scope_fewblank,
            "showcase_swap_moves": sstats.get("moves", -1),
            "showcase_swap_solves": ssol is not None,
            "showcase_swap_cf": bool(ssol is not None and _valid(ssol, sag)),
            "showcase_cbs_opt": sopt.cost if sopt is not None else -1,
            "showcase_bibox_suboptimal": bool(
                ssol is not None and sopt is not None
                and ssol.cost >= sopt.cost)}


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


def _run_bcp_rectangle() -> dict:
    # Rectangle cuts for BCP (bcp.py's rectangle=True), the specialized cut family
    # Lam, Le Bodic, Harabor & Stuckey layer on the branch-and-cut-and-price frame
    # ("Branch-and-Cut-and-Price for MAPF", IJCAI 2019), reusing the rectangle
    # symmetry geometry of Li et al. AAAI'19 (rectangle.py). When two agents cross
    # an open rectangle in the SAME direction, every pair of their Manhattan
    # paths collides inside, and plain branch-and-price must enumerate the
    # symmetric crossings by branching -- the same blowup CBS suffers. A single
    # rectangle cut sum_{B1} y_{a1} + sum_{B2} y_{a2} <= 1 (B1, B2 the two exit
    # barriers, each an anti-diagonal an optimal crossing path hits exactly once)
    # forbids both agents crossing at once, collapsing the whole symmetry. The cut
    # is separated lazily from the MDD barriers, and its dual is priced exactly via
    # a per-barrier "already-crossed" bit so the LP bound stays valid.
    #
    # Same explicit same-direction anti-diagonal crossings as the CBS
    # rectangle_symmetry gate (random instances almost never contain one). Pins:
    # (1) OPTIMALITY -- rectangle ON matches both plain BCP (OFF) and cbs on every
    # scenario (opt_match), proving the cut drops no solution; (2) the COLLAPSE --
    # the branch-and-bound nodes fall from an aggregate 78 (OFF) to 4 (ON, one root
    # node per scenario), each closed by a single rectangle cut (rcuts_on == 4),
    # and the lazily separated vertex/edge cuts drop 354 -> 29; (3) collision-free
    # throughout. If a rectangle cut ever dropped the optimum opt_match falls; if
    # detection/pricing regressed nodes_on climbs back toward nodes_off.
    from mrn_coord.mapf import GridWorld
    from mrn_coord.mapf.bcp import bcp
    from mrn_coord.mapf.cbs import cbs
    from mrn_coord.mapf.conflicts import detect_first_conflict

    scenarios = [
        ("cross6", 6, 6, {0: ((2, 0), (4, 5)), 1: ((1, 1), (4, 2))}),
        ("cross7c", 7, 7, {0: ((1, 1), (5, 6)), 1: ((0, 2), (6, 3))}),
        ("cross7b", 7, 7, {0: ((2, 0), (6, 6)), 1: ((0, 2), (6, 4))}),
        ("cross7a", 7, 7, {0: ((2, 0), (4, 6)), 1: ((0, 2), (5, 6))}),
    ]
    scn = opt_match = rcuts_on = 0
    nodes_off = nodes_on = cuts_off = cuts_on = 0
    all_cf = True
    for _, w, h, ag in scenarios:
        grid = GridWorld(w, h)
        base = cbs(grid, ag, max_expansions=20000)
        soff: dict = {}
        off = bcp(grid, ag, rectangle=False, stats=soff, max_nodes=20000)
        son: dict = {}
        on = bcp(grid, ag, rectangle=True, stats=son, max_nodes=20000)
        if base is None or off is None or on is None:
            continue
        scn += 1
        opt_match += int(on.cost == off.cost == base.cost)
        nodes_off += soff["nodes"]
        nodes_on += son["nodes"]
        cuts_off += soff["cuts"]
        cuts_on += son["cuts"]
        rcuts_on += son["rcuts"]
        all_cf = all_cf and detect_first_conflict(on.paths) is None

    return {
        "case": "bcp_rectangle",
        "scenarios": scn,
        "opt_match": opt_match,
        "nodes_off": nodes_off,
        "nodes_on": nodes_on,
        "cuts_off": cuts_off,
        "cuts_on": cuts_on,
        "rcuts_on": rcuts_on,
        "all_collision_free": all_cf,
        "optimal_matches_cbs": opt_match == scn and scn > 0,
        "rectangle_cuts_collapse_branching": nodes_on < nodes_off and rcuts_on > 0,
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


def _run_rhc_reorder() -> dict:
    # The receding-horizon RE-ORDERING half of Berndt, van Duijkeren, Palmieri,
    # Kleiner & Keviczky, "Receding Horizon Re-ordering of Multi-Agent Execution
    # Schedules" (T-RO 2024). The reactive Switchable ADG above (mapf_switchable_adg)
    # flips ONE passing-order edge at a time, myopically. The T-RO method instead
    # solves, every step, a small integer program over the switchable edges in a
    # horizon: pick the acyclic (deadlock-free) orientation that MINIMIZES the
    # cumulative route-completion time, predicted by rolling the schedule out under
    # the observed delays -- re-solved as execution proceeds (receding horizon).
    #
    # Reproduced as build_sadg + simulate_rhc(horizon=H). A KEY correctness point,
    # found the hard way: build_adg records only CONSECUTIVE passing-order edges, so
    # at a cell shared by THREE+ agents the order a1->a2->a3 is only transitive --
    # reversing the middle edge leaves a1, a3 unconstrained and a re-ordering can put
    # both on the cell at once (the reactive greedy on build_adg is genuinely NOT
    # collision-free there; it only ever got validated on 2-agent cells). build_sadg
    # materialises ALL pairs, so any acyclic orientation is a true total order per
    # cell -- collision-free under arbitrary re-ordering. This gate pins:
    # (1) HARD GUARANTEES on a random battery (40 prioritized plans x one-robot
    #     delays): every run, fixed AND re-ordered, is collision-free, deadlock-free
    #     and finishes (cf_ok == deadlock_free == finished == battery_runs).
    # (2) RE-ORDERING REDUCES cumulative completion (demo, hardcoded plan): the
    #     receding-horizon optimum drops it 26 -> 24 with no makespan regression,
    #     while a horizon of 1 sees no improvement -- so deeper LOOKAHEAD is what
    #     pays (horizon_helps).
    # (3) THE ALL-PAIRS FIX is load-bearing: on a constructed 3-agent-shared-cell
    #     plan the all-pairs SADG re-orders (a switch fires) and stays collision-free
    #     where the consecutive-edge reactive greedy executes a real collision
    #     (sadg_cf and not adg_greedy_cf).
    import random

    from mrn_coord.mapf import GridWorld, prioritized_planning
    from mrn_sim.switchable_adg import (build_adg, build_sadg,
                                        cumulative_completion,
                                        schedule_is_collision_free, simulate,
                                        simulate_rhc)

    def _rhc(paths, delay, h):
        cells, edges = build_sadg(paths)
        return cells, simulate_rhc(cells, edges, delay, horizon=h,
                                   keep_history=True)

    # (2) Demonstrator plan (a prioritized-planning solution, frozen here so the
    # gate is self-contained). Delay the robot scheduled last through a junction;
    # only a multi-edge horizon finds the helpful re-ordering.
    demo_paths = {
        0: [(4, 0), (5, 0), (5, 1)],
        1: [(4, 2), (3, 2), (2, 2), (2, 1), (2, 0)],
        2: [(0, 2), (1, 2), (1, 3), (2, 3), (3, 3), (4, 3), (4, 4), (4, 5)],
        3: [(2, 4), (3, 4), (4, 4), (4, 3), (4, 2), (4, 1)],
    }
    dc, d_fix = _rhc(demo_paths, {3: 4}, 0)
    _, d_h1 = _rhc(demo_paths, {3: 4}, 1)
    _, d_rhc = _rhc(demo_paths, {3: 4}, 8)
    fix_cc = cumulative_completion(dc, d_fix.history)
    h1_cc = cumulative_completion(dc, d_h1.history)
    rhc_cc = cumulative_completion(dc, d_rhc.history)

    # (3) Constructed plan with a 3-agent-shared cell: all-pairs SADG stays safe,
    # consecutive-edge reactive greedy collides.
    safe_paths = {
        0: [(0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (5, 2)],
        1: [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (4, 4), (4, 3), (4, 2), (4, 1)],
        2: [(5, 4), (4, 4), (3, 4), (3, 4), (3, 3)],
        3: [(3, 5), (4, 5), (4, 4), (4, 3), (4, 4), (4, 3), (5, 3), (4, 3)],
        4: [(1, 4), (1, 4), (1, 3)],
    }
    sc, s_rhc = _rhc(safe_paths, {2: 3}, 8)
    ac, ae = build_adg(safe_paths)
    s_greedy = simulate(ac, ae, {2: 3}, switchable=True, keep_history=True)

    # (1) Random hard-guarantee battery.
    def _inst(seed, n=4, w=6, h=6):
        rng = random.Random(seed)
        free = [(x, y) for x in range(w) for y in range(h)]
        cells = rng.sample(free, 2 * n)
        return GridWorld(w, h), {i: (cells[i], cells[n + i]) for i in range(n)}

    runs = cf_ok = dl_free = fin_ok = reduced = 0
    for seed in range(40):
        g, agents = _inst(seed)
        sol = prioritized_planning(g, agents)
        if sol is None:
            continue
        for da in agents:
            for d in (2, 4):
                cells, edges = build_sadg(sol.paths)
                import copy
                rf = simulate_rhc(cells, copy.deepcopy(edges), {da: d},
                                  horizon=0, keep_history=True)
                rr = simulate_rhc(cells, copy.deepcopy(edges), {da: d},
                                  horizon=8, keep_history=True)
                runs += 1
                cf_ok += int(schedule_is_collision_free(rf.history)
                             and schedule_is_collision_free(rr.history))
                dl_free += int(not rf.deadlock and not rr.deadlock)
                fin_ok += int(rf.finished and rr.finished)
                reduced += int(cumulative_completion(cells, rr.history)
                               < cumulative_completion(cells, rf.history))

    return {
        "case": "mapf_rhc_reorder",
        "battery_runs": runs,
        "battery_collision_free": cf_ok,
        "battery_deadlock_free": dl_free,
        "battery_finished": fin_ok,
        "battery_reorder_reduced_cc": reduced,
        "demo_fixed_cc": fix_cc,
        "demo_horizon1_cc": h1_cc,
        "demo_rhc_cc": rhc_cc,
        "demo_rhc_makespan": d_rhc.makespan,
        "demo_fixed_makespan": d_fix.makespan,
        "demo_rhc_switches": d_rhc.switches,
        "safety_sadg_switches": s_rhc.switches,
        "safety_sadg_collision_free": int(schedule_is_collision_free(s_rhc.history)),
        "safety_sadg_finished": int(s_rhc.finished),
        "safety_adg_greedy_collision_free": int(
            schedule_is_collision_free(s_greedy.history)),
        "hard_guarantees": (runs > 0 and cf_ok == dl_free == fin_ok == runs),
        "reorder_reduces_completion": (rhc_cc < fix_cc
                                       and d_rhc.makespan <= d_fix.makespan),
        "horizon_helps": (h1_cc == fix_cc and rhc_cc < h1_cc),
        "all_pairs_fix_is_load_bearing": (
            s_rhc.switches > 0
            and schedule_is_collision_free(s_rhc.history)
            and not schedule_is_collision_free(s_greedy.history)),
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
    # RHC re-ordering (T-RO 2024): the receding-horizon MILP over the Switchable
    # ADG that minimizes cumulative completion -- globally smarter than the myopic
    # single-flip, collision-free via an all-pairs SADG (fixes a 3+-agent-cell leak)
    ("mapf_rhc_reorder", _run_rhc_reorder),
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
    # BCBS: ECBS's sibling from the same paper -- focal at both levels but the
    # high-level bound is on the best COST, so factors multiply (w_high*w_low);
    # kept as a gated contrast (fewer expansions, higher cost than tighter ECBS)
    ("mapf_bcbs", _run_bcbs),
    ("mapf_highway", _run_highway),
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
    # Meta-Agent CBS: merge over-conflicting agents into a coupled meta-agent --
    # same optimum as CBS for any conflict bound B, but a bottleneck that explodes
    # the CBS tree collapses into one joint solve (B interpolates decoupled<->coupled)
    ("mapf_macbs", _run_macbs),
    # WHCA*: windowed hierarchical cooperative A* (Silver 2005) -- RRA* true
    # distance + a rolling cooperation window with rotating priority. Scales,
    # collision-free by construction, and resolves transient deadlocks that a
    # single fixed priority order (prioritized planning) livelocks on
    ("mapf_whca", _run_whca),
    # CBS bypassing conflicts (ICAPS 2015): adopt a same-cost, fewer-conflicts
    # child's path instead of splitting -- same optimum as CBS, fewer high-level
    # expansions and far fewer generated tree nodes, never worse
    ("mapf_cbs_bypass", _run_cbs_bypass),
    # DDM (RA-L 2020): the optimal sub-problem solution database + path
    # diversification. A database-driven, collision-free-by-construction resolver
    # -- gate pins the verified mechanisms (optimal local database, translation
    # reuse, canonical maneuvers, diversification), incomplete by design
    ("mapf_ddm", _run_ddm),
    # EPEA* (JAIR 2014): partial expansion with an Operator Selection Function --
    # generate only the children matching the node's f, same optimum as CBS, far
    # fewer generated nodes than the fully-expanding joint A*
    ("mapf_epea", _run_epea),
    # SIPPS: the safe-interval, collision-minimizing low level of MAPF-LNS2 --
    # same (collisions, length) optimum as the time-expanded planner, far fewer
    # states (one per safe interval, not per timestep)
    ("mapf_sipps", _run_sipps),
    # k-robust CBS: plans that survive delays -- a k-step buffer at every shared
    # cell so no single agent's delay (up to k) can cause a collision; k=0 == cbs
    ("mapf_k_robust", _run_k_robust_cbs),
    # CBM/TAPF: target assignment + path finding for TEAMS -- per-team anonymous
    # max-flow low level under CBS-over-teams; interpolates flow (one team) and
    # labeled MAPF (singleton teams), makespan-optimal
    ("mapf_cbm", _run_cbm_tapf),
    # CBS-TA: CBS with optimal Target Assignment -- a forest of roots (one per
    # assignment, unfolded lazily by Murty's K-best) over CBS's constraint tree,
    # jointly optimal over assignment + paths; degenerates to cbs (one goal each)
    ("mapf_cbs_ta", _run_cbs_ta),
    # LaCAM2 swap: the swap operation that improves PIBT successor generation --
    # detect a required+possible exchange and pull the partner through a pocket,
    # resolving narrow-corridor swaps that base PIBT livelocks on (swap=False==PIBT)
    ("mapf_pibt_swap", _run_pibt_swap),
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
    ("mapf_push_and_swap", _run_push_and_swap),
    # Bibox: constructive polynomial COMPLETE solver on biconnected graphs via an
    # open ear decomposition (solve ears in reverse by cycle rotation, then the
    # basic cycle); valid by construction, solves packed formations CBS busts on
    ("mapf_bibox", _run_bibox),
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
    # BCP rectangle cuts: Lam et al.'s specialized cut family on the branch-price
    # frame -- a single cut sum_B1 y_a1 + sum_B2 y_a2 <= 1 collapses the
    # same-direction rectangle-crossing symmetry that branching enumerates
    ("bcp_rectangle", _run_bcp_rectangle),
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
