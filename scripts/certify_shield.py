#!/usr/bin/env python3
"""Adversarial certification harness for the runtime safety shield.

A safety shield's guarantee is only worth what survives an adversary. This throws
thousands of *deterministic* rollouts at three controllers on identical obstacle
fields, with a nominal command engineered to crash — it steers straight at the
nearest obstacle at full speed every tick — and measures the **robot body**:

    unshielded    the adversarial command, only actuation-clamped (the attack)
    lookahead     the first-order look-ahead CBF filter (mrn_sim.cbf)
    shield        the certified body-true braking shield (mrn_sim.shield)

For each it reports body collisions (a rollout whose body clearance ever drops
below zero), the worst body clearance reached, and the mean deviation from the
nominal command (how hard it had to intervene). The contract: the certified
shield collides **zero** times — driving the body, not a look-ahead point, out of
the obstacle for any input the adversary picks. The look-ahead filter is included
to show its body-frame guarantee is softer (it protects a point ahead of the
axle, so a hard turn can swing the body across the boundary).

    python3 scripts/certify_shield.py            # print the table
    python3 scripts/certify_shield.py --check     # exit non-zero on a shield collision

Pure and deterministic (seeded RNG, list iteration only), so the numbers are
identical across processes and can back a benchmark-gate contract.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir))
sys.path.insert(0, os.path.join(_REPO, "mrn_sim"))


def _body_clearance(pose, obstacles, robot_radius):
    return min(math.hypot(pose[0] - o[0], pose[1] - o[1]) - o[2] - robot_radius
               for o in obstacles)


def _random_field(rng):
    """A clear-at-origin field the adversary will try to drive the robot into."""
    n = rng.randint(1, 5)
    return [(rng.uniform(1.0, 9.0), rng.uniform(-3.0, 3.0), rng.uniform(0.3, 0.8))
            for _ in range(n)]


def certify(seed: int = 0, trials: int = 3000, steps: int = 250,
            dt: float = 0.1) -> dict:
    """Run the adversarial suite; return a flat, deterministic metrics dict."""
    from mrn_sim.cbf import CBFConfig, cbf_filter
    from mrn_sim.kinematics import unicycle_step
    from mrn_sim.shield import ShieldConfig, shield_step

    scfg = ShieldConfig()
    ccfg = CBFConfig(robot_radius=scfg.robot_radius, max_v=scfg.max_v,
                     max_omega=scfg.max_omega)
    rr = scfg.robot_radius
    rng = random.Random(seed)

    names = ("unshielded", "lookahead", "shield")
    collisions = {k: 0 for k in names}
    min_clear = {k: float("inf") for k in names}
    dev_sum = {k: 0.0 for k in names}
    dev_n = {k: 0 for k in names}
    ran = 0

    for _ in range(trials):
        obstacles = _random_field(rng)
        theta0 = rng.uniform(-0.3, 0.3)
        if _body_clearance((0.0, 0.0), obstacles, rr) < 0.2:
            continue                                  # not clear at start; skip
        ran += 1
        # identical rollout per controller from the same start + field
        for name in names:
            pose = (0.0, 0.0, theta0)
            v = 0.0
            worst = float("inf")
            for _k in range(steps):
                # adversarial nominal: aim at the nearest obstacle, full speed
                near = min(obstacles, key=lambda o: math.hypot(
                    pose[0] - o[0], pose[1] - o[1]))
                ang = math.atan2(near[1] - pose[1], near[0] - pose[0])
                derr = math.atan2(math.sin(ang - pose[2]),
                                  math.cos(ang - pose[2]))
                u_nom = (scfg.max_v, 4.0 * derr)
                if name == "unshielded":
                    cv = max(-scfg.max_v, min(scfg.max_v, u_nom[0]))
                    cw = max(-scfg.max_omega, min(scfg.max_omega, u_nom[1]))
                elif name == "lookahead":
                    cv, cw = cbf_filter(pose, u_nom, obstacles, ccfg)
                else:
                    cv, cw = shield_step((pose[0], pose[1], pose[2], v),
                                         u_nom, obstacles, dt, scfg)
                dev_sum[name] += abs(cv - u_nom[0]) + abs(cw - u_nom[1])
                dev_n[name] += 1
                pose = unicycle_step(pose, cv, cw, dt)
                v = cv
                worst = min(worst, _body_clearance(pose, obstacles, rr))
            min_clear[name] = min(min_clear[name], worst)
            if worst < -1e-3:
                collisions[name] += 1

    out = {"case": "shield_certify", "trials": ran}
    for name in names:
        out[name + "_collisions"] = collisions[name]
        out[name + "_min_clearance"] = round(min_clear[name], 4)
        out[name + "_mean_deviation"] = round(dev_sum[name] / max(1, dev_n[name]), 4)
    return out


def certify_reciprocal(seed: int = 0, trials: int = 800, n_robots: int = 4,
                       steps: int = 300, dt: float = 0.1) -> dict:
    """Reciprocal certification: ``n_robots`` shielded robots in *adversarial*
    mutual pursuit (each steers straight at its nearest neighbour at full speed),
    each treating the others as moving obstacles with no shared coordination.

    This is the multi-robot analogue of the single-robot certificate, and the
    moving-obstacle case in its hardest form — every "obstacle" is itself trying
    to collide. The contract: with every robot shielded, none ever touch; the
    same adversarial pursuit with no shield collides in (almost) every rollout.
    """
    from mrn_sim.kinematics import unicycle_step
    from mrn_sim.shield import ShieldConfig, shield_step

    cfg = ShieldConfig()
    rr = cfg.robot_radius
    rng = random.Random(seed)
    n = n_robots

    def min_gap(rob):
        return min(math.hypot(rob[i][0] - rob[j][0], rob[i][1] - rob[j][1])
                   - 2.0 * rr
                   for i in range(n) for j in range(i + 1, n))

    coll = {"shield": 0, "unshielded": 0}
    worst = {"shield": float("inf"), "unshielded": float("inf")}
    ran = 0
    for _ in range(trials):
        start = []
        for i in range(n):
            ang = 2.0 * math.pi * i / n + rng.uniform(-0.3, 0.3)
            radius = rng.uniform(4.0, 6.0)
            start.append([radius * math.cos(ang), radius * math.sin(ang),
                          ang + math.pi, 0.0])     # x, y, theta, v facing centre
        if min_gap(start) < 0.3:
            continue
        ran += 1
        for mode in ("shield", "unshielded"):
            rob = [list(s) for s in start]
            run_worst = float("inf")
            for _k in range(steps):
                cmds = []
                for i in range(n):
                    others = [(rob[j][0], rob[j][1], rr,
                               rob[j][3] * math.cos(rob[j][2]),
                               rob[j][3] * math.sin(rob[j][2]))
                              for j in range(n) if j != i]
                    near = min(others, key=lambda o: math.hypot(
                        rob[i][0] - o[0], rob[i][1] - o[1]))
                    ang = math.atan2(near[1] - rob[i][1], near[0] - rob[i][0])
                    derr = math.atan2(math.sin(ang - rob[i][2]),
                                      math.cos(ang - rob[i][2]))
                    u = (cfg.max_v, 4.0 * derr)
                    if mode == "shield":
                        cmds.append(shield_step(
                            (rob[i][0], rob[i][1], rob[i][2], rob[i][3]),
                            u, others, dt, cfg))
                    else:
                        cmds.append((max(-cfg.max_v, min(cfg.max_v, u[0])),
                                     max(-cfg.max_omega, min(cfg.max_omega, u[1]))))
                for i in range(n):
                    p = unicycle_step((rob[i][0], rob[i][1], rob[i][2]),
                                      cmds[i][0], cmds[i][1], dt)
                    rob[i] = [p[0], p[1], p[2], cmds[i][0]]
                run_worst = min(run_worst, min_gap(rob))
            worst[mode] = min(worst[mode], run_worst)
            if run_worst < -1e-3:
                coll[mode] += 1

    out = {"case": "shield_certify_reciprocal", "trials": ran, "n_robots": n}
    for mode in ("unshielded", "shield"):
        out[mode + "_collisions"] = coll[mode]
        out[mode + "_min_gap"] = round(worst[mode], 4)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the certified shield ever collides")
    parser.add_argument("--mode", choices=("static", "reciprocal"),
                        default="static",
                        help="static single-robot adversary, or reciprocal "
                        "multi-robot mutual pursuit")
    parser.add_argument("--robots", type=int, default=4,
                        help="number of robots in --mode reciprocal")
    parser.add_argument("--trials", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "reciprocal":
        m = certify_reciprocal(seed=args.seed, trials=args.trials,
                               n_robots=args.robots)
        hdr = f"{'controller':12s} {'collisions':>11s} {'min gap (m)':>12s}"
        print(f"reciprocal certification — {m['trials']} rollouts, "
              f"{m['n_robots']} robots in adversarial mutual pursuit\n")
        print(hdr)
        print("-" * len(hdr))
        for name in ("unshielded", "shield"):
            print(f"{name:12s} {m[name + '_collisions']:>11d} "
                  f"{m[name + '_min_gap']:>12.4f}")
        if args.check:
            if m["shield_collisions"] != 0:
                print(f"\nFAIL: shielded robots collided in "
                      f"{m['shield_collisions']} rollouts")
                return 1
            print(f"\nok: {m['n_robots']} shielded robots collision-free across "
                  f"{m['trials']} adversarial rollouts")
        return 0

    m = certify(seed=args.seed, trials=args.trials)
    hdr = f"{'controller':12s} {'collisions':>11s} {'min body clr':>13s} {'mean dev':>9s}"
    print(f"adversarial certification — {m['trials']} rollouts "
          f"(steer at nearest obstacle, full speed)\n")
    print(hdr)
    print("-" * len(hdr))
    for name in ("unshielded", "lookahead", "shield"):
        print(f"{name:12s} {m[name + '_collisions']:>11d} "
              f"{m[name + '_min_clearance']:>13.4f} "
              f"{m[name + '_mean_deviation']:>9.4f}")
    if args.check:
        if m["shield_collisions"] != 0:
            print(f"\nFAIL: certified shield collided in "
                  f"{m['shield_collisions']} rollouts")
            return 1
        print(f"\nok: certified shield collision-free across {m['trials']} "
              f"adversarial rollouts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
