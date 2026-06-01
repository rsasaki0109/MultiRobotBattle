#!/usr/bin/env python3
"""Bounded-suboptimality contract: our ECBS vs. the reference libMultiRobotPlanning.

Where ``compare_mapf_libmrp.py`` pins our *optimal* CBS to the reference's exact
sum-of-costs, this pins our *bounded-suboptimal* ECBS to the guarantee that makes
it useful: with suboptimality factor ``w``, Enhanced CBS (Barer et al.) returns a
solution whose sum-of-costs is at most ``w`` times the optimum. That is a property
with teeth — a single inequality per instance — so it can be gated exactly.

Both our ``mrn_coord.mapf.ecbs`` and Wolfgang Hönig's ``libMultiRobotPlanning``
``ecbs`` run the same discrete model (4-connected grid, wait, unit cost, vertex +
edge-swap conflicts, agents stay on goal) and take the same ``w``. The optimum is
our CBS cost, which the equivalence contract already proved equal to the reference
``cbs`` — so it is a trustworthy denominator.

For each instance we check the contract on **both** implementations:

  ratio = cost / optimal   must satisfy   ratio <= w   (the ECBS guarantee)

The two ECBS solvers need not return the *same* cost — focal-search tie-breaking
differs — so unlike the CBS contract we do not gate equality. We gate the bound
each honors, and report both costs and their ratios so the suboptimality actually
taken is visible (often well below the w ceiling).

The reference is optional: build the ``ecbs`` binary (see ``docs/coordination.md``)
and point ``LIBMRP_ECBS`` at it (or have it on ``PATH``). Absent, the script skips.

    python3 scripts/compare_ecbs_libmrp.py            # print the table
    python3 scripts/compare_ecbs_libmrp.py --check     # exit non-zero if the bound is violated
    python3 scripts/compare_ecbs_libmrp.py --write      # (re)write the artifact

Pure and deterministic.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir))
sys.path.insert(0, os.path.join(_REPO, "mrn_coord"))
sys.path.insert(0, _HERE)

from compare_mapf_libmrp import (MAX_EXPANSIONS, _parse_output,  # noqa: E402
                                 _scenarios, _write_input)
from mrn_coord.mapf.cbs import cbs                       # noqa: E402
from mrn_coord.mapf.ecbs import ecbs                     # noqa: E402
from mrn_coord.mapf.grid import GridWorld                # noqa: E402

WEIGHT = 1.5            # suboptimality factor used on both sides
_EPS = 1e-9             # integer costs, fractional bound — guard the comparison


def _ecbs_binary() -> str | None:
    env = os.environ.get("LIBMRP_ECBS")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    return shutil.which("ecbs")


def _run_reference(binary, width, height, obstacles, agents, w):
    """Solve one instance with the reference ``ecbs``; return ``(soc, makespan)``
    or ``None`` if it reports the instance unsolvable."""
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "in.yaml")
        out_path = os.path.join(tmp, "out.yaml")
        _write_input(in_path, width, height, obstacles, agents)
        proc = subprocess.run([binary, "-i", in_path, "-o", out_path,
                              "-w", str(w)], capture_output=True, text=True)
        if not os.path.isfile(out_path):
            return None
        with open(out_path, "r", encoding="utf-8") as fh:
            parsed = _parse_output(fh.read())
        if parsed is None:
            return None
        if proc.returncode != 0:
            raise RuntimeError(f"ecbs exited {proc.returncode}: {proc.stderr}")
        return parsed


def compare(name, width, height, obstacles, agents, binary, w):
    """Solve one instance with both ECBS solvers and our CBS for the optimum."""
    grid = GridWorld(width, height, blocked=frozenset(obstacles))
    adict = {str(i): (s, g) for i, (s, g) in enumerate(agents)}
    opt = cbs(grid, adict, max_expansions=MAX_EXPANSIONS)
    ours = ecbs(grid, adict, w=w, max_expansions=MAX_EXPANSIONS)
    ref = _run_reference(binary, width, height, obstacles, agents, w)

    optimal = opt.cost if opt is not None else None
    soc_ours = ours.cost if ours is not None else None
    soc_lib = ref[0] if ref is not None else None
    return {
        "scenario": name,
        "agents": len(agents),
        "optimal": optimal,
        "soc_ours": soc_ours,
        "soc_lib": soc_lib,
        "ratio_ours": (soc_ours / optimal) if (optimal and soc_ours is not None) else None,
        "ratio_lib": (soc_lib / optimal) if (optimal and soc_lib is not None) else None,
    }


def run_all(binary, w=WEIGHT):
    return [compare(name, wd, ht, obs, agents, binary, w)
            for (name, wd, ht, obs, agents) in _scenarios()]


_REPORT = os.path.join(_REPO, "benchmarks", "ecbs_libmrp.md")


def _violations(rows, w):
    """Gated failures: a solver that exceeded the w·optimal bound, or that
    failed to solve a solvable instance."""
    bad = []
    for r in rows:
        if r["optimal"] is None:
            continue                                  # CBS budget exhausted; skip
        for who, ratio in (("ours", r["ratio_ours"]), ("lib", r["ratio_lib"])):
            if ratio is None:
                bad.append(f"{r['scenario']}: {who} found no solution "
                           f"(optimal exists, cost {r['optimal']})")
            elif ratio > w + _EPS:
                bad.append(f"{r['scenario']}: {who} ratio {ratio:.3f} > w={w} "
                           f"(cost {r['soc_ours'] if who == 'ours' else r['soc_lib']}"
                           f" vs optimal {r['optimal']})")
    return bad


def _fmt(v, spec="{}"):
    return "-" if v is None else spec.format(v)


def build_report(rows, w):
    lines = [
        "# Our ECBS vs. the reference libMultiRobotPlanning",
        "",
        "Our `mrn_coord.mapf.ecbs` is bounded-suboptimal Enhanced CBS (Barer, "
        "Sharon, Stern & Felner): with suboptimality factor `w` it returns a "
        "solution whose sum-of-costs is at most `w` times the optimum. This "
        "report turns that guarantee into a measured contract against the "
        "reference implementation, Wolfgang Hönig's `libMultiRobotPlanning` "
        "`ecbs` — same discrete model, same `w`. The optimum is our CBS cost, "
        "which [`benchmarks/mapf_libmrp.md`](mapf_libmrp.md) already proved equal "
        "to the reference `cbs`. Regenerate with "
        "`python3 scripts/compare_ecbs_libmrp.py --write` (see "
        "`docs/coordination.md`).",
        "",
        f"`w = {w}`. The gated property is `cost <= w · optimal` for **both** "
        "solvers (`ratio <= w`). The two need not agree on cost — focal-search "
        "tie-breaking differs — so, unlike the optimal CBS contract, equality is "
        "not gated; the ratios show the suboptimality actually taken.",
        "",
        "| scenario | N | optimal | ours (ratio) | lib (ratio) |",
        "| :-- | :-: | --: | --: | --: |",
    ]
    for r in rows:
        ours = (f"{_fmt(r['soc_ours'])} ({_fmt(r['ratio_ours'], '{:.3f}')})")
        lib = (f"{_fmt(r['soc_lib'])} ({_fmt(r['ratio_lib'], '{:.3f}')})")
        lines.append(f"| {r['scenario']} | {r['agents']} | {_fmt(r['optimal'])} | "
                     f"{ours} | {lib} |")
    lines += [
        "",
        f"Gate (`--check`): every `ratio <= {w}`, on both implementations, on "
        "every solvable instance. The bound holds across the suite — and the "
        "ratios sit at or below it, so ECBS is taking only the slack it needs. "
        "This is the bounded-suboptimal sibling of the exact-optimal CBS contract.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _format(rows):
    out = [f"  {'scenario':18} {'N':>2} {'optimal':>7} {'ours':>6} {'r_ours':>7} "
           f"{'lib':>6} {'r_lib':>7}"]
    for r in rows:
        out.append(f"  {r['scenario']:18} {r['agents']:>2} {_fmt(r['optimal']):>7} "
                   f"{_fmt(r['soc_ours']):>6} {_fmt(r['ratio_ours'], '{:.3f}'):>7} "
                   f"{_fmt(r['soc_lib']):>6} {_fmt(r['ratio_lib'], '{:.3f}'):>7}")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if either solver exceeds w·optimal")
    parser.add_argument("--write", action="store_true",
                        help="(re)write the checked-in benchmarks/ecbs_libmrp.md")
    parser.add_argument("-w", "--weight", type=float, default=WEIGHT)
    args = parser.parse_args()

    binary = _ecbs_binary()
    if binary is None:
        print("the reference libMultiRobotPlanning `ecbs` binary was not found "
              "(set LIBMRP_ECBS or put it on PATH); skipping.\n"
              "Build it to run this check (see docs/coordination.md).")
        return

    rows = run_all(binary, w=args.weight)
    print(f"=== our ECBS vs. the reference libMultiRobotPlanning "
          f"(w={args.weight}) ===")
    print(_format(rows))

    if args.write:
        os.makedirs(os.path.dirname(_REPORT), exist_ok=True)
        with open(_REPORT, "w", encoding="utf-8") as fh:
            fh.write(build_report(rows, args.weight))
        print(f"\nwrote {_REPORT}")

    if args.check:
        bad = _violations(rows, args.weight)
        if bad:
            print("\nBOUND VIOLATION:")
            for b in bad:
                print(f"  {b}")
            sys.exit(1)
        print(f"\nok: both solvers stay within w={args.weight}·optimal on every "
              "instance")


if __name__ == "__main__":
    main()
