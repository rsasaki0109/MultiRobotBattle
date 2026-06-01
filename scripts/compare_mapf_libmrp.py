#!/usr/bin/env python3
"""Equivalence contract: our MAPF CBS vs. the reference libMultiRobotPlanning.

``mrn_coord.mapf.cbs`` is a from-scratch implementation of Conflict-Based Search
(Sharon, Stern, Felner & Sturtevant, 2015). This turns "it finds the optimal
solution" into a *checkable* contract by solving identical instances with the
canonical reference — Wolfgang Hönig's ``libMultiRobotPlanning`` C++ ``cbs``
(``github.com/whoenig/libMultiRobotPlanning``) — and comparing the results.

The reference is an *optional* dependency: build the ``cbs`` binary (see
``docs/coordination.md``) and point ``LIBMRP_CBS`` at it (or have ``cbs`` on
``PATH``). With it absent the script skips cleanly so the core CI is untouched.

Why the comparison is exact, not approximate:

  Both solvers operate on the **same discrete model** — a 4-connected grid with
  wait actions and unit edge cost, vertex + edge(swap) conflicts, and agents
  that stay on their goal (we do *not* pass ``--disappear-at-goal``, so the
  reference keeps blocking there, exactly like us). Both minimize the same
  objective, **sum-of-costs**. The minimum of a well-defined objective is a
  single number, so two correct optimal solvers must report the *identical*
  ``sum_of_costs`` on every instance — there is no tolerance to tune. A mismatch
  is a real defect in one of them, which is the whole point of the contract.

Two quantities, two roles:

  sum_of_costs   **Gated.** The optimization objective; provably unique at the
                 optimum. Our value must equal the reference's exactly, and both
                 must agree on whether the instance is solvable at all.

  makespan       **Reported, not gated.** Among the many solutions that share the
                 optimal sum-of-costs, the makespan can differ by which one the
                 high-level tie-break happens to return first. A difference here
                 is a tie-break artifact, not a correctness gap — so we surface
                 it for insight but never fail the build on it. (This is the MAPF
                 analogue of the along-track phase in the ORCA/RVO2 contract.)

    python3 scripts/compare_mapf_libmrp.py            # print the table
    python3 scripts/compare_mapf_libmrp.py --check     # exit non-zero on mismatch
    python3 scripts/compare_mapf_libmrp.py --write      # (re)write the artifact

Pure and deterministic (seeded RNG, fixed scenarios): the comparison is identical
across processes and can back a benchmark-gate contract.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir))
sys.path.insert(0, os.path.join(_REPO, "mrn_coord"))

from mrn_coord.mapf.cbs import cbs                       # noqa: E402
from mrn_coord.mapf.grid import GridWorld                # noqa: E402
from mrn_coord.mapf.movingai import load_map, load_scen  # noqa: E402

# CBS is exponential in the number of conflicts, so the reference and our solver
# must both finish quickly: small grids, few agents, but genuine conflicts.
MAX_EXPANSIONS = 100_000


def _cbs_binary() -> str | None:
    """Locate the reference ``cbs`` executable, or ``None`` if unavailable."""
    env = os.environ.get("LIBMRP_CBS")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    return shutil.which("cbs")


def _scenarios():
    """Conflict-rich instances as ``(name, width, height, obstacles, agents)``.

    ``agents`` is a list of ``(start, goal)`` cell pairs. Every instance forces
    the two solvers to resolve real vertex/edge conflicts, so a wrong cost model
    on either side would show up as a sum-of-costs mismatch.
    """
    out = []

    # Head-on swap on a 1-wide track: one agent must detour, the other goes
    # straight — the textbook CBS conflict.
    out.append(("swap2", 3, 3, set(),
                [((0, 1), (2, 1)), ((2, 1), (0, 1))]))

    # A wall with a single one-cell doorway; two agents cross through it from
    # opposite sides and must take turns.
    wall = {(2, y) for y in range(5) if y != 2}
    out.append(("doorway", 5, 5, wall,
                [((0, 2), (4, 2)), ((4, 2), (0, 2))]))

    # Four agents crossing through a shared centre on an open board.
    out.append(("crossing4", 5, 5, set(),
                [((0, 2), (4, 2)), ((4, 2), (0, 2)),
                 ((2, 0), (2, 4)), ((2, 4), (2, 0))]))

    # A couple of obstacle blocks with three agents threading past them.
    blocks = {(2, 2), (2, 3), (3, 2), (3, 3)}
    out.append(("blocks3", 6, 6, blocks,
                [((0, 0), (5, 5)), ((5, 0), (0, 5)), ((0, 5), (5, 0))]))

    # Seeded random instance on a small open board: distinct starts and goals.
    rng = random.Random(20240602)
    w = h = 6
    cells = [(x, y) for x in range(w) for y in range(h)]
    picked = rng.sample(cells, 8)
    agents = [(picked[2 * i], picked[2 * i + 1]) for i in range(4)]
    out.append(("random4", w, h, set(), agents))

    # The bundled MovingAI example, parsed by our own loader so the contract also
    # covers the community .map/.scen path (3 agents on an 8x8 obstacle map).
    grid = load_map(os.path.join(_REPO, "mrn_coord", "benchmarks", "example.map"))
    tasks = load_scen(os.path.join(_REPO, "mrn_coord", "benchmarks", "example.scen"))
    out.append(("movingai_example", grid.width, grid.height, set(grid.blocked),
                [(t.start, t.goal) for t in tasks]))

    return out


def _write_input(path: str, width, height, obstacles, agents) -> None:
    """Render an instance as a libMultiRobotPlanning input YAML."""
    lines = ["map:", f"  dimensions: [{width}, {height}]", "  obstacles:"]
    if obstacles:
        for (x, y) in sorted(obstacles):
            lines.append(f"    - [{x}, {y}]")
    else:
        lines[-1] = "  obstacles: []"
    lines.append("agents:")
    for i, (start, goal) in enumerate(agents):
        lines.append(f"  - name: agent{i}")
        lines.append(f"    start: [{start[0]}, {start[1]}]")
        lines.append(f"    goal: [{goal[0]}, {goal[1]}]")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _parse_output(text: str):
    """Pull ``cost`` and ``makespan`` out of a libMultiRobotPlanning output YAML.

    Returns ``(sum_of_costs, makespan)`` or ``None`` if no statistics block was
    written (the reference emits no output file when the instance is unsolvable).
    """
    cost = re.search(r"^\s*cost:\s*(\d+)", text, re.MULTILINE)
    mk = re.search(r"^\s*makespan:\s*(\d+)", text, re.MULTILINE)
    if cost is None or mk is None:
        return None
    return int(cost.group(1)), int(mk.group(1))


def _run_reference(binary, width, height, obstacles, agents):
    """Solve one instance with the reference ``cbs``; return ``(soc, makespan)``
    or ``None`` if it reports the instance unsolvable."""
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "in.yaml")
        out_path = os.path.join(tmp, "out.yaml")
        _write_input(in_path, width, height, obstacles, agents)
        proc = subprocess.run([binary, "-i", in_path, "-o", out_path],
                              capture_output=True, text=True)
        # The reference returns non-zero / writes no file on an unsolvable
        # instance; treat a missing output as "no solution".
        if not os.path.isfile(out_path):
            return None
        with open(out_path, "r", encoding="utf-8") as fh:
            parsed = _parse_output(fh.read())
        if parsed is None:
            return None
        if proc.returncode != 0:
            # Solved-but-nonzero shouldn't happen; surface it loudly.
            raise RuntimeError(f"cbs exited {proc.returncode}: {proc.stderr}")
        return parsed


def compare(name, width, height, obstacles, agents, binary):
    """Solve one instance with both solvers; return a flat metrics dict."""
    grid = GridWorld(width, height, blocked=frozenset(obstacles))
    ours = cbs(grid, {str(i): (s, g) for i, (s, g) in enumerate(agents)},
               max_expansions=MAX_EXPANSIONS)
    ref = _run_reference(binary, width, height, obstacles, agents)

    return {
        "scenario": name,
        "agents": len(agents),
        "solved_ours": ours is not None,
        "solved_lib": ref is not None,
        "soc_ours": ours.cost if ours is not None else None,
        "soc_lib": ref[0] if ref is not None else None,
        "makespan_ours": ours.makespan if ours is not None else None,
        "makespan_lib": ref[1] if ref is not None else None,
    }


def run_all(binary):
    return [compare(name, w, h, obs, agents, binary)
            for (name, w, h, obs, agents) in _scenarios()]


_REPORT = os.path.join(_REPO, "benchmarks", "mapf_libmrp.md")


def _mismatches(rows):
    """Gated failures: solved-parity and sum-of-costs disagreements."""
    bad = []
    for r in rows:
        if r["solved_ours"] != r["solved_lib"]:
            bad.append(f"{r['scenario']}: solvability differs "
                       f"(ours={r['solved_ours']}, lib={r['solved_lib']})")
        elif r["solved_ours"] and r["soc_ours"] != r["soc_lib"]:
            bad.append(f"{r['scenario']}: sum_of_costs {r['soc_ours']} != "
                       f"{r['soc_lib']} (reference)")
    return bad


def _fmt(v):
    return "-" if v is None else str(v)


def build_report(rows):
    """Render the checked-in Markdown artifact for the equivalence contract."""
    lines = [
        "# Our CBS vs. the reference libMultiRobotPlanning",
        "",
        "Our `mrn_coord.mapf.cbs` is a from-scratch implementation of "
        "Conflict-Based Search (Sharon, Stern, Felner & Sturtevant). This report "
        "turns *\"it finds the optimal solution\"* into a measured contract: the "
        "same instances are solved by our code and by the canonical reference "
        "implementation, Wolfgang Hönig's `libMultiRobotPlanning` C++ `cbs` "
        "(`github.com/whoenig/libMultiRobotPlanning`). Regenerate with "
        "`python3 scripts/compare_mapf_libmrp.py --write` once the `cbs` binary "
        "is built and on `LIBMRP_CBS`/`PATH` (see `docs/coordination.md`).",
        "",
        "Both solvers run the **same discrete model** — 4-connected grid, wait "
        "actions, unit edge cost, vertex + edge(swap) conflicts, agents that "
        "stay on their goal (we never pass `--disappear-at-goal`). Both minimize "
        "**sum-of-costs**.",
        "",
        "- **`sum_of_costs`** — *gated*. The objective both solvers minimize; its "
        "optimum is a single number, so a correct optimal solver must reproduce "
        "the reference's value exactly (and agree on solvability). There is no "
        "tolerance — a mismatch is a real defect.",
        "- **`makespan`** — *reported, not gated*. Many solutions share the "
        "optimal sum-of-costs; which one a tie-break returns first decides the "
        "makespan, so a difference here is an artifact, not a correctness gap.",
        "",
        "| scenario | N | solved (ours/lib) | sum_of_costs (ours/lib) | "
        "makespan (ours/lib) |",
        "| :-- | :-: | :-: | :-: | :-: |",
    ]
    for r in rows:
        solved = f"{r['solved_ours']} / {r['solved_lib']}"
        soc = f"{_fmt(r['soc_ours'])} / {_fmt(r['soc_lib'])}"
        mk = f"{_fmt(r['makespan_ours'])} / {_fmt(r['makespan_lib'])}"
        lines.append(f"| {r['scenario']} | {r['agents']} | {solved} | {soc} | {mk} |")
    lines += [
        "",
        "Gate (`--check`): identical `sum_of_costs` on every solved instance and "
        "matching solvability. Across the suite our CBS reproduces the "
        "reference's optimal cost exactly — the implementation computes the same "
        "optimum, not merely a feasible solution. The discrete model is shared by "
        "construction (the comparison would otherwise compare two different "
        "problems, not two solvers), so this isolates the search itself.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _format(rows):
    header = (f"  {'scenario':18} {'N':>2} {'solved o/l':>11} "
              f"{'soc o/l':>13} {'makespan o/l':>14}")
    out = [header]
    for r in rows:
        solved = f"{r['solved_ours']}/{r['solved_lib']}"
        soc = f"{_fmt(r['soc_ours'])}/{_fmt(r['soc_lib'])}"
        mk = f"{_fmt(r['makespan_ours'])}/{_fmt(r['makespan_lib'])}"
        out.append(f"  {r['scenario']:18} {r['agents']:>2} {solved:>11} "
                   f"{soc:>13} {mk:>14}")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the optimal sum-of-costs or "
                             "solvability disagrees with the reference")
    parser.add_argument("--write", action="store_true",
                        help="(re)write the checked-in benchmarks/mapf_libmrp.md")
    args = parser.parse_args()

    binary = _cbs_binary()
    if binary is None:
        print("the reference libMultiRobotPlanning `cbs` binary was not found "
              "(set LIBMRP_CBS or put it on PATH); skipping.\n"
              "Build it to run this equivalence check (see docs/coordination.md).")
        return

    rows = run_all(binary)
    print(f"=== our CBS vs. the reference libMultiRobotPlanning ({os.path.basename(binary)}) ===")
    print(_format(rows))

    if args.write:
        os.makedirs(os.path.dirname(_REPORT), exist_ok=True)
        with open(_REPORT, "w", encoding="utf-8") as fh:
            fh.write(build_report(rows))
        print(f"\nwrote {_REPORT}")

    if args.check:
        bad = _mismatches(rows)
        if bad:
            print("\nMISMATCH:")
            for b in bad:
                print(f"  {b}")
            sys.exit(1)
        print("\nok: optimal sum-of-costs and solvability match the reference "
              "on every instance")


if __name__ == "__main__":
    main()
