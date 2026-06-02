"""Load the standard MovingAI MAPF benchmark format and run a solver on it.

MovingAI (https://movingai.com/benchmarks/mapf.html) is the de-facto benchmark
format for multi-agent path finding. Supporting it lets this repo's CBS /
prioritized solvers be evaluated on the same maps and scenarios the MAPF
community uses — drop in any downloaded ``.map`` / ``.scen`` pair.

- :func:`load_map` — parse a ``.map`` (octile grid) into a :class:`GridWorld`.
- :func:`load_scen` — parse a ``.scen`` into a list of (start, goal) tasks.
- :func:`run_mapf_benchmark` — solve the first ``num_agents`` tasks and report
  success / makespan / sum-of-costs.

Note CBS is *optimal* but scales to small teams; for many agents use the
prioritized solver (fast, incomplete). The loaders are pure; the runner reuses
:mod:`mrn_coord.mapf.cbs` / :mod:`mrn_coord.mapf.prioritized`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cbs import cbs
from .ecbs import ecbs
from .grid import Cell, GridWorld
from .lacam import lacam
from .lns import mapf_lns
from .pbs import pbs
from .prioritized import prioritized_planning
from .sipp import plan_sipp
from .solution import makespan, sum_of_costs

# MovingAI passable terrain: '.' and 'G' (ground); everything else is blocked.
_PASSABLE = {".", "G"}


def parse_map(text: str) -> GridWorld:
    """Parse MovingAI ``.map`` text into a :class:`GridWorld`."""
    lines = text.splitlines()
    height = width = 0
    map_start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("height"):
            height = int(s.split()[1])
        elif s.startswith("width"):
            width = int(s.split()[1])
        elif s == "map":
            map_start = i + 1
            break
    rows = lines[map_start:map_start + height]
    blocked = set()
    for y, row in enumerate(rows):
        for x in range(min(width, len(row))):
            if row[x] not in _PASSABLE:
                blocked.add((x, y))
    return GridWorld(width, height, blocked=blocked)


def load_map(path: str) -> GridWorld:
    with open(path, "r", encoding="utf-8") as fh:
        return parse_map(fh.read())


@dataclass(frozen=True)
class ScenTask:
    """One row of a ``.scen`` file: a start/goal pair (+ reference length)."""

    start: Cell
    goal: Cell
    optimal_length: float = 0.0


def parse_scen(text: str) -> list:
    """Parse MovingAI ``.scen`` text into a list of :class:`ScenTask`.

    Each task row is ``bucket map width height sx sy gx gy optimal`` (cells are
    column/row); the leading ``version`` line is skipped.
    """
    tasks = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("version"):
            continue
        f = s.split("\t") if "\t" in s else s.split()
        if len(f) < 9:
            continue
        sx, sy, gx, gy = int(f[4]), int(f[5]), int(f[6]), int(f[7])
        tasks.append(ScenTask((sx, sy), (gx, gy), float(f[8])))
    return tasks


def load_scen(path: str) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        return parse_scen(fh.read())


def run_mapf_benchmark(
    grid: GridWorld,
    tasks,
    *,
    num_agents: int | None = None,
    solver: str = "cbs",
    max_expansions: int = 100_000,
    weight: float = 1.5,
) -> dict:
    """Solve the first ``num_agents`` tasks and return benchmark metrics.

    ``solver`` is ``"cbs"`` (optimal, small teams), ``"ecbs"`` (bounded-
    suboptimal, ``cost <= weight * optimal``; scales further), ``"lacam"``
    (complete satisficing search over configurations; scales to large teams),
    ``"lns"`` (anytime large-neighborhood search, polishing toward the optimum),
    ``"pbs"`` (priority-ordering search, suboptimal; reorders past the deadlocks
    fixed-order prioritized planning hits), ``"prioritized"`` (fast, incomplete),
    or ``"prioritized_sipp"`` (the same, but with the safe-interval low-level
    planner). Returns a dict with
    ``solved`` / ``num_agents`` / and, when solved, ``makespan`` and
    ``sum_of_costs``.
    """
    chosen = tasks if num_agents is None else tasks[:num_agents]
    agents = {str(i): (t.start, t.goal) for i, t in enumerate(chosen)}
    if solver == "cbs":
        solution = cbs(grid, agents, max_expansions=max_expansions)
    elif solver == "ecbs":
        solution = ecbs(grid, agents, w=weight, max_expansions=max_expansions)
    elif solver == "lacam":
        solution = lacam(grid, agents, max_iterations=max_expansions)
    elif solver == "lns":
        solution = mapf_lns(grid, agents, iterations=100, seed=0)
    elif solver == "pbs":
        solution = pbs(grid, agents, max_nodes=max_expansions)
    elif solver == "prioritized":
        solution = prioritized_planning(grid, agents)
    elif solver == "prioritized_sipp":
        solution = prioritized_planning(grid, agents, low_level=plan_sipp)
    else:
        raise ValueError(f"unknown solver: {solver!r}")

    result = {"num_agents": len(agents), "solver": solver, "solved": solution is not None}
    if solution is not None:
        result["makespan"] = makespan(solution.paths)
        result["sum_of_costs"] = sum_of_costs(solution.paths)
    return result
