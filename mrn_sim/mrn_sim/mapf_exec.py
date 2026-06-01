"""Execute a discrete MAPF plan in the continuous world — plan vs. reality.

A MAPF solver (``mrn_coord.mapf``) returns paths that are collision-free *on the
grid, in discrete time*: no two agents share a cell or swap at any integer step.
But robots are discs with non-holonomic kinematics moving in continuous time, so
a coordinated grid plan is only as good as its execution. This module closes the
loop end to end: it turns a grid plan into continuous waypoints, drops the agents
into the deterministic :mod:`mrn_sim` world (grid obstacles become circular
ones), drives each robot along *its own* planned path, and measures the gap
between the discrete guarantee and what actually happens.

The headline is the comparison of how the *same* plan is executed:

- ``"pursuit"`` — free-running pure pursuit that ignores the plan's *timing*:
  it keeps the spatial route but discards the schedule, so robots reach a shared
  cell at the same wall-clock moment and their discs collide. This is the gap.
- ``"tpg"`` — pure pursuit gated by a **Temporal Plan Graph**: from the plan we
  extract, for every cell, the order in which agents occupy it, and a robot may
  advance into its next cell only once the previous occupant has left. That
  precedence makes the discrete coordination transfer to continuous time —
  collision-free by construction (with cell size >= 2·radius), at the cost of
  some makespan stretch while robots wait for kinematics. This is how you bridge
  the gap.
- ``"dwa"`` — a reactive alternative: keep the spatial route but treat the other
  robots as moving obstacles, recovering safety without the schedule.
- ``"shield"`` — free-running pure pursuit (the *same* schedule-ignoring nominal
  as ``"pursuit"``) with the **certified safety shield** (:mod:`mrn_sim.shield`)
  underneath: the nominal still drives robots into the shared cell together, but
  the shield's braking cap keeps their bodies provably apart. It shows the
  discrete-to-continuous gap closed by a *runtime guarantee* rather than by
  re-deriving the schedule (TPG) or by a soft reactive cost (DWA).

Pure and deterministic; reuses the planners, controllers, and world already in
the stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .kinematics import normalize_angle
from .world import Obstacle, Robot, World, step


@dataclass
class ExecResult:
    """Metrics from executing a MAPF plan in the continuous world."""

    solved: bool
    success: bool                 # every robot reached its goal
    robot_collisions: int         # steps with any robot-robot disc overlap
    discrete_makespan: int        # the grid plan's makespan (steps)
    continuous_steps: int         # steps the execution actually took
    makespan_sec: float           # continuous time to the last arrival
    max_path_deviation: float     # furthest a robot strayed from its planned line

    def as_dict(self) -> dict:
        return {
            "solved": self.solved,
            "success": self.success,
            "robot_collisions": self.robot_collisions,
            "discrete_makespan": self.discrete_makespan,
            "continuous_steps": self.continuous_steps,
            "makespan_sec": round(self.makespan_sec, 3),
            "max_path_deviation": round(self.max_path_deviation, 3),
        }


def _center(c, cell_size):
    return ((c[0] + 0.5) * cell_size, (c[1] + 0.5) * cell_size)


def _world_from_grid(grid, cell_size, robot_radius, starts):
    obstacles = [Obstacle((x + 0.5) * cell_size, (y + 0.5) * cell_size,
                          0.5 * cell_size)
                 for (x, y) in grid.blocked]
    robots = {a: Robot(a, (*_center(c, cell_size), 0.0), robot_radius)
              for a, c in starts.items()}
    return World(grid.width * cell_size, grid.height * cell_size, robots,
                 obstacles)


def _milestones(paths):
    """Per-agent ordered distinct cells (waits collapsed) and arrival steps."""
    cells, steps = {}, {}
    for a, path in paths.items():
        ms, st = [], []
        for t, c in enumerate(path):
            if not ms or ms[-1] != c:
                ms.append(c)
                st.append(t)
        cells[a], steps[a] = ms, st
    return cells, steps


def _build_tpg(paths):
    """Temporal Plan Graph precedence: ``prereq[(a, k)] = (p, mp)`` means agent
    ``a`` may enter its milestone ``k`` only once agent ``p`` has reached its
    milestone ``mp`` (i.e. left the shared cell)."""
    cells, steps = _milestones(paths)
    occ: dict = {}
    for a in paths:
        for k, c in enumerate(cells[a]):
            occ.setdefault(c, []).append((steps[a][k], a, k))
    prereq: dict = {}
    for c, lst in occ.items():
        lst.sort()                                   # by arrival step (distinct)
        for idx in range(1, len(lst)):
            _, a, k = lst[idx]
            _, p, kp = lst[idx - 1]
            if p != a:
                prereq[(a, k)] = (p, kp + 1)         # a enters after p departs
    return cells, prereq


def _goto(pose, target, *, v_nominal=1.0, max_omega=3.0, slow_radius=0.6):
    """Simple go-to-point unicycle command toward ``target``."""
    dx, dy = target[0] - pose[0], target[1] - pose[1]
    dist = math.hypot(dx, dy)
    heading_err = normalize_angle(math.atan2(dy, dx) - pose[2])
    omega = max(-max_omega, min(max_omega, 2.0 * heading_err))
    if abs(heading_err) > 1.2:                       # turn in place first
        return (0.0, omega)
    v = v_nominal * min(1.0, dist / slow_radius) * max(0.0, math.cos(heading_err))
    return (v, omega)


def execute_mapf_plan(grid, agents, *, solver="lacam", solution=None,
                      controller="tpg", cell_size=1.0, robot_radius=0.2,
                      lookahead=1.0, dt=0.1, max_steps=None):
    """Solve a MAPF instance (or take ``solution``) and execute it in the world.

    ``agents`` maps id -> ``(start_cell, goal_cell)``. ``controller`` is
    ``"pursuit"`` (free-running, ignores schedule), ``"tpg"`` (schedule-gated,
    collision-free by construction), ``"dwa"`` (reactive), or ``"shield"``
    (free-running pursuit under the certified safety shield). Returns an
    :class:`ExecResult` with the discrete-vs-continuous metrics.
    """
    from mrn_coord.mapf import cbs, ecbs, lacam, mapf_lns, prioritized_planning
    from mrn_coord.mapf.path_follower import carrot_point, pure_pursuit
    from mrn_coord.mapf.solution import makespan

    if solution is None:
        solution = {
            "cbs": lambda: cbs(grid, agents),
            "ecbs": lambda: ecbs(grid, agents, w=1.5),
            "lacam": lambda: lacam(grid, agents),
            "lns": lambda: mapf_lns(grid, agents, iterations=50),
            "prioritized": lambda: prioritized_planning(grid, agents),
        }[solver]()
    if solution is None:
        return ExecResult(False, False, 0, 0, 0, 0.0, 0.0)

    ids = list(agents)
    starts = {a: agents[a][0] for a in ids}
    goals = {a: agents[a][1] for a in ids}
    wpts = {a: [_center(c, cell_size) for c in solution.paths[a]] for a in ids}
    goal_xy = {a: _center(goals[a], cell_size) for a in ids}
    disc_makespan = makespan(solution.paths)
    if max_steps is None:
        max_steps = (disc_makespan + grid.width + grid.height + 10) * 8

    world = _world_from_grid(grid, cell_size, robot_radius, starts)
    state = {a: (0.0, 0.0) for a in ids}
    static_obs = [(o.x, o.y, o.radius) for o in world.obstacles]
    goal_tol = 0.3 * cell_size

    ms_cells = prereq = done = None
    if controller == "tpg":
        ms_cells, prereq = _build_tpg(solution.paths)
        done = {a: 0 for a in ids}
    dwa_cfg = None
    if controller == "dwa":
        from .dwa import DWAConfig
        dwa_cfg = DWAConfig(robot_radius=robot_radius, goal_tolerance=goal_tol)
        from .dwa import dwa_command
    shield_cfg = None
    if controller == "shield":
        from .shield import ShieldConfig, shield_step
        shield_cfg = ShieldConfig(robot_radius=robot_radius)

    collisions = 0
    max_dev = 0.0
    arrived = {}
    stepnum = 0
    for stepnum in range(max_steps):
        poses = {a: world.robots[a].pose for a in ids}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pi, pj = poses[ids[i]], poses[ids[j]]
                if math.hypot(pi[0] - pj[0], pi[1] - pj[1]) < 2 * robot_radius:
                    collisions += 1
        for a in ids:
            if len(wpts[a]) > 1:
                max_dev = max(max_dev, min(
                    _seg_dist(poses[a], wpts[a][i], wpts[a][i + 1])
                    for i in range(len(wpts[a]) - 1)))
            if a not in arrived and math.hypot(
                    goal_xy[a][0] - poses[a][0],
                    goal_xy[a][1] - poses[a][1]) <= goal_tol:
                arrived[a] = stepnum
        if len(arrived) == len(ids):
            break

        cmds = {}
        for a in ids:
            pose = poses[a]
            if controller == "tpg":
                cells = ms_cells[a]
                M = len(cells) - 1
                if done[a] >= M:
                    cmds[a] = (0.0, 0.0)
                    continue
                nk = done[a] + 1
                allowed = True
                if (a, nk) in prereq:
                    p, mp = prereq[(a, nk)]
                    allowed = done[p] >= mp
                tk = nk if allowed else done[a]
                target = _center(cells[tk], cell_size)
                cmds[a] = _goto(pose, target, max_omega=3.0)
                if allowed and math.hypot(target[0] - pose[0],
                                          target[1] - pose[1]) <= goal_tol:
                    done[a] = nk
                continue
            if a in arrived:
                cmds[a] = (0.0, 0.0)
                continue
            if controller == "pursuit":
                v, omega, _ = pure_pursuit(pose, wpts[a], lookahead=lookahead,
                                           v_nominal=1.0, goal_tolerance=goal_tol)
            elif controller == "shield":
                # same schedule-ignoring nominal as "pursuit", but the certified
                # shield rides underneath: others enter as moving obstacle discs.
                v, omega, _ = pure_pursuit(pose, wpts[a], lookahead=lookahead,
                                           v_nominal=1.0, goal_tolerance=goal_tol)
                others = [(poses[b][0], poses[b][1], robot_radius,
                           state[b][0] * math.cos(poses[b][2]),
                           state[b][0] * math.sin(poses[b][2]))
                          for b in ids if b != a]
                v, omega = shield_step((pose[0], pose[1], pose[2], state[a][0]),
                                       (v, omega), static_obs + others, dt,
                                       shield_cfg)
            else:  # dwa
                carrot = carrot_point(pose, wpts[a], lookahead)
                others = [(poses[b][0], poses[b][1], robot_radius)
                          for b in ids if b != a]
                v, omega = dwa_command(pose, state[a][0], state[a][1], carrot,
                                       static_obs + others, world, dwa_cfg)
            state[a] = (v, omega)
            cmds[a] = (v, omega)

        world = step(world, cmds, dt)

    if controller == "tpg":
        success = all(math.hypot(goal_xy[a][0] - world.robots[a].pose[0],
                                 goal_xy[a][1] - world.robots[a].pose[1])
                      <= goal_tol for a in ids)
    else:
        success = len(arrived) == len(ids)
    makespan_sec = (max(arrived.values()) * dt) if (arrived and
                    len(arrived) == len(ids)) else (stepnum * dt if success
                                                    else 0.0)
    return ExecResult(
        solved=True,
        success=success,
        robot_collisions=collisions,
        discrete_makespan=disc_makespan,
        continuous_steps=stepnum,
        makespan_sec=makespan_sec,
        max_path_deviation=max_dev,
    )


def _seg_dist(p, a, b):
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    seg = dx * dx + dy * dy
    if seg < 1e-12:
        return math.hypot(p[0] - ax, p[1] - ay)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / seg))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))
