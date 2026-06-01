"""A reusable multi-robot benchmark environment over the deterministic world.

Turn ``mrn_sim`` from a demo into a test-bed others can plug their own algorithm
into and get comparable numbers:

- :class:`Scenario` — a declarative spec (world size, obstacles, robots, goals),
  loadable from a dict/YAML.
- a **policy** — any callable ``policy(world) -> {robot_id: (v, omega)}`` (your
  planner/controller). The env steps the collision-aware world with it.
- :func:`run_scenario` — runs the closed loop and returns a
  :class:`BenchmarkResult` with standard, reproducible metrics (success,
  makespan, path length, min obstacle clearance, min inter-robot distance,
  collisions).

Everything is pure and deterministic, so a benchmark run is reproducible and
CI-checkable. Example policies (:func:`navigate_policy`) show how to plug one in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .world import Obstacle, Robot, World, step


@dataclass
class Scenario:
    """A multi-robot benchmark scenario."""

    name: str
    width: float
    height: float
    robots: dict                 # id -> (x, y, theta)
    obstacles: list = field(default_factory=list)   # (x, y, radius)
    goals: dict = field(default_factory=dict)        # id -> (x, y)
    robot_radius: float = 0.25

    def world(self) -> World:
        robots = {
            rid: Robot(rid, (p[0], p[1], p[2] if len(p) > 2 else 0.0), self.robot_radius)
            for rid, p in self.robots.items()
        }
        obstacles = [Obstacle(o[0], o[1], o[2]) for o in self.obstacles]
        return World(self.width, self.height, robots, obstacles)

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return cls(
            name=str(d.get("name", "scenario")),
            width=float(d["width"]),
            height=float(d["height"]),
            robots={str(k): tuple(v) for k, v in d["robots"].items()},
            obstacles=[tuple(o) for o in d.get("obstacles", [])],
            goals={str(k): tuple(v) for k, v in d.get("goals", {}).items()},
            robot_radius=float(d.get("robot_radius", 0.25)),
        )


def load_scenario(path: str) -> Scenario:
    """Load a :class:`Scenario` from a YAML file (yaml imported lazily)."""
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return Scenario.from_dict(yaml.safe_load(fh))


@dataclass
class BenchmarkResult:
    """Standard metrics from a benchmark run (all reproducible)."""

    scenario: str
    steps: int
    success: bool                 # all goals reached within tolerance
    goals_reached: int
    goals_total: int
    makespan_sec: float           # time until the last robot reached its goal
    path_length: dict             # id -> distance travelled
    total_path_length: float
    min_obstacle_clearance: float  # min over time of (dist to obstacle surface - radius)
    min_robot_distance: float      # min over time of pairwise center distance
    collisions: int                # steps with an obstacle entry or robot-robot overlap

    def as_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "steps": self.steps,
            "success": self.success,
            "goals_reached": self.goals_reached,
            "goals_total": self.goals_total,
            "makespan_sec": round(self.makespan_sec, 3),
            "total_path_length": round(self.total_path_length, 3),
            "min_obstacle_clearance": round(self.min_obstacle_clearance, 3),
            "min_robot_distance": round(self.min_robot_distance, 3),
            "collisions": self.collisions,
        }


def run_scenario(
    scenario: Scenario,
    policy,
    *,
    dt: float = 0.1,
    max_steps: int = 600,
    goal_tolerance: float = 0.3,
) -> BenchmarkResult:
    """Run ``policy`` on ``scenario`` and return standard metrics.

    ``policy(world) -> {robot_id: (v, omega)}`` is called each tick; the
    collision-aware world is stepped with its commands. Deterministic.
    """
    world = scenario.world()
    ids = list(world.robots)
    radius = scenario.robot_radius
    goals = scenario.goals

    path_length = {a: 0.0 for a in ids}
    reached_step = {a: None for a in ids}
    min_clear = float("inf")
    min_pair = float("inf")
    collisions = 0
    prev = {a: world.robots[a].pose for a in ids}

    def _record(w, k):
        nonlocal min_clear, min_pair, collisions
        hit = False
        for a in ids:
            x, y, _ = w.robots[a].pose
            for o in w.obstacles:
                clr = math.hypot(x - o.x, y - o.y) - o.radius - radius
                min_clear = min(min_clear, clr)
                if clr < 0.0:
                    hit = True
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pi, pj = w.robots[ids[i]].pose, w.robots[ids[j]].pose
                d = math.hypot(pi[0] - pj[0], pi[1] - pj[1])
                min_pair = min(min_pair, d)
                if d < 2.0 * radius:
                    hit = True
        # goal arrivals
        for a in ids:
            if reached_step[a] is None and a in goals:
                gx, gy = goals[a]
                if math.hypot(w.robots[a].pose[0] - gx, w.robots[a].pose[1] - gy) <= goal_tolerance:
                    reached_step[a] = k
        if hit:
            collisions += 1

    _record(world, 0)
    steps = 0
    for k in range(1, max_steps + 1):
        commands = policy(world)
        world = step(world, commands, dt)
        for a in ids:
            x, y, _ = world.robots[a].pose
            path_length[a] += math.hypot(x - prev[a][0], y - prev[a][1])
            prev[a] = world.robots[a].pose
        _record(world, k)
        steps = k
        if goals and all(reached_step[a] is not None for a in goals):
            break

    goals_reached = sum(1 for a in goals if reached_step[a] is not None)
    success = bool(goals) and goals_reached == len(goals)
    last_arrival = max((reached_step[a] for a in goals if reached_step[a] is not None),
                       default=steps)
    return BenchmarkResult(
        scenario=scenario.name,
        steps=steps,
        success=success,
        goals_reached=goals_reached,
        goals_total=len(goals),
        makespan_sec=last_arrival * dt,
        path_length=path_length,
        total_path_length=sum(path_length.values()),
        min_obstacle_clearance=(min_clear if min_clear != float("inf") else 0.0),
        min_robot_distance=(min_pair if min_pair != float("inf") else 0.0),
        collisions=collisions,
    )


def navigate_policy(scenario: Scenario, *, lookahead: float = 0.9,
                    max_speed: float = 1.6, w_mutual: float = 1.6,
                    mutual_radius: float = 1.6):
    """An example policy: per-robot A* plan + pursuit with reciprocal avoidance.

    Plans each robot's path once (grid A* around the scenario obstacles), then
    each tick pulls toward the path carrot while avoiding obstacles and the other
    robots. A turnkey baseline to benchmark against — or a template for your own.
    """
    from mrn_coord.flocking import (
        mutual_avoidance,
        obstacle_avoidance,
        velocity_to_unicycle,
    )
    from mrn_coord.mapf.path_follower import carrot_point

    from .navigate import plan_world_path

    world0 = scenario.world()
    paths = {}
    for a, g in scenario.goals.items():
        start = (world0.robots[a].pose[0], world0.robots[a].pose[1])
        paths[a] = plan_world_path(world0, start, g, cell_size=0.5, inflation=0.4)

    def policy(world):
        ids = list(world.robots)
        positions = [(world.robots[a].pose[0], world.robots[a].pose[1]) for a in ids]
        obs = obstacle_avoidance(
            positions, [(o.x, o.y, o.radius) for o in world.obstacles],
            influence=1.5, strength=2.0)
        mut = mutual_avoidance(positions, radius=mutual_radius)
        cmds = {}
        for i, a in enumerate(ids):
            path = paths.get(a)
            pose = world.robots[a].pose
            if not path:
                cmds[a] = (0.0, 0.0)
                continue
            gx, gy = path[-1]
            if math.hypot(gx - pose[0], gy - pose[1]) <= 0.3:
                cmds[a] = (0.0, 0.0)
                continue
            cx, cy = carrot_point(pose, path, lookahead)
            d = math.hypot(cx - pose[0], cy - pose[1]) or 1.0
            vx = (cx - pose[0]) / d * max_speed + 1.2 * obs[i][0] + w_mutual * mut[i][0]
            vy = (cy - pose[1]) / d * max_speed + 1.2 * obs[i][1] + w_mutual * mut[i][1]
            cmds[a] = velocity_to_unicycle(pose[2], vx, vy, max_v=max_speed, max_omega=3.0)
        return cmds

    return policy


def orca_policy(scenario: Scenario, *, lookahead: float = 1.2,
                max_speed: float = 1.5, time_horizon: float = 2.5,
                tie_break: float = 0.05):
    """Policy: A* plan + carrot, with **ORCA** reciprocal collision avoidance.

    Same plan-and-follow skeleton as :func:`navigate_policy`, but the local
    avoidance is principled ORCA (:func:`mrn_coord.orca.orca_velocity`) instead
    of summed repulsion: the preferred velocity points at the path carrot, then
    ORCA returns the closest velocity that is provably collision-free for the
    horizon given the other robots (reciprocal) and the obstacles (static). A
    tiny per-robot ``tie_break`` rotation perturbs the preferred velocity to
    dissolve the perfect-symmetry deadlock ORCA is otherwise prone to.
    """
    from mrn_coord.flocking import velocity_to_unicycle
    from mrn_coord.mapf.path_follower import carrot_point
    from mrn_coord.orca import orca_velocity

    from .navigate import plan_world_path

    world0 = scenario.world()
    radius = scenario.robot_radius
    obstacles = [(o.x, o.y, o.radius) for o in world0.obstacles]
    paths = {}
    for a, g in scenario.goals.items():
        start = (world0.robots[a].pose[0], world0.robots[a].pose[1])
        paths[a] = plan_world_path(world0, start, g, cell_size=0.5, inflation=0.4)

    ids = list(world0.robots)
    vel = {a: (0.0, 0.0) for a in ids}              # last holonomic velocity
    # Deterministic, distinct per-robot tie-break rotation (breaks symmetry).
    rot = {a: tie_break * (i - (len(ids) - 1) / 2.0) for i, a in enumerate(ids)}

    def policy(world):
        cmds = {}
        new_vel = {}
        for a in ids:
            pose = world.robots[a].pose
            path = paths.get(a)
            if not path:
                cmds[a] = (0.0, 0.0)
                new_vel[a] = (0.0, 0.0)
                continue
            gx, gy = path[-1]
            if math.hypot(gx - pose[0], gy - pose[1]) <= 0.3:
                cmds[a] = (0.0, 0.0)
                new_vel[a] = (0.0, 0.0)
                continue
            cx, cy = carrot_point(pose, path, lookahead)
            dx, dy = cx - pose[0], cy - pose[1]
            d = math.hypot(dx, dy) or 1.0
            ang = math.atan2(dy, dx) + rot[a]
            pref = (math.cos(ang) * max_speed, math.sin(ang) * max_speed)
            neighbors = [((world.robots[b].pose[0], world.robots[b].pose[1]),
                          vel[b], radius) for b in ids if b != a]
            v = orca_velocity(
                (pose[0], pose[1]), vel[a], pref, neighbors, obstacles,
                radius=radius, max_speed=max_speed, time_horizon=time_horizon)
            new_vel[a] = v
            cmds[a] = velocity_to_unicycle(pose[2], v[0], v[1],
                                           max_v=max_speed, max_omega=3.0)
        vel.update(new_vel)
        return cmds

    return policy


def _plan_for(scenario: Scenario, world0: World, planner: str, *,
              turn_radius: float, inflation: float):
    """Plan every robot's path once with the chosen global planner.

    ``planner`` is ``"grid"`` (4-connected A*) or ``"kino"`` (continuous-space
    Hybrid A* with a bounded turning radius). Returns ``{id: [(x, y), ...]}``;
    a kino plan falls back to the grid plan if it finds nothing.
    """
    from .navigate import plan_world_path

    paths = {}
    for a, g in scenario.goals.items():
        pose = world0.robots[a].pose
        start = (pose[0], pose[1])
        if planner == "kino":
            from .kinodynamic import plan_kinodynamic
            goal_yaw = math.atan2(g[1] - start[1], g[0] - start[0])
            res = plan_kinodynamic(
                world0, (start[0], start[1], pose[2]), (g[0], g[1], goal_yaw),
                turn_radius=turn_radius, robot_radius=scenario.robot_radius,
                clearance=max(0.0, inflation - scenario.robot_radius))
            paths[a] = res.waypoints if res is not None else \
                plan_world_path(world0, start, g, cell_size=0.5, inflation=inflation)
        else:
            paths[a] = plan_world_path(world0, start, g, cell_size=0.5,
                                       inflation=inflation)
    return paths


def kinodynamic_policy(scenario: Scenario, *, turn_radius: float = 1.0,
                       lookahead: float = 0.9, max_speed: float = 1.6,
                       w_mutual: float = 1.6, mutual_radius: float = 1.6,
                       inflation: float = 0.4):
    """Policy: continuous-space **Hybrid A\\*** plan + pursuit with avoidance.

    The drop-in kinodynamic counterpart of :func:`navigate_policy`: each robot's
    route is a bounded-curvature, kinematically feasible path
    (:func:`mrn_sim.kinodynamic.plan_kinodynamic`) instead of a 4-connected grid
    path. Same carrot pursuit + obstacle / reciprocal avoidance on top, so the
    two are directly comparable in the benchmark.
    """
    from mrn_coord.flocking import (
        mutual_avoidance,
        obstacle_avoidance,
        velocity_to_unicycle,
    )
    from mrn_coord.mapf.path_follower import carrot_point

    world0 = scenario.world()
    paths = _plan_for(scenario, world0, "kino",
                      turn_radius=turn_radius, inflation=inflation)

    def policy(world):
        ids = list(world.robots)
        positions = [(world.robots[a].pose[0], world.robots[a].pose[1]) for a in ids]
        obs = obstacle_avoidance(
            positions, [(o.x, o.y, o.radius) for o in world.obstacles],
            influence=1.5, strength=2.0)
        mut = mutual_avoidance(positions, radius=mutual_radius)
        cmds = {}
        for i, a in enumerate(ids):
            path = paths.get(a)
            pose = world.robots[a].pose
            if not path:
                cmds[a] = (0.0, 0.0)
                continue
            gx, gy = path[-1]
            if math.hypot(gx - pose[0], gy - pose[1]) <= 0.3:
                cmds[a] = (0.0, 0.0)
                continue
            cx, cy = carrot_point(pose, path, lookahead)
            d = math.hypot(cx - pose[0], cy - pose[1]) or 1.0
            vx = (cx - pose[0]) / d * max_speed + 1.2 * obs[i][0] + w_mutual * mut[i][0]
            vy = (cy - pose[1]) / d * max_speed + 1.2 * obs[i][1] + w_mutual * mut[i][1]
            cmds[a] = velocity_to_unicycle(pose[2], vx, vy, max_v=max_speed, max_omega=3.0)
        return cmds

    return policy


def dwa_policy(scenario: Scenario, *, planner: str = "grid",
               turn_radius: float = 1.0, lookahead: float = 1.0,
               inflation: float = 0.4, cfg=None):
    """Policy: global plan + **DWA** local control, others as moving obstacles.

    Plans each robot's route once (``planner`` = ``"grid"`` or ``"kino"``), then
    each tick tracks the path carrot with the Dynamic Window Approach
    (:func:`mrn_sim.dwa.dwa_command`) — sampling accel-limited velocities,
    forward-simulating, and scoring for goal progress and clearance. The *other
    robots* are injected into DWA's obstacle set each tick (as discs), so
    avoidance is reactive and respects the robots' acceleration limits, unlike
    the instantaneous repulsion of :func:`navigate_policy`.
    """
    from mrn_coord.mapf.path_follower import carrot_point

    from .dwa import DWAConfig, dwa_command

    cfg = cfg or DWAConfig(robot_radius=scenario.robot_radius)
    world0 = scenario.world()
    paths = _plan_for(scenario, world0, planner,
                      turn_radius=turn_radius, inflation=inflation)
    ids = list(world0.robots)
    state = {a: (0.0, 0.0) for a in ids}            # last (v, omega) per robot
    static_obs = [(o.x, o.y, o.radius) for o in world0.obstacles]

    def policy(world):
        cmds = {}
        for a in ids:
            pose = world.robots[a].pose
            path = paths.get(a)
            if not path:
                cmds[a] = (0.0, 0.0)
                state[a] = (0.0, 0.0)
                continue
            gx, gy = path[-1]
            if math.hypot(gx - pose[0], gy - pose[1]) <= cfg.goal_tolerance:
                cmds[a] = (0.0, 0.0)
                state[a] = (0.0, 0.0)
                continue
            local_goal = carrot_point(pose, path, lookahead)
            others = [(world.robots[b].pose[0], world.robots[b].pose[1],
                       scenario.robot_radius) for b in ids if b != a]
            v, omega = dwa_command(pose, state[a][0], state[a][1], local_goal,
                                   static_obs + others, world, cfg)
            state[a] = (v, omega)
            cmds[a] = (v, omega)
        return cmds

    return policy
