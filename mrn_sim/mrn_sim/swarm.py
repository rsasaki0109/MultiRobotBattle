"""Flocking that actually drives the deterministic 2D world.

This closes the same loop the Gazebo swarm does — Boids → unicycle command →
world step — but in the pure, deterministic ``mrn_sim`` world, so it is
verifiable in CI (no physics engine, no DDS). It combines:

- ``mrn_coord.flocking.flock_velocities`` (separation / alignment / cohesion),
- ``mrn_coord.flocking.obstacle_avoidance`` (repulsion from the world's
  circular obstacles),
- ``mrn_coord.flocking.velocity_to_unicycle`` (holonomic desire → ``v, omega``),
- ``mrn_sim.world.step`` (unicycle kinematics + collision).

``flock_in_world`` advances one step and returns the new world plus the Boids
velocity state. Not imported by ``mrn_sim.__init__`` (it pulls in ``mrn_coord``),
so the sim core stays standalone — import it explicitly.
"""

from __future__ import annotations

import math

from mrn_coord.flocking import (
    flock_velocities,
    goal_seek,
    leader_follow,
    obstacle_avoidance,
    predator_evasion,
    velocity_to_unicycle,
)

from .world import World, step


def _wall_turn(x, y, vx, vy, width, height, margin=2.0, push=1.5):
    if x < margin:
        vx += push
    elif x > width - margin:
        vx -= push
    if y < margin:
        vy += push
    elif y > height - margin:
        vy -= push
    return vx, vy


def flock_in_world(
    world: World,
    velocities,
    *,
    dt: float = 0.1,
    perception: float = 4.0,
    separation: float = 1.4,
    max_speed: float = 1.8,
    w_obstacle: float = 1.0,
    obstacle_influence: float = 2.0,
    obstacle_strength: float = 2.0,
    goal=None,
    w_goal: float = 0.8,
    predator=None,
    predators=(),
    w_predator: float = 1.5,
    predator_influence: float = 6.0,
    leader=None,
    w_leader: float = 1.0,
    max_v: float = 1.8,
    max_omega: float = 2.5,
):
    """Advance the swarm one step; return ``(new_world, new_velocities)``.

    ``velocities`` is the Boids velocity state, one ``(vx, vy)`` per robot in
    ``world.robots`` insertion order. Optional terms: ``goal`` (migrate toward
    it), ``predator`` / ``predators`` (one or many points to flee), and
    ``leader`` (an index — followers steer toward that robot). Deterministic
    given the inputs.
    """
    ids = list(world.robots)
    positions = [(world.robots[a].pose[0], world.robots[a].pose[1]) for a in ids]
    yaws = [world.robots[a].pose[2] for a in ids]

    vel = flock_velocities(
        positions, velocities, perception=perception,
        separation=separation, max_speed=max_speed)
    obs = obstacle_avoidance(
        positions, [(o.x, o.y, o.radius) for o in world.obstacles],
        influence=obstacle_influence, strength=obstacle_strength)
    mig = (goal_seek(positions, goal, max_speed=max_speed)
           if goal is not None else [(0.0, 0.0)] * len(ids))

    # one or many predators
    all_predators = list(predators) + ([predator] if predator is not None else [])
    flee = [(0.0, 0.0)] * len(ids)
    for p in all_predators:
        ev = predator_evasion(positions, p, influence=predator_influence)
        flee = [(flee[i][0] + ev[i][0], flee[i][1] + ev[i][1]) for i in range(len(ids))]

    lead = (leader_follow(positions, leader, max_speed=max_speed)
            if leader is not None else [(0.0, 0.0)] * len(ids))

    new_vel = []
    commands = {}
    for i, a in enumerate(ids):
        vx = (vel[i][0] + w_obstacle * obs[i][0] + w_goal * mig[i][0]
              + w_predator * flee[i][0] + w_leader * lead[i][0])
        vy = (vel[i][1] + w_obstacle * obs[i][1] + w_goal * mig[i][1]
              + w_predator * flee[i][1] + w_leader * lead[i][1])
        vx, vy = _wall_turn(positions[i][0], positions[i][1], vx, vy,
                            world.width, world.height)
        # re-clamp to max_speed
        sp = math.hypot(vx, vy)
        if sp > max_speed and sp > 0.0:
            vx, vy = vx / sp * max_speed, vy / sp * max_speed
        new_vel.append((vx, vy))
        v, omega = velocity_to_unicycle(yaws[i], vx, vy, max_v=max_v, max_omega=max_omega)
        commands[a] = (v, omega)

    return step(world, commands, dt), new_vel
