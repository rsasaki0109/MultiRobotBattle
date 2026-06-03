"""Token Passing with Task Swaps (TPTS) for lifelong MAPF.

Ma, Li, Kumar & Koenig, *Lifelong Multi-Agent Path Finding for Online Pickup
and Delivery Tasks* (AAAI 2017), Algorithm 2 -- the paper's second engine, the
improvement over plain Token Passing (TP, :mod:`mrn_coord.lifelong.token_passing`).

Every task here is a real **pickup -> delivery** pair, not a single goal: an
agent first drives to the pickup, *collects the package*, then drives to the
delivery. That two-leg structure is what makes the swap meaningful. A task moves
through ``open -> assigned -> executing -> done``: ``assigned`` once some agent
is en route to its pickup, ``executing`` the moment that agent reaches the
pickup and grabs the package. **Only an ``assigned`` task is swappable** -- once
a package is in hand the carrier is committed.

The swap (the whole point of TPTS): when an agent becomes free, plain TP only
lets it claim an ``open`` task. TPTS lets it also **steal** a task that is
``assigned`` to another agent that has not yet reached the pickup, whenever the
newcomer is strictly closer to that pickup. The robbed agent becomes free and
re-enters assignment. Tasks therefore migrate to better-positioned robots
instead of being frozen to whoever grabbed them first, which shortens trips and
lifts throughput -- the result the paper reports for TPTS over TP.

Motion is unchanged from TP and stays **collision-free by construction**: agents
update a shared *token* of full space-time paths one at a time, each planning a
reservation-respecting path (the package's space-time A\\*) to its current
target -- pickup if still carrying nothing, delivery once loaded -- so no
per-step rule and no fallback rollout is ever needed. Idle agents **park** on a
home endpoint; the instance must be *well-formed* (homes disjoint from task
endpoints) for the reservation search to stay live, exactly as for TP.

``swaps=False`` recovers plain two-leg Token Passing, so a single run pair
isolates exactly what the task-swap rule buys. Pure and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

from mrn_coord.mapf.space_time_astar import plan_path

from .lifelong import LifelongResult, _bfs_dist
from .token_passing import _reservations


@dataclass
class PickupDelivery:
    """A two-leg task: drive to ``pickup``, collect, then drive to ``delivery``."""

    pickup: tuple
    delivery: tuple
    released: int = 0                  # tick the task entered the system

    # mutable run state
    holder: object = None              # agent currently assigned, or None (open)
    executing: bool = False            # True once the holder has collected it
    done_at: int | None = None

    @property
    def open(self) -> bool:
        return self.holder is None and self.done_at is None

    @property
    def swappable(self) -> bool:
        # assigned to someone but not yet picked up, and not finished.
        return self.holder is not None and not self.executing and self.done_at is None


def run_tpts(
    grid,
    starts: dict,
    tasks,
    *,
    swaps: bool = True,
    max_steps: int = 256,
    keep_history: bool = False,
    horizon: int | None = None,
    homes: dict | None = None,
    stats: dict | None = None,
) -> LifelongResult:
    """Run lifelong MAPF under Token Passing with Task Swaps for ``max_steps`` ticks.

    ``starts`` maps agent id -> current cell; ``tasks`` is a list of
    :class:`PickupDelivery` pairs (released at their ``released`` tick). Agents
    collect and deliver them, moving by committing full reservation-respecting
    paths into a shared token, so motion is collision-free by construction.

    ``swaps`` toggles the defining rule: with it on (TPTS) a free agent may
    *steal* an ``assigned`` task from a farther-away holder; with it off the run
    is plain two-leg Token Passing (a free agent only takes ``open`` tasks).

    ``homes`` gives each agent a parking cell for when it is idle (default: its
    start); for liveness the instance must be **well-formed** (homes disjoint
    from every task endpoint). ``horizon`` caps per-agent planning depth.
    ``stats``, if given, is filled with run diagnostics (swaps fired, replans,
    blocked, steps). Returns a :class:`~mrn_coord.lifelong.LifelongResult` whose
    ``avg_service_time`` is measured from each task's release.
    """
    ids = sorted(starts)
    pos = {a: starts[a] for a in ids}
    home = dict(homes) if homes is not None else {a: starts[a] for a in ids}
    plan = {a: [pos[a]] for a in ids}          # committed path from now (index 0 = current)
    task_of = {a: None for a in ids}           # agent -> task it holds, or None
    elapsed = {a: 0 for a in ids}              # ticks since last (re)assignment / completion
    dist_cache: dict = {}

    def dist_to(g):
        if g not in dist_cache:
            dist_cache[g] = _bfs_dist(grid, g)
        return dist_cache[g]

    def reach(a, cell):                        # obstacle-aware distance, +inf if walled off
        return dist_to(cell).get(pos[a], None)

    H = horizon if horizon is not None else 2 * (grid.width + grid.height) + 4

    def target_of(a):
        t = task_of[a]
        if t is None:
            return home[a]
        return t.delivery if t.executing else t.pickup

    def reassign(step):
        """One fixpoint pass of (TP / TPTS) assignment over the free agents.

        A free agent claims the reachable candidate task whose pickup is nearest.
        Candidates are the ``open`` tasks always, plus -- when ``swaps`` -- any
        ``swappable`` task whose current holder is strictly farther from the
        pickup, which it then steals (freeing that holder to re-enter the pass).
        """
        fired = 0
        changed = True
        while changed:
            changed = False
            free = sorted((a for a in ids if task_of[a] is None),
                          key=lambda a: (-elapsed[a], a))
            for a in free:
                best = None                    # (cost, pickup, task, victim)
                for t in tasks:
                    if t.done_at is not None or t.released > step:
                        continue
                    d = reach(a, t.pickup)
                    if d is None:
                        continue               # pickup unreachable from here
                    if t.open:
                        victim = None
                    elif swaps and t.swappable and t.holder != a:
                        hd = dist_to(t.pickup).get(pos[t.holder], None)
                        if hd is None or d >= hd:
                            continue           # not strictly closer -> no steal
                        victim = t.holder
                    else:
                        continue
                    key = (d, t.pickup)
                    if best is None or key < best[:2]:
                        best = (d, t.pickup, t, victim)
                if best is None:
                    continue
                _, _, t, victim = best
                if victim is not None:
                    task_of[victim] = None
                    plan[victim] = [pos[victim]]
                    elapsed[victim] = 0
                    fired += 1
                t.holder = a
                t.executing = False
                task_of[a] = t
                elapsed[a] = 0
                plan[a] = [pos[a]]
                changed = True
        return fired

    swaps_fired = 0
    completed = 0
    per_agent = {a: 0 for a in ids}
    service_times: list = []
    history: list = []
    goal_history: list = []
    completions: list = []
    replans = 0
    blocked = 0

    swaps_fired += reassign(0)

    for step in range(max_steps):
        # 1. deliveries (task done) and pickups (assigned -> executing).
        done = 0
        for a in ids:
            t = task_of[a]
            if t is None:
                continue
            if t.executing and pos[a] == t.delivery:
                t.done_at = step
                t.holder = None
                task_of[a] = None
                plan[a] = [pos[a]]
                elapsed[a] = 0
                completed += 1
                per_agent[a] += 1
                service_times.append(step - t.released)
                done += 1
            elif not t.executing and pos[a] == t.pickup:
                t.executing = True             # collected -> committed, no longer swappable
                plan[a] = [pos[a]]             # replan toward the delivery
        completions.append(done)

        # 2. (re)assignment with the swap rule.
        swaps_fired += reassign(step)

        if keep_history:
            history.append(dict(pos))
            goal_history.append({a: target_of(a) for a in ids})

        # 3. token update: every agent whose committed plan is spent re-plans
        # one at a time against the others' reservations. Longest-waiting first.
        order = sorted(ids, key=lambda a: (-elapsed[a], a))
        for a in order:
            if len(plan[a]) > 1:
                continue
            target = target_of(a)
            if pos[a] == target:
                plan[a] = [pos[a]]
                continue
            vertex, edge = _reservations(plan, ids, a, H)
            p = plan_path(grid, pos[a], target, vertex, edge, max_time=H)
            replans += 1
            if p is not None and len(p) > 1:
                plan[a] = p
            else:
                plan[a] = [pos[a]]
                blocked += 1

        # 4. advance one tick along the committed plans.
        newpos = {}
        for a in ids:
            if len(plan[a]) > 1:
                newpos[a] = plan[a][1]
                plan[a] = plan[a][1:]
            else:
                newpos[a] = plan[a][0]
        pos = newpos
        for a in ids:
            elapsed[a] += 1

    # final-tick deliveries.
    final_done = 0
    for a in ids:
        t = task_of[a]
        if t is not None and t.executing and pos[a] == t.delivery:
            t.done_at = max_steps
            completed += 1
            per_agent[a] += 1
            final_done += 1
            service_times.append(max_steps - t.released)
    completions.append(final_done)
    if keep_history:
        history.append(dict(pos))
        goal_history.append({a: target_of(a) for a in ids})

    avg_service = (sum(service_times) / len(service_times)) if service_times else 0.0
    result = LifelongResult(
        steps=max_steps,
        agents=len(ids),
        completed=completed,
        throughput=(completed / max_steps if max_steps else 0.0),
        per_agent=per_agent,
        avg_service_time=avg_service,
        max_wait=(max(service_times) if service_times else 0),
        history=history,
        goal_history=goal_history,
        completions=completions,
    )
    result.replans = replans
    result.blocked = blocked
    result.swaps_fired = swaps_fired
    if stats is not None:
        stats.update(steps=max_steps, swaps_fired=swaps_fired, replans=replans,
                     blocked=blocked, completed=completed)
    return result
