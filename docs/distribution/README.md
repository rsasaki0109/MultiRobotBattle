# Distribution — share MultiRobotBattle

One-line hook:

> **576-bot swarm battles driven by a benchmark-gated 45+ algorithm MAPF zoo** — swap only the movement layer (greedy → A* → CBS) and win-rate jumps from ~50% to ~67% on a terrain chokepoint. Pure Python, deterministic, runs in the browser.

Links to lead with:

- Repo: https://github.com/rsasaki0109/MultiRobotBattle
- Live demos: https://rsasaki0109.github.io/MultiRobotBattle/
- Browser battle: https://rsasaki0109.github.io/MultiRobotBattle/demo/battle.html
- MAPF solvers in browser: https://rsasaki0109.github.io/MultiRobotBattle/demo/
- Win-rate ladder: https://rsasaki0109.github.io/MultiRobotBattle/tournament.html
- Hero GIF: `docs/media/battle.gif`
- Headline demo: `docs/media/maneuver_layers.gif`
- Objective modes: `docs/media/objective_triple.gif`

## Show HN

**Title:** Show HN: Swarm robot battles powered by a 45-paper MAPF algorithm zoo

**Body:**

I built a pure-Python multi-robot battle sim where hundreds of Boids-style agents fight with emergent focus fire, terrain, and five objective modes (hill, domination, CTF, base assault, escort).

The interesting part: the same repo faithfully reproduces 45+ MAPF / multi-robot planning papers, each benchmark-gated. The battle is not a separate toy — you swap real algorithm layers (tactics, assignment, formation, maneuver) and measure win-rate differences.

Headline result on a terrain chokepoint: greedy pursuit ≈ 50% vs grid A* / prioritized MAPF ≈ 67% (12 seeds, pinned in CI).

Try it in the browser (Pyodide, no backend): https://rsasaki0109.github.io/MultiRobotBattle/demo/battle.html

Repo: https://github.com/rsasaki0109/MultiRobotBattle

Happy to answer questions on the MAPF reproductions, the battle engine, or how the layers stack.

## r/robotics

**Title:** Swarm battle sim + benchmark-gated MAPF zoo (45+ papers, pure Python, browser demos)

**Body:**

Open-source project combining:

- RoboMaster-style swarm battles (576 bots, terrain, objectives, emergent focus fire)
- A MAPF / multi-robot planning zoo reproduced from papers (CBS family, PIBT/LaCAM, lifelong, humanoid footstep planning, ORCA/BVC, …)
- CI-gated benchmarks for both solvers and battle matchups

Swap only the movement layer on a chokepoint — greedy vs A* vs prioritized MAPF — and red's win-rate goes from ~50% to ~67%. Everything is deterministic pure Python; battles and solvers run in the browser via Pyodide.

- GitHub: https://github.com/rsasaki0109/MultiRobotBattle
- Live demo: https://rsasaki0109.github.io/MultiRobotBattle/

Feedback welcome — especially on which MAPF paradigms to add next.

## Awesome-MAPF PR (draft)

Add under "Tools and Benchmarks" or "Implementations":

```markdown
- [MultiRobotBattle](https://github.com/rsasaki0109/MultiRobotBattle) — benchmark-gated MAPF zoo (45+ algorithms) plus swarm battle demos; pure Python, browser-ready Pyodide demos, CI win-rate gates.
```

Target: https://github.com/atlas-algorithm/awesome-mapf (verify org/name before opening PR).
