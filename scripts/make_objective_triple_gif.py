#!/usr/bin/env python3
"""Headline demo — five objective modes in one RoboMaster-style GIF.

hill · domination · CTF · base assault · escort on the same engine.

    python3 scripts/make_objective_triple_gif.py --out docs/media/objective_triple.gif
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, os.pardir, "mrn_coord"))
sys.path.insert(0, _SCRIPT_DIR)

import _battle_gif_render as render  # noqa: E402

from mrn_coord.battle import TEAM_NAMES, battle_scenario, simulate  # noqa: E402
from mrn_coord.battle_objectives import objective_zone  # noqa: E402

ZONE = "#f5cc4d"
FLAG = "#f5cc4d"
OBJECTIVES = ("hill", "domination", "ctf", "base_assault", "escort")


def _hill_dom_pct(res, cfg, fi):
    if not res.objective_progress or fi >= len(res.objective_progress):
        return 0
    prog = res.objective_progress[fi]
    if not prog:
        return 0
    hold = max(1, cfg.objective_hold_ticks)
    numeric = {k: v for k, v in prog.items() if isinstance(v, (int, float))}
    if not numeric:
        return 0
    return min(100, int(100 * max(numeric.values()) / hold))


def _assault_pct(res, cfg, fi):
    if not res.objective_progress or fi >= len(res.objective_progress):
        return 0
    prog = res.objective_progress[fi]
    hold = max(1, cfg.objective_hold_ticks)
    vals = [v for k, v in prog.items() if str(k).startswith("assault_")]
    if not vals:
        vals = [v for v in prog.values() if isinstance(v, (int, float))]
    if not vals:
        return 0
    return min(100, int(100 * max(vals) / hold))


def _escort_pct(res, fi):
    if not res.objective_progress or fi >= len(res.objective_progress):
        return 0
    prog = res.objective_progress[fi]
    if "escort_pct" in prog:
        return int(prog["escort_pct"])
    gp = prog.get("goal_progress")
    return int(round(100 * gp)) if gp is not None else 0


def _max_ticks(name):
    return 900 if name in ("ctf", "base_assault", "escort") else 650


def _draw_zones(ax, name, cfg, zones):
    if name == "ctf":
        for z in zones or []:
            if z[0] == "flag":
                _, fx, fy, fr = z
                ax.add_patch(Circle((fx, fy), fr, fill=False, edgecolor=FLAG,
                                    linewidth=1.4, linestyle=(0, (4, 4)),
                                    alpha=0.8, zorder=2))
            elif z[0] == "base":
                _, team, bx, by, br = z
                ax.add_patch(Circle((bx, by), br, fill=False,
                                    edgecolor=render.TEAM_HEX.get(team, render.MUTED),
                                    linewidth=1.6, alpha=0.75, zorder=2))
    elif name == "base_assault":
        for z in zones or []:
            if z[0] != "base":
                continue
            _, team, bx, by, br = z
            ax.add_patch(Circle((bx, by), br, fill=False,
                                edgecolor=render.TEAM_HEX.get(team, render.MUTED),
                                linewidth=1.8, alpha=0.85, zorder=2))
    elif name == "escort":
        for z in zones or []:
            tag = z[0]
            if tag == "spawn":
                _, team, bx, by, br = z
                col = render.TEAM_HEX.get(team, render.MUTED)
                ax.add_patch(Circle((bx, by), br, fill=False, edgecolor=col,
                                    linewidth=1.6, alpha=0.75, linestyle="--", zorder=2))
            elif tag in ("goal", "base"):
                _, team, bx, by, br = z
                col = render.TEAM_HEX.get(team, render.MUTED)
                lw = 2.2 if tag == "goal" else 1.4
                ax.add_patch(Circle((bx, by), br, fill=False, edgecolor=col,
                                    linewidth=lw, alpha=0.9, zorder=2))
    else:
        zone = objective_zone(cfg)
        if zone:
            cx, cy, r = zone
            ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=ZONE,
                                linewidth=1.6, linestyle=(0, (5, 4)),
                                alpha=0.85, zorder=2))


class ObjectivePanel:
    def __init__(self, ax, name):
        bots, cfg, title = battle_scenario(name)
        self.mode = name
        self.cfg = cfg
        self.res = simulate(bots, cfg, max_ticks=_max_ticks(name))
        self.deaths = render.collect_deaths(self.res.frames)
        self.zones = list(self.res.objective_zone)
        self.escort_team = getattr(cfg, "escort_team", 0)

        ax.set_facecolor(render.FIELD)
        ax.set_xlim(0, cfg.width)
        ax.set_ylim(0, cfg.height)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(render.MUTED)
        short = {
            "hill": "King of the hill",
            "domination": "Domination",
            "ctf": "Capture the flag",
            "base_assault": "Base assault",
            "escort": "Escort payload",
        }.get(name, title)
        ax.set_title(short, color=render.INK, fontsize=8, pad=4)
        render.draw_arena(ax, cfg, minimal=True)
        if getattr(cfg, "obstacles", ()) or getattr(cfg, "walls", ()):
            render.draw_terrain(ax, cfg)
        _draw_zones(ax, name, cfg, self.zones)

        self.robots = render.RobotLayers(ax, flash_life=7)
        self.prog_txt = ax.text(cfg.width / 2, cfg.height - 0.85, "",
                                ha="center", color=ZONE, fontsize=7, zorder=10)
        self.flag_scat = None
        if name == "ctf":
            self.flag_scat = ax.scatter([], [], s=55, c=FLAG, marker="D",
                                        edgecolors="#fff", linewidths=0.5, zorder=11)
        self.payload_patch = None
        if name == "escort":
            self.payload_patch = FancyBboxPatch((0, 0), 2.2, 1.5,
                                                boxstyle="round,pad=0.08",
                                                facecolor=render.TEAM_HEX.get(
                                                    self.escort_team, "#f5cc4d"),
                                                edgecolor="#f5cc4d",
                                                linewidth=1.2, alpha=0.92, zorder=8)
            ax.add_patch(self.payload_patch)
        self.banner = ax.text(cfg.width / 2, cfg.height / 2, "",
                              ha="center", va="center", fontsize=12,
                              fontweight="bold", alpha=0.0, zorder=12)


def _progress_label(panel, fi_local):
    p = panel
    if p.mode == "ctf":
        return "ctf"
    if p.mode == "base_assault":
        pct = _assault_pct(p.res, p.cfg, fi_local)
        return f"capture {pct}%" if pct else "base assault"
    if p.mode == "escort":
        return f"escort {_escort_pct(p.res, fi_local)}%"
    return f"{p.mode} {_hill_dom_pct(p.res, p.cfg, fi_local)}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media/objective_triple.gif")
    ap.add_argument("--fps", type=int, default=14)
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.8))
    fig.patch.set_facecolor(render.BG)
    fig.suptitle(
        "OBJECTIVE MODES — hill · domination · CTF · base assault · escort",
        color=render.INK, fontsize=11, fontweight="bold", y=0.98,
    )

    flat = axes.flat
    panels = [ObjectivePanel(flat[i], name) for i, name in enumerate(OBJECTIVES)]
    flat[5].axis("off")

    nframes = max(len(p.res.frames) for p in panels)
    ticks = list(range(nframes)) + [nframes - 1] * args.fps

    def update(fi):
        t = ticks[fi]
        for p in panels:
            fi_local = min(t, len(p.res.frames) - 1)
            frame = p.res.frames[fi_local]
            prev = p.res.frames[fi_local - 1] if fi_local > 0 else None
            shots = p.res.shots[fi_local] if fi_local < len(p.res.shots) else ()
            projs = (p.res.projectiles[fi_local]
                     if fi_local < len(getattr(p.res, "projectiles", ())) else ())
            p.robots.update(frame, prev, shots, projs, p.deaths, fi_local)
            p.prog_txt.set_text(_progress_label(p, fi_local))
            if p.mode == "ctf":
                prog = (p.res.objective_progress or [{}])[
                    min(fi_local, len(p.res.objective_progress) - 1)]
                fx, fy = prog.get("flag", [p.cfg.width / 2, p.cfg.height / 2])
                if p.flag_scat is not None:
                    p.flag_scat.set_offsets([[fx, fy]])
            elif p.mode == "escort" and p.payload_patch is not None:
                prog = (p.res.objective_progress or [{}])[
                    min(fi_local, len(p.res.objective_progress) - 1)]
                payload = prog.get("payload", [])
                if len(payload) == 2:
                    px, py = payload
                    p.payload_patch.set_x(px - 1.1)
                    p.payload_patch.set_y(py - 0.75)
            if p.res.winner is not None and fi_local >= len(p.res.frames) - 1:
                if p.mode == "ctf":
                    label = "CAPTURES"
                elif p.mode in ("base_assault", "escort"):
                    label = "WINS"
                else:
                    label = "WINS"
                p.banner.set_text(f"{TEAM_NAMES[p.res.winner].upper()} {label}")
                p.banner.set_color(render.TEAM_HEX[p.res.winner])
                p.banner.set_alpha(0.92)
        return []

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04,
                        hspace=0.14, wspace=0.06)
    anim = FuncAnimation(fig, update, frames=len(ticks),
                         interval=1000 // args.fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=96,
              savefig_kwargs={"facecolor": render.BG})
    plt.close(fig)
    render.optimize_gif(args.out, args.fps, colors=64)
    for p in panels:
        print(f"  {p.mode}: {TEAM_NAMES.get(p.res.winner, 'draw')} in {p.res.ticks} ticks")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
