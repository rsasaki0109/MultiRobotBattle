#!/usr/bin/env python3
"""Headline demo — fog + artillery: scouts spot, indirect splash in limited vision.

Dual panel: full spectator map (left) vs red-team fog vision (right), both with
splash detonation rings.

    python3 scripts/make_fog_artillery_gif.py --out docs/media/fog_artillery.gif
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, os.pardir, "mrn_coord"))
sys.path.insert(0, _SCRIPT_DIR)

import _battle_gif_render as render  # noqa: E402

from mrn_coord.battle import RED, battle_scenario, simulate  # noqa: E402


def _filter_frame(frame, visible_indices, view_team=RED):
    vis = set(visible_indices or [])
    out = []
    for i, row in enumerate(frame):
        x, y, team, hp, alive, kind = row
        if not alive:
            continue
        seen = team == view_team or i in vis
        out.append((x, y, team, hp, alive, kind, 1.0 if seen else 0.12))
    return out


def _frame_for_art(filtered):
    return [(x, y, team, hp, alive, kind) for x, y, team, hp, alive, kind, _a
            in filtered]


def _draw_explosions(ax, expls, splash_patches):
    for p in splash_patches:
        p.remove()
    splash_patches.clear()
    for ex, ey, radius, team in expls:
        col = render.TEAM_HEX.get(team, "#f5cc4d")
        patch = Circle((ex, ey), radius, fill=True, facecolor=col,
                       edgecolor=col, linewidth=1.2, alpha=0.22, zorder=7)
        ax.add_patch(patch)
        splash_patches.append(patch)
        inner = Circle((ex, ey), radius * 0.35, fill=True, facecolor="#fff3a8",
                       edgecolor="none", alpha=0.35, zorder=8)
        ax.add_patch(inner)
        splash_patches.append(inner)


class FogArtilleryPanel:
    def __init__(self, ax, cfg, res, subtitle, *, fog=False):
        self.cfg = cfg
        self.res = res
        self.subtitle = subtitle
        self.fog = fog
        self.deaths = render.collect_deaths(res.frames)
        self.splash_patches = []

        ax.set_facecolor(render.FIELD)
        ax.set_xlim(0, cfg.width)
        ax.set_ylim(0, cfg.height)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(render.MUTED)
        ax.set_title(subtitle, color=render.INK, fontsize=9)
        render.draw_arena(ax, cfg, minimal=True)
        render.draw_terrain(ax, cfg)
        self.ax = ax
        self.robots = render.RobotLayers(ax, flash_life=7)
        self.veil = ax.add_patch(Rectangle((0, 0), cfg.width, cfg.height,
                                           facecolor="#06080d", alpha=0.0, zorder=11))
        self.banner = ax.text(cfg.width / 2, cfg.height / 2, "",
                              ha="center", va="center", fontsize=16,
                              fontweight="bold", alpha=0.0, zorder=12)
        self.hud = ax.text(cfg.width / 2, cfg.height * 0.06, "",
                           ha="center", color="#9aa3bf", fontsize=8, zorder=12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media/fog_artillery.gif")
    ap.add_argument("--fps", type=int, default=14)
    args = ap.parse_args()

    bots, cfg, title = battle_scenario("fog_artillery")
    res = simulate(bots, cfg, max_ticks=900)
    nframes = len(res.frames)
    deaths = render.collect_deaths(res.frames)
    ticks = list(range(nframes)) + [nframes - 1] * args.fps

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.patch.set_facecolor(render.BG)
    fig.suptitle("FOG × ARTILLERY — scouts spot, indirect splash",
                 color=render.INK, fontsize=12, fontweight="bold", y=0.98)
    panels = [
        FogArtilleryPanel(axes[0], cfg, res, "Spectator — full map + splash", fog=False),
        FogArtilleryPanel(axes[1], cfg, res, "Red vision — fog + barrage", fog=True),
    ]

    def update(fi):
        t = ticks[fi]
        frame = res.frames[t]
        prev = res.frames[t - 1] if t > 0 else None
        shots = res.shots[t] if t < len(res.shots) else ()
        projs = res.projectiles[t] if t < len(res.projectiles) else ()
        expls = res.explosions[t] if t < len(res.explosions) else ()
        visible = res.fog_visible[t] if t < len(res.fog_visible) else []

        for panel in panels:
            if panel.fog:
                filt = _filter_frame(frame, visible, RED)
                art_frame = _frame_for_art(filt)
                prev_art = None
                if prev is not None:
                    prev_art = _frame_for_art(_filter_frame(prev, visible, RED))
                panel.robots.update(art_frame, prev_art, shots, projs, deaths, t)
                unseen = sum(1 for row in filt if row[2] != RED and row[6] < 0.5)
                panel.veil.set_alpha(0.08 if unseen else 0.0)
                spotted = len(visible)
                n_expl = len(expls)
                if n_expl:
                    panel.hud.set_text(f"{spotted} spotted · {n_expl} splash")
                elif spotted:
                    panel.hud.set_text(f"{spotted} enemies spotted")
                else:
                    panel.hud.set_text("blind advance")
            else:
                panel.robots.update(frame, prev, shots, projs, deaths, t)
                panel.hud.set_text(f"{len(expls)} splash" if expls else "")

            _draw_explosions(panel.ax, expls, panel.splash_patches)

            if res.winner is not None and t >= res.ticks - 2:
                panel.banner.set_text("RED WINS" if res.winner == RED else "BLUE WINS")
                panel.banner.set_color(render.TEAM_HEX.get(res.winner, render.INK))
                panel.banner.set_alpha(0.95)
            else:
                panel.banner.set_alpha(0.0)

        return []

    anim = FuncAnimation(fig, update, frames=len(ticks), interval=1000 / args.fps,
                         blit=False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    render.optimize_gif(args.out, args.fps)
    print("wrote", args.out, f"({res.ticks} ticks, winner={res.winner})")


if __name__ == "__main__":
    main()
