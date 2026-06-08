"""Shared RoboMaster-style matplotlib layers for battle GIF scripts."""

from __future__ import annotations

import os
import sys

import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle, Rectangle

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import _battle_robot_art as art  # noqa: E402

BG = "#06080d"
FIELD = "#0a0e16"
GRID = "#141c28"
INK = "#eef2f8"
MUTED = "#8b95a7"
FRONT = "#2a3548"
WHEEL_RGB = (0.12, 0.13, 0.16, 0.85)
FLASH_LIFE = 10

TEAM_HEX = {0: "#ff5b5b", 1: "#5b8cff", 2: "#5bd96f", 3: "#f5cc4d"}
ALLIANCE_HEX = {0: "#ff5b4a", 1: "#4d8cff"}


def collect_deaths(frames):
    deaths = {}
    for t in range(len(frames) - 1):
        for b0, b1 in zip(frames[t], frames[t + 1]):
            if b0[4] and not b1[4]:
                deaths.setdefault(t + 1, []).append((b1[0], b1[1], b1[2]))
    return deaths


def draw_arena(ax, cfg, *, minimal=False):
    """Competition-floor background with optional no-man's-land strip."""
    ax.set_facecolor(FIELD)
    step = 10 if minimal else 8
    for x in np.arange(0, cfg.width + 0.1, step):
        ax.plot([x, x], [0, cfg.height], color=GRID, lw=0.35, alpha=0.55, zorder=0)
    for y in np.arange(0, cfg.height + 0.1, step):
        ax.plot([0, cfg.width], [y, y], color=GRID, lw=0.35, alpha=0.55, zorder=0)
    if not minimal:
        ax.add_patch(Rectangle((0, 0), cfg.width, cfg.height, fill=False,
                               edgecolor="#243044", lw=2.0, zorder=0))
        mid = cfg.width / 2
        ax.add_patch(Rectangle((mid - 1.4, 0), 2.8, cfg.height, fill=True,
                               facecolor="#0c1018", edgecolor="#1a2434", lw=0.8, zorder=0))
        for y in np.arange(2, cfg.height, 7):
            ax.plot([mid - 0.55, mid + 0.55], [y, y + 2.5], color=FRONT, lw=0.7,
                    alpha=0.45, zorder=0)
        _draw_lane_markers(ax, cfg)
        for cx, cy in ((6, 6), (cfg.width - 6, 6),
                       (6, cfg.height - 6), (cfg.width - 6, cfg.height - 6)):
            ax.add_patch(Circle((cx, cy), 1.8, fill=False, edgecolor="#2e3d54",
                                lw=1.0, alpha=0.55, zorder=0))


def _draw_lane_markers(ax, cfg):
    """Rear supply pads and outer-lane chevrons (RoboMaster competition cues)."""
    w, h = cfg.width, cfg.height
    zone = "#2e3d54"
    for x0, y0 in ((w * 0.06, h * 0.22), (w * 0.06, h * 0.78),
                   (w * 0.94, h * 0.22), (w * 0.94, h * 0.78)):
        ax.add_patch(Rectangle((x0 - 2.2, y0 - 1.6), 4.4, 3.2, fill=False,
                               edgecolor=zone, lw=0.8, linestyle=(0, (3, 3)),
                               alpha=0.45, zorder=0))
    for y in (h * 0.18, h * 0.82):
        for x, dx in ((w * 0.24, 1.2), (w * 0.76, -1.2)):
            ax.plot([x, x + dx, x], [y, y + 1.4, y + 2.8], color=zone,
                    lw=0.65, alpha=0.35, zorder=0)


def draw_elevation(ax, elevation):
    """Raised platform zones — subtle contour fill beneath cover."""
    if not elevation:
        return
    for cx, cy, hw, hh, _bonus in elevation:
        ax.add_patch(Rectangle((cx - hw, cy - hh), 2 * hw, 2 * hh,
                               facecolor="#121820", edgecolor="#2a3848",
                               lw=0.7, alpha=0.55, zorder=0.5))
        inset = 0.35
        ax.add_patch(Rectangle((cx - hw + inset, cy - hh + inset),
                               2 * (hw - inset), 2 * (hh - inset),
                               fill=False, edgecolor="#3d5068",
                               lw=0.5, linestyle=(0, (4, 3)), alpha=0.45,
                               zorder=0.6))
        cap = min(hw, hh) * 0.55
        ax.add_patch(Rectangle((cx - cap * 0.5, cy + hh - cap * 0.35),
                               cap, cap * 0.18,
                               facecolor="#f5cc4d", edgecolor="none",
                               alpha=0.18, zorder=0.7))


def draw_walls(ax, walls, *, face="#1a222c", edge="#3a4a62"):
    """Rectangular bunker blocks with hazard-cap stripes (RoboMaster barriers)."""
    if not walls:
        return
    for cx, cy, hw, hh in walls:
        ax.add_patch(Rectangle((cx - hw, cy - hh), 2 * hw, 2 * hh,
                               facecolor=face, edgecolor=edge,
                               lw=1.2, zorder=1.5))
        stripe_h = min(hh * 0.28, 0.55)
        ax.add_patch(Rectangle((cx - hw * 0.92, cy + hh - stripe_h),
                               hw * 1.84, stripe_h,
                               facecolor="#f5cc4d", edgecolor="none",
                               alpha=0.32, zorder=2))
        ax.add_patch(Rectangle((cx - hw * 0.75, cy - hh * 0.55),
                               hw * 1.5, hh * 0.12,
                               facecolor="#0e1218", edgecolor="none",
                               alpha=0.5, zorder=2))


def draw_obstacles(ax, obstacles, *, face="#222a32", edge="#33405a"):
    """Bunker discs — large obstacles get a hazard-cap ring like RoboMaster cover."""
    if not obstacles:
        return
    for ox, oy, r in obstacles:
        ax.add_patch(Circle((ox, oy), r, facecolor=face, edgecolor=edge,
                            lw=1.1, zorder=1))
        if r >= 2.2:
            ax.add_patch(Circle((ox, oy), r * 0.62, facecolor="#161b24",
                                edgecolor="#2a3448", lw=0.6, zorder=2))
            cap_w = r * 0.95
            ax.add_patch(Rectangle((ox - cap_w * 0.5, oy + r * 0.42), cap_w, r * 0.22,
                                   facecolor="#f5cc4d", edgecolor="none",
                                   alpha=0.28, zorder=2))
        elif r >= 1.4:
            ax.add_patch(Circle((ox, oy), r * 0.45, fill=False,
                                edgecolor="#3d4a62", lw=0.5, zorder=2))


def draw_terrain(ax, cfg):
    """Elevation pads, wall blocks, then circular cover (bottom to top)."""
    draw_elevation(ax, getattr(cfg, "elevation", ()))
    draw_walls(ax, getattr(cfg, "walls", ()))
    draw_obstacles(ax, cfg.obstacles)


def frame_robots(frame, prev):
    hpoly, hcol, spoly, scol, wpoly, wcol, blines = [], [], [], [], [], [], []
    for i, (x, y, team, hp, alive, kind) in enumerate(frame):
        if not alive:
            continue
        k = kind or "soldier"
        if prev:
            px, py = prev[i][0], prev[i][1]
        else:
            px, py = x, y
        hd = art.infer_heading(x, y, px, py, team)
        hpoly.append(art.hull_polygon(x, y, hd, k))
        hcol.append(art.hull_face_rgba(team, hp))
        spoly.append(art.stripe_polygon(x, y, hd, k))
        scol.append(art.stripe_rgba(team, hp))
        wr = art.wheel_radius(k)
        if wr > 0:
            for wx, wy in art.wheel_offsets(x, y, hd, k):
                wpoly.append([
                    (wx - wr, wy - wr * 0.55), (wx + wr, wy - wr * 0.55),
                    (wx + wr, wy + wr * 0.55), (wx - wr, wy + wr * 0.55),
                ])
                wcol.append(WHEEL_RGB)
        blines.append(art.barrel_segment(x, y, hd, k))
    return hpoly, hcol, spoly, scol, wpoly, wcol, blines


def _apply_polys(collection, verts, colors):
    collection.set_verts(verts if verts else np.zeros((0, 4, 2)))
    collection.set_facecolors(colors if colors else [(0, 0, 0, 0)])


class RobotLayers:
    """RoboMaster chassis / turret / tracer collections bound to one axes."""

    def __init__(self, ax, *, flash_life=FLASH_LIFE, fire_alpha=0.75):
        self.ax = ax
        self.flash_life = flash_life
        self.wheels = PolyCollection([], closed=True, linewidths=0.0, zorder=3)
        self.hulls = PolyCollection([], closed=True, linewidths=0.45,
                                    edgecolors="#080a10", zorder=4)
        self.stripes = PolyCollection([], closed=True, linewidths=0.0, zorder=5)
        self.barrels = LineCollection([], linewidths=1.0, colors="#1a1d24",
                                      zorder=6, capstyle="round")
        self.fire_glow = LineCollection([], linewidths=2.4, zorder=7, alpha=0.22,
                                        capstyle="round")
        self.fire = LineCollection([], linewidths=0.9, zorder=7,
                                   alpha=fire_alpha, capstyle="round")
        self.flash = LineCollection([], linewidths=1.2, zorder=9, alpha=0.9)
        for c in (self.wheels, self.hulls, self.stripes, self.barrels,
                  self.fire_glow, self.fire, self.flash):
            ax.add_collection(c)

    def update(self, frame, prev, shots=(), deaths=None, t=0):
        hpoly, hcol, spoly, scol, wpoly, wcol, blines = frame_robots(frame, prev)
        _apply_polys(self.hulls, hpoly, hcol)
        _apply_polys(self.stripes, spoly, scol)
        _apply_polys(self.wheels, wpoly, wcol)
        self.barrels.set_segments(blines if blines else [])
        self.barrels.set_color("#22262e" if blines else "none")

        segs, glow, fcols = [], [], []
        for ax0, ay0, bx, by, team in shots:
            segs.append([(ax0, ay0), (bx, by)])
            r, g, b = art.TEAM_RGB.get(team, (0.9, 0.9, 0.9))
            glow.append((r, g, b, 0.35))
            fcols.append((min(1, r + 0.25), min(1, g + 0.25), min(1, b + 0.2), 0.9))
        self.fire.set_segments(segs)
        self.fire.set_color(fcols if fcols else "none")
        self.fire_glow.set_segments(segs)
        self.fire_glow.set_color(glow if glow else "none")

        fsegs, fcols2 = [], []
        if deaths:
            for dt in range(self.flash_life):
                born = t - dt
                for x, y, team in deaths.get(born, []):
                    age = dt / self.flash_life
                    rad = 0.35 + 1.1 * age
                    fsegs.extend([
                        [(x - rad, y), (x + rad, y)],
                        [(x, y - rad), (x, y + rad)],
                    ])
                    rr, gg, bb = art.TEAM_RGB.get(team, (1, 1, 1))
                    fcols2.extend([(rr, gg, bb, 0.85 * (1 - age))] * 2)
        self.flash.set_segments(fsegs)
        self.flash.set_color(fcols2 if fcols2 else "none")


def optimize_gif(path, fps, *, colors=96):
    try:
        from PIL import Image, ImageSequence
    except Exception:
        return
    im = Image.open(path)
    frames = [f.convert("RGB").quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
              for f in ImageSequence.Iterator(im)]
    if not frames:
        return
    frames[0].save(path, save_all=True, append_images=frames[1:], optimize=True,
                   duration=int(1000 / fps), loop=0, disposal=2)
