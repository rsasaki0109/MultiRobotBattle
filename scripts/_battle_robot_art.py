"""RoboMaster-style robot silhouettes for battle GIF renderers.

Chassis + turret + barrel drawn from position, heading, unit class, and team.
Pure geometry — no simulation coupling.
"""

from __future__ import annotations

import math

# Team accent (RoboMaster-like saturated alliance colours on dark hulls)
TEAM_RGB = {
    0: (0.96, 0.28, 0.28),   # red
    1: (0.28, 0.58, 0.98),   # blue
    2: (0.28, 0.88, 0.48),   # green
    3: (0.98, 0.82, 0.22),   # yellow
}

HULL = {
    "": (1.35, 0.95, 0.18),
    "scout": (0.95, 0.62, 0.14),
    "soldier": (1.35, 0.95, 0.18),
    "tank": (2.15, 1.55, 0.24),
    "sniper": (1.15, 0.82, 0.16),
}

BARREL = {
    "": 0.72,
    "scout": 0.55,
    "soldier": 0.78,
    "tank": 1.05,
    "sniper": 1.35,
}

TURRET_R = {
    "": 0.28,
    "scout": 0.20,
    "soldier": 0.28,
    "tank": 0.38,
    "sniper": 0.24,
}


def default_heading(team, *, western=(0, 2), eastern=(1, 3)):
    """Face the enemy flank when velocity is unknown."""
    if team in western:
        return 0.0
    if team in eastern:
        return math.pi
    return 0.0


def infer_heading(x, y, px, py, team):
    dx, dy = x - px, y - py
    if dx * dx + dy * dy > 2e-3:
        return math.atan2(dy, dx)
    return default_heading(team)


def hull_polygon(x, y, heading, kind):
    """Four corners of a rounded-rect hull in world space."""
    L, W, _ = HULL.get(kind, HULL[""])
    c, s = math.cos(heading), math.sin(heading)
    hx, hy = L * 0.5, W * 0.5
    local = ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
    out = []
    for lx, ly in local:
        out.append((x + c * lx - s * ly, y + s * lx + c * ly))
    return out


def hull_face_rgba(team, hp, *, hull_alpha=0.88):
    r, g, b = TEAM_RGB.get(team, (0.7, 0.7, 0.7))
    # Dark RoboMaster-style body with team tint; hull darkens at low HP
    base = 0.22 + 0.18 * max(0.0, min(1.0, hp))
    fr, fg, fb = base + r * 0.55, base + g * 0.55, base + b * 0.55
    return (min(1, fr), min(1, fg), min(1, fb), hull_alpha)


def stripe_polygon(x, y, heading, kind):
    """Front LED / armour stripe (team colour pop)."""
    L, W, _ = HULL.get(kind, HULL[""])
    c, s = math.cos(heading), math.sin(heading)
    fx = L * 0.42
    hw = W * 0.38
    local = ((fx, -hw), (L * 0.48, -hw * 0.55), (L * 0.48, hw * 0.55), (fx, hw))
    return [(x + c * lx - s * ly, y + s * lx + c * ly) for lx, ly in local]


def stripe_rgba(team, hp):
    r, g, b = TEAM_RGB.get(team, (0.8, 0.8, 0.8))
    a = 0.55 + 0.45 * max(0.0, min(1.0, hp))
    return (r, g, b, a)


def turret_center(x, y, heading, kind):
    L, _, _ = HULL.get(kind, HULL[""])
    c, s = math.cos(heading), math.sin(heading)
    ox = L * 0.08
    return (x + c * ox, y + s * ox)


def barrel_segment(x, y, heading, kind):
    cx, cy = turret_center(x, y, heading, kind)
    bl = BARREL.get(kind, BARREL[""])
    c, s = math.cos(heading), math.sin(heading)
    return (cx, cy), (cx + c * bl, cy + s * bl)


def wheel_offsets(x, y, heading, kind):
    """Two tread contact points for tank / soldier silhouettes."""
    L, W, _ = HULL.get(kind, HULL[""])
    if kind not in ("", "soldier", "tank"):
        return ()
    c, s = math.cos(heading), math.sin(heading)
    spread = L * 0.28
    side = W * 0.38
    pts = []
    for sign in (-1, 1):
        lx, ly = sign * spread, -side
        pts.append((x + c * lx - s * ly, y + s * lx + c * ly))
        lx, ly = sign * spread, side
        pts.append((x + c * lx - s * ly, y + s * lx + c * ly))
    return pts


def wheel_radius(kind):
    if kind == "tank":
        return 0.16
    if kind in ("", "soldier"):
        return 0.11
    return 0.0
