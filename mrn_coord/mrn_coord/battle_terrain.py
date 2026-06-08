"""RoboMaster-style battlefield terrain — cover discs, wall blocks, elevation."""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Geometry helpers (pure — shared by battle sim, maneuver grid, tests)
# ---------------------------------------------------------------------------

def _segment_point_distance(a0, a1, c):
    """Minimum distance from segment ``a0->a1`` to point ``c``."""
    dx, dy = a1[0] - a0[0], a1[1] - a0[1]
    dd = dx * dx + dy * dy
    if dd <= 1e-15:
        return math.hypot(a0[0] - c[0], a0[1] - c[1])
    t = ((c[0] - a0[0]) * dx + (c[1] - a0[1]) * dy) / dd
    t = max(0.0, min(1.0, t))
    px, py = a0[0] + t * dx, a0[1] + t * dy
    return math.hypot(px - c[0], py - c[1])


def _segments_intersect(a, b, c, d):
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def _point_in_rect(x, y, cx, cy, hw, hh):
    return (cx - hw <= x <= cx + hw) and (cy - hh <= y <= cy + hh)


def point_clearance_rect(x, y, cx, cy, hw, hh):
    """Signed clearance to an axis-aligned rectangle (negative = inside)."""
    left = x - (cx - hw)
    right = (cx + hw) - x
    bottom = y - (cy - hh)
    top = (cy + hh) - y
    if left >= 0 and right >= 0 and bottom >= 0 and top >= 0:
        return -min(left, right, bottom, top)
    px = max(cx - hw, min(x, cx + hw))
    py = max(cy - hh, min(y, cy + hh))
    return math.hypot(x - px, y - py)


def segment_intersects_rect(ax, ay, bx, by, cx, cy, hw, hh):
    """True when the segment crosses or lies inside the rectangle."""
    if _point_in_rect(ax, ay, cx, cy, hw, hh) or _point_in_rect(bx, by, cx, cy, hw, hh):
        return True
    x0, x1 = cx - hw, cx + hw
    y0, y1 = cy - hh, cy + hh
    edges = (
        ((x0, y0), (x1, y0)),
        ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)),
        ((x0, y1), (x0, y0)),
    )
    a, b = (ax, ay), (bx, by)
    return any(_segments_intersect(a, b, e0, e1) for e0, e1 in edges)


def segment_rect_distance(ax, ay, bx, by, cx, cy, hw, hh):
    """Minimum distance from a segment to a rectangle (0 when intersecting)."""
    if segment_intersects_rect(ax, ay, bx, by, cx, cy, hw, hh):
        return 0.0
    corners = (
        (cx - hw, cy - hh), (cx + hw, cy - hh),
        (cx + hw, cy + hh), (cx - hw, cy + hh),
    )
    a, b = (ax, ay), (bx, by)
    d = min(
        max(0.0, point_clearance_rect(ax, ay, cx, cy, hw, hh)),
        max(0.0, point_clearance_rect(bx, by, cx, cy, hw, hh)),
    )
    for c in corners:
        d = min(d, _segment_point_distance(a, b, c))
    return d


def cover_along_segment_rect(ax, ay, tx, ty, cx, cy, hw, hh, cover_margin):
    """Cover factor for one rectangular blocker (0 = blocked, 1 = clear)."""
    if segment_intersects_rect(ax, ay, tx, ty, cx, cy, hw, hh):
        return 0.0
    d = segment_rect_distance(ax, ay, tx, ty, cx, cy, hw, hh)
    if d >= cover_margin:
        return 1.0
    return d / cover_margin


def push_out_of_walls(x, y, walls, *, body: float = 0.55):
    """Hard-push a point outside rectangular terrain."""
    for cx, cy, hw, hh in walls:
        left = x - (cx - hw)
        right = (cx + hw) - x
        bottom = y - (cy - hh)
        top = (cy + hh) - y
        if left > 0 and right > 0 and bottom > 0 and top > 0:
            m = min(left, right, bottom, top)
            if m == left:
                x = cx - hw - body
            elif m == right:
                x = cx + hw + body
            elif m == bottom:
                y = cy - hh - body
            else:
                y = cy + hh + body
            continue
        clr = point_clearance_rect(x, y, cx, cy, hw, hh)
        if clr < body:
            px = max(cx - hw, min(x, cx + hw))
            py = max(cy - hh, min(y, cy + hh))
            dx, dy = x - px, y - py
            d = math.hypot(dx, dy)
            if d > 1e-9:
                s = body / d
                x, y = px + dx * s, py + dy * s
            elif left <= 0:
                x = cx - hw - body
            elif right <= 0:
                x = cx + hw + body
            elif bottom <= 0:
                y = cy - hh - body
            else:
                y = cy + hh + body
    return x, y


def wall_avoidance(
    positions,
    walls,
    *,
    influence: float = 2.0,
    strength: float = 2.0,
    max_accel: float = 6.0,
) -> list:
    """Repulsion from axis-aligned wall blocks, per agent."""
    out = []
    for (px, py) in positions:
        ax = ay = 0.0
        for (cx, cy, hw, hh) in walls:
            clr = point_clearance_rect(px, py, cx, cy, hw, hh)
            if clr < influence:
                if clr < 0:
                    left = px - (cx - hw)
                    right = (cx + hw) - px
                    bottom = py - (cy - hh)
                    top = (cy + hh) - py
                    m = min(left, right, bottom, top)
                    if m == left:
                        dx, dy = -1.0, 0.0
                    elif m == right:
                        dx, dy = 1.0, 0.0
                    elif m == bottom:
                        dx, dy = 0.0, -1.0
                    else:
                        dx, dy = 0.0, 1.0
                else:
                    px_cl = max(cx - hw, min(px, cx + hw))
                    py_cl = max(cy - hh, min(py, cy + hh))
                    dx, dy = px - px_cl, py - py_cl
                    d = math.hypot(dx, dy)
                    if d <= 1e-9:
                        dx, dy = 1.0, 0.0
                        d = 1.0
                    else:
                        dx, dy = dx / d, dy / d
                w = strength / (max(abs(clr), 0.2) ** 2)
                ax += dx * w
                ay += dy * w
        mag = math.hypot(ax, ay)
        if mag > max_accel and mag > 0.0:
            ax, ay = ax / mag * max_accel, ay / mag * max_accel
        out.append((ax, ay))
    return out


def elevation_speed_mult(x, y, elevation) -> float:
    """Speed multiplier from the highest bonus zone covering ``(x, y)``."""
    mult = 1.0
    for cx, cy, hw, hh, bonus in elevation:
        if _point_in_rect(x, y, cx, cy, hw, hh):
            mult = max(mult, bonus)
    return mult


def cell_blocked_by_terrain(wx, wy, obstacles, walls, *, inflation=0.35) -> bool:
    """True when a world point lies inside inflated terrain."""
    for ox, oy, r in obstacles:
        if math.hypot(wx - ox, wy - oy) <= r + inflation:
            return True
    for cx, cy, hw, hh in walls:
        if _point_in_rect(wx, wy, cx, cy, hw + inflation, hh + inflation):
            return True
    return False


# ---------------------------------------------------------------------------
# Circular cover presets — ``(x, y, radius)`` discs
# ---------------------------------------------------------------------------

def chokepoint_obstacles(*, width: float = 40.0, height: float = 24.0) -> tuple:
    """Spawn-corner discs — centre lane is rectangular wall blocks."""
    return (
        (width * 0.20, height * 0.24, 1.5),
        (width * 0.20, height * 0.76, 1.5),
        (width * 0.80, height * 0.24, 1.5),
        (width * 0.80, height * 0.76, 1.5),
    )


def chokepoint_walls(*, width: float = 40.0, height: float = 24.0) -> tuple:
    """Compact central slabs plus offset flank bunkers — gaps stay shootable."""
    mid = width * 0.5
    return (
        (mid, height * 0.19, 0.75, 1.10),
        (mid, height * 0.50, 0.75, 1.10),
        (mid, height * 0.81, 0.75, 1.10),
        (mid - 4.2, height * 0.34, 1.00, 0.75),
        (mid + 4.2, height * 0.66, 1.00, 0.75),
    )


def chokepoint_elevation(*, width: float = 40.0, height: float = 24.0) -> tuple:
    """Raised approach pads on each side of the central choke."""
    return (
        (width * 0.36, height * 0.50, 2.8, 4.2, 1.14),
        (width * 0.64, height * 0.50, 2.8, 4.2, 1.14),
    )


def arena_obstacles(*, width: float = 40.0, height: float = 24.0) -> tuple:
    """Scattered lane cover for open-field skirmishes."""
    return (
        (width * 0.34, height * 0.28, 1.9),
        (width * 0.34, height * 0.72, 1.9),
        (width * 0.50, height * 0.50, 2.4),
        (width * 0.66, height * 0.28, 1.9),
        (width * 0.66, height * 0.72, 1.9),
        (width * 0.22, height * 0.50, 1.4),
        (width * 0.78, height * 0.50, 1.4),
    )


def arena_walls(*, width: float = 40.0, height: float = 24.0) -> tuple:
    """Open skirmish arenas use corner discs only — no mid-lane dividers."""
    return ()


def arena_elevation(*, width: float = 40.0, height: float = 24.0) -> tuple:
    """Slight high ground on rear supply lanes."""
    return (
        (width * 0.14, height * 0.50, 2.4, 5.0, 1.08),
        (width * 0.86, height * 0.50, 2.4, 5.0, 1.08),
    )


def kingdom_obstacles(*, width: float = 100.0, height: float = 56.0) -> tuple:
    """Forward berms and mid-lane bunkers for battle-line clashes."""
    return (
        (width * 0.40, height * 0.22, 2.6),
        (width * 0.40, height * 0.78, 2.6),
        (width * 0.50, height * 0.50, 3.2),
        (width * 0.60, height * 0.22, 2.6),
        (width * 0.60, height * 0.78, 2.6),
        (width * 0.28, height * 0.50, 1.8),
        (width * 0.72, height * 0.50, 1.8),
    )


def kingdom_walls(*, width: float = 100.0, height: float = 56.0) -> tuple:
    """Forward berm blocks — compact so battle lines still engage."""
    return (
        (width * 0.40, height * 0.22, 2.4, 1.05),
        (width * 0.40, height * 0.78, 2.4, 1.05),
        (width * 0.60, height * 0.22, 2.4, 1.05),
        (width * 0.60, height * 0.78, 2.4, 1.05),
        (width * 0.50, height * 0.50, 1.8, 1.6),
    )


def kingdom_elevation(*, width: float = 100.0, height: float = 56.0) -> tuple:
    """Raised command strips behind each battle line."""
    return (
        (width * 0.22, height * 0.50, 4.5, 8.0, 1.10),
        (width * 0.78, height * 0.50, 4.5, 8.0, 1.10),
    )


def total_war_obstacles(*, width: float = 140.0, height: float = 72.0) -> tuple:
    """Large competition floor — corner berms and lane chicane anchors."""
    mx = width * 0.5
    return (
        (mx - 5.5, height * 0.50, 2.4),
        (mx + 5.5, height * 0.50, 2.4),
        (width * 0.30, height * 0.11, 1.8),
        (width * 0.30, height * 0.89, 1.8),
        (width * 0.70, height * 0.11, 1.8),
        (width * 0.70, height * 0.89, 1.8),
        (width * 0.14, height * 0.33, 2.0),
        (width * 0.14, height * 0.67, 2.0),
        (width * 0.86, height * 0.33, 2.0),
        (width * 0.86, height * 0.67, 2.0),
    )


def total_war_walls(*, width: float = 140.0, height: float = 72.0) -> tuple:
    """Central barrier slabs, lane chicanes, and rear flank blocks."""
    mx = width * 0.5
    return (
        (mx, height * 0.26, 1.0, 1.85),
        (mx, height * 0.74, 1.0, 1.85),
        (width * 0.43, height * 0.17, 1.65, 0.95),
        (width * 0.43, height * 0.83, 1.65, 0.95),
        (width * 0.57, height * 0.17, 1.65, 0.95),
        (width * 0.57, height * 0.83, 1.65, 0.95),
        (mx - 9.0, height * 0.50, 2.2, 0.85),
        (mx + 9.0, height * 0.50, 2.2, 0.85),
    )


def total_war_elevation(*, width: float = 140.0, height: float = 72.0) -> tuple:
    """Contested centre platform and rear high-ground supply lanes."""
    mx = width * 0.5
    return (
        (mx, height * 0.50, 5.5, 6.5, 1.16),
        (width * 0.20, height * 0.50, 5.0, 10.0, 1.10),
        (width * 0.80, height * 0.50, 5.0, 10.0, 1.10),
    )


def objective_obstacles(*, width: float = 40.0, height: float = 24.0) -> tuple:
    """Corner cover — keeps the centre zone open for hill/CTF."""
    return (
        (width * 0.18, height * 0.30, 1.5),
        (width * 0.18, height * 0.70, 1.5),
        (width * 0.82, height * 0.30, 1.5),
        (width * 0.82, height * 0.70, 1.5),
        (width * 0.28, height * 0.50, 1.6),
        (width * 0.72, height * 0.50, 1.6),
    )


def objective_walls(*, width: float = 40.0, height: float = 24.0) -> tuple:
    """Low corner bunkers framing the open objective ring."""
    return (
        (width * 0.22, height * 0.22, 1.6, 1.0),
        (width * 0.22, height * 0.78, 1.6, 1.0),
        (width * 0.78, height * 0.22, 1.6, 1.0),
        (width * 0.78, height * 0.78, 1.6, 1.0),
    )


def chokepoint_terrain(*, width: float = 40.0, height: float = 24.0) -> dict:
    return {
        "obstacles": chokepoint_obstacles(width=width, height=height),
        "walls": chokepoint_walls(width=width, height=height),
        "elevation": chokepoint_elevation(width=width, height=height),
    }


def arena_terrain(*, width: float = 40.0, height: float = 24.0) -> dict:
    return {
        "obstacles": arena_obstacles(width=width, height=height),
        "walls": arena_walls(width=width, height=height),
        "elevation": arena_elevation(width=width, height=height),
    }


def kingdom_terrain(*, width: float = 100.0, height: float = 56.0) -> dict:
    return {
        "obstacles": kingdom_obstacles(width=width, height=height),
        "walls": kingdom_walls(width=width, height=height),
        "elevation": kingdom_elevation(width=width, height=height),
    }


def total_war_terrain(*, width: float = 140.0, height: float = 72.0) -> dict:
    return {
        "obstacles": total_war_obstacles(width=width, height=height),
        "walls": total_war_walls(width=width, height=height),
        "elevation": total_war_elevation(width=width, height=height),
    }


def objective_terrain(*, width: float = 40.0, height: float = 24.0) -> dict:
    return {
        "obstacles": objective_obstacles(width=width, height=height),
        "walls": objective_walls(width=width, height=height),
        "elevation": (),
    }
