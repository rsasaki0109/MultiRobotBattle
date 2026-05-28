#!/usr/bin/env python3
"""Generate the README hero GIF: cooperative localization in three acts.

This renders a *synthetic, deterministic* animation of the project's core
story — it is not a recording of the live ROS demo (see
``scripts/make_demo_gif.sh`` and ``docs/demo_storyboard.md`` for that). It is a
self-contained concept loop that needs no running stack, only matplotlib, so it
can be regenerated reproducibly in CI or by hand.

Three robots move in formation while exchanging V2V relative-pose constraints:

1. GNSS nominal      — all three localize tightly.
2. robot 2 outage    — robot 2 loses GNSS; its uncertainty ellipse blows up and
                       its estimate drifts.
3. cooperative fix   — V2V constraints from robots 1 and 3 pull robot 2 back;
                       its ellipse collapses again.

Usage::

    python3 scripts/make_hero_gif.py --output docs/media/cooperative_demo.gif

The animation is fully determined by the frame index (no wall-clock, no RNG
state leakage), so repeated runs produce a byte-stable GIF.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse

# --- palette (GitHub dark friendly) ----------------------------------------
BG = "#0b0e14"
PANEL = "#0d1117"
GRID = "#1b2130"
INK = "#c9d1d9"
MUTED = "#6b7689"
R1 = "#38bdf8"  # sky
R2 = "#f472b6"  # pink
R3 = "#a3e635"  # lime
OK = "#34d399"  # gnss healthy
BAD = "#fb7185"  # gnss outage
LINK = "#e2e8f0"  # v2v constraint

ROBOT_COLORS = (R1, R2, R3)
ROBOT_LABELS = ("robot 1", "robot 2", "robot 3")


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Hermite smoothstep, clamped to [0, 1]."""
    if edge0 == edge1:
        return 0.0 if x < edge0 else 1.0
    t = min(1.0, max(0.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def _truth_positions(t: float) -> np.ndarray:
    """Ground-truth positions of the three robots at phase t in [0, 1]."""
    march = 1.5 + 7.0 * t  # left-to-right travel, keeps cluster in frame
    wobble = 0.45 * np.sin(2.0 * np.pi * (t * 1.5))
    return np.array(
        [
            [march + 0.8, 2.1 + wobble],
            [march, 0.0 + 0.6 * np.sin(2.0 * np.pi * (t * 1.5 + 0.3))],
            [march - 0.6, -2.1 + wobble],
        ]
    )


def _robot2_sigma(t: float) -> float:
    """Robot 2's position uncertainty radius over the three acts."""
    base = 0.30
    grow = _smoothstep(0.30, 0.55, t)        # outage onset
    recover = _smoothstep(0.62, 0.86, t)     # cooperative recovery
    return base + 1.35 * grow - 1.25 * recover


def _robot2_drift(t: float) -> np.ndarray:
    """Estimate drift away from truth during the outage, corrected on recovery."""
    grow = _smoothstep(0.30, 0.58, t)
    recover = _smoothstep(0.62, 0.88, t)
    mag = 1.6 * grow - 1.6 * recover
    return np.array([-0.7, 0.85]) * mag


def _phase(t: float) -> tuple[str, str]:
    if t < 0.30:
        return "GNSS nominal", OK
    if t < 0.62:
        return "robot 2 — GNSS outage", BAD
    return "cooperative recovery via V2V", R2


def _glow_ellipse(ax, xy, sigma, color, *, layers=5, peak_alpha=0.30):
    patches = []
    for k in range(layers):
        scale = 1.0 + 0.42 * k
        alpha = peak_alpha * (1.0 - k / layers) ** 1.6
        e = Ellipse(
            xy,
            width=2.0 * sigma * scale,
            height=2.0 * sigma * scale,
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            zorder=2,
        )
        ax.add_patch(e)
        patches.append(e)
    ring = Ellipse(
        xy,
        width=2.0 * sigma,
        height=2.0 * sigma,
        facecolor="none",
        edgecolor=color,
        lw=1.6,
        alpha=0.9,
        zorder=3,
    )
    ax.add_patch(ring)
    patches.append(ring)
    return patches


def render(output: str, frames: int = 110, fps: int = 18) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=100)
    fig.patch.set_facecolor(BG)

    trail_len = 26
    truth_hist: list[np.ndarray] = []

    def draw(frame: int):
        ax.clear()
        t = frame / (frames - 1)

        ax.set_facecolor(PANEL)
        ax.set_xlim(0, 11)
        ax.set_ylim(-4.0, 4.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)
        for gx in range(0, 12):
            ax.axvline(gx, color=GRID, lw=0.5, alpha=0.5, zorder=0)
        for gy in range(-4, 5):
            ax.axhline(gy, color=GRID, lw=0.5, alpha=0.5, zorder=0)

        truth = _truth_positions(t)
        truth_hist.append(truth.copy())
        if len(truth_hist) > trail_len:
            truth_hist.pop(0)

        # estimated positions: robot 2 drifts during the outage
        est = truth.copy()
        est[1] = truth[1] + _robot2_drift(t)
        sig2 = _robot2_sigma(t)
        sigmas = [0.30, sig2, 0.30]

        # --- trails -------------------------------------------------------
        for i, color in enumerate(ROBOT_COLORS):
            if len(truth_hist) > 1:
                pts = np.array([h[i] for h in truth_hist])
                n = len(pts)
                for j in range(1, n):
                    ax.plot(
                        pts[j - 1:j + 1, 0],
                        pts[j - 1:j + 1, 1],
                        color=color,
                        lw=2.2,
                        alpha=0.10 + 0.5 * (j / n),
                        solid_capstyle="round",
                        zorder=1,
                    )

        # --- V2V constraint links ----------------------------------------
        recover = _smoothstep(0.60, 0.84, t)
        pulse = 0.55 + 0.45 * np.sin(2.0 * np.pi * (frame / 11.0))
        pairs = ((0, 1), (2, 1), (0, 2))
        for a, b in pairs:
            to_r2 = 1 in (a, b)
            # links to robot 2 brighten as cooperative recovery kicks in
            alpha = (0.18 + 0.6 * recover) if to_r2 else 0.16
            alpha *= 0.6 + 0.4 * pulse
            ax.plot(
                [est[a, 0], est[b, 0]],
                [est[a, 1], est[b, 1]],
                color=LINK,
                lw=1.4 if not to_r2 else (1.4 + 1.8 * recover),
                alpha=min(0.95, alpha),
                linestyle=(0, (4, 3)),
                zorder=2,
            )

        # --- robots + uncertainty ----------------------------------------
        for i, (color, label) in enumerate(zip(ROBOT_COLORS, ROBOT_LABELS)):
            _glow_ellipse(ax, est[i], sigmas[i], color)
            ax.scatter(
                [est[i, 0]], [est[i, 1]],
                s=120, color=color, edgecolor=BG, linewidth=1.4, zorder=5,
            )
            # GNSS health pip
            healthy = not (i == 1 and 0.30 <= t < 0.66)
            pip = OK if healthy else BAD
            ax.scatter(
                [est[i, 0] + 0.34], [est[i, 1] + 0.34],
                s=34, color=pip, edgecolor=BG, linewidth=0.8, zorder=6,
            )
            ax.text(
                est[i, 0], est[i, 1] - sigmas[i] - 0.42, label,
                color=color, fontsize=8.5, ha="center", va="top",
                zorder=6, weight="bold",
            )

        # --- captions -----------------------------------------------------
        title, accent = _phase(t)
        ax.text(
            0.35, 4.32, "Cooperative Localization",
            color=INK, fontsize=14, weight="bold", va="top", zorder=7,
        )
        ax.text(
            0.35, 3.62, title,
            color=accent, fontsize=11, weight="bold", va="top", zorder=7,
        )
        # progress bar
        ax.add_patch(plt.Rectangle((0.35, -3.74), 10.3, 0.12,
                                   color=GRID, zorder=7))
        ax.add_patch(plt.Rectangle((0.35, -3.74), 10.3 * t, 0.12,
                                   color=accent, zorder=8))

        # legend chips on the bottom strip (clear of the title and robots)
        ax.text(6.55, -3.55, "● GNSS fix", color=OK, fontsize=7.8,
                va="bottom", zorder=7)
        ax.text(8.35, -3.55, "● outage", color=BAD, fontsize=7.8,
                va="bottom", zorder=7)
        ax.text(9.95, -3.55, "╴╴ V2V", color=MUTED, fontsize=7.8,
                va="bottom", zorder=7)

        return ()

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, blit=False)
    anim.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)

    _optimize_gif(output, fps)
    _write_png_fallback(output)


def _optimize_gif(path: str, fps: int, colors: int = 96) -> None:
    """Re-save the GIF quantized to a small palette to shrink the file."""
    from PIL import Image

    src = Image.open(path)
    frames_list = []
    try:
        while True:
            frame = src.copy().convert("RGB").quantize(
                colors=colors, method=Image.FASTOCTREE, dither=Image.Dither.NONE
            )
            frames_list.append(frame)
            src.seek(src.tell() + 1)
    except EOFError:
        pass
    frames_list[0].save(
        path,
        save_all=True,
        append_images=frames_list[1:],
        optimize=True,
        loop=0,
        duration=int(round(1000 / fps)),
    )


def _write_png_fallback(gif_path: str) -> None:
    """Write a single representative frame next to the GIF as a .png fallback."""
    from PIL import Image

    if not gif_path.endswith(".gif"):
        return
    png_path = gif_path[:-4] + ".png"
    src = Image.open(gif_path)
    # ~45% through the loop: robot 2 mid-outage, the most illustrative frame.
    src.seek(int(src.n_frames * 0.45))
    src.convert("RGB").save(png_path)
    print(f"wrote {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="docs/media/cooperative_demo.gif",
        help="output GIF path",
    )
    parser.add_argument("--frames", type=int, default=110)
    parser.add_argument("--fps", type=int, default=18)
    args = parser.parse_args()
    render(args.output, frames=args.frames, fps=args.fps)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
