# Demo Media

This directory holds the README demo GIF and screenshot.

Expected files:

- `cooperative_demo.gif` — short loop (≤ 15 s) showing the synthetic 3-robot
  demo: GNSS outage on robot 2, packet loss, V2V relative constraints, and
  cooperative recovery. Targeted by the `README.md` Demo section.
- `cooperative_demo.png` — single frame fallback referenced by the same
  section for renderers that strip GIFs.

Until those assets are recorded, the README intentionally embeds the missing
files. Markdown renderers will show the alt text in that case. Do not delete
the embed — that is how the placeholder shows up as actionable in CI.

## How to (re)record

The intended capture path is:

1. Launch the demo
   ```bash
   ros2 launch mrn_demos cooperative_localization.launch.py \
     scenario:=gnss_outage_3robots.yaml
   ```
2. Open `mrn_viz/rviz_graph.launch.py` to get the cooperative RViz view.
3. Capture ~15 seconds spanning the outage so the cooperative recovery
   marker turns green. Use any screen recorder; export to GIF at 10 fps
   and ≤ 800 px wide so it stays comfortable in GitHub.
4. Save as `docs/media/cooperative_demo.gif`. Save a representative PNG
   frame as `docs/media/cooperative_demo.png`.

`scripts/make_demo_gif.sh` prints the matching commands so the procedure is
easy to reproduce.

See `docs/demo_storyboard.md` for the storyboard the recording is meant to
follow.
