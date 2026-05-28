# Demo Media

This directory holds the README hero GIF and screenshot.

Files:

- `cooperative_demo.gif` — the README hero loop showing the 3-robot story:
  GNSS outage on robot 2 and cooperative recovery via V2V relative-pose
  constraints. **This is a synthetic concept animation**, rendered
  deterministically by `scripts/make_hero_gif.py` from matplotlib — it is not a
  recording of the live ROS stack. Regenerate it (and the PNG below) with
  `python3 scripts/make_hero_gif.py`.
- `cooperative_demo.png` — single representative frame, written by the same
  script as a fallback for renderers that strip GIFs.

The synthetic loop is intentionally honest about being a concept illustration
(the README caption says so). A recording of the live demo is a separate,
higher-fidelity asset: see the capture procedure below and in
`scripts/make_demo_gif.sh`. When that recording exists it can replace the hero
GIF, but the synthetic generator stays as the reproducible, no-stack fallback.

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
