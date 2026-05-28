#!/usr/bin/env bash
# Print the manual capture procedure for docs/media/cooperative_demo.gif.
#
# Automated capture is intentionally out of scope: the assets are user-facing
# and need a human eye for cropping, framing, and timing. This script just
# documents the reproducible commands so two contributors produce comparable
# GIFs.
set -euo pipefail

cat <<'EOF'
Cooperative demo GIF capture procedure
======================================

In one terminal:

  source /opt/ros/jazzy/setup.bash
  source install/setup.bash
  ros2 launch mrn_demos cooperative_localization.launch.py \
    scenario:=gnss_outage_3robots.yaml

In a second terminal:

  source /opt/ros/jazzy/setup.bash
  source install/setup.bash
  ros2 launch mrn_viz rviz_graph.launch.py

Then capture about 15 seconds spanning the GNSS outage. Suggested workflow:

  - screen recorder of choice -> raw .mp4
  - ffmpeg -i raw.mp4 -vf "fps=10,scale=800:-1:flags=lanczos" \
           -loop 0 docs/media/cooperative_demo.gif
  - ffmpeg -i raw.mp4 -vf "scale=800:-1:flags=lanczos" -frames:v 1 \
           docs/media/cooperative_demo.png

The storyboard the recording should follow lives in
docs/demo_storyboard.md.
EOF
