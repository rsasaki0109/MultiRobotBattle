# Synthetic bag fixture

This directory is **not** a real rosbag2 recording. It contains only:

- `metadata.yaml` — a handwritten stub matching the rosbag2 schema. It declares
  the topics, message types, and storage identifier the experiment runner
  expects when it loads `experiments/bag_replay_smoke.yaml`.
- `manifest.yaml` — the matching MRN manifest declaring those same topics as
  required.

There are no `.mcap` payload files because no actual ROS messages were
recorded. The fixture exists so that `tools/validate_bag.py` and
`mrn_eval.experiment_cli.load_experiment_plan` can be exercised in CI without
needing a real two-robot recording.

Once a real two-robot bag exists (see `docs/bag_capture.md`), point
`experiments/bag_replay_smoke.yaml` at its directory and delete this fixture.
