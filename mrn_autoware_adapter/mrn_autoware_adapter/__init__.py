"""Conservative Autoware adapter for cooperative pose corrections.

Subscribes to ``/<agent_id>/mrn/cooperative_pose``, runs the SE(2) safety
gates from :mod:`mrn_autoware_adapter.correction_gate`, and republishes
the accepted candidate as ``geometry_msgs/PoseWithCovarianceStamped`` on
an Autoware initialpose-style topic.

This adapter is intentionally narrow:

- It does not subscribe to odometry — the cooperative pose is already
  expressed in the cooperative map frame.
- It does not run Autoware's localization stack — that remains in
  Autoware. This adapter is the boundary that hands a re-localization
  hypothesis to Autoware.
- It does not modify Autoware's existing localization output.

See :mod:`docs/autoware_adapter.md` for the parameter table and topic
conventions.
"""
