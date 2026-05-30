"""mrn_gazebo: an optional Gazebo (gz sim) adapter.

The 3D, physics-backed counterpart to ``mrn_sim``: it lets the rest of the stack
treat a Gazebo world as the plant. A ``ros_gz_bridge`` carries model poses and
``cmd_vel`` between gz and ROS, and :class:`gz_pose_adapter_node.GzPoseAdapter`
republishes each model's pose as the ``mrn_msgs/AgentState`` the localization
stack consumes — the same contract ``mrn_sim`` emits.

This package is **optional and not exercised in CI**: it requires Gazebo
(`gz sim`) and `ros_gz`. The pure message builder is unit-tested; the world,
bridge, and launch are documented in ``docs/gazebo.md``.
"""
