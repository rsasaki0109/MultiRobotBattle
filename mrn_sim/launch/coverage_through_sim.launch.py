"""Closed loop: coverage allocation executed by robots in the sim world.

- ``mrn_coverage_allocator`` detects/clusters frontiers on an occupancy grid and
  publishes a ``geometry_msgs/PointStamped`` goal per robot;
- ``mrn_goal_follower`` drives each robot toward its assigned frontier;
- ``mrn_sim_world`` integrates the commands and publishes the poses.

So the allocator assigns frontiers and the robots actually go explore them —
coverage → world. (One allocation; iterative re-mapping is future work.)

    ros2 launch mrn_sim coverage_through_sim.launch.py
    ros2 launch mrn_sim coverage_through_sim.launch.py use_rviz:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    use_rviz = LaunchConfiguration("use_rviz")
    rviz_config = os.path.join(
        get_package_share_directory("mrn_sim"), "rviz", "sim.rviz"
    )
    agent_ids = ["robot_1", "robot_2"]
    # Two unknown pockets (left/right edges) -> two frontier clusters.
    grid_rows = [
        "?.......?",
        "?.......?",
        "?.......?",
        "?.......?",
        "?.......?",
    ]
    return LaunchDescription([
        DeclareLaunchArgument("use_rviz", default_value="false"),
        Node(
            package="mrn_sim", executable="mrn_sim_world", name="mrn_sim_world",
            output="screen",
            parameters=[{
                "agent_ids": agent_ids,
                "initial_poses": ["3.0,1.0,0.0", "5.0,3.0,3.14"],
                "obstacles": [""],
                "width": 9.0,
                "height": 5.0,
                "rate_hz": 20.0,
                "publish_constraints": False,
            }],
        ),
        Node(
            package="mrn_coord", executable="mrn_coverage_allocator",
            name="mrn_coverage_allocator", output="screen",
            parameters=[{
                "grid_rows": grid_rows,
                "robot_ids": agent_ids,
                "robot_positions": ["3,1", "5,3"],
                "method": "hungarian",
                "frame_id": "map",
                "cell_size": 1.0,
            }],
        ),
        Node(
            package="mrn_coord", executable="mrn_goal_follower",
            name="mrn_goal_follower", output="screen",
            parameters=[{
                "agent_ids": agent_ids,
                "goal_topic_template": "coverage/goal/{token}",
                "pose_topic_template": "/{token}/ground_truth/pose",
                "cmd_vel_topic_template": "/{token}/cmd_vel",
                "lookahead": 1.2,
                "v_nominal": 1.0,
                "goal_tolerance": 0.3,
            }],
        ),
        Node(
            package="rviz2", executable="rviz2", name="rviz2",
            arguments=["-d", rviz_config],
            condition=IfCondition(use_rviz),
        ),
    ])
