"""Closed loop: a MAPF plan executed by unicycle robots in the sim world.

Runs three nodes wired on matching agent ids / grid:
- ``mrn_mapf_planner`` plans collision-free paths through a one-cell doorway and
  publishes one ``nav_msgs/Path`` per agent;
- ``mrn_path_follower`` tracks each path with pure pursuit, emitting ``cmd_vel``;
- ``mrn_sim_world`` integrates those commands (unicycle) and publishes the poses.

So the planner plans, the follower steers, and the world moves — planning →
world closed for non-holonomic robots. Set ``use_rviz:=true`` to watch it.

    ros2 launch mrn_sim mapf_through_sim.launch.py
    ros2 launch mrn_sim mapf_through_sim.launch.py use_rviz:=true
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
    agent_ids = ["1", "2", "3"]
    blocked = [f"5,{y}" for y in range(7) if y != 3]
    return LaunchDescription([
        DeclareLaunchArgument("use_rviz", default_value="false"),
        Node(
            package="mrn_sim", executable="mrn_sim_world", name="mrn_sim_world",
            output="screen",
            parameters=[{
                "agent_ids": agent_ids,
                # start at the MAPF start cells (cell_size 1.0)
                "initial_poses": ["1.0,1.0,0.0", "1.0,3.0,0.0", "1.0,5.0,0.0"],
                "obstacles": [""],          # the MAPF paths are collision-free
                "width": 11.0,
                "height": 7.0,
                "rate_hz": 20.0,
                "max_speed": 2.0,
                "publish_constraints": False,
            }],
        ),
        Node(
            package="mrn_coord", executable="mrn_mapf_planner", name="mrn_mapf_planner",
            output="screen",
            parameters=[{
                "agent_ids": agent_ids,
                "starts": ["1,1", "1,3", "1,5"],
                "goals": ["8,5", "8,3", "8,1"],
                "grid_width": 11,
                "grid_height": 7,
                "blocked": blocked,
                "solver": "cbs",
                "cell_size": 1.0,
                "frame_id": "map",
            }],
        ),
        Node(
            package="mrn_coord", executable="mrn_path_follower", name="mrn_path_follower",
            output="screen",
            parameters=[{
                "agent_ids": agent_ids,
                "path_topic_template": "mapf/path/{token}",
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
