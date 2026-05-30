"""Launch the 2D world simulator alone (no controllers).

Publishes per-agent AgentState / ground truth and an RViz MarkerArray. With no
``cmd_vel`` publishers the robots stay put, so this doubles as a launch smoke
test. ``use_rviz:=true`` opens RViz on the world markers.
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
    return LaunchDescription([
        DeclareLaunchArgument("use_rviz", default_value="false"),
        Node(
            package="mrn_sim",
            executable="mrn_sim_world",
            name="mrn_sim_world",
            output="screen",
            parameters=[{
                "agent_ids": ["robot_1", "robot_2", "robot_3"],
                "initial_poses": ["1.0,1.0,0.0", "11.0,1.0,3.14", "1.0,7.0,-1.57"],
                "obstacles": ["6.0,4.0,1.3", "3.0,5.5,0.8", "9.0,5.5,0.9"],
                "width": 12.0,
                "height": 8.0,
                "rate_hz": 20.0,
            }],
        ),
        Node(
            package="rviz2", executable="rviz2", name="rviz2",
            arguments=["-d", rviz_config],
            condition=IfCondition(use_rviz),
        ),
    ])
