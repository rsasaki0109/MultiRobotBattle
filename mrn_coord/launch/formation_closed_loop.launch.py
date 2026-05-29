"""Closed-loop formation demo, entirely inside ROS.

Runs the kinematic agent simulator and the formation controller together:
the sim publishes poses, the controller answers with velocity commands, the sim
integrates them, and the three agents converge into a triangle. Set
``use_rviz:=true`` to also open RViz on the agent markers.

    ros2 launch mrn_coord formation_closed_loop.launch.py
    ros2 launch mrn_coord formation_closed_loop.launch.py use_rviz:=true
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
        get_package_share_directory("mrn_coord"), "rviz", "coordination.rviz"
    )
    agent_ids = ["1", "2", "3"]
    return LaunchDescription([
        DeclareLaunchArgument("use_rviz", default_value="false"),
        Node(
            package="mrn_coord",
            executable="mrn_agent_sim",
            name="mrn_agent_sim",
            output="screen",
            parameters=[{
                "agent_ids": agent_ids,
                "initial_positions": ["0.0,0.0", "5.0,1.0", "1.0,4.0"],
                "rate_hz": 20.0,
            }],
        ),
        Node(
            package="mrn_coord",
            executable="mrn_formation_controller",
            name="mrn_formation_controller",
            output="screen",
            parameters=[{
                "agent_ids": agent_ids,
                "formation_offsets": ["2.0,0.0", "-1.0,1.732", "-1.0,-1.732"],
                "edges": ["1,2", "2,3", "1,3"],
                "gain": 1.2,
                "control_rate_hz": 20.0,
            }],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
            condition=IfCondition(use_rviz),
        ),
    ])
