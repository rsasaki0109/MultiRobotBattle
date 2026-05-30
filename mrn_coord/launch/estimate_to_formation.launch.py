"""End-to-end: localization estimate -> coordination.

Runs the synthetic world (which publishes a per-agent ``mrn_msgs/AgentState``
estimate), bridges those estimates to ``formation/pose/<id>``, and runs the
formation controller on them. The controller publishes ``formation/cmd_vel/<id>``
computed from where localization thinks the agents are — the coordination layer
acting on the live estimate.

    ros2 launch mrn_coord estimate_to_formation.launch.py

Note this is a one-way coupling (estimate -> coordination); the synthetic world
drives its own motion and does not consume the velocity commands.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    scenario = LaunchConfiguration("scenario")
    default_scenario = os.path.join(
        get_package_share_directory("mrn_demos"),
        "config", "scenarios", "gnss_outage_3robots.yaml",
    )
    agent_ids = ["robot_1", "robot_2", "robot_3"]
    return LaunchDescription([
        DeclareLaunchArgument("scenario", default_value=default_scenario),
        Node(
            package="mrn_demos",
            executable="synthetic_world_node.py",
            name="synthetic_world",
            output="screen",
            parameters=[{"scenario_path": scenario}],
        ),
        Node(
            package="mrn_coord",
            executable="mrn_pose_bridge",
            name="mrn_pose_bridge",
            output="screen",
            parameters=[{
                "agent_ids": agent_ids,
                "source": "agent_state",
                "source_topic_template": "/{id}/mrn/agent_state",
                "target_topic_template": "formation/pose/{token}",
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
                "edges": ["robot_1,robot_2", "robot_2,robot_3", "robot_1,robot_3"],
                "gain": 1.0,
                "control_rate_hz": 10.0,
            }],
        ),
    ])
