"""End-to-end: the simulated world feeds the cooperative-localization graph.

Runs ``mrn_sim_world`` (which emits per-agent ``AgentState`` and V2V
``RelativePoseConstraint`` from the true world plus noise) together with the
``mrn_graph`` relative-anchor graph server, which ingests them through its
gates and publishes ``/<id>/mrn/cooperative_pose`` plus ``/mrn/graph/status``.

This is the simulation → localization data path in one command. The three
robots start in a close triangle so they are all within V2V sensing range and
constraints actually flow.

    ros2 launch mrn_sim sim_localization.launch.py
    ros2 launch mrn_sim sim_localization.launch.py use_rviz:=true
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
    agent_ids = ["robot_1", "robot_2", "robot_3"]
    return LaunchDescription([
        DeclareLaunchArgument("use_rviz", default_value="false"),
        Node(
            package="mrn_sim",
            executable="mrn_sim_world",
            name="mrn_sim_world",
            output="screen",
            parameters=[{
                "agent_ids": agent_ids,
                # a close triangle, all within the 5 m sensing radius
                "initial_poses": ["3.0,3.0,0.0", "6.0,3.0,3.14", "4.5,5.5,-1.57"],
                "obstacles": ["9.5,6.0,0.8"],
                "width": 12.0,
                "height": 8.0,
                "rate_hz": 20.0,
                "sense_radius": 5.0,
                "publish_constraints": True,
                # robot_2 has a simulated GNSS outage; cooperative localization
                # should pull its estimate back using V2V constraints from the
                # other two (which stay healthy).
                "degraded_agents": ["robot_2"],
            }],
        ),
        Node(
            package="mrn_graph",
            executable="relative_anchor_graph_node.py",
            name="graph_server",
            output="screen",
            parameters=[{
                "agent_ids": agent_ids,
                "publish_rate_hz": 20.0,
                "stale_timeout_sec": 1.0,
                "max_constraint_age_sec": 2.0,
                "map_frame": "map",
                "use_clock_offset_gate": True,
                "reject_unknown_clock_offset": False,
            }],
        ),
        Node(
            package="rviz2", executable="rviz2", name="rviz2",
            arguments=["-d", rviz_config],
            condition=IfCondition(use_rviz),
        ),
    ])
