"""Launch the formation controller for a three-robot triangle.

The node subscribes to ``formation/pose/<id>`` and publishes
``formation/cmd_vel/<id>``; with no pose publishers running it simply waits, so
this doubles as a launch smoke test. Provide pose sources (e.g. from the
localization estimate) to close the loop.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(
            package="mrn_coord",
            executable="mrn_formation_controller",
            name="mrn_formation_controller",
            output="screen",
            parameters=[{
                "agent_ids": ["1", "2", "3"],
                # equilateral triangle, radius 2
                "formation_offsets": ["2.0,0.0", "-1.0,1.732", "-1.0,-1.732"],
                "edges": ["1,2", "2,3", "1,3"],
                "gain": 1.0,
                "control_rate_hz": 10.0,
            }],
        ),
    ])
