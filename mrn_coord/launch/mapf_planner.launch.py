"""Launch the MAPF planner on the doorway demo scenario.

Three robots cross from the left to the right through a one-cell doorway; CBS
plans collision-free paths and the node publishes one ``nav_msgs/Path`` per
agent on ``mapf/path/<id>`` (latched). This is the same scenario as
``scripts/make_coordination_gif.py``.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    blocked = [f"5,{y}" for y in range(7) if y != 3]
    return LaunchDescription([
        Node(
            package="mrn_coord",
            executable="mrn_mapf_planner",
            name="mrn_mapf_planner",
            output="screen",
            parameters=[{
                "agent_ids": ["1", "2", "3"],
                "starts": ["1,1", "1,3", "1,5"],
                "goals": ["8,5", "8,3", "8,1"],
                "grid_width": 11,
                "grid_height": 7,
                "blocked": blocked,
                "solver": "cbs",
                "frame_id": "map",
                "cell_size": 1.0,
                "publish_rate_hz": 1.0,
            }],
        ),
    ])
