"""Launch the coverage allocator on a small two-pocket map.

Two robots are allocated to the two frontier clusters by travel cost; the node
publishes a ``geometry_msgs/PointStamped`` goal per robot on
``coverage/goal/<id>`` (latched). This is self-contained, so it doubles as a
launch smoke test that actually publishes goals.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    grid_rows = [
        "?.......?",
        "?...#...?",
        "?...#...?",
        "?...#...?",
        "?.......?",
    ]
    return LaunchDescription([
        Node(
            package="mrn_coord",
            executable="mrn_coverage_allocator",
            name="mrn_coverage_allocator",
            output="screen",
            parameters=[{
                "grid_rows": grid_rows,
                "robot_ids": ["1", "2"],
                "robot_positions": ["3,0", "5,4"],
                "method": "hungarian",
                "frame_id": "map",
                "cell_size": 1.0,
                "publish_rate_hz": 1.0,
            }],
        ),
    ])
