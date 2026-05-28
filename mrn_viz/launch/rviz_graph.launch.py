from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    rviz_config = PathJoinSubstitution([
        FindPackageShare("mrn_viz"),
        "rviz",
        "cooperative_localization.rviz",
    ])

    return LaunchDescription([
        Node(
            package="rviz2",
            executable="rviz2",
            name="mrn_rviz",
            arguments=["-d", rviz_config],
            output="screen",
        ),
    ])
