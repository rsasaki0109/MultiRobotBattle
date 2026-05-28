from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    scenario = LaunchConfiguration("scenario")
    scenario_path = LaunchConfiguration("scenario_path")
    default_scenario_path = PathJoinSubstitution([
        FindPackageShare("mrn_demos"),
        "config",
        "scenarios",
        scenario,
    ])

    return LaunchDescription([
        DeclareLaunchArgument("scenario", default_value="gnss_outage_3robots.yaml"),
        DeclareLaunchArgument(
            "scenario_path",
            default_value=default_scenario_path,
            description="Absolute scenario YAML path. Defaults to mrn_demos/config/scenarios/<scenario>.",
        ),
        Node(
            package="mrn_demos",
            executable="synthetic_world_node.py",
            name="synthetic_world",
            output="screen",
            parameters=[{"scenario_path": scenario_path}],
        ),
    ])
