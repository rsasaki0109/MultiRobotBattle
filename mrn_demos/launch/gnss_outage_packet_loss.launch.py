from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    scenario = LaunchConfiguration("scenario")
    scenario_path = LaunchConfiguration("scenario_path")
    network_profile = LaunchConfiguration("network_profile")
    graph_executable = LaunchConfiguration("graph_executable")
    max_clock_offset_sec = LaunchConfiguration("max_clock_offset_sec")
    max_offset_uncertainty_sec = LaunchConfiguration("max_offset_uncertainty_sec")
    reject_unknown_clock_offset = LaunchConfiguration("reject_unknown_clock_offset")
    clock_status_timeout_sec = LaunchConfiguration("clock_status_timeout_sec")

    default_scenario_path = PathJoinSubstitution([
        FindPackageShare("mrn_demos"),
        "config",
        "scenarios",
        scenario,
    ])
    default_network_profile = PathJoinSubstitution([
        FindPackageShare("mrn_netem"),
        "config",
        "loss20_delay80.yaml",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("scenario", default_value="gnss_outage_3robots.yaml"),
        DeclareLaunchArgument(
            "scenario_path",
            default_value=default_scenario_path,
            description="Absolute scenario YAML path. Defaults to mrn_demos/config/scenarios/<scenario>.",
        ),
        DeclareLaunchArgument("network_profile", default_value=default_network_profile),
        DeclareLaunchArgument(
            "graph_executable",
            default_value="relative_anchor_graph_node.py",
            description="mrn_graph executable to run. Use dummy_graph_node.py for pass-through.",
        ),
        DeclareLaunchArgument("max_clock_offset_sec", default_value="0.05"),
        DeclareLaunchArgument("max_offset_uncertainty_sec", default_value="0.01"),
        DeclareLaunchArgument("reject_unknown_clock_offset", default_value="false"),
        DeclareLaunchArgument("clock_status_timeout_sec", default_value="2.0"),
        Node(
            package="mrn_demos",
            executable="synthetic_world_node.py",
            name="synthetic_world",
            output="screen",
            parameters=[
                {
                    "scenario_path": scenario_path,
                    "network_profile_path": network_profile,
                }
            ],
        ),
        Node(
            package="mrn_graph",
            executable=graph_executable,
            name="graph_server",
            output="screen",
            parameters=[
                {
                    "agent_ids": ["robot_1", "robot_2", "robot_3"],
                    "publish_rate_hz": 20.0,
                    "stale_timeout_sec": 1.0,
                    "max_constraint_age_sec": 2.0,
                    "map_frame": "map",
                    "use_clock_offset_gate": True,
                    "reject_unknown_clock_offset": ParameterValue(
                        reject_unknown_clock_offset,
                        value_type=bool,
                    ),
                    "max_clock_offset_sec": ParameterValue(
                        max_clock_offset_sec,
                        value_type=float,
                    ),
                    "max_offset_uncertainty_sec": ParameterValue(
                        max_offset_uncertainty_sec,
                        value_type=float,
                    ),
                    "clock_status_timeout_sec": ParameterValue(
                        clock_status_timeout_sec,
                        value_type=float,
                    ),
                }
            ],
        ),
        Node(
            package="mrn_eval",
            executable="mrn_online_ate",
            name="online_ate",
            output="screen",
            parameters=[
                {
                    "agent_ids": ["robot_1", "robot_2", "robot_3"],
                    "publish_rate_hz": 1.0,
                    "max_samples": 2000,
                    "max_truth_age_sec": 0.25,
                    "experiment_name": "gnss_outage_packet_loss",
                }
            ],
        ),
    ])
