from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _spawn_processes(context):
    bag_dir = LaunchConfiguration("bag_dir").perform(context).strip()
    if not bag_dir:
        raise RuntimeError(
            "bag_replay.launch.py requires bag_dir:=<path> pointing at a "
            "rosbag2 directory (one containing metadata.yaml)."
        )
    play_rate = LaunchConfiguration("play_rate").perform(context).strip() or "1.0"
    storage = LaunchConfiguration("storage").perform(context).strip() or "mcap"
    extra_args_str = LaunchConfiguration("extra_play_args").perform(context).strip()
    extra_args = extra_args_str.split() if extra_args_str else []

    agent_ids_str = LaunchConfiguration("agent_ids").perform(context).strip()
    agent_ids = [
        agent_id.strip()
        for agent_id in agent_ids_str.split(",")
        if agent_id.strip()
    ] or ["robot_1", "robot_2"]

    enable_online_ate = LaunchConfiguration("enable_online_ate").perform(
        context
    ).strip().lower() in {"true", "1", "yes", "on"}

    graph_executable = LaunchConfiguration("graph_executable").perform(context).strip()
    if not graph_executable:
        graph_executable = "relative_anchor_graph_node.py"

    max_clock_offset_sec = LaunchConfiguration("max_clock_offset_sec")
    max_offset_uncertainty_sec = LaunchConfiguration("max_offset_uncertainty_sec")
    reject_unknown_clock_offset = LaunchConfiguration("reject_unknown_clock_offset")
    clock_status_timeout_sec = LaunchConfiguration("clock_status_timeout_sec")

    play_cmd = [
        "ros2",
        "bag",
        "play",
        bag_dir,
        "--rate",
        play_rate,
        "--storage",
        storage,
        "--clock",
        *extra_args,
    ]

    actions = [
        ExecuteProcess(
            cmd=play_cmd,
            name="bag_player",
            output="screen",
        ),
        Node(
            package="mrn_graph",
            executable=graph_executable,
            name="graph_server",
            output="screen",
            parameters=[
                {
                    "agent_ids": agent_ids,
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
    ]

    if enable_online_ate:
        actions.append(
            Node(
                package="mrn_eval",
                executable="mrn_online_ate",
                name="online_ate",
                output="screen",
                parameters=[
                    {
                        "agent_ids": agent_ids,
                        "publish_rate_hz": 1.0,
                        "max_samples": 2000,
                        "max_truth_age_sec": 0.25,
                        "experiment_name": "bag_replay",
                    }
                ],
            )
        )

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "bag_dir",
            default_value="",
            description="Path to a rosbag2 directory (containing metadata.yaml).",
        ),
        DeclareLaunchArgument(
            "play_rate",
            default_value="1.0",
            description="Playback rate passed to ros2 bag play --rate.",
        ),
        DeclareLaunchArgument(
            "storage",
            default_value="mcap",
            description="Storage backend hint for ros2 bag play --storage.",
        ),
        DeclareLaunchArgument(
            "extra_play_args",
            default_value="",
            description="Extra space-separated args appended to ros2 bag play.",
        ),
        DeclareLaunchArgument(
            "agent_ids",
            default_value="robot_1,robot_2",
            description="Comma-separated agent ids the graph server should bind to.",
        ),
        DeclareLaunchArgument(
            "graph_executable",
            default_value="relative_anchor_graph_node.py",
            description="mrn_graph executable to launch alongside the bag player.",
        ),
        DeclareLaunchArgument(
            "enable_online_ate",
            default_value="false",
            description=(
                "Spawn mrn_online_ate. Only useful if the bag carries "
                "/<agent_id>/ground_truth/pose; off by default for real bags."
            ),
        ),
        DeclareLaunchArgument("max_clock_offset_sec", default_value="0.05"),
        DeclareLaunchArgument("max_offset_uncertainty_sec", default_value="0.01"),
        DeclareLaunchArgument("reject_unknown_clock_offset", default_value="false"),
        DeclareLaunchArgument("clock_status_timeout_sec", default_value="2.0"),
        OpaqueFunction(function=_spawn_processes),
    ])
