from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


_DEFAULT_NAV2_AGENT_IDS = "robot_1,robot_2,robot_3"
_DEFAULT_AUTOWARE_AGENT_IDS = "robot_1,robot_2,robot_3"


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "on"}


def _split_agent_ids(value: str) -> list[str]:
    return [agent_id.strip() for agent_id in value.split(",") if agent_id.strip()]


def _spawn_nav2_correction_nodes(context):
    enable = LaunchConfiguration("enable_nav2_correction").perform(context)
    if not _parse_bool(enable):
        return []
    agent_ids = _split_agent_ids(
        LaunchConfiguration("nav2_correction_agents").perform(context)
    )
    if not agent_ids:
        return []

    def _float(name: str) -> float:
        return float(LaunchConfiguration(name).perform(context))

    def _bool(name: str) -> bool:
        return _parse_bool(LaunchConfiguration(name).perform(context))

    params = {
        "max_pose_age_sec": _float("nav2_max_pose_age_sec"),
        "max_translation_jump_m": _float("nav2_max_translation_jump_m"),
        "max_rotation_jump_deg": _float("nav2_max_rotation_jump_deg"),
        "accept_degraded": _bool("nav2_accept_degraded"),
        "publish_rate_hz": _float("nav2_publish_rate_hz"),
    }

    nodes = []
    for agent_id in agent_ids:
        nodes.append(
            Node(
                package="mrn_nav2_adapter",
                executable="mrn_nav2_correction_broadcaster",
                name=f"nav2_correction_broadcaster_{agent_id}",
                output="screen",
                parameters=[{**params, "agent_id": agent_id}],
            )
        )
    return nodes


def _spawn_autoware_pose_publisher_nodes(context):
    enable = LaunchConfiguration("enable_autoware_correction").perform(context)
    if not _parse_bool(enable):
        return []
    agent_ids = _split_agent_ids(
        LaunchConfiguration("autoware_correction_agents").perform(context)
    )
    if not agent_ids:
        return []

    def _float(name: str) -> float:
        return float(LaunchConfiguration(name).perform(context))

    def _bool(name: str) -> bool:
        return _parse_bool(LaunchConfiguration(name).perform(context))

    params = {
        "max_pose_age_sec": _float("autoware_max_pose_age_sec"),
        "max_translation_jump_m": _float("autoware_max_translation_jump_m"),
        "max_rotation_jump_deg": _float("autoware_max_rotation_jump_deg"),
        "accept_degraded": _bool("autoware_accept_degraded"),
        "publish_rate_hz": _float("autoware_publish_rate_hz"),
    }

    nodes = []
    for agent_id in agent_ids:
        nodes.append(
            Node(
                package="mrn_autoware_adapter",
                executable="mrn_autoware_pose_publisher",
                name=f"autoware_pose_publisher_{agent_id}",
                output="screen",
                parameters=[{**params, "agent_id": agent_id}],
            )
        )
    return nodes


def generate_launch_description():
    scenario = LaunchConfiguration("scenario")
    scenario_path = LaunchConfiguration("scenario_path")
    graph_executable = LaunchConfiguration("graph_executable")
    network_profile = LaunchConfiguration("network_profile")
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

    return LaunchDescription([
        DeclareLaunchArgument("scenario", default_value="gnss_outage_3robots.yaml"),
        DeclareLaunchArgument(
            "scenario_path",
            default_value=default_scenario_path,
            description="Absolute scenario YAML path. Defaults to mrn_demos/config/scenarios/<scenario>.",
        ),
        DeclareLaunchArgument(
            "network_profile",
            default_value="",
            description="Optional mrn_netem YAML profile overriding scenario network faults.",
        ),
        DeclareLaunchArgument(
            "graph_executable",
            default_value="relative_anchor_graph_node.py",
            description="mrn_graph executable to run. Use dummy_graph_node.py for pass-through.",
        ),
        DeclareLaunchArgument("max_clock_offset_sec", default_value="0.05"),
        DeclareLaunchArgument("max_offset_uncertainty_sec", default_value="0.01"),
        DeclareLaunchArgument("reject_unknown_clock_offset", default_value="false"),
        DeclareLaunchArgument("clock_status_timeout_sec", default_value="2.0"),
        DeclareLaunchArgument(
            "enable_nav2_correction",
            default_value="false",
            description="If true, spawn mrn_nav2_correction_broadcaster for each agent in nav2_correction_agents.",
        ),
        DeclareLaunchArgument(
            "nav2_correction_agents",
            default_value=_DEFAULT_NAV2_AGENT_IDS,
            description="Comma-separated agent ids that should receive a map->odom correction broadcaster.",
        ),
        DeclareLaunchArgument("nav2_max_pose_age_sec", default_value="1.0"),
        DeclareLaunchArgument("nav2_max_translation_jump_m", default_value="1.5"),
        DeclareLaunchArgument("nav2_max_rotation_jump_deg", default_value="20.0"),
        DeclareLaunchArgument("nav2_accept_degraded", default_value="false"),
        DeclareLaunchArgument("nav2_publish_rate_hz", default_value="10.0"),
        DeclareLaunchArgument(
            "enable_autoware_correction",
            default_value="false",
            description="If true, spawn mrn_autoware_pose_publisher for each agent in autoware_correction_agents.",
        ),
        DeclareLaunchArgument(
            "autoware_correction_agents",
            default_value=_DEFAULT_AUTOWARE_AGENT_IDS,
            description="Comma-separated agent ids that should receive an Autoware-style PoseWithCovarianceStamped publisher.",
        ),
        DeclareLaunchArgument("autoware_max_pose_age_sec", default_value="1.0"),
        DeclareLaunchArgument("autoware_max_translation_jump_m", default_value="1.5"),
        DeclareLaunchArgument("autoware_max_rotation_jump_deg", default_value="20.0"),
        DeclareLaunchArgument("autoware_accept_degraded", default_value="false"),
        DeclareLaunchArgument(
            "autoware_publish_rate_hz",
            default_value="0.0",
            description="0 = emit one PoseWithCovarianceStamped per accepted cooperative pose; >0 = also periodically republish at this rate.",
        ),
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
                    "experiment_name": "gnss_outage_3robots",
                }
            ],
        ),
        OpaqueFunction(function=_spawn_nav2_correction_nodes),
        OpaqueFunction(function=_spawn_autoware_pose_publisher_nodes),
    ])
