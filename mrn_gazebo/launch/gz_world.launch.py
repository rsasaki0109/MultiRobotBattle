"""Launch the Gazebo world + ros_gz bridge + the pose -> AgentState adapter.

Optional: requires Gazebo (`gz sim`) and `ros_gz`. Starts a headless gz server
on the multirobot world, bridges model poses (gz -> ROS) and cmd_vel (ROS ->
gz), and runs the adapter that republishes the model pose as the
`mrn_msgs/AgentState` the localization stack consumes.

    ros2 launch mrn_gazebo gz_world.launch.py
    # then e.g. feed cooperative localization just like mrn_sim:
    #   ros2 run mrn_graph relative_anchor_graph_node.py ...
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory("mrn_gazebo")
    world = os.path.join(pkg, "worlds", "multirobot.sdf")
    bridge_config = os.path.join(pkg, "config", "gz_bridge.yaml")
    gz_sim_share = get_package_share_directory("ros_gz_sim")

    return LaunchDescription([
        # Pass extra gz args (e.g. "-s" for a headless server) via gz_args.
        DeclareLaunchArgument("gz_args", default_value=f"-r {world}"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gz_sim_share, "launch", "gz_sim.launch.py")),
            launch_arguments={"gz_args": LaunchConfiguration("gz_args")}.items(),
        ),
        Node(
            package="ros_gz_bridge", executable="parameter_bridge",
            name="gz_bridge", output="screen",
            parameters=[{"config_file": bridge_config}],
        ),
        Node(
            package="mrn_gazebo", executable="mrn_gz_pose_adapter",
            name="mrn_gz_pose_adapter", output="screen",
            parameters=[{
                "agent_ids": ["robot_1"],
                "gz_pose_topic_template": "/model/{id}/pose",
                "frame_id": "map",
            }],
        ),
    ])
