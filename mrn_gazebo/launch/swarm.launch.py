"""Spawn N differential-drive robots in Gazebo and flock them.

- starts a gz server on the empty swarm world (ground + light);
- spawns ``num_robots`` vehicles on a circle, each with a unique name so its
  topics are ``/model/<name>/{pose,cmd_vel}``;
- bridges every robot's pose (gz->ROS) and cmd_vel (ROS->gz) with ros_gz_bridge;
- runs ``mrn_gz_swarm_controller``, which flocks them (Boids -> unicycle).

Optional: requires Gazebo and ros_gz; not run in CI.

    ros2 launch mrn_gazebo swarm.launch.py num_robots:=8
    ros2 launch mrn_gazebo swarm.launch.py headless:=true     # no GUI
"""

import math
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_VEHICLE = """<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{name}">
    <pose>{x:.3f} {y:.3f} 0.12 0 0 {yaw:.3f}</pose>
    <link name="base">
      <inertial><mass>2.0</mass><inertia><ixx>0.05</ixx><iyy>0.05</iyy><izz>0.08</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
      <collision name="c"><geometry><box><size>0.5 0.3 0.2</size></box></geometry></collision>
      <visual name="v"><geometry><box><size>0.5 0.3 0.2</size></box></geometry><material><ambient>0.22 0.74 0.93 1</ambient><diffuse>0.22 0.74 0.93 1</diffuse></material></visual>
    </link>
    <link name="lw"><pose>0 0.18 0 -1.5707 0 0</pose><inertial><mass>0.2</mass><inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial><collision name="c"><geometry><cylinder><radius>0.1</radius><length>0.05</length></cylinder></geometry></collision><visual name="v"><geometry><cylinder><radius>0.1</radius><length>0.05</length></cylinder></geometry></visual></link>
    <link name="rw"><pose>0 -0.18 0 -1.5707 0 0</pose><inertial><mass>0.2</mass><inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertial><collision name="c"><geometry><cylinder><radius>0.1</radius><length>0.05</length></cylinder></geometry></collision><visual name="v"><geometry><cylinder><radius>0.1</radius><length>0.05</length></cylinder></geometry></visual></link>
    <joint name="lj" type="revolute"><parent>base</parent><child>lw</child><axis><xyz>0 1 0</xyz></axis></joint>
    <joint name="rj" type="revolute"><parent>base</parent><child>rw</child><axis><xyz>0 1 0</xyz></axis></joint>
    <plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive"><left_joint>lj</left_joint><right_joint>rj</right_joint><wheel_separation>0.36</wheel_separation><wheel_radius>0.1</wheel_radius><topic>/model/{name}/cmd_vel</topic></plugin>
    <plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher"><publish_model_pose>true</publish_model_pose><publish_link_pose>false</publish_link_pose><use_pose_vector_msg>false</use_pose_vector_msg></plugin>
  </model>
</sdf>
"""


def _spawn(context):
    n = int(LaunchConfiguration("num_robots").perform(context))
    headless = LaunchConfiguration("headless").perform(context).lower() in ("1", "true", "yes")
    world = os.path.join(
        get_package_share_directory("mrn_gazebo"), "worlds", "swarm.sdf")
    gz_sim_share = get_package_share_directory("ros_gz_sim")
    radius = max(3.0, 0.8 * n)
    names = [f"robot_{i}" for i in range(n)]
    nodes = [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_sim_share, "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": f"-r {'-s' if headless else ''} {world}"}.items(),
    )]
    bridge_args = []
    for i, name in enumerate(names):
        theta = 2.0 * math.pi * i / n
        x, y = radius * math.cos(theta), radius * math.sin(theta)
        sdf = _VEHICLE.format(name=name, x=x, y=y, yaw=theta + math.pi)
        nodes.append(Node(
            package="ros_gz_sim", executable="create", output="screen",
            arguments=["-world", "swarm", "-name", name, "-string", sdf],
        ))
        bridge_args.append(f"/model/{name}/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose")
        bridge_args.append(f"/model/{name}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist")

    nodes.append(Node(
        package="ros_gz_bridge", executable="parameter_bridge",
        name="gz_bridge", output="screen", arguments=bridge_args,
    ))
    nodes.append(Node(
        package="mrn_gazebo", executable="mrn_gz_swarm_controller",
        name="mrn_gz_swarm_controller", output="screen",
        parameters=[{
            "agent_ids": names,
            "arena_half_extent": radius + 4.0,
        }],
    ))
    return nodes


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("num_robots", default_value="6"),
        DeclareLaunchArgument("headless", default_value="false"),
        OpaqueFunction(function=_spawn),
    ])
