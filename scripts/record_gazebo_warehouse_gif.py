#!/usr/bin/env python3
"""Record the Gazebo (gz sim) 3D **warehouse AMR fleet** demo GIF.

The 3D counterpart of ``make_warehouse_gif.py``: a fleet of autonomous mobile
robots works a shelf-and-aisle warehouse, taking an endless stream of
pickup/dropoff tasks. The lifelong-MAPF schedule is **precomputed by mrn_coord**
(the same ``run_lifelong`` / PIBT the 2D demo uses) and replayed in 3D — each
robot tracks its precomputed cell-to-cell waypoints over ``cmd_vel`` through the
kinematic ``VelocityControl`` system, and carries a 360° LiDAR whose returns are
overlaid on the render so you can watch the lasers trace the racking.

Because a 2x3-block warehouse with six robots is far too much SDF to hand-write,
this script **generates** the world (``worlds/warehouse_demo.sdf``) and the
ros_gz bridge (``config/gz_bridge_warehouse.yaml``) from the same
``make_warehouse`` grid the algorithm runs on, then records — so the scene and
the plan can never drift apart. Rendered fully offscreen by the shared
:mod:`_gz_record` harness. Media-generation only; not part of CI.

    python3 scripts/record_gazebo_warehouse_gif.py
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "mrn_sim"))
sys.path.insert(0, os.path.join(_REPO, "mrn_coord"))

from _gz_record import Cam, Scenario, add_cli_and_run  # noqa: E402
from mrn_coord.flocking import velocity_to_unicycle  # noqa: E402
from mrn_coord.lifelong.lifelong import (  # noqa: E402
    TaskStream, make_warehouse, run_lifelong)

ROWS, COLS, AISLE = 2, 3, 1
AGENTS = 6
SIM_STEPS = 56
SCALE = 1.4              # grid cell -> metres (aisle wide enough for the body)
SUB = 6                  # interpolated waypoints per grid step
SHELF_H = 0.9

MAX_V, MAX_W = 3.2, 3.2
TRACK_GAIN, TRACK_SPEED = 2.4, 2.6

# Distinct robot colours, 0..1 for the SDF material (and reused, *255, to tint
# each robot's LiDAR overlay).
_PALETTE = [
    (0.22, 0.74, 0.93),   # cyan
    (0.96, 0.45, 0.71),   # pink
    (0.64, 0.90, 0.21),   # lime
    (0.98, 0.70, 0.20),   # amber
    (0.60, 0.55, 0.98),   # violet
    (0.30, 0.86, 0.70),   # teal
]
IDS = [f"a{i}" for i in range(AGENTS)]
ROBOT_RGB = {IDS[i]: tuple(int(c * 255) for c in _PALETTE[i % len(_PALETTE)])
             for i in range(AGENTS)}

GRID, ENDPOINTS = make_warehouse(ROWS, COLS, aisle=AISLE)


def _starts():
    starts, used = {}, set()
    for i in range(AGENTS):
        for cell in ENDPOINTS:
            if cell not in used:
                starts[IDS[i]] = cell
                used.add(cell)
                break
    return starts


def _lerp(a, b, s):
    return (a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s)


def _build_trajectory():
    """Run the lifelong sim and expand its cell history to metric waypoints."""
    stream = TaskStream(pool=ENDPOINTS)
    result = run_lifelong(GRID, _starts(), stream, max_steps=SIM_STEPS,
                          keep_history=True, allocator="hungarian")
    hist = result.history
    traj = []
    for t in range(len(hist) - 1):
        for k in range(SUB):
            s = k / SUB
            traj.append({a: _lerp(hist[t][a], hist[t + 1][a], s) for a in IDS})
    traj.append({a: (hist[-1][a][0], hist[-1][a][1]) for a in IDS})
    metric = [{a: (p[a][0] * SCALE, p[a][1] * SCALE) for a in IDS} for p in traj]
    print(f"precomputed lifelong trajectory: {len(metric)} waypoints, "
          f"{result.completed} tasks ({result.throughput:.2f}/step)")
    return metric


def _clamp(vx, vy, m):
    s = math.hypot(vx, vy)
    return (vx * m / s, vy * m / s) if s > m else (vx, vy)


def make_step(duration):
    traj = _build_trajectory()
    T = len(traj)

    def step(poses, elapsed):
        idx = min(T - 1, int(elapsed / duration * (T - 1)))
        target = traj[idx]
        cmds = {}
        for rid in IDS:
            px, py, yaw = poses[rid]
            tx, ty = target[rid]
            vx, vy = _clamp(TRACK_GAIN * (tx - px), TRACK_GAIN * (ty - py), TRACK_SPEED)
            cmds[rid] = velocity_to_unicycle(yaw, vx, vy, max_v=MAX_V, max_omega=MAX_W)
        return cmds

    return step


# --- world + bridge generation -------------------------------------------------

_SHELF_MAT = ("<material><ambient>0.20 0.25 0.34 1</ambient>"
              "<diffuse>0.24 0.30 0.42 1</diffuse></material>")


def _robot_model(rid, x, y, rgb):
    r, g, b = rgb
    return f"""    <model name="{rid}">
      <pose>{x:.3f} {y:.3f} 0.1 0 0 0</pose>
      <link name="base">
        <inertial><mass>1.5</mass><inertia><ixx>0.03</ixx><iyy>0.03</iyy><izz>0.05</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><pose>0 0 0.1 0 0 0</pose><geometry><box><size>0.5 0.34 0.2</size></box></geometry></collision>
        <visual name="v"><pose>0 0 0.1 0 0 0</pose><geometry><box><size>0.5 0.34 0.2</size></box></geometry>
          <material><ambient>{r*0.6:.3f} {g*0.6:.3f} {b*0.6:.3f} 1</ambient><diffuse>{r:.3f} {g:.3f} {b:.3f} 1</diffuse><specular>0.3 0.3 0.3 1</specular></material></visual>
        <visual name="nose"><pose>0.22 0 0.17 0 0 0</pose><geometry><box><size>0.1 0.16 0.09</size></box></geometry>
          <material><ambient>0.92 0.95 1 1</ambient><diffuse>0.92 0.95 1 1</diffuse></material></visual>
        <visual name="puck"><pose>0 0 0.22 0 0 0</pose><geometry><cylinder><radius>0.05</radius><length>0.06</length></cylinder></geometry>
          <material><ambient>0.05 0.06 0.08 1</ambient><diffuse>0.08 0.09 0.11 1</diffuse></material></visual>
        <sensor name="lidar" type="gpu_lidar"><pose>0 0 0.25 0 0 0</pose><update_rate>15</update_rate><always_on>1</always_on><topic>/model/{rid}/scan</topic>
          <lidar><scan><horizontal><samples>120</samples><resolution>1</resolution><min_angle>-3.14159</min_angle><max_angle>3.14159</max_angle></horizontal></scan>
            <range><min>0.12</min><max>6.0</max><resolution>0.01</resolution></range></lidar></sensor>
      </link>
      <plugin filename="gz-sim-velocity-control-system" name="gz::sim::systems::VelocityControl"><topic>/model/{rid}/cmd_vel</topic></plugin>
      <plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher">
        <publish_link_pose>false</publish_link_pose><publish_model_pose>true</publish_model_pose>
        <use_pose_vector_msg>false</use_pose_vector_msg><static_publisher>false</static_publisher><update_frequency>30</update_frequency></plugin>
    </model>"""


def _shelf_model(i, x, y):
    return (f'    <model name="shelf_{i}"><static>true</static>'
            f'<pose>{x:.3f} {y:.3f} {SHELF_H/2:.3f} 0 0 0</pose><link name="link">'
            f'<collision name="c"><geometry><box><size>{SCALE:.3f} {SCALE:.3f} {SHELF_H:.3f}</size></box></geometry></collision>'
            f'<visual name="v"><geometry><box><size>{SCALE*0.96:.3f} {SCALE*0.96:.3f} {SHELF_H:.3f}</size></box></geometry>'
            f'{_SHELF_MAT}</visual></link></model>')


def _cam():
    cx = GRID.width * SCALE / 2.0
    cy = GRID.height * SCALE / 2.0
    return Cam((cx, cy - 6.6, 12.5), yaw=1.5707, pitch=1.02,
               width=1000, height=660, hfov=0.95, lidar_z=0.35)


def generate_world(path):
    cam = _cam()
    shelves = "\n".join(_shelf_model(i, x * SCALE, y * SCALE)
                        for i, (x, y) in enumerate(sorted(GRID.blocked)))
    starts = _starts()
    robots = "\n".join(_robot_model(rid, starts[rid][0] * SCALE,
                                    starts[rid][1] * SCALE, _PALETTE[i % len(_PALETTE)])
                       for i, rid in enumerate(IDS))
    gw, gh = GRID.width * SCALE, GRID.height * SCALE
    sdf = f"""<?xml version="1.0" ?>
<!--
  GENERATED by scripts/record_gazebo_warehouse_gif.py from a make_warehouse grid.
  Gazebo (gz sim, Harmonic) 3D warehouse: {AGENTS} AMRs work a shelf-and-aisle
  warehouse on a precomputed lifelong-MAPF (PIBT) schedule, each with a 360°
  LiDAR. Rendered fully offscreen. Media-only; not CI. Do not edit by hand.
-->
<sdf version="1.8">
  <world name="warehouse_demo">
    <physics name="1ms" type="ignored"><max_step_size>0.001</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine><background_color>0.043 0.055 0.078 1</background_color></plugin>
    <scene><ambient>0.5 0.5 0.55 1</ambient><background>0.043 0.055 0.078 1</background><grid>false</grid><shadows>true</shadows></scene>
    <light type="directional" name="sun"><cast_shadows>true</cast_shadows><pose>0 0 10 0 0 0</pose><diffuse>0.9 0.9 0.9 1</diffuse><specular>0.25 0.25 0.25 1</specular><direction>-0.4 0.3 -0.9</direction></light>
    <model name="ground_plane"><static>true</static><link name="link">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>60 60</size></plane></geometry></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>60 60</size></plane></geometry><material><ambient>0.05 0.07 0.10 1</ambient><diffuse>0.07 0.09 0.13 1</diffuse></material></visual></link></model>

{shelves}

{robots}

    <model name="rec_camera"><static>true</static><pose>{cam.pos[0]:.3f} {cam.pos[1]:.3f} {cam.pos[2]:.3f} 0 {cam.pitch} {cam.yaw}</pose>
      <link name="link"><sensor name="rec" type="camera">
        <camera><horizontal_fov>{cam.hfov}</horizontal_fov><image><width>{cam.width}</width><height>{cam.height}</height></image><clip><near>0.1</near><far>120</far></clip></camera>
        <always_on>1</always_on><update_rate>30</update_rate><topic>rec_camera</topic></sensor></link></model>
  </world>
</sdf>
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(sdf)
    print(f"generated {path} ({len(GRID.blocked)} shelves, {AGENTS} robots, "
          f"warehouse {gw:.1f}x{gh:.1f} m)")


def generate_bridge(path):
    blocks = ["# GENERATED by scripts/record_gazebo_warehouse_gif.py — do not edit.",
              "# ros_gz bridge for warehouse_demo.sdf: cmd_vel ROS->gz, pose + scan gz->ROS."]
    for rid in IDS:
        blocks.append(
            f'- ros_topic_name: "/model/{rid}/cmd_vel"\n'
            f'  gz_topic_name: "/model/{rid}/cmd_vel"\n'
            f'  ros_type_name: "geometry_msgs/msg/Twist"\n'
            f'  gz_type_name: "gz.msgs.Twist"\n'
            f'  direction: ROS_TO_GZ\n'
            f'- ros_topic_name: "/model/{rid}/pose"\n'
            f'  gz_topic_name: "/model/{rid}/pose"\n'
            f'  ros_type_name: "geometry_msgs/msg/PoseStamped"\n'
            f'  gz_type_name: "gz.msgs.Pose"\n'
            f'  direction: GZ_TO_ROS\n'
            f'- ros_topic_name: "/model/{rid}/scan"\n'
            f'  gz_topic_name: "/model/{rid}/scan"\n'
            f'  ros_type_name: "sensor_msgs/msg/LaserScan"\n'
            f'  gz_type_name: "gz.msgs.LaserScan"\n'
            f'  direction: GZ_TO_ROS')
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(blocks) + "\n")
    print(f"generated {path}")


WORLD = os.path.join(_REPO, "mrn_gazebo", "worlds", "warehouse_demo.sdf")
BRIDGE = os.path.join(_REPO, "mrn_gazebo", "config", "gz_bridge_warehouse.yaml")

generate_world(WORLD)
generate_bridge(BRIDGE)

SCENARIO = Scenario(
    world=WORLD,
    bridge_cfg=BRIDGE,
    ids=IDS,
    make_step=make_step,
    cam=_cam(),
    crop=(0.16, 0.05),
    use_lidar=True,
    robot_rgb=ROBOT_RGB,
    lidar_step=3,
    colors=72,
)

if __name__ == "__main__":
    add_cli_and_run(SCENARIO, "docs/media/gazebo_warehouse_demo.gif")
