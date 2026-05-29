"""``mrn_mapf_planner``: a thin ROS wrapper around the MAPF core.

Reads a scenario (grid size, obstacles, per-agent start/goal) from parameters,
solves it once with Conflict-Based Search or prioritized planning, and
publishes one ``nav_msgs/Path`` per agent on ``mapf/path/<agent_id>`` with a
latched (transient-local) QoS so RViz and late subscribers receive it. The node
holds no algorithm logic of its own — the planning and grid-to-world conversion
live in the pure, CI-tested :mod:`mrn_coord.mapf.ros_conversion`.

In a live system the agent start cells would come from the cooperative
localization estimate; here they are parameters so the node is self-contained
and launch-smoke-testable.
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from mrn_coord.mapf.ros_conversion import (
    build_agents,
    parse_cells,
    path_to_world_points,
    safe_topic_token,
    solve_scenario,
    yaw_along,
)
from mrn_coord.mapf.solution import pad_paths


def _latched_qos() -> QoSProfile:
    qos = QoSProfile(depth=1)
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    return qos


class MapfPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("mrn_mapf_planner")

        self.declare_parameter("agent_ids", ["1", "2"])
        self.declare_parameter("starts", ["0,0", "0,1"])
        self.declare_parameter("goals", ["3,0", "3,1"])
        self.declare_parameter("grid_width", 4)
        self.declare_parameter("grid_height", 2)
        self.declare_parameter("blocked", [""])
        self.declare_parameter("solver", "cbs")
        self.declare_parameter("max_expansions", 100000)
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("cell_size", 1.0)
        self.declare_parameter("origin_x", 0.0)
        self.declare_parameter("origin_y", 0.0)
        self.declare_parameter("publish_rate_hz", 1.0)

        agent_ids = [str(a) for a in self._param("agent_ids")]
        starts = [str(s) for s in self._param("starts")]
        goals = [str(g) for g in self._param("goals")]
        blocked_raw = [str(b) for b in self._param("blocked") if str(b)]
        self._frame_id = str(self._param("frame_id"))
        self._cell_size = float(self._param("cell_size"))
        self._origin = (float(self._param("origin_x")), float(self._param("origin_y")))

        agents = build_agents(agent_ids, starts, goals)
        solution = solve_scenario(
            int(self._param("grid_width")),
            int(self._param("grid_height")),
            parse_cells(blocked_raw),
            agents,
            solver=str(self._param("solver")),
            max_expansions=int(self._param("max_expansions")),
        )

        self._publishers = {
            a: self.create_publisher(
                Path, f"mapf/path/{safe_topic_token(a)}", _latched_qos()
            )
            for a in agent_ids
        }
        self._paths: dict = {}

        if solution is None:
            self.get_logger().error(
                "MAPF scenario is infeasible (no collision-free solution)"
            )
        else:
            padded = pad_paths(solution.paths)
            for a in agent_ids:
                self._paths[a] = self._build_path_msg(padded[a])
            self.get_logger().info(
                f"planned {len(agent_ids)} agents: "
                f"sum_of_costs={solution.cost} makespan={solution.makespan}"
            )

        self._publish_all()
        rate = float(self._param("publish_rate_hz"))
        if rate > 0.0:
            self._timer = self.create_timer(1.0 / rate, self._publish_all)

    def _param(self, name):
        return self.get_parameter(name).value

    def _build_path_msg(self, cells) -> Path:
        points = path_to_world_points(cells, self._cell_size, self._origin)
        yaws = yaw_along(points)
        msg = Path()
        msg.header.frame_id = self._frame_id
        for (x, y), yaw in zip(points, yaws):
            pose = PoseStamped()
            pose.header.frame_id = self._frame_id
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
            msg.poses.append(pose)
        return msg

    def _publish_all(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for a, msg in self._paths.items():
            msg.header.stamp = stamp
            for pose in msg.poses:
                pose.header.stamp = stamp
            self._publishers[a].publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MapfPlannerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
