"""``mrn_coverage_allocator``: a thin ROS wrapper around frontier allocation.

Reads a partially-explored occupancy grid (text rows) and the robots' current
cells from parameters, detects and clusters frontiers, allocates them to robots
by travel cost (greedy or Hungarian), and publishes each robot's assigned
frontier as a ``geometry_msgs/PointStamped`` goal on ``coverage/goal/<id>`` with
a latched QoS.

The node holds no algorithm logic — frontier detection, clustering, and
allocation are the pure, CI-tested coverage core; this shell only wires the
scenario in and goals out. In a live system the grid would come from a mapping
node and the robot cells from the localization estimate.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from mrn_coord.coverage.allocation import allocate_frontiers
from mrn_coord.coverage.frontier import cluster_frontiers
from mrn_coord.coverage.occupancy import OccupancyGrid
from mrn_coord.coverage.ros_conversion import cell_to_world, parse_robot_positions
from mrn_coord.mapf.ros_conversion import safe_topic_token


def _latched_qos() -> QoSProfile:
    qos = QoSProfile(depth=1)
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    return qos


class CoverageAllocatorNode(Node):
    def __init__(self) -> None:
        super().__init__("mrn_coverage_allocator")

        self.declare_parameter("grid_rows", ["?.......?", "?.......?", "?.......?"])
        self.declare_parameter("robot_ids", ["1", "2"])
        self.declare_parameter("robot_positions", ["1,1", "7,1"])
        self.declare_parameter("method", "hungarian")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("cell_size", 1.0)
        self.declare_parameter("origin_x", 0.0)
        self.declare_parameter("origin_y", 0.0)
        self.declare_parameter("publish_rate_hz", 1.0)

        rows = [str(r) for r in self._param("grid_rows")]
        robot_ids = [str(r) for r in self._param("robot_ids")]
        positions = parse_robot_positions(
            robot_ids, [str(p) for p in self._param("robot_positions")]
        )
        self._frame_id = str(self._param("frame_id"))
        self._cell_size = float(self._param("cell_size"))
        self._origin = (float(self._param("origin_x")), float(self._param("origin_y")))

        grid = OccupancyGrid.from_rows(rows)
        clusters = cluster_frontiers(grid)
        targets = [c.representative for c in clusters]
        assignment = allocate_frontiers(
            grid, positions, targets, method=str(self._param("method"))
        )

        self._goals: dict = {}
        self._pubs = {}
        for rid in robot_ids:
            token = safe_topic_token(rid)
            self._pubs[rid] = self.create_publisher(
                PointStamped, f"coverage/goal/{token}", _latched_qos()
            )
        for rid, cell in assignment.items():
            wx, wy = cell_to_world(cell, self._cell_size, self._origin)
            msg = PointStamped()
            msg.header.frame_id = self._frame_id
            msg.point.x = float(wx)
            msg.point.y = float(wy)
            self._goals[rid] = msg

        self.get_logger().info(
            f"{len(clusters)} frontier cluster(s); allocated {len(assignment)} "
            f"goal(s): " + ", ".join(f"{r}->{c}" for r, c in assignment.items())
        )

        self._publish_all()
        rate = float(self._param("publish_rate_hz"))
        if rate > 0.0:
            self._timer = self.create_timer(1.0 / rate, self._publish_all)

    def _param(self, name):
        return self.get_parameter(name).value

    def _publish_all(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for rid, msg in self._goals.items():
            msg.header.stamp = stamp
            self._pubs[rid].publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CoverageAllocatorNode()
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
