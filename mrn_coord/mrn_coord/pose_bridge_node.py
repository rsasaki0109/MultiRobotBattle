"""``mrn_pose_bridge``: feed the localization estimate into the coordination layer.

This is the seam between the two halves of the project. The localization stack
publishes a per-agent estimate as ``mrn_msgs/AgentState`` (the V2V agent state)
or ``mrn_msgs/CooperativePose`` (the fused cooperative estimate); the
coordination nodes consume a plain ``geometry_msgs/PoseStamped`` on
``formation/pose/<id>``. This node subscribes to the former and republishes the
latter, so the formation controller (and any other coordination consumer) can
act on the live estimate instead of a simulated plant.

It is intentionally thin: per agent, take ``msg.pose.pose`` and restamp it as a
``PoseStamped`` in the estimate's ``map_frame``. The direction is one-way —
estimate in, pose out; acting the resulting commands back on a real plant is a
separate concern.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from mrn_coord.mapf.ros_conversion import safe_topic_token


class PoseBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("mrn_pose_bridge")

        self.declare_parameter("agent_ids", ["robot_1", "robot_2", "robot_3"])
        self.declare_parameter("source", "agent_state")  # or "cooperative_pose"
        self.declare_parameter("source_topic_template", "/{id}/mrn/agent_state")
        self.declare_parameter("target_topic_template", "formation/pose/{token}")
        self.declare_parameter("frame_id", "")  # empty = use the estimate's map_frame

        self._agent_ids = [str(a) for a in self._param("agent_ids")]
        source = str(self._param("source"))
        source_tmpl = str(self._param("source_topic_template"))
        target_tmpl = str(self._param("target_topic_template"))
        self._frame_id = str(self._param("frame_id"))

        if source == "agent_state":
            from mrn_msgs.msg import AgentState as SrcType
        elif source == "cooperative_pose":
            from mrn_msgs.msg import CooperativePose as SrcType
        else:
            raise ValueError(f"unknown source: {source!r}")

        self._pubs = {}
        for a in self._agent_ids:
            token = safe_topic_token(a)
            self._pubs[a] = self.create_publisher(
                PoseStamped, target_tmpl.format(id=a, token=token), 10
            )
            self.create_subscription(
                SrcType, source_tmpl.format(id=a, token=token),
                self._make_cb(a), 10,
            )
        self.get_logger().info(
            f"pose bridge: {len(self._agent_ids)} agents, source={source}"
        )

    def _param(self, name):
        return self.get_parameter(name).value

    def _make_cb(self, agent):
        def cb(msg) -> None:
            out = PoseStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = self._frame_id or getattr(msg, "map_frame", "map")
            out.pose = msg.pose.pose
            self._pubs[agent].publish(out)
        return cb


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PoseBridgeNode()
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
