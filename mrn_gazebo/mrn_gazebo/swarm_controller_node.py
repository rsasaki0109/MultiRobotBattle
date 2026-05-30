"""``mrn_gz_swarm_controller``: flock N Gazebo robots.

Subscribes to every robot's bridged pose (``/model/<id>/pose``), runs the Boids
``flock_velocities`` over their positions, converts each desired holonomic
velocity into a differential-drive ``(v, omega)`` with ``velocity_to_unicycle``,
and publishes it on ``/model/<id>/cmd_vel``. So the same swarm rules that drive
the matplotlib demo drive a Gazebo multi-robot world.

The Boids velocity *state* is the controller's own last desired velocity per
agent (the rule set integrates it via inertia); the robots' actual positions
feed back through Gazebo. A soft inward turn near the arena bounds keeps the
flock in the field.
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from mrn_coord.flocking import flock_velocities, velocity_to_unicycle


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class SwarmController(Node):
    def __init__(self) -> None:
        super().__init__("mrn_gz_swarm_controller")

        self.declare_parameter("agent_ids", ["robot_0", "robot_1", "robot_2"])
        self.declare_parameter("pose_topic_template", "/model/{id}/pose")
        self.declare_parameter("cmd_vel_topic_template", "/model/{id}/cmd_vel")
        self.declare_parameter("perception", 4.0)
        self.declare_parameter("separation", 1.6)
        self.declare_parameter("max_speed", 1.2)
        self.declare_parameter("max_v", 1.2)
        self.declare_parameter("max_omega", 2.5)
        self.declare_parameter("arena_half_extent", 12.0)
        self.declare_parameter("rate_hz", 10.0)

        self._ids = [str(a) for a in self.get_parameter("agent_ids").value]
        self._perception = float(self.get_parameter("perception").value)
        self._separation = float(self.get_parameter("separation").value)
        self._max_speed = float(self.get_parameter("max_speed").value)
        self._max_v = float(self.get_parameter("max_v").value)
        self._max_omega = float(self.get_parameter("max_omega").value)
        self._half = float(self.get_parameter("arena_half_extent").value)
        pose_tmpl = str(self.get_parameter("pose_topic_template").value)
        cmd_tmpl = str(self.get_parameter("cmd_vel_topic_template").value)

        self._pose: dict = {}     # id -> (x, y, yaw)
        self._vel = {a: (0.0, 0.0) for a in self._ids}  # Boids velocity state
        self._cmd_pubs = {}
        for a in self._ids:
            self._cmd_pubs[a] = self.create_publisher(Twist, cmd_tmpl.format(id=a), 10)
            self.create_subscription(
                PoseStamped, pose_tmpl.format(id=a), self._make_cb(a), 10)

        self.create_timer(1.0 / max(float(self.get_parameter("rate_hz").value), 1e-3),
                          self._step)
        self.get_logger().info(f"gz swarm controller: {len(self._ids)} agents")

    def _make_cb(self, agent):
        def cb(msg: PoseStamped) -> None:
            self._pose[agent] = (msg.pose.position.x, msg.pose.position.y,
                                 _yaw(msg.pose.orientation))
        return cb

    def _wall_turn(self, x, y, vx, vy):
        m, push = 2.5, 1.2
        if x < -self._half + m:
            vx += push
        elif x > self._half - m:
            vx -= push
        if y < -self._half + m:
            vy += push
        elif y > self._half - m:
            vy -= push
        return vx, vy

    def _step(self) -> None:
        ids = [a for a in self._ids if a in self._pose]
        if len(ids) < 2:
            return
        positions = [(self._pose[a][0], self._pose[a][1]) for a in ids]
        velocities = [self._vel[a] for a in ids]
        new_vel = flock_velocities(
            positions, velocities, perception=self._perception,
            separation=self._separation, max_speed=self._max_speed)
        for i, a in enumerate(ids):
            vx, vy = self._wall_turn(positions[i][0], positions[i][1], *new_vel[i])
            self._vel[a] = (vx, vy)
            v, omega = velocity_to_unicycle(
                self._pose[a][2], vx, vy, max_v=self._max_v, max_omega=self._max_omega)
            twist = Twist()
            twist.linear.x = float(v)
            twist.angular.z = float(omega)
            self._cmd_pubs[a].publish(twist)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SwarmController()
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
