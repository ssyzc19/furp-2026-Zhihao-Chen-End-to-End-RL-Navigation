#!/usr/bin/env python3
"""
ROS2 (Humble) deployment node: run a trained Habitat PointNav PPO policy on
a TurtleBot3.

Pipeline each control step:
    1. read latest RGB + Depth from camera topics   (TODO: fill camera topics)
    2. read robot pose from /odom -> compute PointGoal [rho, phi]
    3. policy.step(rgb, depth, pointgoal) -> discrete action
    4. action -> /cmd_vel Twist, held open-loop for the mapped duration
    5. stop when action == STOP or rho < success_distance

Camera part is left as TODO until the RGB camera is installed.  Everything
else (odom->pointgoal, action->cmd_vel, episode loop) is complete and can be
tested on the base alone with a dummy image.

Run:
    ros2 run <pkg> pointnav_deploy_node --ros-args \
        -p ckpt_path:=/home/orangepi/furp_code/gaiouka/ckpt.49.pth \
        -p goal_x:=2.0 -p goal_y:=0.0
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
# from sensor_msgs.msg import Image        # TODO: enable when camera is ready
# from cv_bridge import CvBridge           # TODO: enable when camera is ready

from policy_wrapper import (
    PointNavPolicyRunner,
    compute_pointgoal,
    yaw_from_quaternion,
)
from action_mapper import action_to_twist_cmd, ACTION_NAMES, STOP


class PointNavDeployNode(Node):
    def __init__(self):
        super().__init__("pointnav_deploy")

        # ── parameters ──
        self.declare_parameter("ckpt_path", "")
        self.declare_parameter("goal_x", 1.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("success_distance", 0.2)
        self.declare_parameter("control_period", 0.1)  # sec between decisions
        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")

        ckpt = self.get_parameter("ckpt_path").value
        self.goal_x = self.get_parameter("goal_x").value
        self.goal_y = self.get_parameter("goal_y").value
        self.success_distance = self.get_parameter("success_distance").value

        # ── policy ──
        self.get_logger().info(f"Loading policy: {ckpt}")
        self.policy = PointNavPolicyRunner(ckpt_path=ckpt, device="cpu")
        self.policy.reset()
        self.get_logger().info("Policy loaded.")

        # ── robot state ──
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.have_odom = False

        # ── camera state (TODO) ──
        # self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None

        # ── ROS interfaces ──
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(
            Odometry, "/odom", self._odom_cb, qos_profile_sensor_data
        )
        # TODO: enable camera subscriptions once the RGB camera is installed
        # self.create_subscription(Image, self.get_parameter("rgb_topic").value,
        #                          self._rgb_cb, qos_profile_sensor_data)
        # self.create_subscription(Image, self.get_parameter("depth_topic").value,
        #                          self._depth_cb, qos_profile_sensor_data)

        # ── control loop ──
        self._busy_until = self.get_clock().now()
        self._current_twist = Twist()
        period = self.get_parameter("control_period").value
        self.create_timer(period, self._control_step)
        self.finished = False

    # ── callbacks ──

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot_x = p.x
        self.robot_y = p.y
        self.robot_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self.have_odom = True

    # def _rgb_cb(self, msg):
    #     img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
    #     self.latest_rgb = self._resize_to_256(img)   # TODO resize/crop to 256x256
    #
    # def _depth_cb(self, msg):
    #     d = self.bridge.imgmsg_to_cv2(msg, "32FC1")
    #     self.latest_depth = self._normalize_depth(d) # TODO -> 256x256x1 in 0..1

    # ── main loop ──

    def _control_step(self):
        if self.finished:
            return
        if not self.have_odom:
            self.get_logger().warn("Waiting for /odom ...", throttle_duration_sec=2.0)
            return

        # If we are still executing a timed action, keep publishing it.
        now = self.get_clock().now()
        if now < self._busy_until:
            self.cmd_pub.publish(self._current_twist)
            return

        # Compute PointGoal
        pointgoal = compute_pointgoal(
            self.robot_x, self.robot_y, self.robot_yaw, self.goal_x, self.goal_y
        )
        rho, phi = float(pointgoal[0]), float(pointgoal[1])

        # Success check
        if rho < self.success_distance:
            self.get_logger().info(f"SUCCESS: reached goal (rho={rho:.3f} m)")
            self._stop_robot()
            self.finished = True
            return

        # ── camera gate ──
        if self.latest_rgb is None or self.latest_depth is None:
            self.get_logger().warn(
                "No camera frames yet — camera not installed/enabled. "
                "odom->pointgoal is working (rho=%.2f, phi=%.2f rad). "
                "Fill in _rgb_cb/_depth_cb to run the policy." % (rho, phi),
                throttle_duration_sec=3.0,
            )
            return

        # Policy inference
        action = self.policy.step(self.latest_rgb, self.latest_depth, pointgoal)
        self.get_logger().info(
            f"rho={rho:.2f} phi={phi:.2f} -> {ACTION_NAMES.get(action, action)}"
        )

        if action == STOP:
            self.get_logger().info("Policy chose STOP.")
            self._stop_robot()
            self.finished = True
            return

        # Map action -> timed Twist
        lin, ang, dur = action_to_twist_cmd(action)
        tw = Twist()
        tw.linear.x = lin
        tw.angular.z = ang
        self._current_twist = tw
        self._busy_until = now + rclpy.duration.Duration(seconds=dur)
        self.cmd_pub.publish(tw)

    def _stop_robot(self):
        self.cmd_pub.publish(Twist())  # all zeros


def main(args=None):
    rclpy.init(args=args)
    node = PointNavDeployNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
