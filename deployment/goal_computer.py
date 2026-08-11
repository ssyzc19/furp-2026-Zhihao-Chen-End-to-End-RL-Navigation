#!/usr/bin/env python3
"""
M2 — 相对目标坐标计算 ROS 节点 (goal_computer)

功能：
  1. 从 /robot_pose_ekf/odom_combined 获取机器人当前位姿
  2. 接受目标点（odom 坐标系下的 x,y），可通过 ROS param 或话题动态设置
  3. 计算极坐标相对目标：distance（米）+ angle（弧度，相对于机器人朝向）
  4. 发布 /ppo/goal_polar（geometry_msgs/Point: x=distance, y=angle）

对齐训练时的 PointGoalWithGPSCompassSensor (goal_format=POLAR, dimensionality=2):
  - channel 0: 径向距离（米）
  - channel 1: 相对角度（弧度），范围 [-pi, pi]，正值为左转，负值为右转

设置目标点的方式（优先级从高到低）：
  - 方式1：发布 geometry_msgs/Point 到 /ppo/goal_xy（odom 坐标系）
  - 方式2：启动时设置 ROS param ~goal_x, ~goal_y

运行方式（Jetson 上）：
  conda activate wheeltec
  python goal_computer.py _goal_x:=2.0 _goal_y:=0.0

依赖：
  rospy, numpy, nav_msgs, geometry_msgs, tf (可选)
"""

import math
import rospy
import numpy as np

from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped


class GoalComputer:
    """实时计算机器人当前位姿到目标点的极坐标相对位置。"""

    def __init__(self):
        # --- 目标点（odom 坐标系）---
        self.goal_x = rospy.get_param("~goal_x", 2.0)
        self.goal_y = rospy.get_param("~goal_y", 0.0)
        self._goal_valid = True  # 是否已收到有效目标

        # --- 机器人当前位姿（odom 坐标系）---
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_yaw = 0.0  # 弧度
        self._pose_valid = False  # 是否已收到有效位姿

        # --- 订阅 / 发布 ---
        # 1. EKF 融合里程计（实际类型是 PoseWithCovarianceStamped）
        rospy.Subscriber(
            "/robot_pose_ekf/odom_combined",
            PoseWithCovarianceStamped,
            self._on_odom,
        )
        # 2. 动态更新目标点（可选，geometry_msgs/Point in odom frame）
        rospy.Subscriber("/ppo/goal_xy", Point, self._on_goal_xy)
        # 3. 也可以用 PoseStamped 设定目标
        rospy.Subscriber("/ppo/goal_pose", PoseStamped, self._on_goal_pose)

        # 发布极坐标目标
        self.goal_pub = rospy.Publisher("/ppo/goal_polar", Point, queue_size=5)

        # 发布 odom 系下的目标点（可视化用）
        self.goal_viz_pub = rospy.Publisher(
            "/ppo/goal_marker", Point, queue_size=5
        )

        rospy.loginfo(
            f"GoalComputer 初始化完成，目标点: odom({self.goal_x:.2f}, "
            f"{self.goal_y:.2f})，等待 /robot_pose_ekf/odom_combined ..."
        )

        # --- 定时发布（10Hz，即使没有新 odom 数据也尝试发布最新计算结果）---
        self._pub_timer = rospy.Timer(
            rospy.Duration(0.1), self._publish_goal
        )

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------
    def _on_odom(self, msg: PoseWithCovarianceStamped):
        """从 EKF 融合里程计提取位姿。"""
        pose = msg.pose.pose
        self._robot_x = pose.position.x
        self._robot_y = pose.position.y

        # 从四元数提取 yaw
        q = pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        self._pose_valid = True

    def _on_goal_xy(self, msg: Point):
        """动态更新目标点（geometry_msgs/Point -> odom x,y）。"""
        self.goal_x = msg.x
        self.goal_y = msg.y
        self._goal_valid = True
        rospy.loginfo(f"目标点已更新: odom({self.goal_x:.2f}, {self.goal_y:.2f})")

    def _on_goal_pose(self, msg: PoseStamped):
        """动态更新目标点（PoseStamped -> 取 position.x, position.y）。"""
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self._goal_valid = True
        rospy.loginfo(
            f"目标点已更新 (PoseStamped): odom({self.goal_x:.2f}, {self.goal_y:.2f})"
        )

    # ------------------------------------------------------------------
    # 定时发布
    # ------------------------------------------------------------------
    def _publish_goal(self, event):
        """计算极坐标相对目标并发布。"""
        if not self._pose_valid:
            rospy.logwarn_throttle(5.0, "等待位姿数据 (/robot_pose_ekf/odom_combined)...")
            return
        if not self._goal_valid:
            rospy.logwarn_throttle(5.0, "等待目标点设置...")
            return

        # 相对向量（odom 坐标系下的差值）
        dx = self.goal_x - self._robot_x
        dy = self.goal_y - self._robot_y

        # 径向距离（米）
        distance = math.hypot(dx, dy)

        # 全局方向角（robot → goal，在 odom 坐标系下的绝对角度）
        global_angle = math.atan2(dy, dx)

        # 相对于机器人朝向的角度
        # relative_angle > 0: 目标在机器人左侧（需要左转）
        # relative_angle < 0: 目标在机器人右侧（需要右转）
        relative_angle = global_angle - self._robot_yaw

        # 归一化到 [-pi, pi]
        relative_angle = math.atan2(
            math.sin(relative_angle), math.cos(relative_angle)
        )

        # 发布极坐标目标
        polar_msg = Point()
        polar_msg.x = distance
        polar_msg.y = relative_angle
        polar_msg.z = 0.0
        self.goal_pub.publish(polar_msg)

        # 发布目标点标记（可视化，在 odom 系下的绝对坐标）
        viz_msg = Point()
        viz_msg.x = self.goal_x
        viz_msg.y = self.goal_y
        viz_msg.z = 0.0
        self.goal_viz_pub.publish(viz_msg)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    rospy.init_node("goal_computer", anonymous=False)
    node = GoalComputer()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("goal_computer 停止")


if __name__ == "__main__":
    main()
