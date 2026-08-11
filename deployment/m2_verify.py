#!/usr/bin/env python3
"""
M2 — 观测链路验证脚本

在不启动完整推理 pipeline 的情况下，验证 M2 产出的数据格式、范围和统计
特性是否与训练时的期望一致。

检查项：
  1. /ppo/rgb            — 分辨率 256x256, rgb8, 数值范围 [0,255]
  2. /ppo/depth           — 分辨率 256x256, 32FC1, 数值范围 [0.0, 1.0]
  3. /ppo/goal_polar      — geometry_msgs/Point, x=distance, y=angle
  4. /ppo/rgb/camera_info — 缩放后的内参
  5. 同步性              — RGB/Depth 帧的时间戳对齐

用法（Jetson 上，conda wheeltec 环境）：
  python m2_verify.py

也可以在等待过程中按 Ctrl-C 停止，脚本会打印已收到的统计信息。
"""

import sys
import time
import math
import rospy
import numpy as np

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from cv_bridge import CvBridge


# ---------- 期望参数 ----------
EXPECTED_H = 256
EXPECTED_W = 256
EXPECTED_RGB_ENCODING = "rgb8"
EXPECTED_DEPTH_ENCODING = "32FC1"
EXPECTED_DEPTH_MIN = 0.0
EXPECTED_DEPTH_MAX = 1.0
RGB_MIN = 0
RGB_MAX = 255


class M2Verifier:
    """订阅 M2 的产出话题，检查格式与数值范围。"""

    def __init__(self):
        self.bridge = CvBridge()

        # 累计统计
        self._rgb_count = 0
        self._depth_count = 0
        self._goal_count = 0
        self._rgb_info_count = 0
        self._depth_info_count = 0

        # 错误计数
        self._errors = []

        # 用于检查同步
        self._last_rgb_stamp = None
        self._last_depth_stamp = None
        self._max_stamp_diff = 0.0

        # 订阅
        rospy.Subscriber("/ppo/rgb", Image, self._check_rgb)
        rospy.Subscriber("/ppo/depth", Image, self._check_depth)
        rospy.Subscriber("/ppo/goal_polar", Point, self._check_goal)
        rospy.Subscriber("/ppo/rgb/camera_info", CameraInfo, self._check_rgb_info)
        rospy.Subscriber("/ppo/depth/camera_info", CameraInfo, self._check_depth_info)

        rospy.loginfo("M2 验证器启动，等待 /ppo/* 话题 ...")
        rospy.loginfo(
            "期望: RGB={}x{} rgb8 [0,255], Depth={}x{} 32FC1 [0,1], "
            "Goal=Point(distance,angle)".format(
                EXPECTED_H, EXPECTED_W, EXPECTED_H, EXPECTED_W
            )
        )

        # 定时打印状态
        rospy.Timer(rospy.Duration(5.0), self._print_status)

    # ------------------------------------------------------------------
    # 检查回调
    # ------------------------------------------------------------------
    def _check_rgb(self, msg: Image):
        self._rgb_count += 1
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as e:
            self._record_error(f"RGB 解码失败: {e}")
            return

        h, w, c = img.shape
        if h != EXPECTED_H or w != EXPECTED_W:
            self._record_error(
                f"RGB 分辨率错误: 期望 {EXPECTED_H}x{EXPECTED_W}, 实际 {h}x{w}"
            )
        if msg.encoding != EXPECTED_RGB_ENCODING:
            self._record_error(
                f"RGB 编码错误: 期望 {EXPECTED_RGB_ENCODING}, 实际 {msg.encoding}"
            )

        vmin, vmax = img.min(), img.max()
        if vmin < RGB_MIN or vmax > RGB_MAX:
            self._record_error(f"RGB 数值越界: range=[{vmin}, {vmax}]")

        if self._rgb_count == 1:
            rospy.loginfo(
                f"[OK] /ppo/rgb: {h}x{w}, {msg.encoding}, "
                f"range=[{vmin}, {vmax}], mean={img.mean():.1f}"
            )

        # 时间戳追踪
        self._last_rgb_stamp = msg.header.stamp

    def _check_depth(self, msg: Image):
        self._depth_count += 1
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self._record_error(f"Depth 解码失败: {e}")
            return

        h, w = img.shape
        if h != EXPECTED_H or w != EXPECTED_W:
            self._record_error(
                f"Depth 分辨率错误: 期望 {EXPECTED_H}x{EXPECTED_W}, 实际 {h}x{w}"
            )
        if msg.encoding != EXPECTED_DEPTH_ENCODING:
            self._record_error(
                f"Depth 编码错误: 期望 {EXPECTED_DEPTH_ENCODING}, 实际 {msg.encoding}"
            )

        # 数据类型检查
        if img.dtype != np.float32:
            self._record_error(f"Depth dtype 错误: 期望 float32, 实际 {img.dtype}")

        vmin, vmax = float(img.min()), float(img.max())
        if vmin < EXPECTED_DEPTH_MIN - 0.01 or vmax > EXPECTED_DEPTH_MAX + 0.01:
            self._record_error(f"Depth 数值越界: range=[{vmin:.4f}, {vmax:.4f}]")

        if self._depth_count == 1:
            nonzero = (img > 0).sum()
            rospy.loginfo(
                f"[OK] /ppo/depth: {h}x{w}, {msg.encoding}, "
                f"range=[{vmin:.4f}, {vmax:.4f}], "
                f"nonzero={nonzero} ({100*nonzero/img.size:.1f}%)"
            )

        self._last_depth_stamp = msg.header.stamp

        # 同步检查
        if self._last_rgb_stamp is not None and self._last_depth_stamp is not None:
            diff = abs(
                self._last_rgb_stamp.to_sec() - msg.header.stamp.to_sec()
            )
            if diff > self._max_stamp_diff:
                self._max_stamp_diff = diff

    def _check_goal(self, msg: Point):
        self._goal_count += 1
        if self._goal_count == 1:
            rospy.loginfo(
                f"[OK] /ppo/goal_polar: distance={msg.x:.3f}m, "
                f"angle={msg.y:.4f}rad ({math.degrees(msg.y):.1f}deg)"
            )
        # 基本合理性检查
        if msg.x < 0:
            self._record_error(f"goal distance 为负数: {msg.x}")
        if abs(msg.y) > math.pi:
            self._record_error(f"goal angle 超出 [-pi, pi]: {msg.y}")

    def _check_rgb_info(self, msg: CameraInfo):
        if self._rgb_info_count == 0:
            rospy.loginfo(
                f"[OK] /ppo/rgb/camera_info: {msg.width}x{msg.height}, "
                f"K=[{msg.K[0]:.2f},{msg.K[2]:.2f}; {msg.K[4]:.2f},{msg.K[5]:.2f}]"
            )
        self._rgb_info_count += 1

    def _check_depth_info(self, msg: CameraInfo):
        self._depth_info_count += 1

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _record_error(self, msg: str):
        rospy.logerr(msg)
        self._errors.append(msg)

    def _print_status(self, event):
        rospy.loginfo(
            f"--- 状态: rgb={self._rgb_count}, depth={self._depth_count}, "
            f"goal={self._goal_count}, rgb_info={self._rgb_info_count}, "
            f"depth_info={self._depth_info_count}, "
            f"errors={len(self._errors)}, "
            f"max_rgb_depth_stamp_diff={self._max_stamp_diff*1000:.1f}ms ---"
        )

    def summary(self):
        """打印最终汇总并返回是否全部通过。"""
        rospy.loginfo("=" * 60)
        rospy.loginfo("M2 验证汇总")
        rospy.loginfo("=" * 60)

        checks = []

        # 1. RGB
        rgb_ok = self._rgb_count > 0
        checks.append((f"/ppo/rgb 已收到 {self._rgb_count} 帧", rgb_ok))

        # 2. Depth
        depth_ok = self._depth_count > 0
        checks.append((f"/ppo/depth 已收到 {self._depth_count} 帧", depth_ok))

        # 3. Goal
        goal_ok = self._goal_count > 0
        checks.append((f"/ppo/goal_polar 已收到 {self._goal_count} 条", goal_ok))

        # 4. CameraInfo
        info_ok = self._rgb_info_count > 0
        checks.append(
            (f"/ppo/rgb/camera_info 已收到 {self._rgb_info_count} 条", info_ok)
        )

        # 5. 同步
        sync_ok = self._max_stamp_diff < 0.5  # 小于 500ms 视为同步 OK
        checks.append(
            (
                f"RGB/Depth 最大时间戳偏差 {self._max_stamp_diff*1000:.1f}ms "
                f"({'< 500ms OK' if sync_ok else '> 500ms WARN'})",
                sync_ok,
            )
        )

        # 6. 错误数
        err_ok = len(self._errors) == 0
        checks.append((f"验证错误: {len(self._errors)}", err_ok))

        # 打印
        all_ok = True
        for desc, ok in checks:
            status = "PASS" if ok else "FAIL"
            rospy.loginfo(f"  [{status}] {desc}")
            if not ok:
                all_ok = False

        if self._errors:
            rospy.loginfo("--- 错误详情 ---")
            for e in self._errors:
                rospy.loginfo(f"  - {e}")

        if all_ok:
            rospy.loginfo("\n✅ M2 验证全部通过！可以进入 M3 推理节点开发。")
        else:
            rospy.loginfo(
                "\n⚠️  M2 验证有未通过的检查项，请先排查上述 FAIL 项目再进入 M3。"
            )

        return all_ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    rospy.init_node("m2_verifier", anonymous=False)
    verifier = M2Verifier()

    duration = rospy.get_param("~duration", 10.0)  # 默认运行 10 秒
    rospy.loginfo(f"验证将运行 {duration} 秒，可随时按 Ctrl-C 提前退出")

    try:
        if duration > 0:
            rospy.sleep(duration)
        else:
            rospy.spin()  # duration=0 表示一直运行
    except KeyboardInterrupt:
        rospy.loginfo("收到中断信号")

    verifier.summary()


if __name__ == "__main__":
    main()
