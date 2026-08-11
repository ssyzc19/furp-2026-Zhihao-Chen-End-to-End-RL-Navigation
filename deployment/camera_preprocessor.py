#!/usr/bin/env python3
"""
M2 — 相机预处理 ROS 节点 (camera_preprocessor)

功能：
  1. 订阅 /camera/rgb/image_raw (640x480, rgb8) 和
     /camera/depth/image_raw (640x480, 16UC1, 毫米)
  2. 用 ApproximateTimeSynchronizer 对齐两路图像（容忍 ~100ms 偏差）
  3. RGB:  resize 640x480 → 256x256，保持 uint8 [0,255]
  4. Depth: 16UC1 mm → float32 meters → clip(0,10) / 10.0 → resize 256x256
  5. 发布处理后的图像到 /ppo/rgb 和 /ppo/depth

深度归一化公式（必须严格复现训练时逻辑）：
  depth_meters = depth_raw_mm / 1000.0
  depth_normalized = np.clip(depth_meters, 0.0, 10.0) / 10.0

运行方式（Jetson 上）：
  conda activate wheeltec
  python camera_preprocessor.py

依赖：
  rospy, cv_bridge, message_filters, cv2, numpy
"""

import sys
import rospy
import numpy as np
import cv2

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters

# ---------------------------------------------------------------------------
# 训练时的预处理参数（来自 config.yaml + load_and_infer.py）
# ---------------------------------------------------------------------------
TARGET_H = 256
TARGET_W = 256
MAX_DEPTH_M = 10.0          # max_depth
MIN_DEPTH_M = 0.0           # min_depth

# 原始相机分辨率 (Astra S)
RAW_H = 480
RAW_W = 640

# 深度编码：Astra S 的 16UC1 输出是毫米，需要先转米再归一化
DEPTH_MM_TO_M = 1.0 / 1000.0


class CameraPreprocessor:
    """订阅 RGB + Depth 原始话题，预处理后发布对齐的 256x256 图像。"""

    def __init__(self):
        self.bridge = CvBridge()

        # --- 发布者 ---
        self.rgb_pub = rospy.Publisher(
            "/ppo/rgb", Image, queue_size=5
        )
        self.depth_pub = rospy.Publisher(
            "/ppo/depth", Image, queue_size=5
        )
        # 可选：发布 camera_info（把 K 矩阵按缩放比例调整后转发）
        self.rgb_info_pub = rospy.Publisher(
            "/ppo/rgb/camera_info", CameraInfo, queue_size=5
        )
        self.depth_info_pub = rospy.Publisher(
            "/ppo/depth/camera_info", CameraInfo, queue_size=5
        )

        # --- 订阅者（用 message_filters 做近似时间同步） ---
        rgb_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image)
        depth_sub = message_filters.Subscriber("/camera/depth/image_raw", Image)

        # 同步器：容忍 100ms 偏差，队列深度 10
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self._on_synced_frames)

        # camera_info 缓存（用于缩放后转发）
        self._last_rgb_info = None
        self._last_depth_info = None
        rospy.Subscriber(
            "/camera/rgb/camera_info", CameraInfo, self._on_rgb_info
        )
        rospy.Subscriber(
            "/camera/depth/camera_info", CameraInfo, self._on_depth_info
        )

        rospy.loginfo(
            "CameraPreprocessor 初始化完成，等待 /camera/rgb/image_raw + "
            "/camera/depth/image_raw ..."
        )

    # ------------------------------------------------------------------
    # camera_info 回调（缓存原始内参，用于缩放后发布）
    # ------------------------------------------------------------------
    def _on_rgb_info(self, msg: CameraInfo):
        self._last_rgb_info = msg

    def _on_depth_info(self, msg: CameraInfo):
        self._last_depth_info = msg

    # ------------------------------------------------------------------
    # 同步回调
    # ------------------------------------------------------------------
    def _on_synced_frames(self, rgb_msg: Image, depth_msg: Image):
        """收到一对时间戳接近的 RGB + Depth 帧。"""
        try:
            # --- 解码 ---
            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            depth_cv = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )  # 16UC1, shape (480, 640), values in mm

            # --- RGB 预处理：只 resize ---
            rgb_resized = cv2.resize(
                rgb_cv, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR
            )
            # rgb_resized: (256, 256, 3), uint8, [0,255] — 与训练时对齐

            # --- Depth 预处理：mm → meters → clip → normalize → resize ---
            # 1. 转 float32，单位毫米 → 米
            depth_meters = depth_cv.astype(np.float32) * DEPTH_MM_TO_M
            # 2. clip + normalize（精确复现训练公式）
            depth_normalized = np.clip(depth_meters, MIN_DEPTH_M, MAX_DEPTH_M) / MAX_DEPTH_M
            # depth_normalized: (480, 640), float32, [0.0, 1.0]
            # 3. resize 到 256x256
            depth_resized = cv2.resize(
                depth_normalized, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR
            )
            # depth_resized: (256, 256), float32, [0.0, 1.0]

            # --- 发布处理后的图像 ---
            rgb_out = self.bridge.cv2_to_imgmsg(rgb_resized, encoding="rgb8")
            rgb_out.header = rgb_msg.header  # 保留原始时间戳和 frame_id
            self.rgb_pub.publish(rgb_out)

            depth_out = self.bridge.cv2_to_imgmsg(
                depth_resized, encoding="32FC1"
            )
            depth_out.header = depth_msg.header
            self.depth_pub.publish(depth_out)

            # --- 发布缩放后的 camera_info ---
            self._publish_scaled_info(rgb_msg.header, rgb_msg.header.frame_id, "rgb")
            self._publish_scaled_info(depth_msg.header, depth_msg.header.frame_id, "depth")

        except Exception as e:
            rospy.logerr(f"预处理出错: {e}")

    # ------------------------------------------------------------------
    # 缩放后的 camera_info
    # ------------------------------------------------------------------
    def _publish_scaled_info(self, header, frame_id: str, kind: str):
        """把原始 camera_info 的内参按 256/640 的比例缩放后发布。

        K 矩阵缩放规则（纯 resize，无裁剪）：
          fx' = fx * (256 / 640) = fx * 0.4
          fy' = fy * (256 / 480) = fy * 0.5333...
          cx' = cx * 0.4
          cy' = cy * (256 / 480)
        """
        src_info = self._last_rgb_info if kind == "rgb" else self._last_depth_info
        if src_info is None:
            return

        scale_x = TARGET_W / RAW_W   # 256 / 640 = 0.4
        scale_y = TARGET_H / RAW_H   # 256 / 480 = 0.5333...

        info = CameraInfo()
        info.header = header
        info.header.frame_id = frame_id
        info.height = TARGET_H
        info.width = TARGET_W
        info.distortion_model = src_info.distortion_model
        info.D = src_info.D  # 畸变系数不变

        # K 矩阵缩放
        K = list(src_info.K)
        K[0] *= scale_x  # fx
        K[2] *= scale_x  # cx
        K[4] *= scale_y  # fy
        K[5] *= scale_y  # cy
        info.K = K

        # P 矩阵缩放
        P = list(src_info.P)
        P[0] *= scale_x  # fx
        P[2] *= scale_x  # cx
        P[5] *= scale_y  # fy
        P[6] *= scale_y  # cy
        info.P = P

        info.R = src_info.R
        info.binning_x = src_info.binning_x
        info.binning_y = src_info.binning_y
        info.roi = src_info.roi

        if kind == "rgb":
            self.rgb_info_pub.publish(info)
        else:
            self.depth_info_pub.publish(info)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    rospy.init_node("camera_preprocessor", anonymous=False)
    node = CameraPreprocessor()
    rospy.loginfo(
        f"相机预处理参数: {RAW_W}x{RAW_H} → {TARGET_W}x{TARGET_H}, "
        f"depth: mm→m→clip[0,{MAX_DEPTH_M}]→/{MAX_DEPTH_M}"
    )
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("camera_preprocessor 停止")


if __name__ == "__main__":
    main()
