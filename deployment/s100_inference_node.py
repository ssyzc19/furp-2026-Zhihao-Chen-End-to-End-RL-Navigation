#!/home/wheeltec/anaconda3/envs/wheeltec/bin/python3
"""
S100 PointNav PPO inference node (ROS 1 Noetic).

Subscribes to RGB + depth camera topics, odometry, and navigation goals.
Runs the PointNavResNetPolicy model (from habitat-baselines) at a fixed
control rate and publishes velocity commands to /cmd_vel.

Usage:
    roslaunch s100_deploy s100_deploy.launch

    # Dry-run (no cmd_vel publishing):
    roslaunch s100_deploy s100_deploy.launch dry_run:=true

Target hardware:  NVIDIA Jetson Orin Nano
Conda environment: wheeltec (torch 1.14 + CUDA)
"""

import math
import os
import sys
import time

# --- habitat_sim / magnum stub (仅 habitat 模型需要) ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
import torch

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Twist, PoseWithCovarianceStamped

from policy_wrappers import HabitatWrapper, JitWrapper, ActionData
from preprocess import preprocess_rgb, preprocess_depth
from action_controller import (
    ContinuousActionController,
    StepActionController,
    STOP,
    ACTION_NAMES,
)


# ---------------------------------------------------------------------------
# Goal computation
# ---------------------------------------------------------------------------


def quat_to_yaw(orientation) -> float:
    """Convert geometry_msgs/Quaternion to yaw angle (radians)."""
    q = orientation
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def compute_pointgoal(current_pose: tuple, goal_pos: tuple) -> tuple:
    """
    Compute polar goal coordinates matching Habitat's
    PointGoalWithGPSCompassSensor (goal_format=POLAR, dimensionality=2).

    Args:
        current_pose: (x, y, yaw) in the odometry frame.
        goal_pos: (gx, gy) in the same frame.

    Returns:
        (distance, angle) — 2-element tuple for the policy input.
        distance in metres, angle in radians relative to robot heading.
    """
    dx = goal_pos[0] - current_pose[0]
    dy = goal_pos[1] - current_pose[1]
    distance = math.hypot(dx, dy)

    # Angle to goal in agent frame
    c = math.cos(current_pose[2])
    s = math.sin(current_pose[2])
    forward = c * dx + s * dy
    right = -s * dx + c * dy
    angle = math.atan2(right, forward)

    return (distance, angle)


def _normalize_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# ROS node
# ---------------------------------------------------------------------------


class S100PPONode:
    """ROS 1 node that wraps the PointNav PPO policy."""

    def __init__(self):
        rospy.init_node("s100_ppo_nav", anonymous=False)

        # --- Parameters ---
        ckpt_path = rospy.get_param("~ckpt_path", "ckpt.49.pth")
        self.model_type = rospy.get_param("~model_type", "habitat")
        device_str = rospy.get_param("~device", "cuda")
        control_rate = rospy.get_param("~control_rate", 10.0)
        execution_mode = rospy.get_param("~execution_mode", "continuous")
        forward_speed = rospy.get_param("~forward_speed", 0.25)
        turn_speed = rospy.get_param("~turn_speed", 0.5)
        forward_step = rospy.get_param("~forward_step", 0.25)
        turn_angle_deg = rospy.get_param("~turn_angle_deg", 10.0)
        self.success_distance = rospy.get_param("~success_distance", 0.2)
        self.max_steps = rospy.get_param("~max_steps", 500)
        self.dry_run = rospy.get_param("~dry_run", False)
        self.min_depth_for_safety = rospy.get_param("~min_depth_for_safety", 0.3)

        # Topics (configurable)
        rgb_topic = rospy.get_param("~rgb_topic", "/ppo/rgb")
        depth_topic = rospy.get_param("~depth_topic", "/ppo/depth")
        odom_topic = rospy.get_param("~odom_topic", "/robot_pose_ekf/odom_combined")
        goal_topic = rospy.get_param("~goal_topic", "/move_base_simple/goal")
        cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")

        # --- Device ---
        if device_str == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        rospy.loginfo(f"Using device: {self.device}")

        # --- Load model ---
        self._load_model(ckpt_path, device_str)

        # --- Action controller ---
        if execution_mode == "step":
            self.controller = StepActionController(
                forward_step=forward_step,
                turn_angle_deg=turn_angle_deg,
                forward_speed=forward_speed,
                turn_speed=turn_speed,
            )
        else:
            self.controller = ContinuousActionController(
                forward_speed=forward_speed, turn_speed=turn_speed
            )
        rospy.loginfo(f"Action controller: {type(self.controller).__name__}")

        # --- RNN / GRU state ---
        self._reset_rnn()

        # --- State ---
        self.current_goal = None  # (gx, gy)
        self.current_pose = (0.0, 0.0, 0.0)  # (x, y, yaw)
        self.latest_rgb = None
        self.latest_depth = None
        self.step_count = 0
        self.episodes_completed = 0

        # --- Subscribers ---
        self.rgb_sub = rospy.Subscriber(rgb_topic, Image, self._rgb_cb, queue_size=2)
        self.depth_sub = rospy.Subscriber(depth_topic, Image, self._depth_cb, queue_size=2)
        self.odom_sub = rospy.Subscriber(
            odom_topic, PoseWithCovarianceStamped, self._odom_cb, queue_size=5
        )
        self.goal_sub = rospy.Subscriber(goal_topic, PoseStamped, self._goal_cb, queue_size=2)

        # --- Publisher ---
        self.cmd_pub = rospy.Publisher(cmd_topic, Twist, queue_size=1)

        # --- Timer ---
        period = rospy.Duration(1.0 / control_rate)
        self.timer = rospy.Timer(period, self._control_loop)

        # --- Latency tracking ---
        self._inference_times_ms = []

        rospy.loginfo(
            f"S100 PPO node ready. "
            f"ckpt={ckpt_path}, control={control_rate}Hz, "
            f"dry_run={self.dry_run}, max_steps={self.max_steps}"
        )
        rospy.loginfo(
            f"Subscribing: rgb={rgb_topic}, depth={depth_topic}, "
            f"odom={odom_topic}, goal={goal_topic}"
        )
        rospy.loginfo(f"Publishing: cmd_vel={cmd_topic}")

    # ------------------------------------------------------------------
    # Model loading (factory pattern — selects wrapper by model_type)
    # ------------------------------------------------------------------

    def _load_model(self, ckpt_path: str, device_str: str):
        if device_str == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if self.model_type == "habitat":
            # 需要 habitat_stub + habitat-baselines
            import habitat_stub
            habitat_stub.install()
            self.policy = HabitatWrapper(ckpt_path, self.device)
        elif self.model_type == "jit_rgbd":
            self.policy = JitWrapper(ckpt_path, self.device, num_channels=4)
        elif self.model_type == "jit_depth":
            self.policy = JitWrapper(ckpt_path, self.device, num_channels=1)
        else:
            raise ValueError(
                f"Unknown model_type: {self.model_type}. "
                f"Valid: habitat, jit_rgbd, jit_depth"
            )
        rospy.loginfo(
            f"Model loaded: type={self.model_type}, path={ckpt_path}"
        )

    # ------------------------------------------------------------------
    # RNN state
    # ------------------------------------------------------------------

    def _reset_rnn(self):
        self.rnn_hidden = self.policy.get_hidden_state()
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.masks = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        self.step_count = 0

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _rgb_cb(self, msg: Image):
        self.latest_rgb = msg

    def _depth_cb(self, msg: Image):
        self.latest_depth = msg

    def _odom_cb(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        yaw = quat_to_yaw(msg.pose.pose.orientation)
        self.current_pose = (p.x, p.y, yaw)

    def _goal_cb(self, msg: PoseStamped):
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        self.current_goal = (gx, gy)
        self._reset_rnn()
        rospy.loginfo(f"New goal: ({gx:.2f}, {gy:.2f}), RNN reset")

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self, event):
        # No goal — no motion
        if self.current_goal is None:
            self._publish_stop()
            return

        # Step-based mode: wait for current action to finish
        if isinstance(self.controller, StepActionController) and not self.controller.is_done:
            twist, done = self.controller.update(self.current_pose)
            if not self.dry_run:
                self.cmd_pub.publish(twist)
            return

        # Need both frames to infer
        if self.latest_rgb is None or self.latest_depth is None:
            self._publish_stop()
            if self.step_count == 0:
                rospy.logwarn_throttle(30, "Waiting for camera frames...")
            return

        # Timeout check
        if self.step_count >= self.max_steps:
            dist = math.hypot(
                self.current_goal[0] - self.current_pose[0],
                self.current_goal[1] - self.current_pose[1],
            )
            rospy.logwarn(
                f"Episode timeout ({self.max_steps} steps). "
                f"Final distance to goal: {dist:.2f}m"
            )
            self.current_goal = None
            self._publish_stop()
            return

        # --- Preprocess ---
        try:
            rgb_np = preprocess_rgb(self.latest_rgb)  # (256,256,3) uint8
            depth_np = preprocess_depth(self.latest_depth)  # (256,256) float32
        except Exception as e:
            rospy.logerr(f"Preprocessing error: {e}")
            self._publish_stop()
            return

        # Safety: depth-based virtual bumper
        h, w = depth_np.shape
        center_depth = depth_np[int(h * 0.3) : int(h * 0.7), int(w * 0.3) : int(w * 0.7)]
        # Exclude zero-value holes from safety check (structured-light artifacts)
        valid_depth = center_depth[center_depth > 0.001]
        if len(valid_depth) > 100 and np.min(valid_depth) * 10.0 < self.min_depth_for_safety:
            rospy.logwarn_throttle(1.0, f"Safety stop: min depth < {self.min_depth_for_safety}m")
            self._publish_stop()
            return

        # --- Goal ---
        goal_vec = compute_pointgoal(self.current_pose, self.current_goal)
        goal_dist, goal_angle = goal_vec

        # Success check
        if goal_dist < self.success_distance:
            rospy.loginfo(
                f"Goal reached! dist={goal_dist:.3f}m < {self.success_distance}m "
                f"(steps={self.step_count})"
            )
            self.current_goal = None
            self.episodes_completed += 1
            self._publish_stop()
            return

        # --- Build observation tensors ---
        rgb_t = torch.from_numpy(rgb_np).unsqueeze(0).to(self.device)  # (1,256,256,3)
        depth_t = (
            torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(-1).to(self.device)
        )  # (1,256,256,1)
        goal_t = torch.tensor([[goal_dist, goal_angle]], dtype=torch.float32, device=self.device)

        observations = {
            "rgb": rgb_t,
            "depth": depth_t,
            "pointgoal_with_gps_compass": goal_t,
        }

        # --- Inference ---
        try:
            t0 = time.perf_counter()
            action_data = self.policy.act(
                observations,
                self.rnn_hidden,
                self.prev_actions,
                self.masks,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._inference_times_ms.append(elapsed_ms)
        except Exception as e:
            rospy.logerr(f"Inference error: {e}")
            self._reset_rnn()
            self._publish_stop()
            return

        action_idx = int(action_data.actions.item())

        # --- Update RNN state ---
        self.rnn_hidden = action_data.rnn_hidden_states
        self.prev_actions = action_data.actions
        self.masks = torch.ones(1, 1, dtype=torch.bool, device=self.device)

        # --- Action → cmd_vel ---
        if isinstance(self.controller, ContinuousActionController):
            twist = self.controller.action_to_twist(action_idx)
        else:
            self.controller.start_action(action_idx, self.current_pose)
            twist, _ = self.controller.update(self.current_pose)

        # --- Publish ---
        self.step_count += 1

        if not self.dry_run:
            self.cmd_pub.publish(twist)

        # Periodic status log
        if self.step_count % 50 == 0:
            avg_lat = (
                sum(self._inference_times_ms[-50:]) / min(len(self._inference_times_ms), 50)
                if self._inference_times_ms
                else 0
            )
            rospy.loginfo(
                f"Step {self.step_count}: "
                f"pose=({self.current_pose[0]:.1f},{self.current_pose[1]:.1f}) "
                f"goal=({self.current_goal[0]:.1f},{self.current_goal[1]:.1f}) "
                f"dist={goal_dist:.1f}m action={ACTION_NAMES[action_idx]} "
                f"lat={elapsed_ms:.0f}ms avg={avg_lat:.0f}ms"
            )

    def _publish_stop(self):
        if not self.dry_run:
            self.cmd_pub.publish(Twist())


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    node = S100PPONode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down.")
    finally:
        rospy.signal_shutdown("Node terminated")


if __name__ == "__main__":
    main()
