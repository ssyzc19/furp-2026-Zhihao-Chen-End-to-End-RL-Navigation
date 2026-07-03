"""
Compute the PointGoal observation (relative goal in the robot's frame) from
odometry, and load / run the trained PointNav PPO policy.

PointGoal sensor format used by ppo_pointnav_example:
    GOAL_FORMAT = "POLAR", DIMENSIONALITY = 2
    -> observation = [rho, phi]
       rho = straight-line distance robot->goal (m)
       phi = bearing to goal in robot frame (rad), 0 = straight ahead,
             positive = left (counter-clockwise), following Habitat's
             convention.

We take the goal as an (x, y) point in the ODOM frame, supplied at start.
Every step we read the robot pose from /odom and recompute [rho, phi].
"""

import math
import numpy as np
import torch


# ── PointGoal from odom ──────────────────────────────────────────────────────

def compute_pointgoal(robot_x, robot_y, robot_yaw, goal_x, goal_y):
    """
    Return np.array([rho, phi], dtype=float32) — the polar PointGoal
    observation in the robot frame.

    robot_yaw: heading in odom frame (rad), from quaternion.
    """
    dx = goal_x - robot_x
    dy = goal_y - robot_y
    rho = math.hypot(dx, dy)

    # bearing of goal in world frame, minus robot heading -> robot frame
    bearing_world = math.atan2(dy, dx)
    phi = bearing_world - robot_yaw
    # wrap to [-pi, pi]
    phi = math.atan2(math.sin(phi), math.cos(phi))

    return np.array([rho, phi], dtype=np.float32)


def yaw_from_quaternion(qx, qy, qz, qw):
    """Extract yaw (rotation about z) from a quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


# ── Policy wrapper ───────────────────────────────────────────────────────────

class PointNavPolicyRunner:
    """
    Loads a trained PointNavResNetPolicy checkpoint and runs deterministic
    inference on CPU (OrangePi has no CUDA).

    NOTE: importing habitat_baselines requires the habitat packages to be
    installed on the robot.  If that is too heavy for the OrangePi, an
    alternative is to export the policy to TorchScript / ONNX on the training
    machine and load that instead — see README "Lightweight export".
    """

    def __init__(self, ckpt_path, config_path=None, device="cpu"):
        self.device = torch.device(device)
        self._build_policy(ckpt_path, config_path)

        hidden = self.actor_critic.recurrent_hidden_size
        layers = self.actor_critic.num_recurrent_layers
        self.rnn_hidden = torch.zeros(layers, 1, hidden, device=self.device)
        self.prev_action = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        # not-done mask: 1.0 means "carry hidden state"
        self.not_done = torch.ones(1, 1, dtype=torch.float32, device=self.device)

    def _build_policy(self, ckpt_path, config_path):
        from hydra import compose, initialize_config_module
        from hydra.core.global_hydra import GlobalHydra
        from habitat_baselines.rl.ddppo.policy.resnet_policy import (
            PointNavResNetPolicy,
        )
        import gym.spaces as spaces

        GlobalHydra.instance().clear()
        with initialize_config_module(
            config_module="habitat_baselines.config", version_base=None
        ):
            cfg = compose(
                config_name="pointnav/ppo_pointnav_example",
                overrides=["habitat_baselines.num_environments=1"],
            )

        # Build observation/action spaces matching the trained policy.
        # RGB+Depth 256x256 + pointgoal(2). ADJUST if your camera differs.
        obs_space = spaces.Dict({
            "rgb": spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
            "depth": spaces.Box(low=0.0, high=1.0, shape=(256, 256, 1), dtype=np.float32),
            "pointgoal_with_gps_compass": spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            ),
        })
        action_space = spaces.Discrete(4)

        self.actor_critic = PointNavResNetPolicy.from_config(
            cfg, obs_space, action_space
        )
        self.actor_critic.to(self.device)

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        stripped = {k.replace("actor_critic.", ""): v for k, v in sd.items()}
        self.actor_critic.load_state_dict(stripped, strict=False)
        self.actor_critic.eval()

    def reset(self):
        """Call at the start of every new navigation episode."""
        self.rnn_hidden.zero_()
        self.prev_action.zero_()
        self.not_done.fill_(1.0)

    @torch.no_grad()
    def step(self, rgb, depth, pointgoal):
        """
        rgb:   np.uint8   HxWx3
        depth: np.float32 HxWx1  (normalized 0..1)
        pointgoal: np.float32 [rho, phi]

        Returns discrete action id (int).
        """
        obs = {
            "rgb": torch.from_numpy(rgb).unsqueeze(0).to(self.device),
            "depth": torch.from_numpy(depth).unsqueeze(0).to(self.device),
            "pointgoal_with_gps_compass": torch.from_numpy(pointgoal)
            .unsqueeze(0)
            .to(self.device),
        }
        action_data = self.actor_critic.act(
            obs, self.rnn_hidden, self.prev_action, self.not_done,
            deterministic=True,
        )
        action = action_data.actions
        self.rnn_hidden = action_data.rnn_hidden_states
        self.prev_action.copy_(action)
        self.not_done.fill_(1.0)  # stays 1 within an episode
        return int(action.squeeze().item())
