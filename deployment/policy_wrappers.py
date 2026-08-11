"""
多模型统一接口 — Policy Wrappers

三个模型，一个 act() 签名：
  HabitatWrapper  — ckpt.49.pth (habitat-baselines, RGB+Depth, state_dict)
  JitWrapper      — policy_rgbd_jit.pt / policy_depth_jit.pt (TorchScript)

用法：
  wrapper = HabitatWrapper("/path/to/ckpt.49.pth", device)
  # 或
  wrapper = JitWrapper("/path/to/policy_depth_jit.pt", device, num_channels=1)

  hx = wrapper.get_hidden_state()
  action_data = wrapper.act(observations, hx, prev_actions, masks)
  action_idx = action_data.actions.item()
  hx = action_data.rnn_hidden_states
"""

import math
from collections import namedtuple

import numpy as np
import torch
from gym import spaces

# ---------------------------------------------------------------------------
# 统一返回类型（对齐 habitat 的 action_data）
# ---------------------------------------------------------------------------
ActionData = namedtuple(
    "ActionData", ["actions", "values", "rnn_hidden_states"]
)

# Habitat 模型的 observation space（复刻自 load_and_infer.py）
OBSERVATION_SPACE = spaces.Dict(
    {
        "rgb": spaces.Box(
            low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
        ),
        "depth": spaces.Box(
            low=0.0, high=1.0, shape=(256, 256, 1), dtype=np.float32
        ),
        "pointgoal_with_gps_compass": spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        ),
    }
)
ACTION_SPACE = spaces.Discrete(4)


# ---------------------------------------------------------------------------
# Habitat-baselines checkpoint wrapper
# ---------------------------------------------------------------------------
class HabitatWrapper:
    """加载 habitat-baselines PointNavResNetPolicy 的 .pth checkpoint。"""

    def __init__(self, ckpt_path: str, device: torch.device):
        # 必须在 import 前装好 habitat_stub（由主节点在启动时处理）
        from habitat_baselines.rl.ddppo.policy.resnet_policy import (
            PointNavResNetPolicy,
        )

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        hb = config.habitat_baselines
        agent_name = config.habitat.simulator.agents_order[0]
        policy_config = hb.rl.policy[agent_name]

        self._policy = PointNavResNetPolicy(
            observation_space=OBSERVATION_SPACE,
            action_space=ACTION_SPACE,
            hidden_size=hb.rl.ppo.hidden_size,
            num_recurrent_layers=hb.rl.ddppo.num_recurrent_layers,
            rnn_type=hb.rl.ddppo.rnn_type,
            backbone=hb.rl.ddppo.backbone,
            normalize_visual_inputs="rgb" in OBSERVATION_SPACE.spaces,
            policy_config=policy_config,
            aux_loss_config=hb.rl.auxiliary_losses,
            fuse_keys=None,
        )
        self._policy.load_state_dict(ckpt["state_dict"])
        self._policy.to(device)
        self._policy.eval()

        self._device = device
        self._num_layers = hb.rl.ddppo.num_recurrent_layers
        self._hidden_size = hb.rl.ppo.hidden_size

        print(
            f"[HabitatWrapper] loaded: hidden={self._hidden_size}, "
            f"backbone={hb.rl.ddppo.backbone}, rnn={hb.rl.ddppo.rnn_type}"
        )

    def get_hidden_state(self) -> torch.Tensor:
        """返回零初始化的 RNN 隐状态 (1, num_layers, hidden_size)。"""
        return torch.zeros(
            1, self._num_layers, self._hidden_size, device=self._device
        )

    def act(
        self,
        observations: dict,
        rnn_hidden: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
    ) -> ActionData:
        """调用 habitat policy.act()。"""
        with torch.no_grad():
            return self._policy.act(
                observations,
                rnn_hidden,
                prev_actions,
                masks,
                deterministic=True,
            )


# ---------------------------------------------------------------------------
# TorchScript JIT wrapper (RGBD or Depth-only)
# ---------------------------------------------------------------------------
class JitWrapper:
    """加载 TorchScript 模型，支持 RGBD (4ch) 和深度 (1ch) 两种。"""

    def __init__(
        self, ckpt_path: str, device: torch.device, num_channels: int = 4
    ):
        self._model = torch.jit.load(ckpt_path, map_location=device)
        self._model.eval()
        self._device = device
        self._num_channels = num_channels

        # 从 GRU 权重推断 hidden_size 和 num_layers
        w = self._model.gru.weight_ih_l0
        self._hidden_size = w.shape[0] // 3  # 3 gates × hidden
        self._num_layers = 0
        while hasattr(self._model.gru, f"weight_ih_l{self._num_layers}"):
            self._num_layers += 1

        print(
            f"[JitWrapper] loaded: channels={num_channels}, "
            f"hidden={self._hidden_size}, gru_layers={self._num_layers}"
        )

    def get_hidden_state(self) -> torch.Tensor:
        """返回零初始化的 GRU 隐状态 (num_layers, 1, hidden_size)。"""
        return torch.zeros(
            self._num_layers, 1, self._hidden_size, device=self._device
        )

    def act(
        self,
        observations: dict,
        rnn_hidden: torch.Tensor,
        prev_actions: torch.Tensor,  # JIT 不用，保留接口兼容
        masks: torch.Tensor,  # JIT 不用，保留接口兼容
    ) -> ActionData:
        """调用 JIT forward(x, goal, hx) → (logits, h_new)。"""
        with torch.no_grad():
            # --- 图像 tensor: HWC → CHW ---
            if self._num_channels == 4:
                # RGB (uint8→float[0,1]) + Depth (float[0,1]) → 4ch
                rgb = observations["rgb"].float() / 255.0  # (1,256,256,3)
                depth = observations["depth"]  # (1,256,256,1)
                x = torch.cat([rgb, depth], dim=-1)  # (1,256,256,4)
            else:
                # Depth-only
                x = observations["depth"]  # (1,256,256,1)

            x = x.permute(0, 3, 1, 2)  # NHWC → NCHW

            # --- Goal: (r, θ) → (r, cosθ, sinθ) ---
            g = observations["pointgoal_with_gps_compass"]  # (1, 2)
            r = g[:, 0:1]
            theta = g[:, 1:2]
            goal_3d = torch.cat(
                [r, torch.cos(theta), torch.sin(theta)], dim=-1
            )  # (1, 3)

            logits, h_new = self._model(x, goal_3d, rnn_hidden)

        action_idx = torch.argmax(logits, dim=-1, keepdim=True)  # (1, 1)

        return ActionData(
            actions=action_idx,
            values=torch.zeros(1, 1, device=self._device),
            rnn_hidden_states=h_new,
        )
