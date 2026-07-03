#!/usr/bin/env python3
"""
Export a trained PointNav PPO policy to TorchScript so it can run on the
robot (OrangePi / ARM) with ONLY `torch` installed — no habitat packages.

Why a wrapper:
    actor_critic.act() returns a PolicyActionData dataclass, takes a dict of
    observations, and carries RNN hidden state. torch.jit.trace cannot handle
    dataclass outputs or dict logic cleanly, so we wrap the network in a thin
    nn.Module whose forward() takes flat tensors and returns flat tensors:

        forward(rgb, depth, pointgoal, rnn_hidden, prev_action, masks)
            -> (action, rnn_hidden_out)

Run on the TRAINING machine (where habitat_baselines is installed):
    python export_policy.py \
        --ckpt data/new_checkpoints/seed300/ckpt.49.pth \
        --out  pointnav_policy_ts.pt

Then copy pointnav_policy_ts.pt to the robot. The robot loads it with
plain torch.jit.load — see load snippet at the bottom of this file.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn


class PolicyTSWrapper(nn.Module):
    """Flat-tensor-in, flat-tensor-out wrapper around PointNavResNetPolicy."""

    def __init__(self, actor_critic):
        super().__init__()
        self.net = actor_critic.net
        self.action_distribution = actor_critic.action_distribution

    def forward(self, rgb, depth, pointgoal, rnn_hidden, prev_action, masks):
        # Rebuild the observation dict the network expects.
        observations = {
            "rgb": rgb,
            "depth": depth,
            "pointgoal_with_gps_compass": pointgoal,
        }
        features, rnn_hidden_out, _ = self.net(
            observations, rnn_hidden, prev_action, masks
        )
        distribution = self.action_distribution(features)
        # deterministic action = argmax of the categorical logits
        action = distribution.mode()
        return action, rnn_hidden_out


def build_actor_critic(cfg, obs_space, action_space, ckpt_path, device):
    from habitat_baselines.rl.ddppo.policy.resnet_policy import (
        PointNavResNetPolicy,
    )

    actor_critic = PointNavResNetPolicy.from_config(cfg, obs_space, action_space)
    actor_critic.to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    stripped = {k.replace("actor_critic.", ""): v for k, v in sd.items()}
    missing, unexpected = actor_critic.load_state_dict(stripped, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)} (first: {missing[:3]})")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")
    actor_critic.eval()
    return actor_critic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="pointnav_policy_ts.pt")
    p.add_argument("--rgb-h", type=int, default=256)
    p.add_argument("--rgb-w", type=int, default=256)
    p.add_argument("--method", choices=["trace", "script"], default="trace")
    args = p.parse_args()

    device = torch.device("cpu")  # export on CPU so it loads on ARM CPU

    # ── build config + spaces (must match training) ──
    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    import gym.spaces as spaces

    GlobalHydra.instance().clear()
    with initialize_config_module(
        config_module="habitat_baselines.config", version_base=None
    ):
        cfg = compose(
            config_name="pointnav/ppo_pointnav_example",
            overrides=["habitat_baselines.num_environments=1"],
        )

    obs_space = spaces.Dict({
        "rgb": spaces.Box(0, 255, (args.rgb_h, args.rgb_w, 3), dtype=np.uint8),
        "depth": spaces.Box(0.0, 1.0, (args.rgb_h, args.rgb_w, 1), dtype=np.float32),
        "pointgoal_with_gps_compass": spaces.Box(
            -np.inf, np.inf, (2,), dtype=np.float32
        ),
    })
    action_space = spaces.Discrete(4)

    print(f"Loading checkpoint: {args.ckpt}")
    actor_critic = build_actor_critic(
        cfg, obs_space, action_space, args.ckpt, device
    )

    wrapper = PolicyTSWrapper(actor_critic).eval()

    hidden = actor_critic.recurrent_hidden_size
    layers = actor_critic.num_recurrent_layers

    # ── example inputs for tracing ──
    example = (
        torch.zeros(1, args.rgb_h, args.rgb_w, 3, dtype=torch.float32),   # rgb
        torch.zeros(1, args.rgb_h, args.rgb_w, 1, dtype=torch.float32),   # depth
        torch.zeros(1, 2, dtype=torch.float32),                          # pointgoal
        torch.zeros(layers, 1, hidden, dtype=torch.float32),             # rnn_hidden
        torch.zeros(1, 1, dtype=torch.long),                             # prev_action
        torch.ones(1, 1, dtype=torch.float32),                           # masks
    )

    # sanity check the wrapper runs
    with torch.no_grad():
        act, hid = wrapper(*example)
    print(f"Wrapper OK. action shape={tuple(act.shape)}, hidden shape={tuple(hid.shape)}")

    # ── export ──
    with torch.no_grad():
        if args.method == "trace":
            ts = torch.jit.trace(wrapper, example, check_trace=False)
        else:
            ts = torch.jit.script(wrapper)
    ts.save(args.out)
    print(f"Saved TorchScript: {args.out}")

    # ── verify it reloads and matches ──
    reloaded = torch.jit.load(args.out)
    with torch.no_grad():
        act2, _ = reloaded(*example)
    match = torch.equal(act, act2)
    print(f"Reload check: action match = {match}")
    print("\nDone. Copy this file to the robot:")
    print(f"    scp {args.out} orangepi@<robot_ip>:/home/orangepi/furp_code/gaiouka/")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# ROBOT-SIDE LOADING (only needs `torch`, no habitat):
#
#   import torch, numpy as np
#   policy = torch.jit.load("pointnav_policy_ts.pt")
#   policy.eval()
#
#   layers, hidden = 1, 512     # from the trained policy (GRU, 1 layer, 512)
#   rnn_hidden  = torch.zeros(layers, 1, hidden)
#   prev_action = torch.zeros(1, 1, dtype=torch.long)
#   masks       = torch.ones(1, 1)                 # 1 within an episode
#
#   # each step:
#   rgb   = torch.from_numpy(rgb_256x256x3.astype(np.float32)).unsqueeze(0)
#   depth = torch.from_numpy(depth_256x256x1.astype(np.float32)).unsqueeze(0)
#   pg    = torch.from_numpy(np.array([rho, phi], np.float32)).unsqueeze(0)
#
#   with torch.no_grad():
#       action, rnn_hidden = policy(rgb, depth, pg, rnn_hidden, prev_action, masks)
#   action_id = int(action.squeeze().item())
#   prev_action.copy_(action)
# ─────────────────────────────────────────────────────────────────────────────
