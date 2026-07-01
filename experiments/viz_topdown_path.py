"""
Visualize 2D top-down navigation path for a single episode.
Shows: navigable floor, obstacles, robot trajectory, start/goal markers.

Usage (from habitat-lab/ root):
    python experiments/viz_topdown_path.py \
        --ckpt data/new_checkpoints/ckpt.49.pth \
        --seed 300 \
        --episode-id 23 \
        --out experiments/baseline/seed300/viz/ep23_topdown.png [--gif]

Requirements: conda activate habitat39
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import habitat
from habitat.utils.visualizations import maps
from habitat.config.default_structured_configs import TopDownMapMeasurementConfig
from omegaconf import OmegaConf, open_dict


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_cfg(seed: int):
    """Load ppo_pointnav_example config and inject TopDownMap measurement.

    initialize_config_dir only adds the file path but misses
    pkg://habitat_baselines.config (registered via SearchPathPlugin during
    hydra.main but not via the compose API).  initialize_config_module goes
    straight to the installed Python package, so the full defaults chain
    resolves correctly.
    """
    from hydra import compose
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()

    # Primary: package-based — works wherever the script is invoked from
    try:
        from hydra import initialize_config_module
        init_ctx = initialize_config_module(
            config_module="habitat_baselines.config",
            version_base=None,
        )
    except (ImportError, AttributeError):
        # Fallback: file path (must run from habitat-lab/ root)
        from hydra import initialize_config_dir
        cfg_dir = os.path.abspath("habitat-baselines/habitat_baselines/config")
        init_ctx = initialize_config_dir(config_dir=cfg_dir, version_base=None)

    with init_ctx:
        cfg = compose(
            config_name="pointnav/ppo_pointnav_example",
            overrides=[
                f"habitat.seed={seed}",
                "habitat_baselines.num_environments=1",
            ],
        )

    # open_dict disables struct-mode so we can add new measurement keys
    with open_dict(cfg):
        cfg.habitat.task.measurements["top_down_map"] = OmegaConf.structured(
            TopDownMapMeasurementConfig(
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=True,
                draw_view_points=False,
                draw_goal_positions=True,
            )
        )
    return cfg


def _load_policy(cfg, obs_space, action_space, ckpt_path: str, device):
    """Reconstruct PointNavResNetPolicy and load checkpoint weights."""
    # Import from the correct module — resnet_policy, not ppo
    from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy

    actor_critic = PointNavResNetPolicy.from_config(cfg, obs_space, action_space)
    actor_critic.to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", {})

    # Strip "actor_critic." prefix that PPOTrainer adds when saving
    stripped = {k.replace("actor_critic.", ""): v for k, v in state_dict.items()}
    missing, unexpected = actor_critic.load_state_dict(stripped, strict=False)
    if missing:
        print(f"[WARN] missing keys ({len(missing)}): {missing[:3]} ...")
    if unexpected:
        print(f"[WARN] unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")

    actor_critic.eval()
    print(f"Loaded: {ckpt_path}")
    return actor_critic


def _obs_to_tensor(obs_dict, device):
    return {
        k: torch.tensor(np.array(v), dtype=torch.float32, device=device).unsqueeze(0)
        for k, v in obs_dict.items()
    }


# ── main API ─────────────────────────────────────────────────────────────────

def build_env_and_policy(ckpt_path: str, seed: int, episode_id: str):
    cfg = _build_cfg(seed)
    env = habitat.Env(cfg.habitat)
    env.seed(seed)

    # Jump to the requested episode (best-effort)
    for ep in env.episodes:
        if ep.episode_id == str(episode_id):
            env._current_episode = ep
            break
    else:
        print(f"[WARN] episode_id={episode_id} not found; using first episode")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_critic = _load_policy(
        cfg, env.observation_space, env.action_space, ckpt_path, device
    )
    return env, actor_critic, device


def run_episode(env, actor_critic, device, max_steps: int = 500):
    """Roll out one episode deterministically; return frames, final info, step count."""
    hidden_size = actor_critic.recurrent_hidden_size
    n_layers = actor_critic.num_recurrent_layers

    # Shape: (num_layers, batch=1, hidden_size)
    rnn_hidden = torch.zeros(n_layers, 1, hidden_size, device=device)
    # Shape: (batch=1, 1) — long for embedding lookup
    prev_actions = torch.zeros(1, 1, dtype=torch.long, device=device)
    # Float mask: 1 = not done (carry hidden state), 0 = done (reset)
    not_done_masks = torch.ones(1, 1, dtype=torch.float32, device=device)

    obs = env.reset()
    frames, info = [], {}

    for step in range(max_steps):
        # Capture top-down map before the step
        info = env.get_metrics()
        if "top_down_map" in info:
            frame = maps.colorize_draw_agent_and_fit_to_height(
                info["top_down_map"], 512
            )
            frames.append(frame)

        obs_t = _obs_to_tensor(obs, device)
        with torch.no_grad():
            # act() returns PolicyActionData (not a tuple)
            action_data = actor_critic.act(
                obs_t, rnn_hidden, prev_actions, not_done_masks,
                deterministic=True,
            )

        # Unpack PolicyActionData fields
        action = action_data.actions            # shape [1, 1]
        rnn_hidden = action_data.rnn_hidden_states
        prev_actions.copy_(action)

        # env.step expects a scalar int
        obs, _, done, info = env.step(action.squeeze().item())
        if done:
            break

    # Capture final frame after last step
    if "top_down_map" in info:
        frames.append(
            maps.colorize_draw_agent_and_fit_to_height(info["top_down_map"], 512)
        )

    return frames, info, step + 1


# ── output ───────────────────────────────────────────────────────────────────

def save_png(frames, info, out_path: str, episode_id, seed, n_steps):
    if not frames:
        print("[ERROR] No frames collected — top_down_map measurement missing?")
        return

    success = bool(info.get("success", 0))
    dtg = float(info.get("distance_to_goal", -1))
    spl = float(info.get("spl", 0))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(frames[-1])
    ax.axis("off")
    ax.set_title(
        f"Seed {seed} | Episode {episode_id} | {'SUCCESS' if success else 'FAILURE'}\n"
        f"Steps: {n_steps} | dtg: {dtg:.3f} m | SPL: {spl:.3f}",
        color="green" if success else "red",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.legend(
        handles=[
            mpatches.Patch(color="#33BB33", label="Start"),
            mpatches.Patch(color="#FF3333", label="Goal (0.2 m radius)"),
            mpatches.Patch(color="#3399FF", label="Agent path"),
            mpatches.Patch(color="#888888", label="Obstacle / wall"),
        ],
        loc="lower right",
        fontsize=9,
        framealpha=0.8,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved PNG: {out_path}")


def save_gif(frames, out_path: str):
    if not frames:
        return
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(
        out_path,
        save_all=True,
        append_images=imgs[1:],
        duration=100,
        loop=0,
    )
    print(f"Saved GIF: {out_path}")


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--episode-id", type=str, default="23")
    parser.add_argument("--out", default="experiments/topdown_path.png")
    parser.add_argument("--gif", action="store_true", help="Also save animated GIF")
    args = parser.parse_args()

    print(f"Checkpoint : {args.ckpt}")
    print(f"Seed={args.seed}  Episode={args.episode_id}")

    env, actor_critic, device = build_env_and_policy(
        args.ckpt, args.seed, args.episode_id
    )
    frames, info, n_steps = run_episode(env, actor_critic, device)
    env.close()

    print(
        f"Done: {n_steps} steps | "
        f"success={info.get('success')} | "
        f"dtg={info.get('distance_to_goal', -1):.3f} m | "
        f"spl={info.get('spl', 0):.3f}"
    )

    save_png(frames, info, args.out, args.episode_id, args.seed, n_steps)
    if args.gif:
        save_gif(frames, args.out.replace(".png", ".gif"))
