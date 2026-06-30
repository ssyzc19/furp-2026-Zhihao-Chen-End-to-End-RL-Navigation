"""
Visualize 2D top-down navigation path for a single episode.
Shows: navigable floor, obstacles, robot trajectory, start/goal markers.

Usage (on cloud GPU, from habitat-lab/ root):
    python experiments/viz_topdown_path.py \
        --ckpt data/new_checkpoints/ckpt.49.pth \
        --seed 300 \
        --episode-id 23 \
        --out experiments/baseline/seed300/viz_ep23_topdown.png

Requirements: conda activate habitat39
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

import habitat
from habitat.config.default_structured_configs import (
    TopDownMapMeasurementConfig,
)
from habitat.tasks.nav.nav import TopDownMap
from habitat.utils.visualizations import maps
from habitat_baselines.config.default_structured_configs import (
    PPOConfig,
)
from omegaconf import OmegaConf, DictConfig
import torch

# ── colour constants (from habitat maps.py) ──────────────────────────────────
MAP_INVALID_POINT = 0    # black / out-of-bounds
MAP_VALID_POINT   = 1    # navigable floor
MAP_BORDER_INDICATOR = 2
MAP_SOURCE_POINT_INDICATOR = 4   # start  (green)
MAP_TARGET_POINT_INDICATOR = 6   # goal   (red)
MAP_SHORTEST_PATH_COLOR = 7
MAP_VIEW_POINT_INDICATOR = 8     # current agent pos (blue)
MAP_TARGET_BOUNDING_BOX = 9


def build_env(ckpt_path: str, seed: int, episode_id: str):
    """
    Build a Habitat Env with TopDownMap measurement.
    Returns (env, policy).
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()

    cfg_dir = os.path.abspath(
        "habitat-baselines/habitat_baselines/config"
    )
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(
            config_name="pointnav/ppo_pointnav",
            overrides=[
                f"habitat.seed={seed}",
                "habitat_baselines.num_environments=1",
                "habitat_baselines.eval.use_ckpt_config=True",
                f"habitat_baselines.eval.video_option=[]",  # no video here
            ],
        )

    # Inject TopDownMap measurement
    tdm_cfg = TopDownMapMeasurementConfig(
        map_resolution=1024,
        draw_source=True,
        draw_border=True,
        draw_shortest_path=True,
        draw_view_points=False,
        draw_goal_positions=True,
    )
    OmegaConf.update(
        cfg,
        "habitat.task.measurements.top_down_map",
        OmegaConf.structured(tdm_cfg),
        merge=True,
    )

    env = habitat.Env(cfg.habitat)
    env.seed(seed)

    # Jump to target episode
    episodes = env.episodes
    target_ep = None
    for ep in episodes:
        if ep.episode_id == str(episode_id):
            target_ep = ep
            break

    if target_ep is None:
        print(f"[WARN] episode_id={episode_id} not found; using first episode")
    else:
        env._current_episode = target_ep

    # Load policy
    from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We'll run the env manually without the full trainer.
    # Load checkpoint weights manually.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    return env, ckpt, device, cfg


def run_episode_collect_maps(env, ckpt, device, cfg):
    """Run one episode, collect top-down map at each step."""
    from habitat_baselines.rl.ppo import PointNavResNetPolicy

    # Build policy
    obs_space = env.observation_space
    action_space = env.action_space

    # Load actor-critic from checkpoint
    actor_critic = ckpt.get("extra_state", {})
    # Try loading via trainer shortcut
    state_dict = ckpt.get("state_dict", None)

    obs = env.reset()

    # Simple greedy rollout using checkpoint's actor_critic
    # We recreate the policy architecture to load weights
    from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
    from hydra.utils import instantiate

    # Minimal policy reconstruction
    policy_cfg = cfg.habitat_baselines.rl.policy
    hidden_size = 512

    # Import policy class
    from habitat_baselines.rl.ppo.policy import PointNavResNetPolicy as PNRP
    actor_critic_model = PNRP.from_config(
        cfg,
        obs_space,
        action_space,
    )
    actor_critic_model.to(device)
    if state_dict is not None:
        # strip "actor_critic." prefix if present
        new_sd = {}
        for k, v in state_dict.items():
            new_key = k.replace("actor_critic.", "")
            new_sd[new_key] = v
        actor_critic_model.load_state_dict(new_sd, strict=False)
    actor_critic_model.eval()

    # Convert obs to tensors
    import gym.spaces as spaces

    def obs_to_tensor(obs_dict):
        result = {}
        for k, v in obs_dict.items():
            arr = np.array(v)
            result[k] = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)
        return result

    # Recurrent state
    num_recurrent_layers = actor_critic_model.net.num_recurrent_layers
    hidden = torch.zeros(
        num_recurrent_layers, 1, hidden_size, device=device
    )
    prev_action = torch.zeros(1, 1, dtype=torch.long, device=device)
    not_done = torch.ones(1, 1, device=device)

    topdown_frames = []
    positions = []

    MAX_STEPS = 500
    step = 0
    done = False

    while not done and step < MAX_STEPS:
        # Collect top-down map
        info = env.get_metrics()
        if "top_down_map" in info:
            tdm = info["top_down_map"]
            frame = maps.colorize_draw_agent_and_fit_to_height(
                tdm, 512
            )
            topdown_frames.append(frame)

        # Get action
        obs_tensor = obs_to_tensor(obs)
        with torch.no_grad():
            _, action, _, hidden = actor_critic_model.act(
                obs_tensor,
                hidden,
                prev_action,
                not_done,
                deterministic=True,
            )
        prev_action.copy_(action)

        obs, reward, done, info = env.step(action.item())
        step += 1

    # Final frame
    if "top_down_map" in info:
        tdm = info["top_down_map"]
        frame = maps.colorize_draw_agent_and_fit_to_height(tdm, 512)
        topdown_frames.append(frame)

    return topdown_frames, info, step


def save_final_map(topdown_frames, info, out_path, episode_id, seed, n_steps):
    """Save the final top-down map with path as PNG."""
    if not topdown_frames:
        print("No frames collected.")
        return

    final_frame = topdown_frames[-1]
    success = info.get("success", 0)
    dtg = info.get("distance_to_goal", -1)
    spl = info.get("spl", 0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(final_frame)
    ax.axis("off")

    result_str = "SUCCESS" if success else "FAILURE"
    color = "green" if success else "red"
    ax.set_title(
        f"Seed {seed} | Episode {episode_id} | {result_str}\n"
        f"Steps: {n_steps} | dtg: {dtg:.3f}m | SPL: {spl:.3f}",
        fontsize=13,
        color=color,
        fontweight="bold",
        pad=10,
    )

    # Legend
    legend_elements = [
        mpatches.Patch(color="#33BB33", label="Start"),
        mpatches.Patch(color="#FF3333", label="Goal (0.2m radius)"),
        mpatches.Patch(color="#3399FF", label="Agent path"),
        mpatches.Patch(color="#888888", label="Obstacle / wall"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=9,
        framealpha=0.8,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def save_all_frames_gif(topdown_frames, out_path):
    """Optionally save as animated GIF (requires pillow)."""
    from PIL import Image
    imgs = [Image.fromarray(f) for f in topdown_frames]
    imgs[0].save(
        out_path,
        save_all=True,
        append_images=imgs[1:],
        duration=100,
        loop=0,
    )
    print(f"Saved GIF: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--episode-id", type=str, default="23",
                        help="Episode ID to visualize")
    parser.add_argument("--out", default="experiments/topdown_path.png",
                        help="Output PNG path")
    parser.add_argument("--gif", action="store_true",
                        help="Also save animated GIF")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    print(f"Seed: {args.seed}, Episode: {args.episode_id}")

    env, ckpt, device, cfg = build_env(args.ckpt, args.seed, args.episode_id)

    print("Running episode...")
    frames, info, n_steps = run_episode_collect_maps(env, ckpt, device, cfg)
    env.close()

    print(f"Episode done in {n_steps} steps | success={info.get('success')} | dtg={info.get('distance_to_goal'):.3f}m")

    out_png = args.out
    save_final_map(frames, info, out_png, args.episode_id, args.seed, n_steps)

    if args.gif:
        gif_path = out_png.replace(".png", ".gif")
        save_all_frames_gif(frames, gif_path)
