import argparse, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch, habitat
from habitat.utils.visualizations import maps
from omegaconf import OmegaConf, open_dict


def build_env_and_policy(ckpt_path, seed, episode_id):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()

    cfg_dir = os.path.abspath(
        "habitat-baselines/habitat_baselines/config"
    )

    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(
            config_name="pointnav/ppo_pointnav_example",
            overrides=[
                f"habitat.seed={seed}",
                "habitat_baselines.num_environments=1",
            ],
        )

    # enable topdown map
    from habitat.config.default_structured_configs import TopDownMapMeasurementConfig

    tdm_cfg = OmegaConf.structured(
        TopDownMapMeasurementConfig(
            map_resolution=1024,
            draw_source=True,
            draw_border=True,
            draw_shortest_path=True,
            draw_goal_positions=True,
        )
    )

    with open_dict(cfg):
        cfg.habitat.task.measurements.top_down_map = tdm_cfg

    env = habitat.Env(cfg.habitat)
    env.seed(seed)

    for ep in env.episodes:
        if str(ep.episode_id) == str(episode_id):
            env._current_episode = ep
            break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    from habitat_baselines.rl.ppo.policy import PointNavResNetPolicy

    actor_critic = PointNavResNetPolicy.from_config(
        cfg, env.observation_space, env.action_space
    )

    actor_critic.to(device)

    sd = ckpt.get("state_dict", {})
    new_sd = {k.replace("actor_critic.", ""): v for k, v in sd.items()}
    actor_critic.load_state_dict(new_sd, strict=False)
    actor_critic.eval()

    print(f"Loaded ckpt: {ckpt_path}")
    return env, actor_critic, device, cfg


def run_episode(env, actor_critic, device):
    hidden_size = 512
    n_layers = actor_critic.net.num_recurrent_layers

    hidden = torch.zeros(n_layers, 1, hidden_size, device=device)
    prev_action = torch.zeros(1, 1, dtype=torch.long, device=device)
    not_done = torch.ones(1, 1, device=device)

    def to_tensor(obs):
        return {
            k: torch.tensor(np.array(v), dtype=torch.float32, device=device).unsqueeze(0)
            for k, v in obs.items()
        }

    obs = env.reset()

    frames, info = [], {}

    for step in range(500):
        info = env.get_metrics()

        if "top_down_map" in info:
            frames.append(
                maps.colorize_draw_agent_and_fit_to_height(
                    info["top_down_map"], 512
                )
            )

        with torch.no_grad():
            _, action, _, hidden = actor_critic.act(
                to_tensor(obs), hidden, prev_action, not_done, deterministic=True
            )

        prev_action.copy_(action)

        obs, _, done, info = env.step(action.item())

        if done:
            break

    if "top_down_map" in info:
        frames.append(
            maps.colorize_draw_agent_and_fit_to_height(
                info["top_down_map"], 512
            )
        )

    return frames, info, step + 1


def save_png(frames, info, out, episode_id, seed, n_steps):
    if not frames:
        print("No frames collected")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(frames[-1])
    ax.axis("off")

    ok = bool(info.get("success", 0))
    dtg = info.get("distance_to_goal", -1)
    spl = info.get("spl", 0)

    ax.set_title(
        f"Seed {seed} | Episode {episode_id} | {'SUCCESS' if ok else 'FAILURE'}\n"
        f"Steps: {n_steps} | dtg: {dtg:.3f}m | SPL: {spl:.3f}",
        color="green" if ok else "red",
        fontsize=13,
        fontweight="bold",
    )

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved PNG: {out}")


def save_gif(frames, out):
    from PIL import Image

    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=100, loop=0)
    print(f"Saved GIF: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--episode-id", default="23")
    parser.add_argument("--out", default="experiments/topdown.png")
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()

    env, ac, dev, cfg = build_env_and_policy(
        args.ckpt, args.seed, args.episode_id
    )

    frames, info, n_steps = run_episode(env, ac, dev)
    env.close()

    print(
        f"Done: {n_steps} steps | success={info.get('success')} | dtg={info.get('distance_to_goal',-1):.3f}"
    )

    save_png(frames, info, args.out, args.episode_id, args.seed, n_steps)

    if args.gif:
        save_gif(frames, args.out.replace(".png", ".gif"))
