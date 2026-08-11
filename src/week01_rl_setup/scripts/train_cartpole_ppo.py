"""Week 1 smoke test — CartPole PPO (Stable-Baselines3).

Track 2 (RL Navigation) 的最小 RL 训练验证：训练 PPO 学会平衡 CartPole，
保存模型，再加载回放并统计 Episode Reward。

对应笔记: ../notes/rl_concepts_qa.md
训练日志: ../results/cartpole_train_log_10k.txt / cartpole_train_log_50k.txt
配置:     ../configs/cartpole_ppo_config.md

环境:
    conda env: amr (Python 3.10) — 本地 Windows / WSL2
    依赖: torch, gymnasium, stable-baselines3

运行:
    python train_cartpole_ppo.py            # 训练 + 保存
    python train_cartpole_ppo.py --timesteps 50000
"""

import argparse

import gymnasium as gym
from stable_baselines3 import PPO


def train(timesteps: int, save_path: str) -> PPO:
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
    return model


def evaluate(model: PPO, episodes: int = 5) -> None:
    """加载好的策略自主运行，打印每个 episode 的总奖励。"""
    env = gym.make("CartPole-v1", render_mode="human")
    obs, info = env.reset()
    total_reward = 0
    finished = 0
    while finished < episodes:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print("Episode total reward:", total_reward)
            total_reward = 0
            finished += 1
            obs, info = env.reset()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--save-path", type=str, default="ppo_cartpole")
    parser.add_argument("--eval-episodes", type=int, default=5)
    args = parser.parse_args()

    model = train(args.timesteps, args.save_path)
    evaluate(model, args.eval_episodes)
