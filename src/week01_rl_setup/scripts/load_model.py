"""Week 1 — 加载已训练的 CartPole PPO 模型并回放。

说明（来自 ../notes/rl_concepts_qa.md）:
    模型文件 (ppo_cartpole.zip) 保存的是"训练成果"（网络参数）；
    代码文件保存的是"训练方法"。重复运行本脚本不会让模型变强——
    因为没有再调用 learn()，只是加载已有参数做推理。

运行:
    python load_model.py                 # 默认加载 ppo_cartpole.zip
    python load_model.py --model ppo_cartpole
"""

import argparse

import gymnasium as gym
from stable_baselines3 import PPO


def replay(model_path: str, episodes: int = 5) -> None:
    env = gym.make("CartPole-v1", render_mode="human")
    model = PPO.load(model_path)

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
    parser.add_argument("--model", type=str, default="ppo_cartpole")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    replay(args.model, args.episodes)
