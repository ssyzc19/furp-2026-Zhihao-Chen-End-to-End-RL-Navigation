环境

Conda Environment: amr
Python: 3.10.20

Libraries

torch
gymnasium
stable-baselines3

PPO Config

PPO(
    "MlpPolicy",
    env,
    verbose=1
)

Training

total_timesteps=10000