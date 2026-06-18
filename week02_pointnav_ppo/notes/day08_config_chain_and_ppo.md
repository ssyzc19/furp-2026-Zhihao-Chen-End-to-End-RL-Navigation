# Day 8 — 训练入口与配置链

## 训练入口
```
python -m habitat_baselines.run --config-name=pointnav/ppo_pointnav.yaml
```
调用链：`run.py → main(cfg) → execute_exp(cfg,"train") → 读 trainer_name → PPOTrainer → trainer.train()`

## 配置链（4 层 defaults 叠加）
```
ppo_pointnav.yaml        PPO 超参 + 实验设置（lr/clip/num_steps、num_environments、total_num_steps、ckpt/tb 路径）
  └ pointnav_gibson.yaml 指定 Gibson 数据集
     └ pointnav_base.yaml  RGB+Depth agent
        └ pointnav.yaml    任务定义：动作/观测/奖励/指标
```
`pointnav.yaml` 任务定义：
- 动作：stop / move_forward / turn_left / turn_right
- 观测：pointgoal_with_gps_compass_sensor（+ RGB/Depth）
- 奖励：distance_to_goal_reward
- 指标：success / spl

## 关键超参含义
| 参数 | 含义 |
|---|---|
| num_environments | 并行环境数 |
| total_num_steps | 总训练步数 |
| lr / clip_param | 学习率 / PPO 裁剪范围 |
| num_steps | 每次更新前采样步数 |
| ppo_epoch / num_mini_batch | 同批数据训练轮数 / 小批数 |
| gamma | 折扣因子 |
| checkpoint_folder / tensorboard_dir | 模型 / 日志目录 |

## PPO 为什么出现
改进策略梯度方法的不足：TRPO 稳定但实现复杂（二阶优化）；vanilla PG 数据效率低、易崩；Q-learning 不擅长连续控制。PPO 用**一阶优化 + 带裁剪的代理目标 L_CLIP** 限制更新步长，兼顾稳定性与实现简易。

## PPO 在本配置里怎么跑（Gibson 配置）
6 env × 128 步 = 一次 rollout 收 768 条经验 → GAE(gamma0.99,tau0.95) 算 advantage → clip_param0.2 构造裁剪目标 → 拆 num_mini_batch=2 小批 × ppo_epoch=4 轮 → Adam(lr2.5e-4) 更新。
