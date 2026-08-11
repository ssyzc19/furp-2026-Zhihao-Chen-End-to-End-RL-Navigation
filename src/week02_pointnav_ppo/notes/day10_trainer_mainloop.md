# Day 10 — PPOTrainer 主循环与三大件

## 三大件
| 组件 | 角色 | 内容 |
|---|---|---|
| actor_critic | 机器人大脑 | RGB+Depth+Goal → Action + Value |
| rollouts | 经验记录本 | obs / action / reward / done … |
| PPO (updater) | 教练 | 分析经验、算损失、更新参数 |

创建顺序：`train() → _create_agent() → SingleAgentAccessMgr → _init_policy_and_updater() → _create_policy()(actor_critic) + _create_updater()(PPO)`

## 主循环
```
PPOTrainer.train()
 └ for update:
     _compute_actions_and_step_envs()   actor_critic.act() → env.step()
     _collect_environment_result()      → RolloutStorage
     _update_agent()                    compute_returns() → PPO.update() → optimizer.step()
     _training_log()                    → TensorBoard
     save_checkpoint()
```

## 一次更新的数据流
```
Observation → actor_critic → Action → Habitat Env → Reward / Next Obs
   → Rollout Buffer → PPO Optimizer → 更新 actor_critic 参数
```
