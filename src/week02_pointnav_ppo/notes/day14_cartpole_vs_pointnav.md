# Day 14 — CartPole PPO ↔ PointNav PPO 对应表（Goal 3）

> 用已知的 CartPole（SB3 封装）作参照系，理解 Habitat PointNav 里每一步对应什么、代码在哪。

## 对应表

| 维度 | CartPole (SB3) | PointNav (Habitat) | 代码位置 |
|---|---|---|---|
| Observation | 4 维状态向量 | RGB(256×256) + Depth(256×256) + PointGoal(2维极坐标) | `resnet_policy.py` |
| Action | 离散 2（左/右推） | 离散 4：Stop / Forward / Left / Right | `pointnav.yaml` |
| Reward | 每步 +1 | DistanceToGoalReward + SuccessReward(2.5) + Slack(-0.01) | `nav.py` |
| Policy | MLP | ResNet18 + GRU | `resnet_policy.py` |
| Memory | 无 | GRU Hidden State | `resnet_policy.py` |
| Rollout Buffer | SB3 内部 | RolloutStorage | `rollout_storage.py` |
| Advantage | SB3 内部 GAE | `compute_returns()`（GAE, gamma0.99 tau0.95） | `rollout_storage.py` |
| PPO Loss | SB3 封装 | Clip(0.1) + Value(0.5) + Entropy(0.01) | `ppo.py` |
| Update | `model.learn()` 一行 | `PPOTrainer.train()` 显式循环 | `ppo_trainer.py` |
| Metrics | Episode Reward | Success / SPL / DistanceToGoal | `nav.py` |

**一句话总结**：SB3 在 CartPole 里替我封装掉的（rollout buffer、GAE、PPO loss、训练循环），在 Habitat 里都被显式拆成 `RolloutStorage` / `compute_returns` / `ppo.py` / `ppo_trainer.py`，可以逐个读到。

---

## 数据流图（Goal 2 验收图）

```text
RGB / Depth / PointGoal / PrevAction
              │
              ▼
     PointNavResNetPolicy
              │
   ┌──────────┼─────────────────┐
   ├─ ResNet18 (RGB+Depth → 512维视觉特征)
   ├─ Goal Embedding (distance,cosθ,sinθ → 32维)
   ├─ PrevAction Embedding (→ 32维)
              │  torch.cat → 576维
              ▼
            GRU (→ 512维隐藏状态，提供记忆)
              │
      ┌───────┴───────┐
      ▼               ▼
    Actor           Critic
      │               │
      ▼               ▼
   Action           Value
      │
      ▼
   Env.step()
      │
      ▼
   Reward  ──►  RolloutStorage
                    │
                    ▼
              GAE → Return / Advantage
                    │
                    ▼
              PPO.update()
                    │
              ┌─────┼─────┐
            Clip   Value  Entropy   (三项 loss 相加)
              └─────┼─────┘
                    ▼
              backward() → optimizer.step()
                    │
                    ▼
              更新 actor_critic 网络参数
```

**完整链路口述**：`Obs → Policy(ResNet18+GRU) → Action → Env → Reward → RolloutStorage → GAE → PPO.update(Clip+Value+Entropy) → optimizer.step → 更新 Policy`。
