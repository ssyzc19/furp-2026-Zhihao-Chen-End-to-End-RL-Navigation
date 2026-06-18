# 训练证据（Day 11）

## 配置
- Config: `pointnav/ppo_pointnav_example.yaml`
- Overrides: `total_num_steps=500000  num_environments=4  num_checkpoints=10  checkpoint_folder=data/new_checkpoints/run1  tensorboard_dir=tb/run1`
- Hardware: RTX A4000 · Python 3.9.25 · seed=100

## 指标变化
| Metric | Start (update 20, 2560 frames) | End (update 3900, 499200 frames) |
|---|---|---|
| Reward | -0.037 | 3.735 |
| Success | 0.000 | 0.852 |
| SPL | 0.000 | 0.736 |
| DistanceToGoal | 7.141 | 0.303 |

Success 从 0 升到 85.2%，到目标距离从 7.14m 降到 0.30m —— PPO 成功学会 PointNav 导航。

## Checkpoint
`data/new_checkpoints/run1/`：ckpt.0~9 + latest.pth，各 ~23M，共 246M。TensorBoard 日志 5.3M（`tb/run1`）。

```
ckpt.0.pth … ckpt.9.pth   latest.pth
```

> 曲线见 `curves/`，训练+评估完整汇总见 `results_summary.md`。
