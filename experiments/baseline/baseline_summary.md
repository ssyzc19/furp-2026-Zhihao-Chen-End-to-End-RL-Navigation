# PointNav PPO Baseline — 三 seed 汇总

Config: `pointnav/ppo_pointnav_example.yaml` (habitat-test-scenes) · 500k steps · num_environments=4 · RTX A4000
所有 seed 均为干净从零训练（未加载彼此 checkpoint）。评估均为 ckpt 最末、10 episodes、num_environments=2。

| seed | Success Rate | SPL | NE (m) | Reward |
|---|---|---|---|---|
| 100 | 0.80 | 0.6465 | 0.116 | 4.5619 |
| 200 | 1.00 | 0.8508 | 0.1155 | 4.0708 |
| 300 | 0.80 | 0.7209 | 0.1327 | 5.3155 |
| **mean** | **0.867** | **0.7394** | **0.121** | — |
| **std**  | **0.115** | **0.103** | **0.009** | — |

> 注: seed100 使用 num_checkpoints=10，seed200/300 使用 50（仅存档频率不同，不影响训练）。
