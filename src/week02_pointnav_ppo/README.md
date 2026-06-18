# Week 2 — Habitat PointNav PPO Baseline（交付物）

**Track:** Track 2 — End-to-End RL Navigation for AMR
**区间:** Week 2 (Day 8–14)
**主线:** ① 跑通官方 PointNav PPO baseline ② 读懂 Habitat Baselines 架构 ③ 建立 CartPole↔PointNav 对应

## 进度自评

| 目标 | 状态 |
|---|---|
| Goal 1 — 跑通 Habitat PPO（训练 + 评估 + 视频） | ✅ 100% |
| Goal 2 — 读懂源码调用链与架构 | ✅ 95% |
| Goal 3 — CartPole↔PointNav 对应表 + 数据流图 | ✅ 完成 |

**一句话成果**：在云 GPU 上完成 PointNav PPO 训练（success 0→0.85），评估得 **SR 0.80 / SPL 0.6465**，导出成功/失败导航视频，并从配置链到核心源码（Policy/RolloutStorage/PPO/Reward）打通了完整训练链路的理解。

## 目录结构

```
week02_pointnav_ppo/
├── README.md                          本文件：索引 + 进度 + 结果
├── notes/                             一周学习笔记
│   ├── day08_config_chain_and_ppo.md    配置链 + PPO 动机 + 命令→配置→任务映射
│   ├── day09_call_chain_obs_action.md   obs/action 来源 + PPO 调用链
│   ├── day10_trainer_mainloop.md        PPOTrainer 主循环 + 三大件关系
│   ├── day13_policy_resnet_gru.md       Policy 架构：ResNet18 + PointGoal embed + GRU
│   ├── day14_cartpole_vs_pointnav.md    ★ 对应表 + 数据流图（Goal 2/3 验收）
│   └── reading_R2R_Anderson_CVPR2018.md R2R 论文阅读笔记（指标/Seq2Seq/泛化瓶颈）
├── experiments/                       运行过程日志
│   ├── smoke_test_1000steps_log.txt     训练版 smoke test（1000 步，验证管线）
│   └── eval_ckpt9_log.txt               Day12 评估完整日志（含完整 config）
├── results/                           结果与证据
│   ├── results_summary.md               ★ 训练 + 评估指标汇总（含范围说明）
│   ├── train_evidence_raw.md            训练原始证据（ckpt 列表 / 指标摘录）
│   ├── eval_summary_day12_raw.md        评估原始小结
│   └── curves/                          4 张指标曲线 + TensorBoard 合图
│       ├── success.png  spl.png  distance_to_goal.png  reward.png
│       └── tensorboard_all_4metrics.png
└── videos/                            评估导航视频（ckpt.9）
    ├── success_ep33_spl0.95.mp4  success_ep29_spl0.90.mp4
    └── failure_ep3.mp4           failure_ep31.mp4
```

## 关键结果（速览）

| 阶段 | SR | SPL | DistanceToGoal | Reward |
|---|---|---|---|---|
| 训练末（running avg, 500k 步） | 0.852 | 0.736 | 0.303 m | 3.735 |
| 评估（ckpt.9, 10 episodes） | **0.80** | **0.6465** | 0.116 m | 4.5619 |

> ⚠️ 基于 habitat-test-scenes（3 场景）的流程验证，**非** Gibson 标准 benchmark；详见 `results/results_summary.md` 的范围说明。

## 复现要素

- 配置：`pointnav/ppo_pointnav_example.yaml` · seed=100 · RTX A4000 / Python 3.9 (habitat39)
- 训练 overrides：`total_num_steps=500000 num_environments=4 num_checkpoints=10`
- 评估：`evaluate=True eval_ckpt_path_dir=.../ckpt.9.pth num_environments=2 test_episode_count=10`
- 完整日志见 `experiments/`，指标/ckpt 证据见 `results/`

## 参考论文（PDF 仍在上层 Week2/ 目录）

PPO `1707.06347` · R2R `Anderson CVPR2018` · Speaker-Follower `1806.02724` · VLN-CE `2004.02857` · RxR `2010.07954` · Sutton & Barto《RL: An Introduction》

## Week 3 方向（来自交付自评建议）

先把 PPOTrainer / Policy / RolloutStorage / Evaluator 完全讲通，**暂不下载 Gibson**（避免一次性增加 20~100GB、收益不高）；之后再进入 Gibson / HM3D / DD-PPO benchmark / VLN。
