# Week 2 实验结果汇总（Training + Evaluation）

> Track 2 — End-to-End RL Navigation for AMR · Habitat PointNav PPO
> 硬件：云 GPU AutoDL RTX A4000 / Ubuntu 22.04 · `conda env habitat39` (Python 3.9)
> 配置：`pointnav/ppo_pointnav_example.yaml`（habitat-test-scenes）· seed=100

---

## 1. 训练（Day 11）

**命令 overrides**
```
total_num_steps=500000   num_environments=4
checkpoint_folder=data/new_checkpoints/run1
tensorboard_dir=tb/run1  num_checkpoints=10
```

**训练指标变化（running average，跨 3900 updates / 499200 frames）**

| Metric | Start (update 20) | End (update 3900) |
|---|---|---|
| Reward | -0.037 | **3.735** |
| Success | 0.000 | **0.852** |
| SPL | 0.000 | **0.736** |
| DistanceToGoal | 7.141 | **0.303** |

**结论**：reward 大幅上升，success 从 0 升到 85.2%，到目标距离从 7.14m 降到 0.30m —— PPO 策略**成功学会了 PointNav 导航**。

**Checkpoint**：`data/new_checkpoints/run1/` 共 11 个文件（ckpt.0~9 + latest.pth），每个 ~23M，共 246M。TensorBoard 日志 5.3M。

**曲线**（见 `curves/`）：
- `success.png` — 0 → ~0.85，约 100k 步后明显爬升
- `spl.png` — 0 → ~0.73，稳定上升
- `distance_to_goal.png` — 从 5+ 快速降到 ~0.3
- `reward.png` — 稀疏正奖励（到达瞬间），整体噪声大属正常
- `tensorboard_all_4metrics.png` — 四指标合图

---

## 2. 评估（Day 12）

**命令**
```
habitat_baselines.evaluate=True
eval_ckpt_path_dir=data/new_checkpoints/run1/ckpt.9.pth
num_environments=2   test_episode_count=10
```
（ckpt.9 实际训练步数 450560）

**评估结果（10 个 episode，val split）**

| Metric | Value |
|---|---|
| Success Rate (SR) | **0.80** |
| SPL | **0.6465** |
| Distance To Goal | 0.116 m |
| Average Reward | 4.5619 |

成功 8 / 失败 2。

**视频证据**（见 `videos/`）：
| 文件 | 类型 | SPL |
|---|---|---|
| success_ep33_spl0.95.mp4 | ✅ 成功 | 0.95 |
| success_ep29_spl0.90.mp4 | ✅ 成功 | 0.90 |
| failure_ep3.mp4 | ❌ 失败（distance 0.21，未触发 stop） | 0.00 |
| failure_ep31.mp4 | ❌ 失败（distance 0.22，未触发 stop） | 0.00 |

> 失败案例观察：两个 fail 的 distance_to_goal 都在 0.21~0.22m（接近 success_distance=0.2m 的判定阈值），属于"差一点点没停在目标区域内"的**停止判定边界失败**，而非完全走错 —— 这是后续改进的一个候选切入点。

---

## 3. 评估踩坑记录（环境数不匹配）

| 尝试 | num_environments | 结果 |
|---|---|---|
| 1 | 1 | ❌ PointNavResNetPolicy 内 tensor shape mismatch（模型按 4 env 训练）|
| 2 | 4 | ❌ IndexError（evaluator 只起了 2 个 inference workers）|
| 3 | **2** | ✅ 成功，产出 SR/SPL + 视频 |

**教训**：评估时 `num_environments` 受 val 场景数（2 个）与 inference worker 数约束，设为 2 才跑通。

---

## 4. 诚实的范围说明（reproducibility note）

- 本结果基于 **habitat-test-scenes**（仅 3 个场景，val 仅 2 个），**不是** Gibson/HM3D 标准 benchmark。SR/SPL 数值只在该小场景集内有意义，不可直接与论文 benchmark 对比。
- 用途定位：**证明训练+评估全流程跑通、策略确实在学习**（Week 1-2 目标）。正经 benchmark 留待 Week 3+（需另下 Gibson 数据集，20~100GB）。
- 复现要素已记录：配置文件、overrides、seed=100、硬件、ckpt 列表、完整训练/评估日志（见 `experiments/`）。
