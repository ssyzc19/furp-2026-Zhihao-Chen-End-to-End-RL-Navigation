# Weekly Progress Log — Week 2

> 接续 `00_weekly.md`。本文件记录 Week 2 进度，格式沿用模板（Progress / Challenges / Next steps）。

---

### Week 2 — 2026-06-18

**Track:** Track 2 — End-to-End RL Navigation for AMR (Habitat PointNav)

**Attended this week's meeting:** Yes / No (if No, did you email leave? Yes / No)

**Progress this week**
- **Goal 1 — 跑通官方 PointNav PPO Baseline（✅ 100%）**：在云 GPU（RTX A4000）上用 `pointnav/ppo_pointnav_example.yaml` 完成「训练 → 评估 → 视频」全流程。先跑 1000 步训练版 smoke test 验证管线，再正式训练 `total_num_steps=500000 num_environments=4`。
  - 训练末（running avg）：Success 0→**0.852**、SPL 0→**0.736**、DistanceToGoal 7.14m→**0.30m**、Reward -0.04→**3.74**。
  - 评估（ckpt.9，10 episodes）：**SR 0.80 / SPL 0.6465 / NE 0.116m / Reward 4.56**，成功 8 失败 2，导出成功与失败导航视频。
- **Goal 2 — 读懂 Habitat Baselines 架构（✅ 95%）**：从配置链（`run.py → ppo_pointnav.yaml → pointnav_base.yaml → pointnav.yaml`）到调用链（`train → _collect_rollout_step → act → step → insert → compute_returns(GAE) → update`），再到核心源码：`PointNavResNetPolicy`(ResNet18+GRU)、`RolloutStorage`、`ppo.py`(clip/value/entropy)、`nav.py`(DistanceToGoalReward/Success/SPL)。
- **Goal 3 — CartPole↔PointNav 对应表（✅ 完成）**：产出 10 维对应表 + 完整数据流图 `Obs → Policy(ResNet+GRU) → Action → Env → Reward → RolloutStorage → GAE → PPO.update → optimizer`。
- **文献**：读 PPO 论文（L_CLIP 动机）+ R2R 论文（SR/SPL/NE 定义、Seq2Seq baseline、跨场景泛化瓶颈）。

**Challenges & blockers**
- **（已解决）评估环境数不匹配**：ckpt 按 4 env 训练，评估用 1 env → PointNavResNetPolicy tensor shape mismatch；用 4 env → IndexError（只有 2 个 inference workers）；最终 `num_environments=2`（受 val 场景数限制）跑通。
- **范围说明（诚实记录）**：结果基于 habitat-test-scenes（仅 3 场景，val 仅 2），属**流程验证而非标准 benchmark**，SR/SPL 不可与论文直接对比。

**Next steps**
- 把 PPOTrainer / Policy / RolloutStorage / Evaluator 四个组件完全讲通（巩固 Goal 2 剩余 5%）。
- 暂不下载 Gibson（避免一次性 +20~100GB、收益不高）；待源码吃透后再进入 Gibson / HM3D / DD-PPO benchmark。
- 确定本周期「10% 创新」候选方向（如奖励塑形 / 观测设计 / 停止判定改进——失败案例多为差 0.02m 没停在目标区内）。

**Hours spent (optional):**

**Links (optional):**
- 交付文件夹 `Week2/week02_pointnav_ppo/`（notes / experiments / results）
- `results/results_summary.md`（训练+评估指标汇总）、`results/train_evidence.md`、`results/curves/`（4 张指标曲线）
- `results/videos/`：success_ep33_spl0.95 / success_ep29_spl0.90 / failure_ep3 / failure_ep31
