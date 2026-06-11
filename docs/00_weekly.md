# Weekly Progress Log

> Update this file **every week**. Add a new entry at the top for each week.
> This is the first thing we check during review. Keep it honest and specific — it also feeds your attendance record (Rule 1).

**How to use:** copy the *Week template* block below for each new week. Newest week goes at the top.

---

## Week template — copy me

### Week N — YYYY-MM-DD

**Attended this week's meeting:** Yes / No (if No, did you email leave? Yes / No)

**Progress this week**
- _What did you actually do / finish?_

**Challenges & blockers**
- _What got in the way? What are you stuck on?_

**Next steps**
- _What will you do next week?_

**Hours spent (optional):** _e.g. 6h_

**Links (optional):** _commits, notebooks, docs, datasets..._

---

<!-- =================  YOUR ENTRIES BELOW  ================= -->

### Week 1 — 2026-06-11

**Track:** Track 2 — End-to-End RL Navigation for AMR (Habitat PointNav)

**Attended this week's meeting:** Yes 

**Progress this week**
- 选定主方向：Track 2（端到端强化学习导航，Habitat PointNav），并完成 Week 1 环境搭建 + smoke test，已 100% 达标。
- **RL 基础栈跑通**：用 Stable-Baselines3 训练 CartPole PPO（`ppo_cartpole.py`，MlpPolicy，10k 步），Episode Reward 稳定到 500；保存/加载模型并 `render_mode="human"` 回放（`load_model.py`）。
- **环境记录**：云 GPU 服务器（RTX A4000 / 原生 Ubuntu 22.04，真实 EGL）+ 本地 WSL2 做读码/CartPole；conda env `habitat39`（Python 3.9）；Habitat-Sim **0.3.3** ✅、Habitat-Lab **0.3.3** ✅、PyTorch ✅、Gymnasium ✅，全部 `import` 通过。
- **数据集就位**：`habitat_test_scenes` + `habitat_test_pointnav_dataset` 下载并迁移到云服务器，修正目录结构 `data/scene_datasets/habitat-test-scenes/`。
- **Habitat smoke test 全链路打通**：`python examples/shortest_path_follower_example.py` 成功输出 `Environment creation successful` / `Agent stepping around inside environment` / `Episode finished`，并生成带 RGB/Depth 渲染的 **`trajectory.mp4`**；运行时确认 `Renderer: NVIDIA RTX A4000` / `OpenGL version: 4.6`，GPU 渲染通路正常。
- **概念理解**：读懂 PointNav 配置链（`pointnav_habitat_test.yaml → pointnav_base.yaml → pointnav.yaml`）、任务定义（actions：`stop/move_forward/turn_left/turn_right`；measurements：`distance_to_goal/success/spl/distance_to_goal_reward`；sensor：`pointgoal_with_gps_compass`）；建立数据流心智模型 `Observation(RGB+Depth+PointGoal) → PPO → Action → Env → Reward`。

**Challenges & blockers**
- **（已解决）WSL2 EGL/OpenGL 渲染**：本地 WSL2 + RTX 4060 下 `CameraSensor` 无法创建 EGL 上下文（`unable to find CUDA device 0 among 1 EGL devices`，`glxinfo` 显示 `llvmpipe` 软件渲染）。用分层测试法（Dataset→Sim→Scene→Sensor）定位到故障在渲染链路而非 RL/Habitat-Lab 本身 → 按 time-box 原则不在 WSL2 死磕，**切换到云 GPU 实例（RTX A4000 / 原生 Ubuntu）后一次成功**。
- **（已解决）数据集下载**：云端访问 `huggingface.co` 超时 → 改用「本地 WSL → Windows 桌面 → Jupyter 上传 → 云服务器」中转 `.tar.gz`。
- **（已解决）`*.glb not found`**：把场景文件从 `data/versioned_data/...` 复制到 Habitat 要求的 `data/scene_datasets/habitat-test-scenes/`。
- **遗留小项**：habitat-lab commit hash 待补（`git -C ~/habitat-lab rev-parse HEAD`）以满足可复现要求。

**Next steps**
- **Goal 1 — 跑通官方 PointNav PPO Baseline（第一次导航训练）**：`python -m habitat_baselines.run --config-name=.../ppo_pointnav.yaml`，先小步数 + 少 env（`total_num_steps≈5e4–1e5`、`num_environments=4`）打通 `rollout → update → checkpoint → eval`，拿到训练日志、TensorBoard 曲线、ckpt 和一份 eval 的 SR/SPL。
- **Goal 2 — 读懂 Habitat Baselines 架构**：按「训练循环→数据结构→网络→损失」读 `ppo_trainer.py` / `policy.py` / `rollout_storage.py` / reward / observation。
- **Goal 3 — 建立 CartPole PPO ↔ PointNav PPO 对应表**：每读一段就问「这在 CartPole 里对应哪一步？SB3 替我封装了什么？」，产出对应表 + `Obs→PPO→Action→Env→Reward` 数据流图。

**Hours spent (optional):**

**Links (optional):**
- `trajectory.mp4`（PointNav 最短路径导航轨迹，带 RGB/Depth 渲染）
- `ppo_cartpole.py` / `load_model.py` / `ppo_cartpole.zip`（CartPole PPO 训练 + 回放）
- 详细排查与复盘见进度报告（WSL2 EGL blocker 复盘、Day8–14 计划、Goal 1–3）
