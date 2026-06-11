# Week 1 — RL Setup & Smoke Test

**Track:** Track 2 — End-to-End RL Navigation for AMR (Habitat PointNav)
**Date:** 2026-06-11
**目标:** 完成 Week 1 环境搭建 + 两条 smoke test（CartPole PPO RL 栈 / Habitat PointNav 渲染栈），为 Week 2 跑 PointNav PPO baseline 打底。

本文件夹收纳 Week 1 的全部代码、笔记、配置与结果证据。

## 目录结构

```
week01_rl_setup/
├── scripts/                         可运行脚本
│   ├── train_cartpole_ppo.py        训练 PPO 学会 CartPole + 回放统计 reward
│   ├── load_model.py                加载已保存模型回放
│   └── shortest_path_follower_example.py  Habitat PointNav 最短路基线（渲染 trajectory.mp4）
├── configs/
│   └── cartpole_ppo_config.md       实验环境与 PPO 超参记录
├── notes/                           学习笔记（研究者视角）
│   ├── rl_concepts_qa.md            RL / PPO 概念问答（PPO 学到了什么、何时更新）
│   ├── pointnav_example_analysis.md shortest_path_follower_example.py 逐段解读
│   ├── conda_env_entry.md           进入 conda amr 环境 + 验证 torch
│   └── commands_reference.md        Gym/PPO/Linux/Conda/Habitat 安装与数据集命令汇总
└── results/                         运行证据
    ├── cartpole_ppo_results.md      CartPole PPO 训练/测试 reward 汇总
    ├── cartpole_train_log_10k.txt   10k 步训练完整 REPL 日志
    ├── cartpole_train_log_50k.txt   50k 步训练完整 REPL 日志（reward 收敛到 500）
    └── pointnav_smoke_test_log.txt  Habitat PointNav 渲染 smoke test 完整运行日志
```

## 如何运行（CartPole smoke test）

```bash
conda activate amr            # Python 3.10；依赖: torch, gymnasium, stable-baselines3
cd scripts
python train_cartpole_ppo.py             # 训练 10k 步并回放
python train_cartpole_ppo.py --timesteps 50000   # 训练更久，reward 收敛到 500
python load_model.py                     # 加载 ppo_cartpole.zip 回放
```

## 如何运行（Habitat PointNav smoke test）

> 在云 GPU 服务器（RTX A4000 / Ubuntu 22.04，conda env `habitat39` / Python 3.9）运行，
> 需先按 `notes/commands_reference.md` 安装 habitat-sim / habitat-lab 0.3.3 并下载测试数据集。

```bash
conda activate habitat39
cp scripts/shortest_path_follower_example.py ~/habitat-lab/examples/   # 放回 habitat-lab examples 下
cd ~/habitat-lab                                                       # 配置/数据为相对路径，须在仓库根目录
python examples/shortest_path_follower_example.py
# 输出: examples/images/shortest_path_example/{00,01,02}/trajectory.mp4
```

## Week 1 关键结果

| 项目 | 状态 | 证据 |
|---|---|---|
| CartPole PPO 训练（10k 步） | ✅ reward 上升 21→59 | `results/cartpole_train_log_10k.txt` |
| CartPole PPO 训练（50k 步） | ✅ reward 收敛到 **500**（满分） | `results/cartpole_train_log_50k.txt` |
| 模型保存 / 加载 / 回放 | ✅ | `scripts/load_model.py` |
| Habitat-Sim / Habitat-Lab 0.3.3 安装 | ✅ import 成功 | `notes/commands_reference.md` |
| Habitat PointNav smoke test | ✅ 生成带渲染 `trajectory.mp4`（云 GPU RTX A4000） | `scripts/shortest_path_follower_example.py` / `results/pointnav_smoke_test_log.txt` |

> 注：CartPole 跑在本地 conda env `amr`（Python 3.10）；Habitat 渲染/训练跑在云 GPU 服务器（RTX A4000 / Ubuntu 22.04，conda env `habitat39` / Python 3.9）。详见上层 `docs/00_weekly.md`。
> Habitat 数据集（`habitat_test_scenes` / pointnav 测试集）体积较大，未提交本仓库，按 `src/README.md` 要求仅记录获取命令（见 `notes/commands_reference.md`）。
