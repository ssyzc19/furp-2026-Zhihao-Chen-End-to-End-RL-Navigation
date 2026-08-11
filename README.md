# PointNav Sim-to-Real: Stop-Aware Reward + Extended Training

> **HM3D PointGoal Navigation with PPO — from simulation to real robot**
>
> FURP 2026 | Zhihao Chen
>
> [📄 Final Report](REPORT.md) | [📊 Results](results/) | [🤖 Deployment](deployment/) | [🎥 Demo Video](video/)

---

## What This Is

A complete, reproducible research package for PointGoal navigation using deep RL:

1. **Simulation**: Train PPO PointNav on Habitat + HM3D (800 scenes)
2. **Improvement**: Stop-Aware Reward fixes bad-stop and near-miss failures
3. **Ablation**: Training steps (5e7) > PPO hyperparameter tuning
4. **Deployment**: Real robot (WHEELTEC S100) navigates autonomously

## Key Numbers

| Stage | SR | SPL | DTG |
|-------|:--:|:---:|:---:|
| baseline (1e7) | 0.845 | 0.687 | 0.843m |
| + stop_aware | 0.890 | 0.731 | 0.780m |
| **+ 5e7 steps** | **0.945** | **0.820** | **0.515m** |

| Model | Sim | Real Robot |
|-------|:---:|:----------:|
| baseline | 0.895 | ✅ Goal reached |
| **5e7** | **0.945** | ✅ **Goal reached** |

## Reproduce Simulation

```bash
# 1. Setup
conda create -n habitat39 python=3.9 -y && conda activate habitat39
pip install torch==2.5.0 habitat-sim==0.3.3 habitat-lab==0.3.3 habitat-baselines==0.3.3
# Download HM3D pointnav_hm3d_v1.zip → data/scene_datasets/hm3d/

# 2. Train
cd habitat-lab
bash scripts/train_hm3d_baseline.sh 300

# 3. Evaluate
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
bash scripts/eval_hm3d.sh checkpoints/latest.pth

# 4. Failure analysis
python scripts/failure_analysis.py checkpoints/latest.pth
```

## Reproduce Real Robot

Hardware needed: WHEELTEC S100 + Jetson Orin Nano + Astra S RGBD + ROS1 Noetic

```bash
# T1: roscore
# T2: roslaunch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch
# T3: roslaunch turn_on_wheeltec_robot wheeltec_camera.launch
# T4: roslaunch ppo_navigation start_m2.launch
# T5: roslaunch s100_deploy s100_deploy.launch model_type:=habitat ckpt_path:=ckpt.49.pth
# T6: rostopic pub /move_base_simple/goal ...
```

Full guide: [deployment/STARTUP.md](deployment/STARTUP.md)

## What's Inside

```
├── README.md           ← you are here
├── REPORT.md           ← final research report
├── scripts/            ← training, eval, failure analysis
├── results/            ← eval CSV + failure analysis JSONs
├── deployment/         ← real robot ROS nodes + launch files
└── video/              ← demo video script
```

## Improvement Attempts

| # | What | Why | Result |
|:--:|------|-----|:------:|
| 1 | Stop-Aware Reward | Fix near-miss + bad_stop | ✅ SR +4.5pp |
| 2 | 5e7 training steps | Fix lost (data coverage) | ✅ SR +4.5pp, SPL +10.1% |
| 3 | LR decay | Stabilize critic | ⚠️ Superficial |
| 4 | Adv normalization | Reduce variance | ❌ Failed |
| 5 | GAE smoothing (tau=0.99) | Reduce variance | ❌ Harmful |

## Platform

- **Training**: NVIDIA RTX 4090D (24GB), Ubuntu 22.04, CUDA 12.4
- **Robot**: WHEELTEC S100, Jetson Orin Nano, ROS1 Noetic
- **Camera**: Orbbec Astra S RGBD
