#!/bin/bash
# HM3D PointNav PPO Baseline Training
# Run on cloud server: RTX 4090D, 24GB VRAM
# Usage: bash train_hm3d_baseline.sh [SEED]
#
# Uses the benchmark/nav/pointnav=pointnav_hm3d config group (NOT manual
# data_path/scenes_dir overrides — those get silently overridden back to
# habitat-test-scenes by the ppo_pointnav.yaml defaults list; confirmed via
# the FileNotFoundError on data/datasets/pointnav/habitat-test-scenes/v1/train).
# pointnav_hm3d.yaml pulls in habitat/dataset/pointnav/hm3d.yaml, whose
# data_path template (data/datasets/pointnav/hm3d/v1/{split}/{split}.json.gz)
# matches this server's actual v1 download — verified directly.

SEED=${1:-100}
CKPT_DIR="/root/autodl-tmp/exp/baseline_seed${SEED}"

# Exclude Mesa software EGL ICD; only NVIDIA device enumerated.
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
PYTHON="${PYTHON:-/root/miniconda3/envs/habitat39/bin/python}"
cd "$(dirname "$0")"

echo "=== HM3D PointNav Baseline | seed=${SEED} ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Checkpoint dir: ${CKPT_DIR}"

"${PYTHON}" -u -m habitat_baselines.run \
    --config-name=pointnav/ppo_pointnav.yaml \
    benchmark/nav/pointnav=pointnav_hm3d \
    habitat_baselines.num_environments=10 \
    habitat_baselines.total_num_steps=1e7 \
    habitat_baselines.num_updates=-1 \
    habitat_baselines.num_checkpoints=50 \
    habitat_baselines.checkpoint_folder="${CKPT_DIR}/checkpoints" \
    habitat_baselines.eval_ckpt_path_dir="${CKPT_DIR}/checkpoints" \
    habitat_baselines.tensorboard_dir="${CKPT_DIR}/tb" \
    habitat_baselines.video_dir="${CKPT_DIR}/video" \
    habitat_baselines.log_interval=100 \
    habitat.seed=${SEED}
