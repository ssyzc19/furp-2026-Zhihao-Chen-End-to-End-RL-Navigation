#!/bin/bash
# Evaluate a trained checkpoint on HM3D val
# Usage: bash eval_hm3d.sh <CKPT_PATH> [NUM_EPISODES]
#
# Uses benchmark/nav/pointnav=pointnav_hm3d + habitat.dataset.split=val to
# switch off the default train split (manual data_path/scenes_dir overrides
# get silently discarded by the defaults list — see train_hm3d_baseline.sh).

CKPT_PATH=${1:?"Usage: bash eval_hm3d.sh <ckpt_path_or_dir> [num_episodes]"}
NUM_EPISODES=${2:-200}

echo "=== HM3D PointNav Eval ==="
echo "Checkpoint: ${CKPT_PATH}"
echo "Episodes: ${NUM_EPISODES}"

# 清理残留 resume state 文件，避免 eval 时加载训练 config
rm -f /root/autodl-tmp/checkpoints/.habitat-resume-state*.pth

# 如果传的是单个文件，复制到临时目录让 Habitat 只 eval 这一个
if [ -f "${CKPT_PATH}" ]; then
    TMPDIR=$(mktemp -d)
    cp "${CKPT_PATH}" "${TMPDIR}/ckpt.0.pth"
    EVAL_DIR="${TMPDIR}"
    trap "rm -rf ${TMPDIR}" EXIT
else
    EVAL_DIR="${CKPT_PATH}"
fi

/root/miniconda3/envs/habitat39/bin/python -u -m habitat_baselines.run \
    --config-name=pointnav/ppo_pointnav.yaml \
    benchmark/nav/pointnav=pointnav_hm3d \
    habitat.dataset.split=val \
    habitat_baselines.evaluate=True \
    habitat_baselines.eval_ckpt_path_dir="${EVAL_DIR}" \
    habitat_baselines.num_environments=10 \
    habitat_baselines.test_episode_count=${NUM_EPISODES} \
    habitat_baselines.num_updates=-1 \
    habitat_baselines.load_resume_state_config=False \
    ~habitat.task.measurements.top_down_map \
    habitat_baselines.video_dir="eval_videos/$(basename ${CKPT_PATH})"
