#!/bin/bash
# Generate top-down trajectory PNGs for paper/video
# Run on AutoDL server:  ssh root@<AUTODL_IP> "bash -s" < generate_topdown.sh

cd /root/habitat-lab
conda activate habitat39
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

CKPT=/root/autodl-tmp/exp/baseline_seed300/checkpoints/latest.pth
OUTDIR=/root/autodl-tmp/topdown_figures
mkdir -p $OUTDIR

# ── Fig A: LOST — seed=300, DTG=17.16m (most dramatic failure) ──
python experiments/baseline/viz_topdown_path.py \
  --ckpt $CKPT --seed 300 --episode-id 5 \
  --out $OUTDIR/topdown_lost_17m.png --gif

# ── Fig B: NEAR-MISS — seed=100, DTG=0.201m (0.001m from success!) ──
python experiments/baseline/viz_topdown_path.py \
  --ckpt /root/autodl-tmp/exp/baseline_seed100/checkpoints/latest.pth \
  --seed 100 --episode-id 3 \
  --out $OUTDIR/topdown_nearmiss_0201m.png --gif

# ── Fig C: SUCCESS — seed=200, DTG=0.019m, SPL=0.976 (near-perfect) ──
python experiments/baseline/viz_topdown_path.py \
  --ckpt /root/autodl-tmp/exp/baseline_seed200/checkpoints/latest.pth \
  --seed 200 --episode-id 4 \
  --out $OUTDIR/topdown_success_0019m.png --gif

echo "Done. Outputs in $OUTDIR/"
ls -lh $OUTDIR/
