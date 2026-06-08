#!/bin/bash
#SBATCH --job-name=srgnn_regime_srgnn
#SBATCH --account=PGS0407
#SBATCH --partition=nextgen
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/users/PGS0407/binben14/GPU_ticket/logs/srgnn_regime_srgnn_%j.out
#
# PREPARED BY ML 2026-05-31 — DO NOT auto-submit. To launch: copy THIS file into
# /users/PGS0407/binben14/GPU_ticket/  (watcher submits within ~1 min). PM approval
# required before copying (per operating procedure).
#
# SR-GNN v3.3 CANONICAL (no flags = detached readout, the locked default) on the
# strict-clean synthetic_regime benchmark. 3 seeds x 20 epochs x 1 dataset.
# feat_dim is read dynamically from the npz (=16); no code change needed.
set -euo pipefail

module load miniconda3
source activate vlm-new

cd /users/PGS0407/binben14/VietHuy/Hoang/SR-GNN/experiments

python LAB/v3_3/run_v3_3_benchmark.py \
    --datasets synthetic_regime \
    --seeds 42,123,7 \
    --epochs 20 \
    --hidden 128 \
    --batch 500 \
    --lr 1e-3 \
    --p0_fix off \
    --out LAB/v3_3/results/regime_srgnn_canonical.json

echo "DONE srgnn_regime_srgnn"
