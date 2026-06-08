#!/bin/bash
#SBATCH --job-name=srgnn_regime_baselines
#SBATCH --account=PGS0407
#SBATCH --partition=nextgen
#SBATCH --time=05:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/users/PGS0407/binben14/GPU_ticket/logs/srgnn_regime_baselines_%j.out
#
# PREPARED BY ML 2026-05-31 — DO NOT auto-submit. To launch: copy THIS file into
# /users/PGS0407/binben14/GPU_ticket/ (PM approval required first).
#
# Per-NODE-memory baselines (tgat, jodie, cawn, tgn) on synthetic_regime.
# 4 models x 3 seeds x 20 epochs x 1 dataset = 12 runs. Same splits/seeds as the
# SR-GNN job so the SR-GNN(per-pair) vs baseline(per-node) gap is apples-to-apples.
# feat_dim read dynamically (=16).
set -euo pipefail

module load miniconda3
source activate vlm-new

cd /users/PGS0407/binben14/VietHuy/Hoang/SR-GNN/experiments

python run_baselines_benchmark.py \
    --datasets synthetic_regime \
    --models tgat,jodie,cawn,tgn \
    --seeds 42,123,7 \
    --epochs 20 \
    --hidden 128 \
    --batch 500 \
    --lr 1e-3 \
    --out results/regime_baselines.json

echo "DONE srgnn_regime_baselines"
