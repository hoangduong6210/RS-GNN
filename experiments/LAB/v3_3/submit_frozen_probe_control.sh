#!/bin/bash
#SBATCH --job-name=srgnn_frozen_probe_ctrl
#SBATCH --account=PGS0407
#SBATCH --partition=nextgen
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/users/PGS0407/binben14/GPU_ticket/logs/srgnn_frozen_probe_ctrl_%j.out
#
# PREPARED BY ML 2026-06-09 (reviewer #1 decisive control) — DO NOT auto-submit.
# To launch: copy THIS file into /users/PGS0407/binben14/GPU_ticket/ (watcher submits
# within ~1 min). PM approval required before copying (per operating procedure).
#
# FREEZE-THEN-PROBE CONTROL: separate SR-GNN "decoupling-by-construction" from the
# classic "freeze-then-probe / linear-probing transfer".
#   ARM 1 (decoupling) : config-B; backbone NEVER sees link-pred grad (--p0_fix off).
#   ARM 2 (FtP)        : pretrain e2e (backbone shaped by link-pred) -> freeze backbone
#                        -> train fresh link head -> measure inductive AP (--frozen_probe).
# Both arms share the SAME backbone stack; ONLY the training protocol differs.
# 3 seeds (42,1,7) x {coedit, wikipedia} x 2 arms. B-protocol: chrono 70/15/15, fair
# inductive neg, PRE-update leak-free, sklearn AP — all enforced inside train.run_epoch.
set -euo pipefail

module load miniconda3/24.1.2-py310
source activate vlm-new

cd /users/PGS0407/binben14/VietHuy/Hoang/SR-GNN/experiments

RES=LAB/v3_3/results
COMMON="--seeds 42,1,7 --epochs 20 --hidden 128 --batch 500 --lr 1e-3 \
        --design correct_decoupled --p0_fix off --fsm_arch v3 --fsm_decode hier \
        --decol_hier_v2 --causal_batch --hier_causal_policy --lambda_edge_trans 0.5"

# ── ARM 1: decoupling-by-construction (config-B) ──────────────────────────────
python LAB/v3_3/run_v3_3_benchmark.py \
    --datasets coedit,wikipedia \
    $COMMON \
    --out $RES/v3_3_frozen_probe_ARM1_decoupling.json

# ── ARM 2: freeze-then-probe (same stack + --frozen_probe; PHASE-1 forces e2e) ─
python LAB/v3_3/run_v3_3_benchmark.py \
    --datasets coedit,wikipedia \
    $COMMON \
    --frozen_probe \
    --out $RES/v3_3_frozen_probe_ARM2_ftp.json

# ── AGGREGATE -> single control file + verdict ────────────────────────────────
python LAB/v3_3/aggregate_frozen_probe_control.py \
    --arm1 $RES/v3_3_frozen_probe_ARM1_decoupling.json \
    --arm2 $RES/v3_3_frozen_probe_ARM2_ftp.json \
    --out  results/v3_3_frozen_probe_control_3seed.json

echo "DONE srgnn_frozen_probe_ctrl"
