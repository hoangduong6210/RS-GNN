# SR-GNN: Regularization-by-Decoupling for Inductive Temporal Link Prediction

Clean research artifacts for the **SR-GNN ED02** paper — a two-stream temporal
link-prediction model whose continuous backbone is shaped only by a VAE-KL
regularizer (gradient-decoupled from the link-prediction head), paired with a
faithful, intervene-able symbolic lifecycle readout.

**Authors:** Duong Viet Hoang* and Shih, Lun-Min — Department of Computer Science,
Da-Yeh University, Taiwan. *Corresponding author.*

## Contents

- `SR_GNN_ED02.md` / `.pdf` — paper (English) + `_VI` Vietnamese version.
- `figs/` — paper figures and their generator scripts (`make_figs.py`,
  `make_schematics.py`).
- `experiments/` — model code, runners, and results JSONs (3-seed counterfactual
  kill battery, `hier_causal_policy` AP-neutrality, config-B cross-dataset
  baselines). Datasets (`*.npz`) are not tracked — see `.gitignore`.

## Reproducibility

Headline numbers trace to the JSON files under `experiments/results/`. GPU runs
target an Ascend `nextgen` SLURM partition with a CUDA 12.4 environment.

> This repository contains only the clean, publishable artifacts; internal team
> coordination, context, and communication files are intentionally excluded.
