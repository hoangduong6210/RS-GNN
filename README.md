# RS-GNN V2 (ED02)

**Regularization-by-Decoupling for Inductive Temporal Link Prediction, with a Faithful Intervene-able Lifecycle Readout.**

Duong Viet Hoang\* and Shih, Lun-Min - Department of Computer Science, Da-Yeh University, Taiwan. *\*Corresponding author.*

A two-stream temporal link-prediction model whose continuous backbone is shaped only by a VAE-KL regularizer - decoupled by construction from the link-prediction objective - paired with a faithful, intervene-able symbolic lifecycle readout. The decoupling is the measured driver of inductive generalization; the lifecycle readout is a clean interpretability add-on that rides on the same stop-gradient.

## Repository layout

```
RS-GNN/
├── paper/                           RS-GNN V2 documents:
│   ├── SR_GNN_ED02.md / .pdf            Paper (English) - the RS-GNN V2 submission draft
│   ├── SR_GNN_ED02_VI.md / .pdf         Paper (Vietnamese, mirrored)
│   ├── SR_GNN_ED02_slides.md            Talk slides (Marp, English)
│   └── _header_breakcode.tex           LaTeX header used when rendering the papers
├── figs/                            Paper figures (1-8) + appendix schematics (A1-A5) + generators
└── experiments/                     ED02 model + baselines + cited driver scripts and result JSONs
```

## Headline result (measured)

On CoEdit inductive link prediction, removing the end-to-end link head (`enable_main_predictor=False`)
**alone** accounts for +21.1 pp inductive AP (single-variable knob ablation, 3-seed). A freeze-then-probe
control shows the contamination from end-to-end coupling is **irreversible** (FtP ≈ coupled ≪ decoupled),
on two datasets. Five-seed CoEdit headline: B − C = +22.8 pp, B − TGAT = +14.1 pp.

## Reproducibility

Every number in the paper traces to a results JSON (shipped under `experiments/results/` and
`experiments/LAB/v3_3/results/`) or a job ID in Appendix A. Large binaries (`*.npz` datasets and
faithfulness dumps) are not redistributed here; they regenerate from the code, after which the
lifecycle/counterfactual demos consume the dumps under `experiments/LAB/v3_3/results/`. GPU runs
target an Ascend `nextgen` SLURM partition with a CUDA 12.4 environment.

## Scope (honest)

The decoupling mechanism (Contribution 1) is a general training principle, shown across CoEdit / Wikipedia /
MOOC. The per-pair causal lifecycle readout (Contribution 2) is **CoEdit-scoped by design** - lifecycle
dynamics are driven by non-commensurable, domain-specific quantities, so a faithful intervene-able readout
must specialize per dataset; globalization is principled future work (see `PERPAIR_GLOBALIZATION_DESIGN.md`).
