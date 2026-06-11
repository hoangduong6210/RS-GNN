"""Visual summary for CHANGELOG_v1_to_ED02 - makes the v1->ED02 transformation legible."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.42, wspace=0.24)

# ---- (A) the measured headline: CoEdit inductive AP ----
axA = fig.add_subplot(gs[0, 0])
names = ["EdgeBank\n(floor)", "DyGFormer\n(re-impl)", "TGAT\n(best base)", "e2e coupled\n(K1)",
         "freeze-\nthen-probe", "RS-GNN\ndecoupled"]
vals = [0.590, 0.612, 0.847, 0.779, 0.768, 0.988]
cols = ["#bbbbbb", "#bbbbbb", "#8aa1b1", "#e76f51", "#e76f51", "#2a9d8f"]
b = axA.bar(range(len(names)), vals, color=cols)
axA.set_xticks(range(len(names))); axA.set_xticklabels(names, fontsize=8)
axA.set_ylabel("CoEdit inductive AP"); axA.set_ylim(0.5, 1.02)
axA.set_title("(A) ED02 measured headline: decoupling wins inductive", fontsize=11, fontweight="bold")
for i, v in enumerate(vals):
    axA.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=8)
axA.annotate("", xy=(5, 0.985), xytext=(3, 0.782),
             arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
axA.text(4, 0.90, "+21.1pp\n(single-variable\nmeasured)", ha="center", fontsize=8, color="#1d6f63")

# ---- (B) irreversibility, 2 datasets ----
axB = fig.add_subplot(gs[0, 1])
x = np.arange(2); w = 0.26
dec = [0.988, 0.996]; ftp = [0.768, 0.897]; cpl = [0.779, 0.909]
axB.bar(x - w, dec, w, label="decoupled (by construction)", color="#2a9d8f")
axB.bar(x, ftp, w, label="freeze-then-probe", color="#f4a261")
axB.bar(x + w, cpl, w, label="coupled (e2e)", color="#e76f51")
axB.set_xticks(x); axB.set_xticklabels(["CoEdit", "Wikipedia"])
axB.set_ylabel("inductive AP"); axB.set_ylim(0.5, 1.02)
axB.set_title("(B) Contamination is irreversible (2 datasets):\nFtP ≈ coupled ≪ decoupled", fontsize=11, fontweight="bold")
axB.legend(fontsize=8, loc="lower right")

# ---- (C) retraction ledger ----
axC = fig.add_subplot(gs[1, 0]); axC.axis("off")
axC.set_title("(C) Retracted from v1 (integrity)", fontsize=11, fontweight="bold", loc="left")
retr = [
    "R1  echo/resonance memory  →  VAE-KL regularizer",
    "R2  transition-CE term  →  de-collapse CE (detached)",
    "R3  regime-switch advantage  →  FALSIFIED (loses to CAWN)",
    "R4  causal-confidence = error flag  →  retracted (anti-calibrated)",
    "R5  do(·) = Pearl identification  →  typed forward re-evaluation",
    "-   Proposition 1 optimality theorem  →  removed",
    "-   29.7%-causal-chain headline  →  removed",
]
for i, t in enumerate(retr):
    axC.text(0.0, 0.92 - i * 0.135, t, fontsize=9.5, family="monospace",
             color="#8a2a2a" if i < 5 else "#555555")

# ---- (D) new measured evidence ----
axD = fig.add_subplot(gs[1, 1]); axD.axis("off")
axD.set_title("(D) New in ED02 (measured, 3-5 seed)", fontsize=11, fontweight="bold", loc="left")
adds = [
    "knob ablation: enable_main_predictor  →  +21.1pp (1 flag)",
    "freeze-then-probe irreversibility  →  2 datasets",
    "5-seed headline  →  B−C +22.8 / B−TGAT +14.1",
    "cross-dataset knob  →  wiki +8.6 / mooc +0.85",
    "trained-θ ladder  →  survives training (non-tautological)",
    "EdgeBank floor + DyGFormer baselines (measured)",
    "Fig 6/7: faithful + intervene-able readout",
]
for i, t in enumerate(adds):
    axD.text(0.0, 0.92 - i * 0.135, t, fontsize=9.5, family="monospace", color="#1d6f63")

fig.suptitle("RS-GNN:  v1 (Resonance-Symbolic)  →  v2 ED02 :  from argued compiler to measured decoupling + honest scope",
             fontsize=13, fontweight="bold", y=0.985)
plt.savefig("changelog_summary.png", dpi=140, bbox_inches="tight")
print("saved figs/changelog_summary.png")
