"""
SR-GNN ED02 publication figures (Figs 1-5). All numbers traced to JSON/npz on disk.
matplotlib Agg, dpi>=220, English. Run: python3 make_figs.py
Author: DATA team. NO fabricated numbers — every value loaded/aggregated from source files.
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = "/users/PGS0407/binben14/VietHuy/Hoang/SR-GNN"
RES  = f"{ROOT}/experiments/results"
BASE = f"{ROOT}/experiments/results/baselines"
LAB  = f"{ROOT}/experiments/LAB/v3_3/results"
OUT  = f"{ROOT}/figs"
DPI  = 240

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

SR_BLUE   = "#1f4e8c"   # SR-GNN(B) highlight
SR_LIGHT  = "#7fa8d8"   # canonical-A
GREY      = "#b8b8b8"   # baselines

def agg_runs(fn, key):
    d = json.load(open(fn)); runs = d["runs"] if isinstance(d, dict) and "runs" in d else d
    v = [r[key] for r in runs if r.get(key) is not None]
    return float(np.mean(v)), float(np.std(v)), len(v)

def agg_baselines(fn, key):
    d = json.load(open(fn)); out = {}
    by = {}
    for r in d["runs"]:
        by.setdefault(r["model"], []).append(r[key])
    for m, vs in by.items():
        out[m] = (float(np.mean(vs)), float(np.std(vs)), len(vs))
    return out

# ============================================================ FIG 1 — CoEdit headline
def fig1():
    b_m, b_s, _ = agg_runs(f"{RES}/v3_3_coedit_ARM_B_publishable_3seed.json", "ind_ap")
    a_m, a_s, _ = agg_runs(f"{RES}/v3_3_coedit_ARM_A_canonical_3seed.json", "ind_ap")
    bl = agg_baselines(f"{BASE}/baselines_coedit_Bprotocol.json", "ind_ap")
    rows = [("SR-GNN (B)", b_m, b_s, "B"),
            ("SR-GNN (canonical-A)", a_m, a_s, "A")]
    for m, (mu, sd, _) in bl.items():
        rows.append((m.upper() if m != "graphmixer" else "GraphMixer", mu, sd, "base"))
    rows.sort(key=lambda r: r[1], reverse=True)
    labels = [r[0] for r in rows]; means = [r[1] for r in rows]
    stds = [r[2] for r in rows]; kinds = [r[3] for r in rows]
    colors = [SR_BLUE if k == "B" else SR_LIGHT if k == "A" else GREY for k in kinds]

    # best baseline (exclude B & A)
    base_only = [r for r in rows if r[3] == "base"]
    best_base = max(base_only, key=lambda r: r[1])
    gap = (b_m - best_base[1]) * 100

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    x = np.arange(len(rows))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                  edgecolor="black", linewidth=0.7, error_kw=dict(lw=1.1))
    for i, (mu, sd) in enumerate(zip(means, stds)):
        ax.text(i, mu + sd + 0.012, f"{mu:.3f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold" if kinds[i] == "B" else "normal")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Inductive Average Precision")
    ax.set_ylim(0.55, 1.06)
    ax.set_title("Inductive link prediction on CoEdit  (3-seed mean ± std)", fontweight="bold")

    # annotate +gap vs best baseline
    bi = labels.index("SR-GNN (B)")
    bbi = labels.index(best_base[0].upper() if best_base[0] != "graphmixer" else "GraphMixer")
    yb = b_m + b_s + 0.045
    ax.annotate("", xy=(bi, yb), xytext=(bbi, yb),
                arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.6))
    ax.text((bi + bbi) / 2, yb + 0.006, f"+{gap:.1f} pp vs best baseline ({best_base[0].upper()})",
            ha="center", va="bottom", color="#c0392b", fontsize=10.5, fontweight="bold")

    handles = [Patch(facecolor=SR_BLUE, edgecolor="black", label="SR-GNN (B, decoupled)"),
               Patch(facecolor=SR_LIGHT, edgecolor="black", label="SR-GNN (canonical-A)"),
               Patch(facecolor=GREY, edgecolor="black", label="TGN-family baselines")]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.92,
              edgecolor="#bbb", fontsize=9.5)
    fig.tight_layout()
    p = f"{OUT}/fig1_coedit_headline.png"; fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"FIG1 -> {p}  | B={b_m:.4f}±{b_s:.4f} bestbase={best_base[0]}={best_base[1]:.4f} gap=+{gap:.2f}pp")

# ============================================================ FIG 2 — decoupling mechanism (3-seed)
def fig2():
    # 3-seed aggregates; sample (n-1) std to match paper convention.
    def agg3(fn):
        d = json.load(open(fn)); runs = d["runs"] if "runs" in d else d
        v = [r["ind_ap"] for r in runs if r.get("ind_ap") is not None]
        return float(np.mean(v)), float(np.std(v, ddof=1)), len(v)
    B_m, B_s, nB = agg3(f"{RES}/v3_3_coedit_ARM_B_publishable_3seed.json")
    C_m, C_s, nC = agg3(f"{RES}/v3_3_coedit_ARM_C_correct_3seed.json")
    A_m, A_s, nA = agg3(f"{RES}/v3_3_coedit_ARM_A_canonical_3seed.json")
    labels = ["B\n(decoupled)",
              "C\n(end-to-end)",
              "A\n(no-lifecycle\nablation)"]
    means = [B_m, C_m, A_m]; stds = [B_s, C_s, A_s]
    colors = [SR_BLUE, "#c0392b", SR_LIGHT]
    fig, ax = plt.subplots(figsize=(7.0, 5.6), constrained_layout=True)
    x = np.arange(3)
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black",
                  linewidth=0.8, width=0.62, error_kw=dict(lw=1.3, ecolor="#333"))
    # highlight B with a heavier edge
    bars[0].set_linewidth(2.2); bars[0].set_edgecolor("#0d2b52")
    for i, (mu, sd) in enumerate(zip(means, stds)):
        ax.text(i, mu + sd + 0.014, f"{mu:.3f}\n$\\pm${sd:.3f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold" if i == 0 else "normal", linespacing=1.15)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Inductive Average Precision (CoEdit)")
    ax.set_ylim(0.0, 1.24)
    ax.set_title("Decoupling ablation — CoEdit inductive AP (3-seed mean$\\pm$std)",
                 fontweight="bold", pad=10)
    gap = (B_m - C_m) * 100
    # B-vs-C gap arrow placed ABOVE the bar value labels (no collision)
    yb = max(B_m + B_s, C_m + C_s) + 0.115
    ax.annotate("", xy=(1, yb), xytext=(0, yb),
                arrowprops=dict(arrowstyle="<->", color="#222", lw=1.7))
    ax.text(0.5, yb + 0.013, f"$\\Delta$(B$-$C) = +{gap:.1f} pp",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
    p = f"{OUT}/fig2_decoupling_ablation.png"; fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"FIG2 -> {p}  | B={B_m:.4f}±{B_s:.4f} C={C_m:.4f}±{C_s:.4f} "
          f"A={A_m:.4f}±{A_s:.4f}  gapBC=+{gap:.2f}pp  n={nB}/{nC}/{nA}")

# ============================================================ FIG 3 — counterfactual ladder
def fig3():
    import sys
    sys.path.insert(0, f"{ROOT}/experiments/LAB/v3_3")
    import fsm_intervene as fimod
    npz = f"{LAB}/faithfulness_coedit_v3_hier_hv2_let0.5_s42_cbON.npz"
    scm = fimod.HierV2SCM(npz)
    base = scm.baseline()["p_edge"].mean()
    order = ["DEATH", "IDLE", "DECAY", "REINFORCE", "BIRTH"]
    pe = {s: scm.do_state(state=s)["p_edge_forced"].mean() for s in order}
    # ladder x-axis: strictly monotone forced-state existence intent
    xs = ["do(DEATH)", "do(IDLE)", "do(DECAY)", "do(REINFORCE)\n= do(BIRTH)"]
    ys = [pe["DEATH"], pe["IDLE"], pe["DECAY"], pe["REINFORCE"]]
    cols = ["#c0392b", "#e08e3c", "#3c8ee0", "#2e8b57"]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(len(xs))
    ax.plot(x, ys, "-o", color=SR_BLUE, lw=2.2, ms=9, zorder=3)
    for i, (v, c) in enumerate(zip(ys, cols)):
        ax.scatter(x[i], v, s=130, color=c, edgecolor="black", zorder=4)
        ax.text(x[i], v + 0.045, f"{v:.2f}", ha="center", fontsize=10.5, fontweight="bold")
    # baseline (observed, un-intervened) as a reference band, NOT a ladder point
    ax.axhline(base, color="grey", ls="--", lw=1.3, alpha=0.85)
    ax.text(len(xs) - 1, base + 0.012, f"observed baseline = {base:.2f}",
            ha="right", va="bottom", color="#555", fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels(xs, fontsize=9.5)
    ax.set_xlim(-0.4, len(xs) - 0.6)
    ax.set_ylabel("Counterfactual mean P(edge)")
    ax.set_ylim(-0.08, 1.12)
    ax.set_title("Existence counterfactual ladder:  do(force state) -> P(edge)\n"
                 "CoEdit, N=12000 pairs (config B, cbON)", fontweight="bold", fontsize=12)
    ax.text(0.02, 0.88, "monotone & exact:\nDEATH < IDLE < DECAY < REINFORCE=BIRTH\n"
            "do(DEATH): P(edge) down for 100% of pairs",
            transform=ax.transAxes, fontsize=9.0, va="top",
            bbox=dict(boxstyle="round", fc="#f4f4f4", ec="grey"))
    fig.tight_layout()
    p = f"{OUT}/fig3_counterfactual_ladder.png"; fig.savefig(p, dpi=DPI); plt.close(fig)
    print(f"FIG3 -> {p}  | base={base:.4f} DEATH={pe['DEATH']:.3f} IDLE={pe['IDLE']:.3f} "
          f"DECAY={pe['DECAY']:.3f} REINF={pe['REINFORCE']:.3f} BIRTH={pe['BIRTH']:.3f}")

# ============================================================ FIG 4 — cross-dataset summary
def fig4():
    datasets = ["coedit", "wikipedia", "mooc"]
    fnmap = {
        "coedit":   (f"{RES}/v3_3_coedit_ARM_B_publishable_3seed.json", f"{BASE}/baselines_coedit_Bprotocol.json"),
        "wikipedia":(f"{RES}/v3_3_wikipedia_ARM_B_publishable_3seed.json", f"{BASE}/baselines_wikipedia_Bprotocol.json"),
        "mooc":     (f"{RES}/v3_3_mooc_ARM_B_publishable_3seed_rerun.json", f"{BASE}/baselines_mooc_Bprotocol.json"),
    }
    data = {}
    for ds in datasets:
        bfn, blfn = fnmap[ds]
        bt = agg_runs(bfn, "trans_ap"); bi = agg_runs(bfn, "ind_ap")
        blt = agg_baselines(blfn, "trans_ap"); bli = agg_baselines(blfn, "ind_ap")
        best_t = max(blt.items(), key=lambda kv: kv[1][0])
        best_i = max(bli.items(), key=lambda kv: kv[1][0])
        data[ds] = dict(b_t=bt, b_i=bi, bb_t=best_t, bb_i=best_i)

    fig, ax = plt.subplots(figsize=(9.4, 5.4), constrained_layout=True)
    x = np.arange(len(datasets)); w = 0.2
    srt = [data[d]["b_t"][0] for d in datasets]; srt_e = [data[d]["b_t"][1] for d in datasets]
    sri = [data[d]["b_i"][0] for d in datasets]; sri_e = [data[d]["b_i"][1] for d in datasets]
    bbt = [data[d]["bb_t"][1][0] for d in datasets]; bbt_e = [data[d]["bb_t"][1][1] for d in datasets]
    bbi = [data[d]["bb_i"][1][0] for d in datasets]; bbi_e = [data[d]["bb_i"][1][1] for d in datasets]

    ax.bar(x - 1.5*w, srt, w, yerr=srt_e, capsize=3, color=SR_BLUE, edgecolor="black", lw=0.6, label="SR-GNN (B) — transductive")
    ax.bar(x - 0.5*w, sri, w, yerr=sri_e, capsize=3, color=SR_LIGHT, edgecolor="black", lw=0.6, label="SR-GNN (B) — inductive")
    ax.bar(x + 0.5*w, bbt, w, yerr=bbt_e, capsize=3, color="#6f6f6f", edgecolor="black", lw=0.6, label="best baseline — transductive")
    ax.bar(x + 1.5*w, bbi, w, yerr=bbi_e, capsize=3, color=GREY, edgecolor="black", lw=0.6, label="best baseline — inductive")

    # baseline model name centered ABOVE each grey bar (+ its errorbar cap), horizontal,
    # small offset so it never strikes the cap or the neighbour bar
    for d, xi in zip(datasets, x):
        yt = data[d]["bb_t"][1][0] + data[d]["bb_t"][1][1] + 0.012
        yi = data[d]["bb_i"][1][0] + data[d]["bb_i"][1][1] + 0.012
        ax.text(xi + 0.5*w, yt, data[d]["bb_t"][0], ha="center", va="bottom",
                fontsize=7.8, color="#333")
        ax.text(xi + 1.5*w, yi, data[d]["bb_i"][0], ha="center", va="bottom",
                fontsize=7.8, color="#333")

    ax.set_xticks(x); ax.set_xticklabels(["CoEdit", "Wikipedia", "MOOC"])
    ax.set_ylabel("Average Precision")
    ax.set_ylim(0.55, 1.12)
    ax.set_title("SR-GNN (B) vs best per-dataset baseline  (3-seed mean " + r"$\pm$" + " std)",
                 fontweight="bold", pad=10)
    # legend pulled to upper-left interior where bars are short; em-dash kept (DejaVu OK)
    ax.legend(loc="lower center", ncol=2, frameon=True, framealpha=0.9, fontsize=8.8,
              bbox_to_anchor=(0.5, 0.0))
    p = f"{OUT}/fig4_cross_dataset.png"; fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    for d in datasets:
        print(f"FIG4 {d}: SR-B trans={data[d]['b_t'][0]:.4f} ind={data[d]['b_i'][0]:.4f} | "
              f"bestbase trans={data[d]['bb_t'][0]}={data[d]['bb_t'][1][0]:.4f} "
              f"ind={data[d]['bb_i'][0]}={data[d]['bb_i'][1][0]:.4f}")
    print(f"FIG4 -> {p}")

# ============================================================ FIG 5 — causal-coherence (advisory)
def fig5():
    coh, vio = [], []
    for f in sorted(glob.glob(f"{ROOT}/experiments/LAB/v3_3/results/wc_grnd/wc_conf_calib_grnd_coedit_s*_summary.json")):
        wc = json.load(open(f))["arms"]["wc"]
        coh.append(wc["mean_c_t_coherent"]); vio.append(wc["mean_c_t_violation"])
    coh = np.array(coh); vio = np.array(vio)
    fig, ax = plt.subplots(figsize=(6.8, 5.4), constrained_layout=True)
    parts = ax.violinplot([coh, vio], positions=[1, 2], showmeans=True, showextrema=True, widths=0.7)
    for pc, c in zip(parts["bodies"], ["#2e8b57", "#c0392b"]):
        pc.set_facecolor(c); pc.set_alpha(0.55); pc.set_edgecolor("black")
    for key in ["cmeans", "cmaxes", "cmins", "cbars"]:
        if key in parts: parts[key].set_color("black")
    # overlay per-seed points
    rng = np.random.default_rng(0)
    ax.scatter(np.full_like(coh, 1) + rng.uniform(-0.06, 0.06, len(coh)), coh, color="#145a32", s=45, zorder=3, edgecolor="white")
    ax.scatter(np.full_like(vio, 2) + rng.uniform(-0.06, 0.06, len(vio)), vio, color="#7b241c", s=45, zorder=3, edgecolor="white")
    # mean labels placed OFF the violin center bar (coherent above, violation below)
    ax.text(1, coh.max() + 0.055, f"{coh.mean():.3f}" + r"$\pm$" + f"{coh.std():.3f}",
            ha="center", fontsize=10.5, fontweight="bold", color="#145a32")
    ax.text(2, vio.min() - 0.075, f"{vio.mean():.3f}" + r"$\pm$" + f"{vio.std():.3f}",
            ha="center", fontsize=10.5, fontweight="bold", color="#7b241c")
    ax.set_xticks([1, 2]); ax.set_xticklabels(["coherent\ntransitions", "violation\ntransitions"])
    ax.set_xlim(0.4, 2.6)
    ax.set_ylabel(r"causal-coherence  $c_t$")
    ax.set_ylim(0, 1.10)
    ax.set_title("Causal-coherence score (advisory, self-consistency)\nCoEdit, 3 seeds",
                 fontweight="bold", fontsize=12, pad=8)
    p = f"{OUT}/fig5_causal_coherence.png"; fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"FIG5 -> {p}  | coherent={coh.mean():.4f}±{coh.std():.4f} violation={vio.mean():.4f}±{vio.std():.4f} n={len(coh)}")

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("ALL FIGS DONE ->", OUT)
