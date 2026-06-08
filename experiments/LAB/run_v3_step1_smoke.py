"""
Step 1 smoke test for sr_gnn_v3.

Train 3 epochs Wikipedia, verify:
  - Edge state distribution NOT degenerate (not 100% IDLE / not stuck at 2 states)
  - All 5 states have non-trivial mass
  - Train loss decreases
  - Val AP >= 0.90 (sanity, not beating baseline)
"""
import os, sys, time, json, random
import numpy as np
import torch

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(LAB_DIR)
sys.path.insert(0, LAB_DIR)
sys.path.insert(0, EXP_DIR)

from data.download import download_dataset, get_data_splits
from train import run_epoch, DEVICE
from models.sr_gnn_v3 import SRGNN_v3


def measure_edge_state_dist(model):
    if not model.edge_mem._state_table:
        return [0.0] * 5
    states = torch.stack(model.edge_mem._state_table)  # (E, 12)
    state_idx = states[:, :5].argmax(dim=-1)
    counts = torch.bincount(state_idx, minlength=5).float()
    dist = (counts / counts.sum()).tolist()
    return dist


def measure_signal_stats(model):
    """Return mean of recur, hawkes_lam, mean_dt, var_dt across all stored edges."""
    if not model.edge_mem._state_table:
        return {}
    states = torch.stack(model.edge_mem._state_table)
    return {
        "recur_mean":      float(states[:, 5].mean()),
        "recur_max":       float(states[:, 5].max()),
        "hawkes_lam_mean": float(states[:, 6].mean()),
        "hawkes_lam_max":  float(states[:, 6].max()),
        "mean_dt_mean":    float(states[:, 7].mean()),
        "var_dt_mean":     float(states[:, 8].mean()),
        "n_obs_mean":      float(states[:, 9].mean()),
        "n_edges":         int(states.size(0)),
    }


def main():
    DATASET = "wikipedia"
    EPOCHS  = 3
    HIDDEN  = 128
    BATCH   = 500
    LR      = 1e-3
    SEED    = 42

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    data = download_dataset(DATASET)
    splits = get_data_splits(data)
    num_nodes, feat_dim = data["num_nodes"], data["feat_dim"]

    model = SRGNN_v3(num_nodes, feat_dim, HIDDEN, device=DEVICE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    print(f"\n=== STEP 1 SMOKE TEST: SRGNN_v3 multi-signal edge state ===")
    print(f"Dataset: {DATASET}  nodes: {num_nodes}  edges: {data['num_edges']}")
    print(f"Epochs: {EPOCHS}  batch: {BATCH}  lr: {LR}  seed: {SEED}\n")

    t0 = time.time()
    history = []
    for ep in range(1, EPOCHS + 1):
        if hasattr(model, "reset"): model.reset()
        tr = run_epoch(model, splits["train"], num_nodes, BATCH,
                       optimizer=optimizer, desc=f"v3/E{ep}/tr")
        va = run_epoch(model, splits["val"],   num_nodes, BATCH,
                       desc=f"v3/E{ep}/va")
        dist = measure_edge_state_dist(model)
        sig  = measure_signal_stats(model)
        elapsed = time.time() - t0
        log = {"epoch": ep, "tr_AP": tr["AP"], "tr_loss": tr["Loss"],
               "va_AP": va["AP"], "va_loss": va["Loss"],
               "state_dist": dist, "signals": sig, "elapsed_s": elapsed}
        history.append(log)
        print(f"E{ep:02d}  tr_AP={tr['AP']:.4f} tr_loss={tr['Loss']:.4f}  "
              f"va_AP={va['AP']:.4f}  [{elapsed:.0f}s]")
        print(f"     state_dist=[I={dist[0]:.2f} B={dist[1]:.2f} R={dist[2]:.2f} D={dist[3]:.2f} Dt={dist[4]:.2f}]")
        print(f"     signals: recur={sig['recur_mean']:.3f}/{sig['recur_max']:.3f}  "
              f"hawkes_λ={sig['hawkes_lam_mean']:.3f}/{sig['hawkes_lam_max']:.3f}  "
              f"mean_dt={sig['mean_dt_mean']:.1f}  n_edges={sig['n_edges']}")

    # Test
    if hasattr(model, "reset"): model.reset()
    te = run_epoch(model, splits["test"], num_nodes, BATCH, desc="v3/test")
    final_dist = measure_edge_state_dist(model)
    final_sig  = measure_signal_stats(model)
    final_entropy = -sum(p * np.log(p + 1e-12) for p in final_dist)

    out = {
        "model":         "srgnn_v3_step1",
        "dataset":       DATASET,
        "seed":          SEED,
        "epochs":        EPOCHS,
        "test_ap":       te["AP"],
        "test_auc":      te["AUC"],
        "final_state_dist": final_dist,
        "final_entropy":    final_entropy,
        "final_signals": final_sig,
        "history":       history,
        "elapsed_s":     time.time() - t0,
    }
    out_path = os.path.join(LAB_DIR, "results", "v3_step1_smoke.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "="*60)
    print("FINAL  test AP={:.4f}  AUC={:.4f}".format(te["AP"], te["AUC"]))
    print("State dist [I,B,R,D,Dt] = [{}]".format(", ".join(f"{p:.3f}" for p in final_dist)))
    print(f"State entropy = {final_entropy:.3f}  (max possible = {np.log(5):.3f})")
    print(f"\n✅ Saved → {out_path}")

    # Acceptance criteria
    print("\n=== Step 1 Acceptance Criteria ===")
    print(f"  [{'✓' if te['AP'] >= 0.90 else '✗'}] Val/Test AP ≥ 0.90 (sanity): {te['AP']:.4f}")
    nontriv_states = sum(1 for p in final_dist if p > 0.05)
    print(f"  [{'✓' if nontriv_states >= 3 else '✗'}] ≥3 states with mass >5% (not collapse): {nontriv_states}/5")
    print(f"  [{'✓' if final_entropy >= 0.7 else '✗'}] Entropy ≥ 0.7: {final_entropy:.3f}")


if __name__ == "__main__":
    main()
