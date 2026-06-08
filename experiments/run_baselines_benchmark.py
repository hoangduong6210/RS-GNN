"""
External-baseline benchmark for head-to-head vs SR-GNN v3.3.

Runs the registered temporal-graph baselines through the SAME train.py harness
that produces the v3.3 numbers (identical get_data_splits chronological split,
identical fair inductive negative sampling, identical AP/AUC metric, identical
epochs/batch). The ONLY thing that varies is the model. This guarantees protocol
parity — the comparison is apples-to-apples.

Metric aggregation is packed INTO the run: after every completed (model,dataset,seed)
run the output json is rewritten with both `runs` (per-seed) and `summary`
(mean±std grouped by model×dataset). A wall-clock timeout therefore still leaves a
valid, aggregated partial result on disk — no separate post-hoc aggregator pass.

Default protocol matches the v3.3 A/B exactly:
  seeds [42,123,7] x 20 epochs x hidden 128 x batch 500 x lr 1e-3.

Excluded by design:
  - dygformer: BROKEN (PatchEncoder.forward signature mismatch, dygformer.py:114) — ML-owned bug.
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import run_experiment, DEVICE

# Working external baselines (all CPU-smoke verified: construct+forward+backward,
# correct (B,) score shapes, nonzero grads). dygformer excluded (broken forward).
DEFAULT_MODELS = ["jodie", "dyrep", "tgat", "tgn", "graphmixer", "cawn"]


def summarize(results):
    """Group by (model, dataset); mean±std (population std, matches v3.3 runner)."""
    by = {}
    for r in results:
        by.setdefault((r["model"], r["dataset"]), []).append(r)
    summary = []
    for (model, ds), rows in by.items():
        def col(key):
            vals = [x[key] for x in rows if not np.isnan(x[key])]
            return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))
        ta_m, ta_s = col("trans_ap"); tu_m, tu_s = col("trans_auc")
        ia_m, ia_s = col("ind_ap");   iu_m, iu_s = col("ind_auc")
        times = [x["train_time_s"] for x in rows]
        summary.append({
            "model": model, "dataset": ds,
            "trans_ap_mean": ta_m, "trans_ap_std": ta_s,
            "trans_auc_mean": tu_m, "trans_auc_std": tu_s,
            "ind_ap_mean": ia_m, "ind_ap_std": ia_s,
            "ind_auc_mean": iu_m, "ind_auc_std": iu_s,
            "time_mean": float(np.mean(times)),
            "num_params": rows[0].get("num_params"),
            "n_seeds": len(rows),
        })
    return summary


def main():
    p = argparse.ArgumentParser(description="External baseline benchmark (v3.3-parity harness)")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS),
                   help="comma-sep baseline keys (default: working 6)")
    p.add_argument("--datasets", default="wikipedia,mooc,coedit")
    p.add_argument("--seeds", default="42,123,7")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out",
                   default="/users/PGS0407/binben14/VietHuy/Hoang/SR-GNN/experiments/results/baselines/baselines_benchmark.json")
    p.add_argument("--dump_dir", default=None,
                   help="If set, write per-edge (y_true,y_score,test_idx) .npz per "
                        "(model,dataset,seed) here for post-CP eval; inline post-CP AP "
                        "computed on synthetic_regime.")
    args = p.parse_args()

    MODELS = [m.strip() for m in args.models.split(",") if m.strip()]
    DATASETS = [d.strip() for d in args.datasets.split(",") if d.strip()]
    SEEDS = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[device] {DEVICE}")
    print(f"[protocol] models={MODELS} datasets={DATASETS} seeds={SEEDS} "
          f"epochs={args.epochs} hidden={args.hidden} batch={args.batch} lr={args.lr}")

    results = []
    total = len(MODELS) * len(DATASETS) * len(SEEDS)
    idx = 0
    for model in MODELS:
        for ds in DATASETS:
            for s in SEEDS:
                idx += 1
                print(f"\n{'='*64}\nRUN {idx}/{total}  model={model}  dataset={ds}  seed={s}\n{'='*64}")
                try:
                    r = run_experiment(model, ds, args.epochs, args.hidden,
                                       args.batch, args.lr, s,
                                       dump_dir=args.dump_dir)
                    results.append(r)
                    # Pack aggregation INTO the run: rewrite full json after each run.
                    with open(args.out, "w") as f:
                        json.dump({"runs": results, "summary": summarize(results)},
                                  f, indent=2, default=str)
                    print(f"  -> {model} {ds} s{s}: Trans AP={r['trans_ap']:.4f} "
                          f"Ind AP={r['ind_ap']:.4f} [{r['train_time_s']:.0f}s]")
                except Exception as e:
                    print(f"  X FAILED {model} {ds} s{s}: {e}")
                    import traceback; traceback.print_exc()

    summary = summarize(results)
    print("\n" + "="*92)
    print(f"BASELINE BENCHMARK  ({len(SEEDS)} seeds x {args.epochs} epochs x {len(DATASETS)} datasets)")
    print("="*92)
    print(f"{'Model':<12} {'Dataset':<11} | {'Trans AP':>15} | {'Ind AP':>15} | {'Time':>7} | seeds")
    print("-"*92)
    for s in summary:
        print(f"{s['model']:<12} {s['dataset']:<11} | "
              f"{s['trans_ap_mean']:.4f}+-{s['trans_ap_std']:.4f} | "
              f"{s['ind_ap_mean']:.4f}+-{s['ind_ap_std']:.4f} | "
              f"{s['time_mean']:>6.0f}s | {s['n_seeds']}")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
