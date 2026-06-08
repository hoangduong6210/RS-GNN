"""
Run SR-GNN v3 (REACT) vs v2 baseline benchmark.

Outputs:
  results/v3_benchmark.json — main comparison
  results/v3_cost.json      — wall-clock + memory
"""
import os
import sys
import json
import time
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from train import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--datasets", nargs="+", default=["wikipedia"])
    parser.add_argument("--models", nargs="+", default=["srgnn_v2", "srgnn_v3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--out", default="results/v3_benchmark.json")
    args = parser.parse_args()

    all_results = []
    t0 = time.time()
    for ds in args.datasets:
        for m in args.models:
            for s in args.seeds:
                print(f"\n{'='*68}\n  {m} / {ds} / seed={s}\n{'='*68}")
                try:
                    r = run_experiment(
                        m, ds,
                        epochs=args.epochs, hidden=args.hidden,
                        batch_size=args.batch, lr=args.lr, seed=s
                    )
                    all_results.append(r)
                except Exception as e:
                    print(f"  [!] {m}/{ds}/seed={s} failed: {e}")
                    import traceback; traceback.print_exc()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[saved] {args.out}  total={time.time()-t0:.0f}s")

    # Summary
    print("\n" + "=" * 78)
    print(f"{'Model':<14} {'Dataset':<12} {'Seed':>4} {'Trans AP':>9} {'Trans AUC':>10} {'Time(s)':>8}")
    print("─" * 78)
    for r in all_results:
        print(f"{r['model']:<14} {r['dataset']:<12} {r['seed']:>4} "
              f"{r['trans_ap']:>9.4f} {r['trans_auc']:>10.4f} {r['train_time_s']:>8.0f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
