"""Diagnostic: train v3_multi 3 epochs, print learned scale_mix, echo_gate,
and inspect predictions on a few test events to see if echo dominates."""
import os, sys, numpy as np, torch
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from data.download import download_dataset, get_data_splits
from train import build_model, run_epoch

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

torch.manual_seed(42); np.random.seed(42)
data = download_dataset("wikipedia")
splits = get_data_splits(data)
N, F = data["num_nodes"], data["feat_dim"]

# Build and train briefly
m = build_model("srgnn_v3_multi", N, F, 128).to(DEVICE)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

for ep in range(3):
    if hasattr(m, "reset"): m.reset()
    out = run_epoch(m, splits["train"], N, 500, optimizer=opt, desc=f"E{ep}")
    print(f"epoch {ep}: train AP={out['AP']:.4f}")

# Inspect learned params
print(f"\n=== Learned params ===")
print(f"scale_mix raw:    {m.scale_mix.data.cpu().tolist()}")
print(f"scale_mix softmax: {torch.softmax(m.scale_mix, 0).data.cpu().tolist()}")
print(f"echo_gate:        {m.echo_gate.data.item():.4f}  → sigmoid={torch.sigmoid(m.echo_gate).item():.4f}")
print(f"echo_norm.weight mean/std: {m.echo_norm.weight.mean().item():.3f} / {m.echo_norm.weight.std().item():.3f}")
print(f"echo_norm.bias  mean/std:  {m.echo_norm.bias.mean().item():.3f} / {m.echo_norm.bias.std().item():.3f}")

# Inspect echo magnitudes for active nodes
print(f"\n=== Echo magnitudes (after training, before test) ===")
for k, e in enumerate(m.echoes):
    nz = (e.echo.abs().sum(-1) > 1e-3).sum().item()
    norm = e.echo.norm(dim=-1)
    print(f"  scale {k} (λ={e.lambda_echo:.3f}): {nz}/{N} non-zero echoes, "
          f"mean_norm={norm[norm>1e-3].mean().item() if nz>0 else 0:.3f}, "
          f"max_norm={norm.max().item():.3f}")

# Run a tiny test batch with diagnostics
print(f"\n=== Test predictions (first batch) ===")
m.eval()
if hasattr(m, "reset"): m.reset()
# Don't load state_dict — just run with current state, but reset
test = splits["test"]
src = torch.tensor(test["sources"][:10], dtype=torch.long, device=DEVICE)
dst = torch.tensor(test["destinations"][:10], dtype=torch.long, device=DEVICE)
t   = torch.tensor(test["timestamps"][:10],   dtype=torch.float, device=DEVICE)
feat= torch.tensor(test["features"][:10],     dtype=torch.float, device=DEVICE)
# Random neg
neg = torch.tensor(np.random.choice(N, 10), dtype=torch.long, device=DEVICE)
with torch.no_grad():
    out = m(src, dst, t, feat, neg)
print(f"  pos_scores: {out['pos_score'].cpu().tolist()}")
print(f"  neg_scores: {out['neg_score'].cpu().tolist()}")
print(f"  R_uv_mean:  {out['R_uv_mean'].item():.4f}")
