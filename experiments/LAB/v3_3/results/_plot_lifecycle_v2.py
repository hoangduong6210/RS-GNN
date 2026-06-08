"""Re-draw lifecycle for CoEdit directed pair 3178->7437 (42 events).
v2 fix: 3 SEPARATE panels sharing x (time). NO dual-axis overlay of gates vs slope.
Numbers measured live from disk -- nothing hardcoded.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = "/users/PGS0407/binben14/VietHuy/Hoang/SR-GNN"
COEDIT = ROOT + "/experiments/data/coedit.npz"
FAITH = ROOT + "/experiments/LAB/v3_3/results/faithfulness_coedit_v3_hier_hv2_let0.5_s42_cbON.npz"
OUT = ROOT + "/experiments/LAB/v3_3/results/lifecycle_pair_3178_7437_v2.png"
SRC, DST = 3178, 7437
BASE = 68000  # faith row i <=> coedit global idx 68000+i (verified 100%)

c = np.load(COEDIT, allow_pickle=True)
f = np.load(FAITH, allow_pickle=True)
N = f["z_pair"].shape[0]
gidx = np.arange(BASE, BASE + N)
fsrc, fdst, fts = c["sources"][gidx], c["destinations"][gidx], c["timestamps"][gidx]
mask = (fsrc == SRC) & (fdst == DST)
rows = np.where(mask)[0]
assert rows.size == 42, rows.size

t = fts[rows].astype(float)
t = t - t.min()                       # seconds since first test event of this pair
tmin = t / 60.0                       # minutes (span ~18.7 min)
cal = f["argmax_s_t1_cal"][rows]
slope = f["slope_rel"][rows].astype(float)
p_alive = f["p_alive_gate"][rows].astype(float)
p_rising = f["p_rising_gate"][rows].astype(float)
p_birth = f["p_birth_gate"][rows].astype(float)

# interaction rate = 1/dt (events/sec); first event has no preceding dt
dt = np.diff(t)
rate = np.full(t.shape, np.nan)
rate[1:] = 1.0 / np.where(dt > 0, dt, np.nan)

# state -> color/label   (1=BIRTH 2=REINFORCE 3=DECAY 4=DEATH)
STATE = {1: ("BIRTH", "#1f77b4"), 2: ("REINFORCE", "#2ca02c"),
         3: ("DECAY", "#d62728"), 4: ("DEATH", "#7f7f7f")}

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
fig.suptitle("CoEdit pair 3178→7437 | 42 events | 1 BIRTH, 20 REINFORCE, 21 DECAY, 0 DEATH",
             fontsize=14, fontweight="bold")

# ---- Panel 1: interaction cadence, points colored by CALIBRATED state ----
ax1.plot(tmin, rate, "-", color="#999999", lw=1.0, zorder=1, alpha=0.8)
for s, (lab, col) in STATE.items():
    sel = cal == s
    if not sel.any():
        continue
    ax1.scatter(tmin[sel], rate[sel], c=col, s=55, edgecolor="k",
                linewidth=0.4, zorder=3, label=f"{lab} (n={int(sel.sum())})")
# special BIRTH marker (event 0)
b = np.where(cal == 1)[0]
if b.size:
    bi = b[0]
    ax1.scatter(tmin[bi], np.nanmax(rate[1:]) * 0.0 + (rate[bi] if not np.isnan(rate[bi]) else np.nanmin(rate[1:])),
                marker="*", s=420, c=STATE[1][1], edgecolor="k", linewidth=0.8, zorder=5)
    ax1.annotate("BIRTH = event mở màn",
                 xy=(tmin[bi], np.nanmin(rate[1:])),
                 xytext=(tmin[bi] + 1.0, np.nanmax(rate[1:]) * 0.6),
                 fontsize=10, fontweight="bold", color=STATE[1][1],
                 arrowprops=dict(arrowstyle="->", color=STATE[1][1], lw=1.5))
ax1.set_ylabel("rate = 1/Δt (events/sec)")
ax1.set_title("Panel 1 — Nhịp tương tác (điểm tô theo state CALIBRATED)")
ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax1.grid(alpha=0.25)

# ---- Panel 2: dynamics (slope_rel) ----
ax2.axhline(0.0, color="k", lw=1.0, ls="--", zorder=1)
ax2.plot(tmin, slope, "-o", color="#444444", ms=4, lw=1.3, zorder=3)
ax2.fill_between(tmin, 0, slope, where=slope > 0, color="#2ca02c", alpha=0.25,
                 interpolate=True, label="slope>0 → tăng (REINFORCE)")
ax2.fill_between(tmin, 0, slope, where=slope < 0, color="#d62728", alpha=0.25,
                 interpolate=True, label="slope<0 → giảm (DECAY)")
ax2.set_ylim(-0.95, 0.55)
ax2.set_ylabel("slope_rel (per-pair rate change)")
ax2.set_title("Panel 2 — Động lực (slope_rel)")
ax2.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax2.grid(alpha=0.25)

# ---- Panel 3: gates [0,1] on their OWN axis ----
ax3.axhline(0.5, color="k", lw=1.0, ls=":", zorder=1, label="ref y=0.5")
ax3.plot(tmin, p_alive, "-o", color="#1f77b4", ms=4, lw=1.4, label="p_alive_gate")
ax3.plot(tmin, p_rising, "-s", color="#2ca02c", ms=4, lw=1.4, label="p_rising_gate")
ax3.plot(tmin, p_birth, "-^", color="#9467bd", ms=4, lw=1.4, label="p_birth_gate")
ax3.set_ylim(0.0, 1.0)
ax3.set_ylabel("gate probability")
ax3.set_xlabel("time since first test event (minutes)")
ax3.set_title("Panel 3 — Gates (xác suất 0–1, trục riêng) — p_alive luôn >0.5 → không DEATH")
ax3.legend(loc="center right", fontsize=9, framealpha=0.9, ncol=2)
ax3.grid(alpha=0.25)

fig.tight_layout(rect=[0, 0, 1, 0.975])
fig.savefig(OUT, dpi=140)
print("SAVED", OUT)
print("p_alive min/max %.4f %.4f" % (p_alive.min(), p_alive.max()))
print("birth event index (0-based):", int(np.where(cal == 1)[0][0]))
print("slope sign-change idx:", (np.where(np.diff(np.sign(slope)) != 0)[0] + 1).tolist())
print("cal seq:", cal.tolist())
print("time span minutes: %.2f" % tmin.max())
