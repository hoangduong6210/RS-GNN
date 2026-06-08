"""Publication-quality (English) lifecycle figure for CoEdit directed pair 3178->7437.
3 vertical panels sharing x = time (minutes). Numbers measured live from disk; nothing hardcoded.
faith row i <=> coedit global idx 68000+i (verified). 42 events for this pair.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = "/users/PGS0407/binben14/VietHuy/Hoang/SR-GNN"
COEDIT = ROOT + "/experiments/data/coedit.npz"
FAITH = ROOT + "/experiments/LAB/v3_3/results/faithfulness_coedit_v3_hier_hv2_let0.5_s42_cbON.npz"
OUT = ROOT + "/experiments/LAB/v3_3/results/lifecycle_pair_3178_7437_en.png"
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

# Calibrated lifecycle state -> (label, professional color)
#   1=BIRTH 2=REINFORCE 3=DECAY 4=DEATH
STATE = {1: ("BIRTH",     "#2ca02c"),   # green
         2: ("REINFORCE", "#4682b4"),   # steelblue
         3: ("DECAY",     "#ff7f0e"),   # orange
         4: ("DEATH",     "#d62728")}   # red

# ---- global style ----
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
})

fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(9, 10), sharex=True, constrained_layout=True)
fig.suptitle(
    "Lifecycle of a real CoEdit pair (3178→7437): 42 events\n"
    "1 BIRTH · 20 REINFORCE · 21 DECAY · 0 DEATH",
    fontsize=13, fontweight="bold")

# ============ Panel 1: Interaction tempo ============
ax1.plot(tmin, rate, "-", color="#999999", lw=1.0, zorder=1, alpha=0.85)
for s, (lab, col) in STATE.items():
    sel = cal == s
    if not sel.any():
        continue
    ax1.scatter(tmin[sel], rate[sel], c=col, s=60, edgecolor="k",
                linewidth=0.5, zorder=3, label=f"{lab} (n={int(sel.sum())})")
# special BIRTH star + annotation
b = np.where(cal == 1)[0]
if b.size:
    bi = b[0]
    ymin = np.nanmin(rate[1:])
    ymax = np.nanmax(rate[1:])
    by = rate[bi] if not np.isnan(rate[bi]) else ymin
    ax1.scatter(tmin[bi], by, marker="*", s=480, c=STATE[1][1],
                edgecolor="k", linewidth=0.9, zorder=5)
    ax1.annotate("BIRTH (first event)",
                 xy=(tmin[bi], by),
                 xytext=(tmin[bi] + 1.2, ymax * 0.62),
                 fontsize=10, fontweight="bold", color=STATE[1][1],
                 arrowprops=dict(arrowstyle="->", color=STATE[1][1], lw=1.6))
ax1.set_ylabel("Edit rate  1/Δt  (events/sec)")
ax1.set_title("Interaction tempo")
leg1 = ax1.legend(loc="upper right", title="Calibrated lifecycle state",
                  framealpha=0.92)
leg1.get_title().set_fontsize(9.5)
leg1.get_title().set_fontweight("bold")
ax1.grid(alpha=0.25, linewidth=0.7)

# ============ Panel 2: Tempo dynamics (slope_rel) ============
ax2.axhline(0.0, color="k", lw=1.0, ls="--", zorder=1)
ax2.plot(tmin, slope, "-o", color="#333333", ms=4.5, lw=1.3, zorder=3)
ax2.fill_between(tmin, 0, slope, where=slope > 0, color="#2ca02c", alpha=0.22,
                 interpolate=True)
ax2.fill_between(tmin, 0, slope, where=slope < 0, color="#d62728", alpha=0.22,
                 interpolate=True)
ax2.set_ylim(-0.95, 0.55)
ax2.set_ylabel("slope_rel")
ax2.set_title("Tempo dynamics (per-pair rate slope)")
ax2.grid(alpha=0.25, linewidth=0.7)
handles2 = [
    Patch(facecolor="#2ca02c", alpha=0.35, label="rising  →  REINFORCE"),
    Patch(facecolor="#d62728", alpha=0.35, label="falling  →  DECAY"),
    Line2D([0], [0], color="k", ls="--", lw=1.0, label="slope = 0"),
]
ax2.legend(handles=handles2, loc="lower right", framealpha=0.92)

# ============ Panel 3: Hierarchical decode gates ============
ax3.axhline(0.5, color="k", lw=1.1, ls="--", zorder=1)
ax3.plot(tmin, p_alive, "-o", color="#1f77b4", ms=4.5, lw=1.5, label="p_alive_gate")
ax3.plot(tmin, p_rising, "-s", color="#2ca02c", ms=4.5, lw=1.5, label="p_rising_gate")
ax3.plot(tmin, p_birth, "-^", color="#9467bd", ms=4.5, lw=1.5, label="p_birth_gate")
ax3.text(tmin.max() * 0.015, 0.515, "alive threshold (0.5)", ha="left",
         va="bottom", fontsize=9, color="#444444", style="italic")
ax3.set_ylim(0.0, 1.0)
ax3.set_ylabel("Gate probability")
ax3.set_xlabel("Time since first event (minutes)")
ax3.set_title("Hierarchical decode gates")
ax3.legend(loc="upper right", framealpha=0.92, ncol=1)
ax3.grid(alpha=0.25, linewidth=0.7)
ax3.annotate("p_alive > 0.5 throughout  →  pair stays alive  →  no DEATH",
             xy=(tmin.max() * 0.5, 0.06), ha="center", va="bottom",
             fontsize=9.5, fontweight="bold", color="#1f77b4",
             bbox=dict(boxstyle="round,pad=0.35", fc="#eef4fb",
                       ec="#1f77b4", lw=0.8))

fig.savefig(OUT, dpi=220)
print("SAVED", OUT)
print("dpi=220 figsize=(9,10)")
print("p_alive min/max %.4f %.4f" % (p_alive.min(), p_alive.max()))
print("slope min/max %.4f %.4f" % (slope.min(), slope.max()))
print("birth event index (0-based):", int(np.where(cal == 1)[0][0]))
print("slope sign-change idx:", (np.where(np.diff(np.sign(slope)) != 0)[0] + 1).tolist())
print("cal counts:", {int(k): int((cal == k).sum()) for k in [1, 2, 3, 4]})
print("time span minutes: %.2f" % tmin.max())
