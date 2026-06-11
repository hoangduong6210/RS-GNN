"""
RS-GNN v3.3 architecture SCHEMATICS (A1-A5). matplotlib Agg patches, dpi=240, English.
Grounded in v3_3_ARCHITECTURE_CURRENT.md + fsm_head.py (CAUSAL_RULE_MATRIX / hier tree).
NO fabricated blocks: every box maps to a real module/flag. NO data numbers invented.
QA: no text overlap, no missing glyph (DejaVu Sans covers lambda/phi/arrows), boxes/arrows clear.
Run: python3 make_schematics.py
Author: DATA team.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

OUT = "/users/PGS0407/binben14/VietHuy/Hoang/SR-GNN/figs"
DPI = 240

plt.rcParams.update({"font.size": 10.5, "figure.facecolor": "white", "axes.facecolor": "white"})

# colours
BACK = "#cfe0f3"   # backbone (stream A) fill
BACK_E = "#1f4e8c"
FSM = "#f6d9c2"    # FSM / readout (stream B) fill
FSM_E = "#c0651f"
SCORE = "#d6efd6"  # AP / link-score
SCORE_E = "#2e8b57"
INTERP = "#efe3f7" # interpretable / lifecycle
INTERP_E = "#7b3fa0"
NEUTRAL = "#eeeeee"

def box(ax, x, y, w, h, text, fc, ec, fs=10, tc="black", lw=1.6, style="round", bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"{style},pad=0.012,rounding_size=0.02",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, fontweight="bold" if bold else "normal")
    return (x, y, w, h)

def arrow(ax, p0, p1, color="black", lw=1.7, style="-|>", ls="-", rad=0.0, mut=14):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=mut, color=color,
                        lw=lw, linestyle=ls, zorder=1,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)

def R(b):  # right-mid of a box
    return (b[0]+b[2], b[1]+b[3]/2)
def L(b):  # left-mid
    return (b[0], b[1]+b[3]/2)
def Tm(b): # top-mid
    return (b[0]+b[2]/2, b[1]+b[3])
def Bm(b): # bottom-mid
    return (b[0]+b[2]/2, b[1])

def base_ax(figsize, xlim=(0, 16), ylim=(0, 9)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    return fig, ax


# ===================================================== A1 - Two-stream + DETACH wall
def a1():
    fig, ax = base_ax((15.5, 6.4), (0, 100), (0, 42))
    ax.set_title("A1  Two-stream RS-GNN v3.3:  detached FSM readout (config B)",
                 fontsize=13.5, fontweight="bold", pad=8)
    y = 22; h = 6
    # backbone (stream A) - left half, one colour
    b_event = box(ax, 1, y, 8, h, "EVENT\n(src,dst,t,feat)", NEUTRAL, "#888", fs=9.5)
    b_csn   = box(ax, 11, y, 9, h, "CSN\n(ResidualCSN)", BACK, BACK_E, fs=9.5)
    b_ectg  = box(ax, 22, y, 14.5, h,
                  "ECTGv3\nHawkes $\\lambda$ / Welford / rate\n[causal_batch]",
                  BACK, BACK_E, fs=9.2)
    b_drgc  = box(ax, 38.5, y, 9, h, "DRGC_v2\ncoupled-GRU\n$\\to$ edge_h", BACK, BACK_E, fs=9.2)
    arrow(ax, R(b_event), L(b_csn), BACK_E)
    arrow(ax, R(b_csn), L(b_ectg), BACK_E)
    arrow(ax, R(b_ectg), L(b_drgc), BACK_E)

    # DETACH wall - placed right AFTER edge_h (hatched barrier)
    xw = 50.5
    wall = Rectangle((xw-0.7, 4), 1.4, 34, facecolor="#f2c9c2", edgecolor="#c0392b",
                     hatch="////", lw=2.2, zorder=4)
    ax.add_patch(wall)
    ax.text(xw, 39.0, "stop-gradient wall", ha="center", va="bottom", color="#c0392b",
            fontsize=10.5, fontweight="bold")
    ax.text(xw, 2.4,
            "edge_h.detach()  $\\Rightarrow$  $\\partial$(link-pred loss)$/\\partial$(backbone) $= 0$",
            ha="center", va="top", color="#c0392b", fontsize=9.6)
    # FORWARD: edge_h passes THROUGH the wall (value flows; only gradient is cut)
    arrow(ax, R(b_drgc), (xw-0.7, y+h/2), BACK_E)
    arrow(ax, (xw+0.7, y+h/2), (53, y+h/2), FSM_E)
    ax.text(xw, y+h+0.6, "forward: edge_h value passes", ha="center", va="bottom",
            fontsize=8.2, color="#555", style="italic", zorder=7,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92))
    # GRADIENT-BLOCK: link-pred grad travels back and is stopped at the wall (X)
    yg = y - 4.2
    ax.annotate("", xy=(xw+1.0, yg), xytext=(60, yg),
                arrowprops=dict(arrowstyle="-|>", color=SCORE_E, lw=2.0, ls=(0,(4,3)),
                                mutation_scale=13, shrinkA=0, shrinkB=0), zorder=3)
    ax.plot([xw-1.4, xw+1.4], [yg-1.4, yg+1.4], color="#c0392b", lw=3.0, zorder=6)
    ax.plot([xw-1.4, xw+1.4], [yg+1.4, yg-1.4], color="#c0392b", lw=3.0, zorder=6)
    ax.text(61, yg, "link-pred gradient (blocked)", ha="left", va="center",
            fontsize=8.4, color=SCORE_E, fontweight="bold")

    # FSM stream (stream B) - right of wall, different colour
    b_so  = box(ax, 53, y, 11, h, "StateObserver\n$s_t$ over 5 states", FSM, FSM_E, fs=9.2)
    b_tp  = box(ax, 66.5, y, 14, h,
                "TransitionPredictor\n$T_{uv}=W+g(\\phi)$", FSM, FSM_E, fs=9.2)
    arrow(ax, R(b_so), L(b_tp), FSM_E)

    # two heads splitting out of TransitionPredictor
    b_pos = box(ax, 83, 30, 15, 6.5,
                "$s_{t1}$ pos\n$\\to$ ExistenceDecoder", FSM, FSM_E, fs=9.0)
    b_cal = box(ax, 83, 13, 15, 6.5,
                "$s_{t1}$ cal\n$\\to$ hier decode", INTERP, INTERP_E, fs=9.0)
    arrow(ax, R(b_tp), L(b_pos), FSM_E, rad=0.18)
    arrow(ax, R(b_tp), L(b_cal), INTERP_E, rad=-0.18)

    # terminal score / interpretable
    b_link = box(ax, 83, 38.5, 15, 3.2, "LINK SCORE  (AP)", SCORE, SCORE_E, fs=9.5, bold=True)
    b_int  = box(ax, 83, 6.0, 15, 3.2, "interpretable lifecycle", INTERP, INTERP_E, fs=9.0)
    arrow(ax, Tm(b_pos), Bm(b_link), SCORE_E)
    arrow(ax, Bm(b_cal), Tm(b_int), INTERP_E)

    # legend / annotation
    leg = [Line2D([0],[0], color=BACK_E, lw=8, label="Stream A: continuous backbone"),
           Line2D([0],[0], color=FSM_E, lw=8, label="Stream B: symbolic FSM readout (stop-grad)"),
           Line2D([0],[0], color=SCORE_E, lw=8, label="AP / link-prediction path"),
           Line2D([0],[0], color=INTERP_E, lw=8, label="interpretable / counterfactual path")]
    ax.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.005, -0.02),
              frameon=False, fontsize=8.6, ncol=2)
    ax.text(50.5, 8.8, "backbone: 0 link-pred grad  $\\Rightarrow$  inductive generalisation",
            ha="center", va="bottom", fontsize=9.4, style="italic", color="#333", zorder=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.92))
    fig.tight_layout()
    p = f"{OUT}/A1_two_stream_detach.png"; fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print("A1 ->", p)


# ===================================================== A2 - Gradient routing LITERAL
def a2():
    # LITERAL physics (v3_3_ARCHITECTURE_CURRENT.md §6, CPU-proven 0/56):
    #   detach wall sits BETWEEN backbone (LEFT) and FSM stream (RIGHT).
    #   ONLY the parsimony KL gradient pierces the wall -> backbone.
    #   BCE / edge_trans CE / self-consist NLL originate in stream B and are
    #   BLOCKED at the wall (each trains only its own stream-B module).
    fig, ax = base_ax((14.0, 7.2), (0, 100), (0, 52))
    ax.set_title("A2  Gradient routing (literal):  only the parsimony KL crosses the detach wall",
                 fontsize=13, fontweight="bold", pad=8)

    # ----- detach wall in the MIDDLE (drawn as a hatched barrier) -----
    xw = 49.0
    wall = Rectangle((xw-0.9, 3.5), 1.8, 45, facecolor="#f2c9c2", edgecolor="#c0392b",
                     hatch="////", lw=2.2, zorder=4)
    ax.add_patch(wall)
    ax.text(xw, 49.6, "DETACH wall  (edge_h.detach())", ha="center", va="bottom",
            color="#c0392b", fontsize=10.5, fontweight="bold")

    # ----- LEFT of wall: backbone (the ONLY module any gradient reaches across) -----
    bb = box(ax, 5, 27.5, 28, 11, "BACKBONE\nCSN / ECTGv3 / DRGC_v2\n(56 param tensors)",
             BACK, BACK_E, fs=9.6, lw=2.2, bold=True)
    ax.text(19, 24.0, "shaped ONLY by KL", ha="center", va="top",
            fontsize=9.0, style="italic", color=BACK_E)

    # ----- RIGHT of wall: stream-B modules, each with its OWN loss feeding it -----
    # (target module, its loss text, edge colour, y) - three blocked rows, well spaced
    rows = [
        ("FSM head ($s_{t1}$ pos)",         "BCE  (link-pred)",             SCORE_E,  FSM,     44),
        ("Hier heads (birth/alive/rising)", "edge_trans CE  (de-collapse)", INTERP_E, INTERP,  28),
        ("$w_{obs}$ scalar (belief trust)", "self-consist NLL  (WC-CONF)",  "#7a7a7a", NEUTRAL, 12),
    ]
    for tgt, losstxt, c, fc, yy in rows:
        tb = box(ax, 80, yy-3.4, 18, 6.8, tgt, fc, c, fs=8.6, lw=2.0)
        lb = box(ax, 55, yy-3.4, 19, 6.8, losstxt, "#ffffff", c, fs=8.8, lw=2.0)
        # forward-training gradient: loss -> its OWN module (stays right of wall)
        arrow(ax, R(lb), L(tb), c, lw=2.2)
        # the BLOCKED attempt: same loss tries to reach backbone but hits the wall
        ax.annotate("", xy=(xw+1.2, yy), xytext=(L(lb)[0]-0.5, yy),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.0, ls=(0,(4,3)),
                                    mutation_scale=14, shrinkA=0, shrinkB=0), zorder=3)
        ax.plot([xw-1.6, xw+1.6], [yy-1.6, yy+1.6], color="#c0392b", lw=3.0, zorder=6)
        ax.plot([xw-1.6, xw+1.6], [yy+1.6, yy-1.6], color="#c0392b", lw=3.0, zorder=6)
        ax.text(xw+2.6, yy, "blocked", ha="left", va="center",
                fontsize=8.0, color="#c0392b", fontweight="bold", zorder=6)

    # ----- KL: the ONE gradient that PIERCES the wall into the backbone -----
    # placed in the gap between row1 (y=44) and row2 (y=28), level with the backbone centre.
    ky = 33.0
    kl = box(ax, 55, ky-3.4, 19, 6.8, "$\\lambda_{echo}\\cdot$ KL  (TIP/VAE parsimony)",
             "#ffffff", BACK_E, fs=8.6, lw=2.4)
    # straight horizontal arrow from KL loss, THROUGH the wall, into the backbone
    ax.annotate("", xy=(R(bb)[0], ky), xytext=(L(kl)[0], ky),
                arrowprops=dict(arrowstyle="-|>", color=BACK_E, lw=2.6,
                                mutation_scale=16, shrinkA=0, shrinkB=0), zorder=5)
    ax.text(xw, 38.2, "crosses", ha="center", va="center",
            fontsize=8.4, color=BACK_E, fontweight="bold", zorder=6)

    # ----- legend for arrow semantics -----
    leg = [Line2D([0],[0], color="black", lw=2.4, ls="-", label="gradient that REACHES its target"),
           Line2D([0],[0], color="black", lw=2.0, ls=(0,(4,3)), label="gradient BLOCKED at the wall")]
    ax.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.005, -0.01),
              frameon=False, fontsize=8.6)

    ax.text(74, 1.4,
            "Only the parsimony KL reaches the backbone; the link-pred BCE, lifecycle CE,\n"
            "and self-consistency NLL are all blocked by the detach $\\Rightarrow$ "
            "backbone never sees the task gradient.",
            ha="center", va="bottom", fontsize=8.9, color="#333")
    fig.tight_layout()
    p = f"{OUT}/A2_gradient_decoupling.png"; fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print("A2 ->", p)


# ===================================================== A3 - Hierarchical decode tree
def a3():
    fig, ax = base_ax((12.5, 7.2), (0, 100), (0, 64))
    ax.set_title("A3  Hierarchical decode of $s_{t1}$cal:  DECAY can win argmax",
                 fontsize=13, fontweight="bold", pad=8)
    # root
    root = box(ax, 38, 55, 24, 6.5, "per-pair PRE signals\n($p_{birth},p_{alive},p_{rising}$)",
               NEUTRAL, "#666", fs=9.2)
    # level 1: p_birth?
    g1 = box(ax, 40, 44, 20, 5.5, "$p_{birth}$ ?", "#fff7e0", "#c79a1f", fs=10)
    arrow(ax, Bm(root), Tm(g1), "#666")
    birth = box(ax, 4, 44, 18, 5.5, "BIRTH", FSM, FSM_E, fs=10, bold=True)
    arrow(ax, L(g1), R(birth), "#c79a1f")
    ax.text(28, 47.6, "yes", ha="center", fontsize=8.6, color="#c79a1f")

    # level 2: p_alive?
    g2 = box(ax, 40, 33, 20, 5.5, "$p_{alive}$ ?", "#fff7e0", "#c79a1f", fs=10)
    arrow(ax, Bm(g1), Tm(g2), "#c79a1f")
    ax.text(51.5, 41.2, "no", ha="left", fontsize=8.6, color="#c79a1f")
    death = box(ax, 78, 33, 18, 5.5, "DEATH", FSM, FSM_E, fs=10, bold=True)
    arrow(ax, R(g2), L(death), "#c79a1f")
    ax.text(72, 36.6, "dead", ha="center", fontsize=8.6, color="#c79a1f")

    # level 3: p_rising? (alive branch)
    g3 = box(ax, 40, 22, 20, 5.5, "$p_{rising}$ ?", "#fff7e0", "#c79a1f", fs=10)
    arrow(ax, Bm(g2), Tm(g3), "#c79a1f")
    ax.text(51.5, 30.2, "alive", ha="left", fontsize=8.6, color="#c79a1f")
    reinf = box(ax, 4, 22, 18, 5.5, "REINFORCE", FSM, FSM_E, fs=9.5, bold=True)
    decay = box(ax, 78, 22, 18, 5.5, "DECAY", FSM, FSM_E, fs=10, bold=True)
    arrow(ax, L(g3), R(reinf), "#c79a1f")
    arrow(ax, R(g3), L(decay), "#c79a1f")
    ax.text(31, 25.6, "rising", ha="center", fontsize=8.6, color="#c79a1f")
    ax.text(70, 25.6, "falling", ha="center", fontsize=8.6, color="#c79a1f")

    # factorisation formula block (bottom, away from tree)
    ax.text(50, 12.5,
            "$P(\\mathrm{BIRTH})=p_{birth}$\n"
            "$P(\\mathrm{REINFORCE})=(1-p_{birth})\\,p_{alive}\\,p_{rising}$\n"
            "$P(\\mathrm{DECAY})=(1-p_{birth})\\,p_{alive}\\,(1-p_{rising})$\n"
            "$P(\\mathrm{DEATH})=(1-p_{birth})\\,(1-p_{alive})$",
            ha="center", va="center", fontsize=10.2,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f4f4f4", ec="#999"))
    ax.text(50, 2.4,
            "DECAY competes ONLY with REINFORCE inside the alive branch (never directly with DEATH).",
            ha="center", va="bottom", fontsize=9.2, style="italic", color="#333")
    fig.tight_layout()
    p = f"{OUT}/A3_hier_decode_tree.png"; fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print("A3 ->", p)


# ===================================================== A4 - Lifecycle FSM (band-5 admissibility)
def a4():
    fig, ax = base_ax((13.0, 6.0), (0, 100), (0, 40))
    ax.set_title("A4  Lifecycle FSM (CAUSAL_RULE_MATRIX, band-5):  single-rung moves only",
                 fontsize=12.5, fontweight="bold", pad=8)
    states = ["IDLE", "BIRTH", "REINFORCE", "DECAY", "DEATH"]
    cols = ["#9aa0a6", "#2e8b57", "#4682b4", "#e08e3c", "#c0392b"]
    xs = [9, 28, 50, 72, 91]
    yc = 22; w = 13; h = 7
    bxs = []
    for s, c, x in zip(states, cols, xs):
        b = box(ax, x-w/2, yc-h/2, w, h, s, "#ffffff", c, fs=9.5, lw=2.4, bold=True)
        bxs.append(b)
        # self-loop (curved arrow on top)
        a = FancyArrowPatch((x-2.2, yc+h/2), (x+2.2, yc+h/2), arrowstyle="-|>",
                            mutation_scale=11, color=c, lw=1.6,
                            connectionstyle="arc3,rad=-1.5", zorder=1)
        ax.add_patch(a)
    # adjacent bidirectional arrows (offset above/below so heads don't overlap)
    for i in range(4):
        x0 = xs[i] + w/2; x1 = xs[i+1] - w/2
        arrow(ax, (x0, yc+1.3), (x1, yc+1.3), "#333", lw=1.8, rad=0.0)
        arrow(ax, (x1, yc-1.3), (x0, yc-1.3), "#333", lw=1.8, rad=0.0)
    ax.text(50, yc+h/2+5.2, "adjacent transitions (both directions) + self-loops = ADMISSIBLE",
            ha="center", fontsize=9.4, color="#333")

    # forbidden non-adjacent jumps (red dashed, crossed)
    forb = [(0, 2), (1, 3), (2, 4), (1, 4)]
    for (i, j) in forb:
        x0, x1 = xs[i], xs[j]
        ax.plot([x0, x1], [yc-h/2-3.5, yc-h/2-3.5], ls=(0, (4, 3)), color="#c0392b", lw=1.5, zorder=1)
        xm = (x0+x1)/2
        ax.plot([xm-0.9, xm+0.9], [yc-h/2-3.5-0.9, yc-h/2-3.5+0.9], color="#c0392b", lw=2.2)
        ax.plot([xm-0.9, xm+0.9], [yc-h/2-3.5+0.9, yc-h/2-3.5-0.9], color="#c0392b", lw=2.2)
    ax.text(50, yc-h/2-8.0,
            "non-adjacent jumps (e.g. IDLE$\\to$REINFORCE, REINFORCE$\\to$DEATH) are FORBIDDEN",
            ha="center", fontsize=9.4, color="#c0392b")
    fig.tight_layout()
    p = f"{OUT}/A4_lifecycle_fsm_band5.png"; fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print("A4 ->", p)


# ===================================================== A5 - Causal-confidence overlay (advisory)
def a5():
    fig, ax = base_ax((13.5, 6.2), (0, 100), (0, 44))
    ax.set_title("A5  WC-CONF causal-confidence overlay:  advisory, never bends prediction",
                 fontsize=12.5, fontweight="bold", pad=8)
    # FREE predict path (top)
    b_state = box(ax, 2, 30, 17, 8, "$s_{t1}$ pos\n(FREE predict)", FSM, FSM_E, fs=9.2)
    b_score = box(ax, 27, 30, 17, 8, "LINK SCORE\n(unchanged)", SCORE, SCORE_E, fs=9.2, bold=True)
    arrow(ax, R(b_state), L(b_score), SCORE_E, lw=2.0)
    ax.text(10.5, 39.6, "prediction path stays FREE  ($\\Delta$ score $= 0$)",
            ha="left", fontsize=9.0, color=SCORE_E)

    # walked-belief path (bottom)
    b_grnd = box(ax, 2, 7, 17, 8, "grounded init\nsoftmax($s_t$ pos)", INTERP, INTERP_E, fs=8.8)
    b_walk = box(ax, 27, 7, 21, 8,
                 "walked belief\n$b_t=$softmax($b_{t-1}T_{uv}$)\non causal ray", INTERP, INTERP_E, fs=8.6)
    arrow(ax, R(b_grnd), L(b_walk), INTERP_E, lw=2.0)

    # coherence merge
    b_ct = box(ax, 56, 18, 18, 9,
               "coherence\n$c_t=\\sum s_{t1}\\cdot$reach$(b)$", "#fff7e0", "#c79a1f", fs=9.0, bold=True)
    arrow(ax, R(b_state), L(b_ct), "#c79a1f", lw=1.6, rad=-0.25, style="-|>")
    arrow(ax, R(b_walk), L(b_ct), "#c79a1f", lw=1.6, rad=0.15, style="-|>")

    b_adv = box(ax, 80, 18, 18, 9, "$c_t$ advisory\n(human inspect)", NEUTRAL, "#777", fs=9.2)
    arrow(ax, R(b_ct), L(b_adv), "#777", lw=1.8)

    ax.text(50, 2.6,
            "$c_t$ = % of free next-state that lands on the causal ray;  validity flag, NOT an error predictor.",
            ha="center", va="bottom", fontsize=9.1, style="italic", color="#333")
    fig.tight_layout()
    p = f"{OUT}/A5_causal_confidence_overlay.png"; fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print("A5 ->", p)


if __name__ == "__main__":
    a1(); a2(); a3(); a4(); a5()
    print("ALL SCHEMATICS DONE ->", OUT)
