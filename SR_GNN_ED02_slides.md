---
marp: true
theme: default
paginate: true
size: 16:9
header: 'RS-GNN - Regularization-by-Decoupling for Inductive Temporal Link Prediction'
footer: 'Duong Viet Hoang & Shih Lun-Min · Da-Yeh University · Draft ED02'
style: |
  section { font-size: 24px; }
  h1 { color: #1a3c6e; }
  h2 { color: #1a3c6e; }
  table { font-size: 20px; }
  .small { font-size: 18px; }
  .cite { font-size: 16px; color: #666; }
---

# RS-GNN
## Regularization-by-Decoupling for Inductive Temporal Link Prediction
### …with a faithful, intervene-able lifecycle readout

**Duong Viet Hoang · Shih Lun-Min**
Department of Computer Science, Da-Yeh University, Taiwan

<span class="cite">Draft ED02 - every reported number traces to a results JSON (Appendix A). Every ± is the sample std (n−1) over the stated seeds.</span>

---

## The problem: inductive temporal link prediction

A temporal graph is a stream of timestamped edges $\{(u_i,v_i,t_i,x_i)\}$. The task: given history, score whether $(u,v)$ occurs at $t$.

**Two protocols reward different biases:**
- **Transductive** - test edges reuse *seen* nodes → a model can lean on a memorized per-node embedding.
- **Inductive** - test edges involve an *unseen* node → any per-node embedding is uninitialized. This is the **deployment-relevant, harder** regime.

> A model that wins transductively but collapses inductively has memorized the population, not learned the process.

**State-of-the-art (JODIE, TGN, DyRep, TGAT, CAWN, GraphMixer, DyGFormer) all train the representation *end-to-end* against the link loss.** We ask whether that is the right default.

---

## Key finding: end-to-end coupling is the wrong default

A **single-variable knob ablation** - flip `enable_main_predictor` False→True, *every other flag fixed* - isolates the cause.

| Arm | flag flipped | inductive AP | Δ vs B |
|---|---|---|---|
| **B (decoupled)** | - | **0.9899 ± 0.0016** | - |
| **K1** | end-to-end head ON | **0.7788 ± 0.0193** | **−21.11 pp** |
| K2 | lfg soft→hard | 0.9868 | −0.31 (inert) |
| K3 | compliance floor | 0.9889 | −0.10 (inert) |

<span class="cite">CoEdit, 3 seeds {42,1,7}. `v3_3_coedit_knob_ablation_3seed.json`.</span>

**One flag carries the entire gap.** Turning on the end-to-end head - letting the link gradient reach the backbone - costs **−21.1 pp** inductively. The two co-varying gates are inert.

---

## The finding visualized

![w:760](figs/fig6_decoupling_ablation.png)

<span class="cite">Decoupling ablation, CoEdit inductive AP. Decoupled full model (B) vs end-to-end (C). At 5 seeds: B 0.9876 / C 0.7593 → **B−C = +22.8 pp**. Arm A (no-lifecycle, 0.928) already clears every baseline.</span>

---

## Not a CoEdit artifact: the same flag on graphs we did not build

Same single-flag knob, two **standard** benchmarks we did not construct:

| Dataset | B (decoupled) | K1 (coupled) | Δ inductive |
|---|---|---|---|
| CoEdit *(built here)* | 0.9899 | 0.7788 | **+21.11 pp** |
| Wikipedia *(standard)* | 0.9957 | 0.9093 | **+8.64 pp** |
| MOOC *(standard)* | 0.9978 | 0.9894 | **+0.85 pp** |

<span class="cite">3 seeds {42,1,7}. `v3_3_knob_ablation_{wikipedia,mooc}_3seed.json`.</span>

**Decoupling wins everywhere; magnitude is honest.** The gap measures *how much training-node identity the coupled head can exploit* - large where unseen-node generalization is hard (CoEdit), small where features already transfer (MOOC). The small MOOC delta is reported plainly.

---

## Irreversibility control: freeze-then-probe cannot recover it

The obvious challenge: *"decoupling is just classic freeze-then-probe."* So we ran exactly that on RS-GNN's own backbone.

| Dataset | decouple-by-construction | **freeze-then-probe** | coupled / end-to-end |
|---|---|---|---|
| **CoEdit** | **0.988** | **0.768** | 0.759 |
| **Wikipedia** | **0.996** (n=3) | **0.897** (n=2) | 0.909 (n=3) |

<span class="cite">`v3_3_frozen_probe_ARM{1,2}_*.json`. FtP ≈ coupled ≪ decoupled - on **both** datasets.</span>

**Once the link loss has shaped the backbone, freezing and re-probing does not recover inductive transfer.** Freeze-then-probe lands on the *coupled* number, not the decoupled one. The contribution is **preventing contamination by construction** - never exposing the backbone to the link gradient. To our knowledge this irreversibility is new for temporal link prediction.

---

## Method: two streams, one detach

![w:820](figs/A1_two_stream_detach.png)

<span class="cite">Backbone representation `edge_h` crosses a stop-gradient (`.detach()`) before the symbolic Stream B; **no link-prediction gradient reaches the backbone.**</span>

---

## Method: where each gradient goes

**Stream A - continuous backbone.** Event encoder + a multi-signal per-pair edge-state operator (Hawkes intensity, Welford gap statistics, recurrence/rate EWMAs) + a coupled-GRU node-memory. Trained **only** by a variational parsimony KL + deterministic update laws.

**Stream B - symbolic readout.** Reads `edge_h.detach()`; a hierarchical 5-state lifecycle decoder; the existence-decoder logit is the **only** thing scored.

$$\mathcal{L} = \mathcal{L}_{\text{BCE}}(\hat p, y) \;+\; \lambda_{\text{kl}}\,\mathrm{KL}(q(z|m)\,\|\,p(z)) \;+\; \lambda_{\text{dc}}\,\mathcal{L}_{\text{decol-CE}}(s^{\text{cal}}_{t+1})$$

Three **disjoint** gradient routes: backbone ← KL · scored head ← BCE · interpretable head ← de-collapse CE.

<span class="cite">A committed graph-ancestry test certifies `pred_loss.backward()` deposits **0.000e+0** on every backbone tensor (`decoupling_invariants_verify.json`). The primitives (stop-gradient, frozen probing, information bottleneck) are prior art; the measured result is not.</span>

---

## A faithful, intervene-able readout - riding free on the same detach

The same `.detach()` places the symbolic decoder **off the scored path**, so interventions never tax AP (eval-time bit-identical).

The readout commits to a **falsifiable** rule - *REINFORCE ⟺ rising edit cadence* - confirmed three ways from the **learned** gates:

- **(A)** Population: REINFORCE-decoded pairs sit at slope −0.49, above DECAY-decoded at −0.86.
- **(B)** Under `do(slope)`, forcing a falling pair's cadence up flips $P(\text{REINFORCE})$ 0→1.
- **(C)** The same flip at single-pair granularity.

> A cadence-blind decoder would fail all three. It does not.

This is a **temporal concept bottleneck** (Koh et al. 2020) - **not** a learned SCM, **not** Pearl-sense causal identification. `do(·)` is an input override, and we say so.

---

## The readout is falsifiable and bidirectional

![w:560](figs/fig6_faithful_falsifiable.png) ![w:560](figs/fig7_intervenable_scm.png)

<span class="cite">Left: the "REINFORCE ⟺ rising cadence" rule, three views (population / do(slope) / single-pair). Right: bidirectional typed control - REINFORCE→do(silence)→DEATH (1.0), DECAY→do(rising)→REINFORCE (1.0); reversibility Δ=0 by exact reconstruction.</span>

**do(state), 3 seeds, N=12000 real pairs:** do(DEATH) drives existence down for **100% of pairs every seed** (mean −0.522 ± 0.001); do(BIRTH/REINFORCE) up for **100% every seed**. The directionality is a **learned** property - the trained per-seed weights preserve DEATH<IDLE<DECAY<{REINFORCE,BIRTH}, not just the hand-set init.

---

## Results: CoEdit is the discriminating benchmark

![w:720](figs/fig4_coedit_headline.png)

<span class="cite">CoEdit inductive AP, 3 seeds. RS-GNN (config B) **0.9885 ± 0.0035** (5-seed 0.9876 ± 0.0030), **+13.5 pts** (95% CI [12.6,14.5]) over the best parametric baseline (TGAT 0.853). DyGFormer frontier 0.612; EdgeBank floor ≈0.59 - the win is **not** memorization.</span>

---

## Results: cross-dataset, protocol-matched, 3 seeds

| Model | CoEdit ind | Wikipedia ind | MOOC ind |
|---|---|---|---|
| **RS-GNN (B)** | **0.9885** | 0.9959 | **0.9978** |
| TGAT | 0.8530 | **0.9981** | 0.9737 |
| JODIE | 0.8147 | 0.9860 | 0.9901 |
| CAWN | 0.7825 | 0.9877 | 0.8101 |
| DyGFormer *(frontier)* | 0.6120 | 0.7859 | - |
| EdgeBank *(floor)* | 0.5899 | 0.6541 | 0.5534 |

<span class="cite">All models through one leak-audited harness, same negative pool. Full per-seed values in Appendix A.</span>

**CoEdit is where models separate.** RS-GNN is #1 both protocols there. On Wikipedia it is #1 transductive / #2 inductive (TGAT's lone inductive win pairs with a collapsed transductive AP 0.658 - the same identity-shortcut signature our knob isolates). **MOOC is near-saturated → weak evidence**, claimed only as "competitive at the ceiling."

---

## Where the tax lands: the identity shortcut

Removing the detach costs **−22.8 pp inductively** but only **−3.8 pp transductively** (5-seed).

> The coupled head reshapes the backbone toward *training-node identity* - rewarded transductively, dead weight on unseen nodes. The tax lands almost entirely on the inductive split.

**Three independent cuts agree, ruling out a generic frozen-probe reading:**
1. single-flag knob on **three** datasets,
2. the **irreversibility** control (freezing after the fact recovers nothing),
3. the **split-localized** tax above.

The mechanism is triangulated - not carried by one self-built benchmark.

---

## A deliberate trade-off: two separable contributions

**C1 - decoupling for inductive AP - is GENERAL.** Shown directionally on three datasets, no reference to CoEdit's domain.

**C2 - the per-pair causal lifecycle FSM - is deliberately dataset-scoped, by design.**

> Lifecycle drivers are *different concepts* across domains, not the same concept at a different scale: banking = {frequency, money-amount}; preference = {frequency, distinct-user degree}; CoEdit = {edit-rate, gap-vs-habit, recurrence}.

A single global FSM would have to discard the per-domain driver knowledge that makes the readout faithful. So the part that **generalizes** (C1) is shown cross-dataset; the part that is **faithful** (C2) is scoped on purpose. One detach, two consequences - C2 complements C1 rather than competing with it.

---

## Honest limitations

- **CoEdit-tuned headline.** Config B was tuned on CoEdit; the +13.5 *AP* number is CoEdit-specific. The decoupling **mechanism** is shown cross-dataset (knob on Wiki/MOOC).
- **The causal structure is designer-imposed.** Gates are hand-specified analytic priors + ~150 residuals; `do(state)` reads a sorted softplus-weight vector. The *ordering* is learned-confirmed, but we do **not** claim Pearl-sense causal discovery.
- **Confidence is self-consistency, not error prediction.** The coherence signal predicts the model's *own* rule violations (AUC 0.9985) but **not** actual misses (AUC 0.405 ± 0.484, seed-fragile). **We retract the error-predictor claim.**
- **Scored path is non-Markov** via an `ever_alive` gate → the typed-transition formalism is bounded to the interpretable path.
- **Frontier comparison is partial:** TCL/NAT unrun; MOOC DyGFormer not finalized. We downscope accordingly rather than claim to beat the full named frontier.

---

## Retracted in this draft (research integrity)

We walked back every claim that did not survive a multi-seed or controlled test:

- **Echo / "resonance" memory** → backbone regularizer is a plain VAE/parsimony KL.
- **Learned transition-CE** → de-collapse CE on the *interpretable* next-state only.
- **Regime-switch / distribution-shift advantage** → **falsified** (does not beat CAWN on post-change-point slices).
- **Confidence-as-error-flag** → retracted (anti-calibrated in the worst seed).
- **Causal-confidence-as-correctness / do-calculus identification** → demoted to "typed input override."
- **A compiler optimality theorem + regulatory-necessity framing** → removed.

> We state the deltas plainly and keep `do(·)` only where it is an input override.

---

## Future work (flagged honestly, none claimed as done)

- 5-seed promotion of the **cross-dataset knob** and the full B-vs-C contrast on standard datasets.
- A **standard-GNN-backbone** (TGAT/TGN) freeze-then-probe to complement the RS-GNN-backbone control.
- An **identical-head** K1 variant (same MLP scored detached vs coupled) for the cleanest toggle.
- A **per-novelty-bin** identity probe within the inductive split.
- Finalize **MOOC DyGFormer** + add **TCL/NAT**, or keep the frontier comparison downscoped.
- **Globalize** the per-pair readout: a normalized-dynamics space with a domain-adaptive driver basis + cross-pair spillover (design-staged in `PERPAIR_GLOBALIZATION_DESIGN.md`).

---

## Conclusion

We reframe a design decision usually treated as obvious - *train the representation on the task loss* - and show the **opposite** is better for inductive temporal link prediction.

- End-to-end coupling is the **wrong default**, and the damage is **irreversible** (FtP ≈ coupled ≪ decoupled, two datasets).
- The mechanism is **triangulated**: a single-flag knob (−21.1 pp), the irreversibility control, and a split-localized tax; the headline is **five-seed**.
- The **same detach** frees a faithful, intervene-able lifecycle readout at zero scored-path cost - a clean add-on, deliberately scoped because lifecycle drivers are non-commensurable across domains.

> Decoupling-by-construction: a reusable principle for temporal graph models that must generalize to unseen entities while remaining interrogable.

**The primitives are prior art. The result is not.**

---

# Thank you

<span class="small">Every number traces to a results JSON (Appendix A). Every ± is the sample std (n−1) over the stated seeds. Code, CoEdit stream, and splits ship with the paper.</span>

**Contact:** Hoangduong4316@icloud.com · Lmshih@mail.dyu.edu.tw
