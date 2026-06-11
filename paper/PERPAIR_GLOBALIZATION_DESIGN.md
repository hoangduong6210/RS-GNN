# Globalizing the Per-Pair Lifecycle Operator - Design Proposal

*Prepared by the ML team (DESIGN ONLY - no code changed, no jobs run). Grounded in the live code: `experiments/LAB/v3_3/models/sr_gnn_v3_3.py`, `.../fsm_head.py`, `.../sr_gnn_v3.py`, `experiments/LAB/v3_3/fsm_intervene.py`, and `v3_3_ARCHITECTURE_CURRENT.md` (verified 2026-06-07). File:line refs are to the on-disk files. No measured numbers are invented; everything below is a PROPOSAL with a stated validation plan handed to TESTBENCH.*

---

## 0. Problem statement and the reviewer concern

Reviewer objection: the per-pair lifecycle dynamics demonstration lives on **CoEdit alone** (a self-constructed graph), so the novelty reads as "one bespoke dataset."

PM position (which this proposal accepts and tries to strengthen, not contradict): per-pair dynamics are **intrinsically per-dataset**. The semantics of a CoEdit DEATH (a wiki page-pair stops being co-edited) are not the semantics of a financial-transaction DEATH or a MOOC-action DEATH. So demonstrating the *mechanism* on the dataset where lifecycle is richest (CoEdit; ind +13.5pp over best baseline, `v3_3_ARCHITECTURE_CURRENT.md` §8) is the **correct unit of analysis**, not a weakness.

The novelty gap is therefore not "wrong dataset" - it is that the current artifact does not separate **what is pair-specific**, **what is dataset-specific**, and **what is genuinely global**. Right now those three are entangled inside one operator that was *tuned on CoEdit* (config B is coedit-TUNED; `v3_3_ARCHITECTURE_CURRENT.md` §11). This document proposes mechanisms that make the per-pair machinery **factor cleanly into a global core + a dataset-level adapter + a per-pair input**, so the SAME mechanism transfers across datasets and the per-dataset-ness becomes an explicit, measured design choice rather than a hidden one.

### What the operator actually is today (the thing we are globalizing)

The interpretable per-pair transition operator (`fsm_head.py:103`, `:133-155`, paper §3.6):

```
T_uv(i→j) = (U@V)(i,j)        # global, h-conditioned low-rank base W
          + g(φ_uv)(i,j)       # per-pair additive bias, zero-init (fsm_head.py:120-129)
```

with the 7-dim per-pair feature vector `φ_uv` assembled PRE-update (anti-leak) at `sr_gnn_v3_3.py:1203-1211`:

```
φ_uv = [ Hawkes λ,  log mean_dt,  log var_dt,  recurrence EWMA,
         log staleness,  ever_alive∈[0,1],  z_pre ]      # (B, 7)
```

Key facts that constrain any redesign:
- `g(·)` is a 2-layer MLP, **output-zero-init** (`fsm_head.py:128-129`), so a fresh model == the global operator and per-pair structure is *learned away from zero by supervision*.
- The per-pair operator drives ONLY the interpretable / counterfactual path `s_t1_cal`; the scored-AP path `s_t1_pos` additionally applies the non-Markov `ever_alive` gate, so the AP number is decoupled from this operator by construction (`v3_3_ARCHITECTURE_CURRENT.md` §4; paper §3.6 scope note). **Consequence: every mechanism below is AP-neutral by construction** - it touches `s_t1_cal` / `g(φ)`, never the scored logit. That is a feature (we can validate interpretability-transfer without risking the headline AP) AND a limitation (transfer claims are about *lifecycle faithfulness*, not about AP).
- Exactly ONE per-pair channel is already a *per-pair-relative* (z-normalized) quantity: `z_pre = (dt − μ_pair)/σ_pair` from the pair's OWN Welford stats (`sr_gnn_v3_3.py:1199-1202`). The other six channels are **raw, dataset-scale-dependent** (Hawkes λ, mean_dt in absolute time units, staleness in absolute time units, recurrence EWMA). This is the crux: **a CoEdit-tuned `g(·)` sees CoEdit-scaled inputs; on a dataset with 100× different inter-event times those same features land in a different region of input space, so `g` mis-fires.** This is the concrete, code-level reason the mechanism does not transfer today.

The shared admissibility matrix `C` (`CAUSAL_RULE_MATRIX`, `fsm_head.py:52-59`; single-rung band) IS already global. So the *constraint topology* transfers; the *quantitative semantics* (where on the rate/staleness axes a pair counts as decaying vs dying) do not.

---

## 1. Mechanism A - Normalized dynamics space (φ_uv made dataset-invariant)  ★RECOMMENDED, do-now★

**Idea.** Make the *operator input* `φ_uv` dataset-scale-invariant, so the SAME `g(·)` weights see comparable inputs on every dataset. We already do this for one channel (`z_pre`); extend it to the whole vector. The operator then lives in a **normalized dynamics space** - every pair of every dataset maps into the same dimensionless coordinate system (rate-ratio, slope, staleness-in-σ-units, recurrence, z-gap), and the global core `W` + global `g` operate on that invariant space.

**Formulation.** Replace the 7-dim raw φ with a dimensionless φ̃ built from quantities that are ALREADY pair-relative or made so by a dataset-level scale:

| current raw channel | invariant replacement | source already in code |
|---|---|---|
| Hawkes λ (abs) | λ / λ̄_dataset (or λ·mean_dt, dimensionless intensity) | λ at `edge_st[:,6]`; λ̄ = running dataset mean |
| log mean_dt (abs time) | log(mean_dt / median_dt_dataset) | `edge_st[:,7]`; dataset median from a streaming accumulator |
| log var_dt (abs) | coefficient of variation √var/mean (dimensionless) | `edge_st[:,7],[:,8]` |
| recurrence EWMA | already in [0,1]-ish, keep | `edge_st[:,5]` |
| log staleness (abs time) | **staleness / σ_pair** (staleness in pair-σ units) OR staleness / median_dt_dataset | `dt_src`, `edge_st[:,8]` |
| ever_alive | already ∈[0,1], keep | `ever_alive_pos` |
| z_pre | already dimensionless, keep | `sr_gnn_v3_3.py:1201` |

Plus `slope_rel = (rate_fast − rate_slow)/(rate_slow+ε)` (`sr_gnn_v3_3.py:1087`) which is *already* dimensionless and is the cleanest rising/falling axis - it should arguably be IN φ̃ (it currently feeds the hier priors but not `g(φ)`).

The single new dataset-level object is a small vector of **streaming scale statistics** `Σ_D = (median_dt, λ̄, …)` maintained by a Welford/quantile accumulator over the training stream (one per dataset, ~5 scalars). Normalization is `φ̃ = normalize(φ_raw; Σ_D)`. `g(·)` and `W` are unchanged in shape; only their *inputs* become invariant.

**Why it stays per-pair yet generalizes cross-dataset.** Per-pair-ness is untouched - every channel is still a function of THIS pair's own history. But because the inputs are now dimensionless, the CoEdit-learned `g` no longer mis-fires on a 100×-slower dataset: a pair that is "decaying" (staleness ≈ 3σ_pair, slope_rel < 0) maps to the same region of φ̃-space regardless of absolute clock. The mechanism - "where on the normalized rate/staleness axes does DECAY hand off to DEATH" - becomes a single global function evaluated per-pair. This is exactly the claim that answers the reviewer: *one mechanism, many datasets, per-pair semantics preserved.*

**Cost.** Tiny. ~5 scalar accumulators per dataset (streaming, O(1)/event, no Python loop - vectorizable like the existing Welford in `EdgeStateStoreV3`). No new trainable params; `g`/`W` shapes unchanged. State_dict gains at most the buffer of scale scalars. Risk of training instability is low because normalization is the standard z/scale trick already proven for `z_pre`.

**Validation (measurable, no hand-wave).**
1. **Train-on-CoEdit, score-φ̃-on-X (zero-shot operator transfer).** Train config B with φ̃ on CoEdit. Freeze `g`. On wiki and mooc, run the FSM head with the frozen `g` but each dataset's own `Σ_D`. Measure lifecycle faithfulness (ρ between `s_t1_cal` argmax/prob and the held-out timing-derived state) and the 5-state distribution spread. **Success = faithfulness on X with frozen-CoEdit-`g` ≈ faithfulness of a `g` trained on X** (gap small), vs. the RAW-φ baseline where the gap should be large. This is the headline transfer experiment.
2. **AP-neutrality regression (mandatory).** CPU probe: `max|Δ pos/neg score| (φ̃ vs φ_raw) == 0.000` - confirm the swap does not move the scored logit (it should not; φ feeds only `g`→`s_t1_cal`). Mirror the existing `_probe_*` style.
3. **Ablation: which channels carry the transfer.** Leave-one-channel-out on φ̃ to show the dimensionless staleness/slope channels are what enable transfer (interpretable contribution table).

**Status: DO-NOW.** Pure input re-parameterization + 5 streaming scalars; isolated to the no_grad φ-assembly block (`sr_gnn_v3_3.py:1183-1216`) and the φ̃ normalizer. Lowest risk, directly attacks the code-level cause of non-transfer, and the transfer experiment is clean and measurable.

---

## 2. Mechanism B - Dataset-conditioned lifecycle semantics (global core, dataset adapter)

**Idea.** Keep per-pair *normalization* (Welford μ/σ per pair - already done) but make the *semantic thresholds* (the decay/death hand-off, the bias matrix shape) an explicit **function of dataset-level statistics**, i.e. one global mechanism instantiated per dataset by a tiny adapter. This is the "FiLM / hypernetwork conditioned on Σ_D" view of Mechanism A, taken one step further: instead of only normalizing inputs, also condition the *operator parameters* on the dataset.

**Formulation.** Two concrete sub-variants (pick the cheaper that works):
- **B1 (input-side, ⊂ A):** condition only via normalized φ̃ - this is literally Mechanism A. Listed here to show B1 is the minimal B.
- **B2 (parameter-side):** a dataset-embedding `e_D = MLP(Σ_D)` (a few scalars → small vector) modulates `g`'s output via FiLM: `g'(φ̃) = γ(e_D) ⊙ g(φ̃) + β(e_D)`. The single global `g` is shared; `e_D` specializes the per-pair gate's gain/offset to the dataset. The dataset-conditioned hier-prior thresholds (`decol_slope_scale`, the p_alive/p_birth gate biases, `sr_gnn_v3_3.py:1333-1345`) can likewise be made `θ(e_D)` instead of CoEdit-tuned scalars.

**Why per-pair yet cross-dataset.** Per-pair input φ̃ preserved; the *semantics* (γ, β, thresholds) are a deterministic function of dataset statistics rather than CoEdit-frozen constants. New dataset ⇒ compute Σ_D ⇒ get its semantics for free, no per-pair refit. Directly turns "config B is coedit-TUNED" (§11 caveat) into "config B's tuned scalars are the values a learned θ(Σ_D) emits for CoEdit's Σ_D."

**Cost.** Small: `MLP(Σ_D)` + FiLM params (~hundreds). More trainable params than A, and - critically - **needs ≥3 datasets in the TRAINING mix** for θ(Σ_D) to be identifiable (with one training dataset, `e_D` is a constant and B2 collapses to A). So B2 requires a multi-dataset training protocol (DATA/TESTBENCH coordination).

**Validation.**
1. **Leave-one-dataset-out.** Train θ(Σ_D) on {coedit, wiki} (or a synthetic family with controlled Σ_D), zero-shot the adapter to mooc via its Σ_D. Measure lifecycle faithfulness vs. (i) RAW baseline, (ii) Mechanism A. **Success = B2 > A only if dataset semantics genuinely differ beyond scale** - if A already closes the gap, B2 is unnecessary (report that honestly).
2. **Recover the tuned scalars.** Check that θ(Σ_D=CoEdit) emits values near the hand-tuned config-B scalars - evidence the adapter learned the right map.

**Status: PARTLY do-now / partly future.** B1==A is do-now. B2 needs a multi-dataset training harness that does not exist yet → **future-work** for this paper round unless DATA/TESTBENCH can stand up a ≥3-dataset joint-training protocol. Recommend B2 as the "obvious next step" framing, validated minimally with a leave-one-out if time permits, else stated as future work with the experiment pre-registered.

---

## 3. Mechanism C - Meta-learned / amortized per-pair operator (few/zero-shot)

**Idea.** Make `g` an *amortized inference network* that, from a pair's short history window, emits that pair's lifecycle-operator parameters - trained across many pairs (and ideally many datasets) so it generalizes to unseen pairs AND unseen datasets few-shot.

**Formulation.** Instead of `g(φ_uv) → bias`, an encoder `q(τ_uv) → ψ_uv` reads the pair's recent event sub-sequence τ_uv (the last k inter-event gaps / states) and produces operator parameters ψ_uv (e.g. the per-pair bias matrix, or low-rank factors). Trained with an outer objective over pairs/datasets (Reptile/MAML-lite, or simple amortized/hypernetwork training - the latter is far cheaper and likely sufficient).

**Why per-pair yet cross-dataset.** Maximally per-pair (each pair's operator is *inferred* from its own trajectory) and, if trained across datasets, the encoder learns a dataset-agnostic map from "trajectory shape" → "operator," giving few-shot transfer.

**Cost / risk.** Highest. (i) Needs a windowed per-pair history buffer - heavier than the current O(1) streaming state. (ii) Meta-training is finicky and multi-dataset. (iii) **CRITICAL RISK - do NOT frame as fast adaptation / regime-switch.** The regime-switch / fast-per-pair-adaptation story was **FALSIFIED** (RS-GNN loses to CAWN on every post-change-point slice; `v3_3_ARCHITECTURE_CURRENT.md` §11, memory "regime-switch FALSIFIED"). Any claim that meta-learning lets the operator *adapt quickly within a stream* re-opens a hole reviewers (and our own falsification) already closed. If C is pursued, it must be scoped to *cross-dataset zero/few-shot transfer of a STATIC operator*, never to within-stream regime adaptation.

**Validation.** Few-shot: fine-tune only the amortization encoder on k% of a held-out dataset's pairs, measure faithfulness vs. k. Zero-shot at k=0. Must beat Mechanism A's frozen-transfer to justify the extra machinery.

**Status: FUTURE-WORK.** Out of scope for one paper round: needs windowed buffers, meta-training infra, ≥3 datasets, and carries the falsified-adaptation framing risk. Mention as future work; do not build now.

---

## 4. Critique - which is risky, which is feasible

- **Mechanism A (normalized dynamics space):** lowest risk, attacks the *actual* code-level cause of non-transfer (6 of 7 φ channels are raw-scale), reuses the already-proven z-norm trick, AP-neutral by construction, single clean transfer experiment. **Feasible now.** This is the strongest one-round contribution.
- **Mechanism B2 (dataset-conditioned adapter):** conceptually the cleanest "one mechanism, many datasets" story and the most *novel-sounding*, but needs ≥3-dataset joint training to be identifiable and may be **redundant if A already closes the transfer gap** (likely, if the gap is mostly scale). Good as the framing/future-work hook; validate minimally via leave-one-out only if the multi-dataset harness materializes.
- **Mechanism C (meta-learned):** most ambitious, highest cost, and carries the **falsified regime-switch risk** if mis-scoped. Future work, narrowly framed as static cross-dataset transfer.
- **Cross-cutting honesty constraint:** ALL of these are **AP-neutral** (they touch `s_t1_cal`/`g(φ)`, not the scored `s_t1_pos` logit; §4 of the architecture doc). So none can be sold as "improves prediction across datasets." The transferable quantity is **lifecycle faithfulness / interpretability**, and the paper claim must be exactly that: *the interpretable per-pair lifecycle mechanism, expressed in a normalized dynamics space, transfers across datasets - demonstrated on the dataset family, anchored on CoEdit where lifecycle is richest.* This is a defensible, falsifiable novelty claim that does not over-reach.

---

## 5. Recommendation + minimal experiment

**Recommend Mechanism A (Normalized Dynamics Space) as the do-now contribution**, with **Mechanism B2 pre-registered as the natural extension** (future work / minimal leave-one-out if the multi-dataset harness lands), and **Mechanism C as future work, narrowly scoped to static transfer (never regime adaptation).**

Rationale: A is the smallest change that converts "demoed on one dataset" into a *measured cross-dataset transfer claim*, it targets the real code-level cause (raw-scale φ), it is AP-neutral and isolated to the φ-assembly no_grad block, and its core experiment is clean and falsifiable.

**Minimal experiment to prove it (hand to TESTBENCH - ML supplies the variant, does not run it):**

1. **Build φ̃** (dimensionless 7-8 dim: keep z_pre/recurrence/ever_alive; convert Hawkes→λ·mean_dt, mean_dt→mean_dt/median_dt_D, staleness→staleness/σ_pair, var→CoV; add slope_rel) behind a flag `--phi_normalized` (default OFF ⇒ byte-identical to current B). Add a streaming `Σ_D` accumulator (vectorized; reuse the causal-batch replay pattern, no Python per-event loop - the EverAliveStore lesson).
2. **AP-neutral CPU probe** (mandatory, ML-owned, no GPU): `max|Δ pos/neg score|(φ̃ vs φ_raw) == 0` on a small batch - confirms the scored logit is untouched. Style after `_probe_hier_causal_policy.py`.
3. **Frozen-`g` transfer run (the headline, TESTBENCH/GPU):**
   - Train config B + `--phi_normalized` on CoEdit (3 seeds 42/1/7, existing protocol).
   - Freeze `g`; evaluate FSM faithfulness on wiki and mooc using each dataset's own `Σ_D`.
   - **Compare three arms per target dataset:** (a) RAW-φ frozen-CoEdit-`g` [expected: poor transfer], (b) φ̃ frozen-CoEdit-`g` [the claim], (c) φ̃ `g` trained on the target [the ceiling]. Metric = lifecycle faithfulness ρ (`s_t1_cal` prob vs timing-derived state) + 5-state spread (no-collapse check) + argmax-DECAY/DEATH fire-rate. **Claim holds iff (b) ≈ (c) ≫ (a).**
   - AP regression (must stay = config-B band: coedit ind 0.9885±0.0028, wiki/mooc per §8) - confirms AP-neutrality on real runs.
4. **Channel ablation** (leave-one-out on φ̃) to attribute the transfer to the dimensionless staleness/slope channels - gives the paper an interpretable "why it transfers" table.

**Do-now vs future, explicitly:**
- **Do-now (this round):** Mechanism A + experiments 1-4 above. Variant + CPU probe = ML; GPU runs = TESTBENCH.
- **Future-work (pre-register, build only if harness lands):** Mechanism B2 dataset-conditioned adapter (needs ≥3-dataset joint training) and Mechanism C amortized/meta operator (needs windowed buffers + meta-training; scope to static transfer ONLY, never regime-switch adaptation).

---

*No code was modified and no jobs were run in producing this proposal. The φ̃ variant, the Σ_D accumulator, and the AP-neutral CPU probe are ML to implement on request; the frozen-`g` transfer runs are a TESTBENCH/GPU hand-off. All transfer claims are about lifecycle faithfulness (the per-pair operator feeds only `s_t1_cal`), explicitly NOT about AP, which is decoupled by the detach design.*
