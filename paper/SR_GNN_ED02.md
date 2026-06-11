# RS-GNN: Regularization-by-Decoupling for Inductive Temporal Link Prediction, with a Faithful Intervene-able Lifecycle Readout

\begin{center}
{\large \textbf{Duong Viet Hoang}\textsuperscript{1,\,*} \qquad \textbf{Shih Lun-Min}\textsuperscript{1}}

\textsuperscript{1}\,Department of Computer Science, Da-Yeh University, Taiwan

\texttt{Hoangduong4316@icloud.com} \qquad \texttt{Lmshih@mail.dyu.edu.tw}

\textsuperscript{*}\,Corresponding author
\end{center}

*Reproducibility note: every reported number traces to an artifact listed in Appendix A; unverified statements are explicitly scoped. Every `±` is the sample standard deviation (n−1) over the reported seeds.*

---

## Abstract

State-of-the-art temporal link predictors train their representation end-to-end against the link loss. We show this is the wrong default for *inductive* generalization. A single-variable knob ablation isolates the cause: turning the end-to-end head on, every other flag fixed, drops inductive average precision (AP) on CoEdit - a co-edit graph we introduce - by −21.1 points (0.9899 → 0.7788), essentially the entire decoupled-vs-coupled gap. The same flag keeps decoupling ahead on two standard graphs we did not build (Wikipedia +8.6, MOOC +0.9), so the mechanism is not a CoEdit artifact. At five seeds, detached RS-GNN reaches 0.9876 inductive AP, +22.8 over the coupled configuration; against baselines it leads by +14.1 over a protocol-matched TGAT *within our shared harness* - our baselines are simplified re-implementations, so this is a within-harness margin, not a claim against published state of the art (§6.1). The primitives are prior art (stop-gradient, frozen probing, temporal concept bottlenecks); the result is not. A decisive control shows classic freeze-then-probe reaches only 0.768 (CoEdit) and 0.897 (Wikipedia), beside the coupled arm, whereas decoupling-by-construction reaches 0.988 / 0.996: once the link loss shapes the backbone the damage is irreversible on both datasets. To our knowledge this irreversibility is new for temporal link prediction; the win is preventing that contamination by construction, localized to the inductive split. Under both historical and inductive hard negatives [Poursafaei et al., 2022] the ordering does not invert but widens to +40-43 points on Wikipedia and CoEdit, as the coupled arm collapses toward chance. The same detach frees a symbolic five-state lifecycle readout from the scored path at zero accuracy cost, exactly score-invariant and falsifiable under do(state) overrides.

**Keywords:** Inductive temporal link prediction · Regularization by decoupling · Stop-gradient feature decoupling · Temporal graph neural networks · Interpretable and intervene-able lifecycle readout · Temporal concept bottleneck · Information bottleneck · Reproducible evaluation

---

## 1. Introduction

Temporal interaction graphs are sequences of timestamped edges. Formally, a stream is an ordered event set $\mathcal{E} = \{(u_i, v_i, t_i, x_i)\}_{i=1}^{N}$, $t_1 \le \dots \le t_N$, with optional edge feature $x_i$. The canonical task is *future link prediction*: given history $\mathcal{H}_t$, score whether a candidate edge $(u, v)$ occurs at $t$; the model outputs $\hat{p} = \sigma(f_\theta(u, v, t, \mathcal{H}_t))$ trained by BCE against observed positives and sampled negatives.

Two protocols reward different inductive biases. **Transductive** test edges reuse training nodes, so a model can lean on a memorized per-node embedding. **Inductive** test edges involve a node unseen in training, so any per-node embedding is uninitialized at test time. Inductive is the deployment-relevant and harder regime: the model must score from *transferable* dynamics rather than node identity. A model that wins transductively but collapses inductively has memorized the population, not learned the process.

State-of-the-art methods are *memory* models (JODIE, TGN, DyRep) or *encoder* models (TGAT, CAWN, GraphMixer); both train the representation *end-to-end* against the link objective (§2). This has two costs. (1) **Identity overfitting:** with write access to the backbone, gradient descent encodes features that merely *identify* training nodes - rewarded transductively, dead weight inductively. (2) **Opacity:** a node-memory vector cannot be asked "is this pair reinforcing or decaying?" nor intervened on, though temporal processes have a natural born→reinforced→decay→die lifecycle.

RS-GNN keeps a rich continuous backbone - an event encoder, a multi-signal per-pair edge-state operator grounded in Hawkes intensity [Hawkes, 1971], and a coupled-GRU node-memory module - but **decouples it from the link loss by an explicit stop-gradient** (a `.detach()` on the backbone representation $h_{uv}$). The backbone is shaped only by a variational parsimony objective (a KL regularizer in the spirit of the information bottleneck [Tishby et al., 2000] and the VAE [Kingma & Welling, 2014]) plus deterministic update laws. The link head reads this *frozen* representation through a symbolic five-state lifecycle decoder, a stop-gradient device of the kind that stabilizes siamese learning against shortcut collapse [Chen & He, 2021]. §3.7 gives the gradient routing, §6.2 the ablation.

Our contributions are:

1. **Regularization-by-decoupling for inductive temporal link prediction (core result).** The building blocks - stop-gradient [Chen & He, 2021], frozen probing [Alain & Bengio, 2017; Kumar et al., 2022], the information bottleneck [Tishby et al., 2000] - are prior art. We establish, by a single-variable ablation and an irreversibility control, that **end-to-end coupling is the wrong default for inductive temporal link prediction, and that the damage it causes is irreversible** - freeze-then-probe cannot recover it. To our knowledge this irreversibility has not previously been shown in this setting. (a) A single-variable three-seed knob ablation isolates the cause: enabling the end-to-end main-prediction head (`enable_main_predictor=True`), every other flag fixed, costs **−21.1 points** inductive AP on CoEdit (0.9899 → 0.7788), essentially the entire decoupled-vs-coupled gap, with two co-varying gates inert (§6.2). (b) The same single flag keeps decoupling ahead on two *standard* graphs we did not construct - Wikipedia +8.6 pp, MOOC +0.9 pp - so the mechanism is not a CoEdit artifact, its magnitude tracking how much node identity the coupled head can exploit (§6.5). (c) A decisive control measures the delta against freeze-then-probe: freezing the backbone *after* the link loss reaches only 0.768 (CoEdit) and 0.897 (Wikipedia) - beside the coupled end-to-end arm, far below the 0.988 / 0.996 of decoupling-by-construction - so the damage is **irreversible** on both datasets, and the contribution is preventing that contamination by construction. To our knowledge this irreversibility has not been shown for temporal link prediction (§6.2). With no link-prediction gradient the head keeps a generic per-pair representation that transfers to unseen nodes, and the identity-shortcut tax it removes lands almost entirely on the inductive split (§7).
2. **A multi-signal per-pair edge-state operator** (Hawkes intensity, Welford gap statistics, recurrence and rate EWMAs) with a *read-before-write* batched estimator (`causal_batch`) that fixes a silent intra-batch staleness bug; the fix is AP-positive (+5.7 pp inductive in config B, §3.2).
3. **A hierarchical five-state lifecycle decoder** (BIRTH / REINFORCE / DECAY / DEATH / IDLE) that makes the intermediate DECAY state argmax-reachable, which the flat readout it replaces empirically cannot (§3.3).
4. **A faithful, intervene-able lifecycle readout that rides free on the same detach (a clean add-on).** The same stop-gradient places the symbolic decoder off the scored path, so `do(state)` interventions - typed forward re-evaluations under an input override - never tax AP. The readout is *falsifiable*: a learned "REINFORCE ⟺ rising edit cadence" rule is confirmed at population, do(slope), and single-pair granularity, and the trained existence weights preserve the DEATH<IDLE<DECAY<active ordering, so the do(state) directionality survives training (§4). It is a temporal concept bottleneck [Koh et al., 2020] with typed forward re-evaluation - a faithful, intervene-able readout that contribution 1's detach makes free, complementing the core result rather than competing with it.

We measure RS-GNN against a field of competitors on CoEdit - parametric temporal-GNN families, two EdgeBank memorization floors [Poursafaei et al., 2022], and the transformer DyGFormer [Yu et al., 2023] - through one leak-audited harness at three seeds (§6.1). We separately study a walked-chain causal-coherence confidence signal and report it honestly as internal self-consistency, not an external error predictor (§5).

---

## 2. Related work

**Memory-based temporal GNNs.** JODIE [Kumar et al., 2019], TGN [Rossi et al., 2020] and DyRep [Trivedi et al., 2019] all maintain a per-node recurrent state trained end-to-end against the link objective (JODIE co-evolves user/item RNNs with a trajectory projection; TGN adds graph-attention over a node-memory; DyRep is a temporally-attentive point process). RS-GNN also carries memory - a coupled-GRU node-memory module - but shapes it by a parsimony objective rather than the link loss, which is exactly the difference our ablation isolates.

**Encoder, walk, and transformer methods.** TGAT [Xu et al., 2020] self-attends over the temporal neighborhood with a Bochner time encoding, giving native inductive support without a stored node state. CAWN [Wang et al., 2021] encodes anonymized causal walks; GraphMixer [Cong et al., 2023] replaces attention with a token-mixing MLP. The recent inductive frontier adds TCL [Wang et al., 2021b] (contrastive neighborhood representations), NAT [Luo & Li, 2022] (dictionary-based neighbor features), and **DyGFormer** [Yu et al., 2023] (a co-occurrence-encoding patching transformer, current temporal-graph-transformer SOTA). We run DyGFormer through our harness on CoEdit and Wikipedia (§6.1); TCL and NAT are not yet run on any dataset and DyGFormer is not yet finalized on MOOC, so our frontier comparison is **partial**, and we downscope the positioning accordingly (§8.2) rather than claim to beat the full named frontier. Inductive behavior is dataset-dependent: TGAT is near-perfect inductively on Wikipedia yet weak transductively, which we attribute to node features that happen to identify unseen Wikipedia nodes.

**Benchmarks, fair evaluation, and memorization floors.** Poursafaei et al. [2022] show much of the apparent skill of parametric temporal models is matched by **EdgeBank**, a non-parametric heuristic predicting an edge if the pair was ever (or recently) seen, and argue any headline must beat this floor under harder negatives; the Temporal Graph Benchmark [Huang et al., 2023] shows rankings reorder under a fair protocol. We adopt both: every model - including two EdgeBank variants - runs through one shared harness with a single negative pool and a leak-audited evaluator, both protocols reported, the floor *measured* (Table 1).

**Point processes and lifecycle modeling.** Hawkes processes [Hawkes, 1971] model self-exciting event intensity and drive neural temporal models [Mei & Eisner, 2017]. We use a per-pair Hawkes intensity $\lambda$ with Welford online moments [Welford, 1962] of the inter-event gap as the substrate the symbolic decoder reads; the BIRTH→REINFORCE→DECAY→DEATH abstraction is a coarse, interpretable summary of where a pair sits in its trajectory.

**Concept bottlenecks and neuro-symbolic readouts.** The closest prior-art to our readout is the **concept bottleneck model (CBM)** [Koh et al., 2020], where a network predicts human-named concepts and a practitioner intervenes by editing one. Our typed lifecycle readout is a *temporal* CBM - five per-pair lifecycle concepts over a point-process substrate, with $\mathrm{do}(\text{state})$ a concept override - differing in that the bottleneck sits *behind* a stop-gradient rather than *on* the prediction path, so it never trades against accuracy. The structure is hand-specified, not discovered (analytic-prior gates with a small learnable residual, fixed admissibility $C$, the `do(state)` directionality a sorted softplus-weight read-off; §4), so we frame it as a faithful, intervene-able *typed readout* riding free on the detach - not a learned SCM or a Pearl-style discovered causal model [Pearl, 2009].

**What "faithful" means here.** The standard notion is that an explanation reflects the *predictor's* computation [Jacovi & Goldberg, 2020; Rudin, 2019]. Our scored $s^{\text{pos}}_{t+1}$ and interpretable $s^{\text{cal}}_{t+1}$ differ (up to 0.999 on a pair). The readout is faithful to its *own supervised objective* - the plotted quantity is exactly what the de-collapse CE optimizes, unlike post-hoc surrogates such as GNNExplainer [Ying et al., 2019] - and the scored path is score-invariant under it (§3.4). It is **not** faithful to the scored distribution, and we do not claim it is.

**Decoupling, frozen representations, and stop-gradient as regularizers.** RS-GNN sits at the intersection of stop-gradient against collapse (SimSiam [Chen & He, 2021]), frozen-encoder linear probing [Alain & Bengio, 2017] where fine-tuning can distort transferable features so a probe transfers better [Kumar et al., 2022], and gradient-stopping as capacity control [Caron et al., 2021], readable through the information bottleneck [Tishby et al., 2000; Alemi et al., 2017]. These primitives are prior art, and "freeze a pretrained encoder, probe with a task head, gain transfer" is a known phenomenon. Our delta over them is sharp on two fronts. First, **relocation**: we move stop-gradient-as-regularizer into the *inductive split of temporal link prediction*, where end-to-end coupling is the universal default - and we measure, for the first time in this setting, that coupling is the wrong default. Second, **isolation**: the backbone is trained only by a parsimony objective from scratch (the regime is task-gradient-vs-none, not pretrain-then-probe), and we localize the tax to an identity shortcut that lands almost entirely on the inductive split (§7). This sharpens the nearest neighbor - Kumar et al.'s "fine-tuning distorts transferable features" - into a stronger, measured claim: the distortion is *irreversible*. Freezing the backbone *after* the link loss does not recover it, on two datasets (§6.2). The contribution is never exposing the backbone to the link gradient in the first place.

**Positioning.** RS-GNN is not the first memory model, nor the first to use Hawkes intensities, nor the first neuro-symbolic temporal model. Its claim is the *combination and its measurement*: **decoupling** the representation from the link loss is what makes a per-pair dynamical representation generalize inductively, with the symbolic readout a faithful, intervene-able add-on that rides free on the same detach.

---

## 3. Method

The two-stream architecture is summarized in Appendix Figure A1; the gradient routing, decode tree, admissibility band, and confidence overlay are Appendix Figures A2-A5.

### 3.1 Two streams, one detach

RS-GNN is a two-stream model. Let an event be a timestamped edge $(u, v, t)$ with optional feature vector $x$.

**Stream A - continuous backbone.** An event encoder (a residual continuous-signal network, CSN) maps event features and source staleness $\Delta t = t - t_{\text{last}}(u)$ to a per-event representation $e_{uv}$. A multi-signal **edge-state operator** (ECTGv3) maintains, per ordered pair $(u,v)$, a small bank of running statistics:

- a **Hawkes self-exciting intensity** $\lambda_{uv}$ updated at each event by $\lambda \leftarrow 1 + (\lambda - 1)\,e^{-\beta \Delta t}$, so each interaction transiently raises the pair's rate and the rate decays between events;
- **Welford online mean and variance** of the inter-event gap, updated by the numerically-stable recurrence $\mu_k = \mu_{k-1} + (g_k - \mu_{k-1})/k$, $M_k = M_{k-1} + (g_k - \mu_{k-1})(g_k - \mu_k)$, with $\sigma_k^2 = M_k/k$ [Welford, 1962];
- a **recurrence EWMA** counting repeated co-occurrence, and **fast/slow rate EWMAs** $r^f, r^s$ whose ratio $r^f/r^s$ is a rising-vs-falling cadence signal;
- a **leaky rate-peak** that tracks the maximum recent rate with slow decay.

A coupled-GRU module (DRGC) updates per-node memory $m_u, m_v$ from $e_{uv}$ and emits a parsimony KL term $\mathrm{KL}(q(z \mid m) \,\|\, p(z))$. The backbone's *only* training signal is this KL (weighted by $\lambda_{\text{kl}}$) plus the deterministic update laws above; **no link-prediction gradient reaches it.**

**Stream B - symbolic lifecycle readout.** From the (detached) edge representation `edge_h`, a `StateObserver` produces a soft *current* state $s_t \in \Delta^4$ over five states $\{\text{IDLE}, \text{BIRTH}, \text{REINFORCE}, \text{DECAY}, \text{DEATH}\}$; a `TransitionPredictor` produces next-state logits; a lifecycle mask - a causal-rule transition matrix $C \in \{0,1\}^{5\times 5}$ encoding admissible transitions (e.g. DEATH may not precede BIRTH) - together with an `ever_alive` gate shape the next-state distribution $s_{t+1}^{\text{pos}}$; an `ExistenceDecoder` maps $s_{t+1}^{\text{pos}}$ to the edge-existence logit that is **scored**.

**The detach.** The input to every Stream-B module is `edge_h.detach()`. The scored logit is the existence-decoder logit, and every path from it to the backbone crosses a stop-gradient. The committed test of §3.4 confirms `pred_loss.backward()` produces exactly zero gradient on every backbone tensor. A non-detached predictor head exists in the code but is used only for the end-to-end *ablation* (config C, §6.2).

### 3.2 The per-pair operator and the read-before-write fix (`causal_batch`)

The operator must read each pair's *pre-event* statistics: scoring event $i$ may use only state from events $j < i$, or evaluation leaks the label. The original batched store snapshotted state once per minibatch, so repeated same-pair events *within* a batch read the same stale row and only the last write persisted. At batch 500 on CoEdit the Welford count capped near 6 even for pairs editing 200+ times, silently disabling the DECAY-vs-REINFORCE distinction (read off a rate ratio that never accumulated).

The `causal_batch` fix replays the deterministic channels event-by-event in stream order, so the $k$-th in-batch occurrence reads the post-state of the $(k{-}1)$-th while scoring stays strictly pre-update (no re-leak; matches a batch=1 reference to max $|\Delta|=0.000$). The fix is **AP-positive**: in config B it lifts inductive AP 0.9312 → 0.9885 (**+5.7 pp**, three seeds) and transductive +0.65 pp, because the previously-collapsed statistics are exactly the ones the decoder reads.

### 3.3 Hierarchical lifecycle decode

We replace the flat five-class readout with a hierarchical one because the *specific* flat head we use cannot surface DECAY as argmax. That head interpolates a single ordered cadence statistic across BIRTH→REINFORCE→DECAY→DEATH, so the middle class DECAY is pinned between its neighbors and essentially never wins (0.04% of pairs; §6.4) - a property of *that interpolating head* (Appendix B), not of softmax. No threshold tuning recovers it, so we change the output *structure*; on the final config decoded DECAY then tracks the per-pair rising-cadence signal (Spearman $\rho \approx -0.59$, $p < 10^{-300}$, recurring $n=9157$).

The hierarchical decoder factors the next-state distribution as a decision tree over per-pair pre-update gates $p_{\text{birth}}, p_{\text{alive}}, p_{\text{rising}} \in [0,1]$:

$$
\begin{aligned}
P(\text{BIRTH}) &= p_{\text{birth}} \\
P(\text{REINFORCE}) &= (1 - p_{\text{birth}})\, p_{\text{alive}}\, p_{\text{rising}} \\
P(\text{DECAY}) &= (1 - p_{\text{birth}})\, p_{\text{alive}}\, (1 - p_{\text{rising}}) \\
P(\text{DEATH}) &= (1 - p_{\text{birth}})\, (1 - p_{\text{alive}})
\end{aligned}
$$

These four terms sum to one by construction, with $P(\text{IDLE})$ carrying the residual pre-birth mass. For DECAY to be the argmax of all four terms it must beat REINFORCE ($r<\tfrac12$), DEATH ($a>\tfrac{1}{2-r}$) and BIRTH ($(1-b)\,a\,(1-r)>b$); these conditions carve out a **non-zero-measure region** of the gate cube $(b,a,r)\in[0,1]^3$, which is the structural property the interpolating flat head denies (§6.4 confirms it: 47.8% DECAY-argmax hierarchical vs. 0.04% flat). The decode tree is drawn in Appendix Figure A3.

Each gate is $\sigma(\text{analytic prior} + \text{small learnable residual})$, zero-initialized (a fresh gate equals its prior); a *de-collapse* cross-entropy trains the residuals against a soft target from the running statistics. The refinement `decol_hier_v2` re-anchors the alive/rising priors on the uncorrupted recurrence-count signal and gates the corruptible mean/staleness terms behind a has-history mask, so feature corruption cannot drive a recurring-active pair to DEATH.

![](../figs/fig1_lifecycle_pair.png)

*Figure 1. Decoded per-pair lifecycle, real CoEdit pair 3178→7437 (42 events, 18.69 min; config B, calibrated next-state $s^{\text{cal}}_{t+1}$). x-axis: event index; curves: edit-rate slope and the three lifecycle gates. Takeaway: the gates track the pair's own cadence (1 BIRTH, 20 REINFORCE, 21 DECAY, 0 DEATH; $p_{\text{alive}}\in[0.579,0.801]$ throughout), with REINFORCE↔DECAY flips at slope sign-changes - the readout follows the data, not a fixed prior.*

### 3.4 Two next-state heads: AP vs interpretation

There are **two** next-state distributions over the five lifecycle states. The *scored* distribution $s^{\text{pos}}_{t+1}$ (`s_t1_pos`; the masked, gated transition softmax) is the **only** input to the existence decoder and therefore the only thing that affects AP. The *interpretable* distribution $s^{\text{cal}}_{t+1}$ (`s_t1_cal`; the hierarchical tree above, optionally causal-policed) feeds the de-collapse CE, the faithfulness measurement, and the intervention battery - but **never** the existence decoder.

The eval-time invariance is a **computation-graph property certified on this implementation**, not a theorem: its content is autograd reachability (a non-ancestor of an output cannot change it), correct but near-tautological. The one non-trivial obligation is premise (i) - that $s^{\text{cal}}_{t+1}$ is not an ancestor of the scored logit - a fact about this code's forward graph that the assertion test below certifies on a single model instance.

> **Property 1 (eval-time score invariance - an autograd-certified property of the trained instance, not a theorem).** Fix a trained model and an eval batch. *Premises (structural, about the forward graph):* (i) no $s^{\text{cal}}_{t+1}$ tensor is an autograd ancestor of the existence logit; (ii) the existence decoder reads *only* $s^{\text{pos}}_{t+1}$. *Then:* for any readout mode $r\in\{\text{flat},\text{hier}\}$ and any causal-policy setting, the positive and negative existence scores - and hence AP - are unchanged ($\max|\Delta\,\text{score}|=0$). This is immediate from reachability: the toggle alters only $s^{\text{cal}}$ sub-graph tensors, non-ancestors of the logit by (i). We do not prove (i)/(ii) over the model class; we certify them on this implementation by the graph-ancestry asserts below.

This is **eval-time** (frozen model, readout toggled). The **training-time** statement is weaker: a model *trained* with vs. without the readout op (e.g. `hier_causal_policy`) has AP unchanged only *within seed noise* ($\max|\Delta_{\text{ind}}|=1.5\mathrm{e}{-3}\ll\pm3.5\mathrm{e}{-3}$ seed std), not identically, because the added op advances the optimizer RNG stream; the symbolic output still never enters the existence loss, so this is RNG jitter, not leakage (§6.2).

**The premises are certified by graph-ancestry asserts.** A committed test runs three asserts on a config-B instance (full detail in Appendix A): **[G]** every parameter with nonzero gradient under `pred_loss.backward()` lies in the head set ($\max|\text{backbone grad}|=\mathbf{0.000\mathrm{e}{+}0}$) - a name-independent *ancestry* test, so a backbone tensor named like a head cannot be silently skipped; **[A]** back-propagating *from the scored logit* deposits $\mathbf{0.000\mathrm{e}{+}0}$ on every cal-path head, certifying premise (i) (that $s^{\text{cal}}_{t+1}$ is not an ancestor of the scored logit) at the graph level; **[S]** a flat↔hier value-equality consistency check ($\mathbf{0.000\mathrm{e}{+}0}$ on 225 positives) which [A], not [S], makes load-bearing. So we report a faithful symbolic readout *and* competitive AP without the usual interpretability-vs-accuracy trade [Rudin, 2019], resting the guarantee on reachability asserts, not a name partition of tensors.

### 3.5 Causal policy on the interpretable state

The interpretable state $s^{\text{cal}}_{t+1}$ is regularized by two soft, differentiable, renormalized steps:

1. a **soft expected-admissibility mask** from the admissibility matrix $C$. Config B uses the **strict band-diagonal $C_{\text{BAND-5}}$** ($|i-j|\le 1$ along the IDLE-BIRTH-REINFORCE-DECAY-DEATH axis). The scored path applies a hard binarized mask from the *full* current-state expectation $\mathbb{E}_{s_t}[C]$ (no argmax - avoiding near-uniform brittleness); the interpretable branch uses the soft expectation with a small floor (forbidden transitions suppressed ~20×).
2. an **`ever_alive` gate** on the interpretable branch: the DEATH leaf is scaled by the pair's ever-alive accumulator and freed mass routed to pre-birth IDLE.

**`ever_alive` is non-Markov and used on *both* paths.** Both paths apply the `ever_alive` gate after the mask, suppressing DEATH when the pair was never alive, so the scored $s^{\text{pos}}_{t+1}$ carries a **non-Markov dependence** $C$ cannot express (bounded to the interpretable path in §3.6, scoped in §8.1). This is orthogonal to Property 1, whose toggles on $s^{\text{cal}}$ never touch `ever_alive` on the scored path, so AP-neutrality is unaffected. The policy is AP-neutral by §3.4 (training-time, within seed noise; §6.2). The admissibility band $C_{\text{BAND-5}}$ is drawn in Appendix Figure A4.

### 3.6 The per-pair transition operator

The interpretable next-state distribution comes from a per-pair operator $T_{uv} = C \odot (W + g(\phi_{uv}))$ that adapts the shared admissibility matrix $C$ to the pair's statistics. Here $W = UV^\top$ is a low-rank learnable base (population tendencies, factored into matrices $U,V$) and $g(\phi_{uv})$ is a small per-pair gate over the feature summary $\phi_{uv}$ - the vector of detached pair statistics (rate ratio, recurrence count, slope, staleness) - that specializes without giving each pair a free full matrix. Because $g$ is a deterministic function of the detached statistics, $T_{uv}$ is reconstructable offline from a stored feature row, which makes the §4 interventions exact.

**Scope of the typed-transition formalism.** $T_{uv}$ over fixed $C$ is a Markov transition operator, and the interventions of §4 act on it. This Markov operator describes the **interpretable** path ($s^{\text{cal}}_{t+1}$) exactly. The **scored** path is *not* the same object: after the soft mask it additionally applies the non-Markov `ever_alive` gate (§3.5), so $s^{\text{pos}}_{t+1}$ does not realize the typed Markov transition the formalism defines. We therefore bound the SCM/typed-transition description to the interpretable path and do not describe the scored path as realizing the same operator (limitation in §8.1).

### 3.7 The decoupling, exactly

The total loss is

$$
\mathcal{L} = \underbrace{\mathcal{L}_{\text{BCE}}(\hat{p}, y)}_{\text{prediction}} \;+\; \lambda_{\text{kl}}\,\mathrm{KL}\big(q(z\mid m)\,\|\,p(z)\big) \;+\; \lambda_{\text{dc}}\,\mathcal{L}_{\text{decol-CE}}(s_{t+1}^{\text{cal}}) ,
$$

with three disjoint gradient routes: the **backbone** receives only the parsimony KL and deterministic laws; the **existence head** (scored $s^{\text{pos}}_{t+1}$ path) only $\mathcal{L}_{\text{BCE}}$; the **hierarchical heads** (interpretable $s^{\text{cal}}_{t+1}$ path) only $\mathcal{L}_{\text{decol-CE}}$. The stop-gradient on $h_{uv}$ blocks every Stream-B gradient from the backbone, so both heads read a frozen representation and the auxiliary objective leaks nowhere - the graph-ancestry asserts of §3.4 certify this. The three disjoint gradient routes are drawn in Appendix Figure A2.

---

## 4. The intervene-able lifecycle readout

**Why a temporal link predictor should be interrogable.** A practitioner often wants *what-if*: if this pair were forced into decline, how far would its edge probability fall? An entangled embedding has no typed state to intervene on. RS-GNN's decoded state comes from a known transition operator, so `do(·)` overrides a named concept and re-propagates it, riding free on the detach (§3.7) at zero cost to prediction.

**A detached temporal concept bottleneck with typed forward re-evaluation.** The interpretable state $s^{\text{cal}}_{t+1}$ comes from a known transition operator $T_{uv}$ (§3.6) over fixed $C$, with analytic-prior gates plus ~150 learnable residuals and a non-Markov `ever_alive` gate - five lifecycle concepts behind the zero-AP-cost detach (§3.4). We describe the interventions as **typed forward re-evaluation under an input override**, keeping $\mathrm{do}(\cdot)$ for the override but *not* claiming Pearl-sense causal identification [Pearl, 2009]: the structure is designer-imposed. The interpretability is **not separable from contribution 1** - the same detach blocks the identity shortcut *and* frees the readout from the scored path: one mechanism, two consequences.

The readout supports two override modes. A **driver override** ($\mathrm{do}(\text{rate}/\text{slope}/\text{staleness})$) overwrites a named statistic and re-propagates the forward graph. A **state override** $\mathrm{do}(\text{state}{=}s)$ replaces the gated draw with a fixed one-hot state and pushes it through the existence decoder with upstream statistics held fixed. The engine reconstructs the gate baseline exactly (residual $1\mathrm{e}{-16}$), so any change is attributable to the override. Measurements are real CoEdit pairs (config B, $N=12000$) over three seeds {1, 7, 42}.

**Existence counterfactual, both directions, per-seed.** Forcing the state moves predicted existence in the designer-intended direction: $\mathrm{do}(\text{DEATH})$ drives existence *down* for **100% of pairs every seed**, and $\mathrm{do}(\text{BIRTH})$ / $\mathrm{do}(\text{REINFORCE})$ drive it *up* for **100% every seed**. The full-population mean $\mathrm{do}(\text{DEATH})$ drop is **−0.522 ± 0.001** (recurring subset −0.384 ± 0.003).

*The directionality survives training - it is not a hardcoded artifact.* The forced-state existence intent is the existence-decoder weight $w_s=\mathrm{softplus}(\theta_s)$, and $\mathrm{do}(\text{state}{=}s)$ moves $p_{\text{edge}}$ monotonically in $w_s$, so the do(state) ordering is the sorted order of $w$. A natural question is whether this reflects only the hand-set init $w=[0.1,1,1,0.3,0]$. The **trained per-seed weights** answer it: after BCE training the partial order **DEATH < IDLE < DECAY < {REINFORCE, BIRTH}** holds on every seed (seed 42: $7.8\mathrm{e}{-5} < 0.060 < 0.31 < \{1.08, 1.27\}$; 3-seed means $0.0001 < 0.044 < 0.381 < \{1.023, 1.347\}$), so the directionality is a **learned property, not an init artifact**. *Honest exception:* the init ties REINFORCE = BIRTH and training **breaks** it (REINFORCE becomes largest, $|\Delta w|$ = 0.19/0.36/0.43 per seed), so the partial order, not a strict REINFORCE = BIRTH equality, is what holds. Figure 2 plots these **trained** per-seed weights (3-seed mean$\pm$std) against the init-equivalent prior, making the learned tie-break explicit. `do(noop)` gives $\Delta=0$, and reversibility is exact ($\max|\Delta p_{\text{edge}}|=0$ after undo, all seeds) by exact input-reconstruction - a numerical, not causal-content, statement.

![](../figs/fig2_counterfactual_ladder.png)

*Figure 2. Trained existence-intent ladder. Per-state existence weight $w_s=\mathrm{softplus}(\theta_s)$ after BCE training (3-seed mean$\pm$std, CoEdit config B; from `cf_trained_theta_3seed.json`) in solid blue, plotted against the init-equivalent prior $w=[0.1,1,1,0.3,0]$ in grey (the dashed line, where REINFORCE = BIRTH). Training preserves the partial order DEATH < IDLE < DECAY < {BIRTH, REINFORCE} on every seed and breaks the init REINFORCE = BIRTH tie (REINFORCE largest, $|\Delta w|$ = 0.19/0.36/0.43 per seed). Since $\mathrm{do}(\text{state})$ moves $p_{\text{edge}}$ monotonically in $w_s$, this is the do(state) ordering: $\mathrm{do}(\text{DEATH})$ drives existence down for 100% of pairs/seed, $\mathrm{do}(\text{BIRTH/REINFORCE})$ up for 100%/seed, $\mathrm{do}(\text{noop})\;\Delta=0$, exact reversibility (§4).*

**Dose-response, sign-correct on real drivers (3-seed, serialized).** Driver interventions move the distribution monotonically and in the physically correct direction; the regression slopes are serialized per-seed and sign-stable across all three seeds: rate ratio raises $P(\text{REINFORCE})$ (slope $+0.017$) and lowers $P(\text{DEATH})$ ($-0.057$); rising-slope raises $P(\text{REINFORCE})$ ($+0.284$, largest); true-occurrence lowers $P(\text{BIRTH})$ ($-0.004$); staleness raises $P(\text{DECAY})$ then $P(\text{DEATH})$ (synthetic axis). The claim is the **sign**, stable across seeds; magnitudes are reported as the serialized slopes.

**Trajectory counterfactuals on real pairs.** On DECAY-decoded pairs a synthetic $\mathrm{do}(\text{slope}{=}+)$ flips DECAY→REINFORCE for **0.9999 ± 0.0002** (per-seed 1.0/1.0/0.9996). On REINFORCE-decoded pairs (recurring subset) the kill decomposes by dose: each single driver is *partial* - isolated $\mathrm{do}(\text{rate}{=}\text{dead})$ → DEATH for **0.597 ± 0.028** (0.574/0.588/0.629), isolated $\mathrm{do}(\text{staleness}{=}\text{high})$ for **0.482 ± 0.007** (0.477/0.479/0.489) - while all drivers dead together is **decisive**: **0.999 ± 0.002** (1.0/1.0/0.997). So $T_{uv}$ supports two-way, three-seed-validated trajectory control (sources in Appendix A).

**The readout encodes a faithful, falsifiable rule.** The lifecycle decoder is not a free-form classifier read post-hoc: it commits to a checkable rule - *REINFORCE iff the edit cadence is rising* - that could be wrong and is confirmed three ways from the **learned** gates (non-tautological). (A) At the population level, the slope-vs-state separation is sign-correct: REINFORCE-decoded pairs sit at slope $\approx-0.49$, above DECAY-decoded pairs at $\approx-0.86$. (B) Under $\mathrm{do}(\text{slope})$, forcing a falling pair's cadence upward flips $P(\text{REINFORCE})$ from 0 to 1. (C) The same flip holds at single-pair granularity. Because all three read the learned residuals (not a fixed prior), a decoder that had ignored cadence would fail (A)-(C); it does not.

![](../figs/fig6_faithful_falsifiable.png)

*Figure 3. The "REINFORCE ⟺ rising cadence" rule, confirmed three ways from learned gates: (A) population slope separation (REINFORCE −0.49 > DECAY −0.86); (B) $\mathrm{do}(\text{slope})\to P(\text{REINFORCE})$ switching 0→1; (C) a single-pair counterfactual flip. Non-tautological: a cadence-blind decoder would fail all three (§4).*

![](../figs/fig7_intervenable_scm.png)

*Figure 4. Bidirectional typed control of the lifecycle readout: REINFORCE→$\mathrm{do}(\text{silence})$→DEATH (1.0) and DECAY→$\mathrm{do}(\text{rising})$→REINFORCE (1.0); dose-response signs correct (rate→REINFORCE +, rate→DEATH −); reversibility $\Delta=0$ by exact reconstruction (§4).*

**AP-neutrality and scope.** The battery reads the interpretable $s^{\text{cal}}_{t+1}$, on the detached side of §3.7's wall, so running it is eval-time bit-identical-AP (Property 1). The readout is exercised in data on the *alive axis* (rate/recurrence → alive → DEATH/REINFORCE), three-seed-validated. The *rising axis* (slope) is wired (synthetic flip 0.9999) but degenerate on real CoEdit (`slope_rel` is essentially always negative), and staleness is synthetic-only; we do *not* claim either as a validated real-data counterfactual (§8.1).

---

## 5. Causal-coherence confidence (honest scope)

We explored whether the model can flag *its own* low-confidence predictions via a **walked-chain causal-coherence** signal. A per-pair belief $b_t$ is carried by $T_{uv}$ projected onto the causal-admissible ray; the coherence $c_t \in [0,1]$ is the agreement between the model's free next-state prediction and this belief. The flag is off by default and byte-identical when off, so it never perturbs AP. Across three seeds (CoEdit, grounded-init), $c_t$ separates cleanly by causal-rule outcome: rule-following 0.891 ± 0.042 vs. rule-violating 0.216 ± 0.174, a well-spread signal with full $[0,1]$ support.

![](../figs/fig3_causal_coherence.png)

*Figure 5. Causal-coherence $c_t$ by outcome (CoEdit, grounded-init, three seeds). Bars: mean $c_t$, whiskers sample std. Takeaway: rule-consistent (0.891 ± 0.042) vs rule-violating (0.216 ± 0.174) separate cleanly - but $c_t$ measures the model's *own* rule violations, not external error (§5).*

**What c_t is - and is not.** Low coherence predicts the model's *own* causal-rule violation almost perfectly and stably (AUC = **0.9985 ± 0.0015**, three seeds), but this is **self-consistency**, not external truth - circular, since a model that confidently makes the same coherent mistake scores high. Tested against *actual* prediction misses (external `posMiss10`), the AUC is **0.405 ± 0.484**, wildly seed-dependent (0.949 / 0.245 / 0.021): a promising single-seed result did **not** replicate. We therefore **retract** any error-predictor claim and report $c_t$ only as a stable internal coherence measure (turning it into a calibrated error predictor needs external supervision we have not built). The confidence mechanism is drawn in Appendix Figure A5.

---

## 6. Experiments

### 6.1 Cross-dataset, protocol-matched, three seeds

**The CoEdit benchmark.** CoEdit is a **non-bipartite co-edit interaction graph**: nodes are contributors, edges are timestamped co-edit events between two contributors on the same artifact, so *both* endpoints carry a lifecycle (unlike the bipartite Wikipedia/MOOC graphs). It uses the same 70/15/15 chronological split and leak-audited negative pool as the standard datasets (construction protocol in Appendix A; stream + splits released with the code). The +13.5 headline is CoEdit-specific (§8.1).

All models run through the **same** harness (`experiments/train.py`), splits, and leak-audited negative pool (§6.3 confirms test AP is not 1.0), built fairly per protocol (transductive negatives from seen→seen, inductive from the unseen-node pool, so an inductive positive is never scored against a trivially-impossible negative). AP is sklearn `average_precision_score` for every model. Every cell is mean ± sample std over {1, 7, 42}; MOOC baselines were re-run on {1, 7, 42} to match (the earlier {7, 42, 123} numbers are not used). RS-GNN is **config B**, tuned on CoEdit only.

**The CoEdit competitor set.** Nine competitors run through the same harness on CoEdit (Tables 1-2): six parametric temporal-GNN families, two EdgeBank memorization-floor variants [Poursafaei et al., 2022], and the inductive-frontier transformer DyGFormer [Yu et al., 2023]. "+13.5" is over the best *measured* CoEdit competitor (TGAT, 0.853); we do **not** claim to beat the full named frontier - TCL/NAT are unrun, so the frontier comparison is partial (§8.2).

**Is CoEdit pessimizing baselines specifically?** No - our simplified DyGFormer re-implementation is below its published level on **Wikipedia** as well (0.786 here vs ≈0.98 published; the gap is architectural, see the baseline-fidelity note after Table 1), so its low CoEdit AP is the re-implementation ceiling, not CoEdit-specific starvation. What we read from Tables 1-2 is the *relative* ordering within one shared harness; the absolute levels are not comparable to published SOTA.

![](../figs/fig4_coedit_headline.png)

*Figure 6. CoEdit inductive AP: RS-GNN (config B) vs. protocol-matched baselines (six parametric + DyGFormer frontier + two EdgeBank floors), three seeds {1, 7, 42}. Bars: mean, whiskers: sample std. Takeaway: RS-GNN 0.9885 ± 0.0035 (three-seed; 0.9876 ± 0.0030 at five seeds), +13.5 pts (paired 95% CI [12.6, 14.5]) over the best parametric baseline (TGAT, 0.853 at three seeds / 0.847 at five); the modern frontier DyGFormer reaches only 0.612, and the EdgeBank floor ≈0.59.*

**Five-seed promotion (de-fragiling the headline).** The CoEdit headline and its core ablation were re-run at five seeds {1, 7, 42, 2, 3} so the central claims do not rest on a three-seed contrast. At $n=5$: RS-GNN config B **0.9876 ± 0.0030**, the coupled config C **0.7593 ± 0.0140**, TGAT **0.8466 ± 0.0133** - giving **B − C = +22.8 pp** and **B − TGAT = +14.1 pp** ($n=5$). Both deltas hold at the larger seed count (they move <1 pp from the three-seed values), so we no longer label the headline fragile; the remaining seed-budget item is promoting Wikipedia/MOOC and the frontier baselines to five seeds (§8.2). Tables 1-2 keep the three-seed values for cross-model comparability (every competitor has three matched seeds); the five-seed numbers above supersede them wherever the *CoEdit headline* magnitude is stated.

**Table 1 - Inductive AP (mean ± std, 3 seeds {1, 7, 42}).**

| Model | CoEdit ind-AP | Wikipedia ind-AP | MOOC ind-AP |
|---|---|---|---|
| **RS-GNN (config B)** | **0.9885 ± 0.0035** | 0.9959 ± 0.0014 | **0.9978 ± 0.0013** |
| JODIE | 0.8147 ± 0.0942 | 0.9860 ± 0.0029 | 0.9901 ± 0.0037 |
| TGAT | 0.8530 ± 0.0012 | **0.9981 ± 0.0013** | 0.9737 ± 0.0062 |
| CAWN | 0.7825 ± 0.0452 | 0.9877 ± 0.0062 | 0.8101 ± 0.2340 |
| TGN ᵇ | 0.6349 ± 0.0065 | 0.8637 ± 0.0459 | 0.9818 ± 0.0054 |
| DyRep | 0.6218 ± 0.0119 | 0.6314 ± 0.0550 | 0.7817 ± 0.2736 |
| GraphMixer | 0.6232 ± 0.0247 | 0.7380 ± 0.0770 | 0.9735 ± 0.0215 |
| DyGFormer (frontier) ᵇ | 0.6120 ± 0.0217 | 0.7859 ± 0.0344 | -ᵃ |
| EdgeBank-∞ (floor) | 0.5899 ± 0.0003 | 0.6541 ± 0.0000 | 0.5534 ± 0.0004 |
| EdgeBank-tw (floor) | 0.5894 ± 0.0003 | 0.6535 ± 0.0000 | 0.5534 ± 0.0004 |

ᵃ MOOC DyGFormer and TCL/NAT (all datasets) are not yet finalized to three seeds under our GPU budget and are omitted rather than shown as unfinished cells (§8.2).

Every cell traces to that model's own JSON at seeds {1, 7, 42} (per-seed values in Table A2). The discriminating dataset is **CoEdit**: RS-GNN leads the nine measured competitors (TCL/NAT unrun; §8.2), with the two EdgeBank floors at ≈0.59 confirming the win is not memorization (CoEdit inductive positives involve unseen nodes EdgeBank cannot have stored). MOOC is near-saturated, so its #1 result is weak evidence (§8.3).

ᵇ **Baseline fidelity (re-implementations vs published).** Our DyGFormer, TGN, and TGAT are simplified single-hop re-implementations run through the shared harness, so their absolute inductive APs sit below the official DyGLib figures (DyGFormer Wikipedia inductive ≈0.98, TGN ≈0.97 [Yu et al., 2023]). We verified this gap is **architectural, not under-training**: re-running DyGFormer and TGN at DyGLib's full published budget (100 epochs, learning rate $10^{-4}$, batch 200, patience 20) leaves Wikipedia inductive AP statistically unchanged (DyGFormer 0.762 ± 0.014, TGN 0.866 ± 0.014, 3 seeds; validation AP also caps at 0.84 / 0.93), so longer training does not close the gap - only the official multi-hop architectures do (`experiments/results/reconcile/reconcile_dyglib_wikipedia_3seed.json`). **Tables 1-2 are therefore an internally-consistent within-harness comparison** - every model, RS-GNN included, uses the same simplified backbones, the same split, and the same protocol - and are *not* a claim against published state of the art. The within-harness CoEdit margin and, above all, the within-model controlled experiments (the single-flag knob, §6.3, and the freeze-then-probe control, §6.2) carry the contribution; none of them depends on the absolute baseline level. Matching published SOTA in-harness would require integrating the official DyGLib models (§8.4).

**Table 2 - Transductive AP (mean ± std, 3 seeds {1, 7, 42}), all models.**

| Model | CoEdit trans-AP | Wikipedia trans-AP | MOOC trans-AP |
|---|---|---|---|
| **RS-GNN (config B)** | **0.9985 ± 0.0004** | **0.9993 ± 0.0002** | **0.9988 ± 0.0002** |
| JODIE | 0.9657 ± 0.0217 | 0.9954 ± 0.0010 | 0.9917 ± 0.0024 |
| TGAT | 0.8690 ± 0.0058 | 0.6578 ± 0.0214 | 0.6360 ± 0.0433 |
| CAWN | 0.8802 ± 0.0128 | 0.9861 ± 0.0017 | 0.9436 ± 0.0643 |
| TGN | 0.9419 ± 0.0081 | 0.9125 ± 0.0042 | 0.9876 ± 0.0007 |
| DyRep | 0.9294 ± 0.0007 | 0.8838 ± 0.0067 | 0.9442 ± 0.0501 |
| GraphMixer | 0.7474 ± 0.0233 | 0.8304 ± 0.0757 | 0.9742 ± 0.0050 |
| DyGFormer (frontier) | 0.8079 ± 0.0017 | 0.8975 ± 0.0080 | -ᵃ |
| EdgeBank-∞ (floor) | 0.6229 ± 0.0001 | 0.8170 ± 0.0001 | 0.6291 ± 0.0002 |
| EdgeBank-tw (floor) | 0.5553 ± 0.0001 | 0.6531 ± 0.0000 | 0.5879 ± 0.0001 |

ᵃ As Table 1: MOOC DyGFormer and TCL/NAT are not yet three-seed finalized and are omitted rather than shown unfinished (§8.2).

*Seed-protocol and std note.* All baselines in Tables 1-2 use the **B-protocol runs** at seeds {1, 7, 42}, the same set RS-GNN uses. Every `±` is the sample std (n−1) recomputed from the per-seed values; where a stored `*_std` field is a population std (÷n) it is overridden (e.g. DyGFormer CoEdit 0.0177→0.0217). So the override is reproducible without trusting the note, **Table A2 (Appendix A) lists the three per-seed AP values for every cell in Tables 1-2.**

**Reading the table.** **CoEdit** is the discriminating benchmark: RS-GNN is #1 both ways, +13.5 inductive over TGAT (0.853; paired 95% CI [12.6, 14.5], n=3), with the DyGFormer frontier (0.612) and EdgeBank floor (≈0.59) lower still. On **Wikipedia** RS-GNN is #1 transductive and #2 inductive (0.9959 vs TGAT 0.9981, whose transductive AP collapses to 0.658), so it is the best all-around model. **MOOC** is near-saturated; we claim only "competitive at the ceiling."

![](../figs/fig5_cross_dataset.png)

*Figure 7. Cross-dataset result summary (the headline visualization): RS-GNN (config B) vs. the best baseline per dataset, transductive and inductive AP, three seeds {1, 7, 42}, with the CoEdit B-vs-C decoupling gap overlaid. Takeaway: a clear CoEdit inductive win (+13.5 over the best measured competitor) that the B-vs-C ablation attributes to the detach (−22.1 inductive vs −3.8 transductive when removed); best-or-co-best on Wikipedia/MOOC, where TGAT's lone inductive win on Wikipedia (0.9981 vs 0.9959) is paired with a collapsed transductive AP (0.658) - the same identity-shortcut signature the ablation isolates. This figure, with Tables 1-3, carries the cross-dataset and mechanism narrative.*

### 6.2 The decoupling ablation (the core experiment)

**Config B is the full model** (detached backbone + multi-signal operator + hierarchical lifecycle readout + intervention battery + causal policy); the arms below are *ablations*, not competing models. The full decoupled-vs-coupled contrast is **B vs. C** (0.9885 → 0.7672, +22.1 ± 1.4 pp), but B and C differ on more than the detach, so on its own +22.1 is a *configuration* effect. We therefore ran a **single-variable, three-seed knob ablation** that decomposes the gap by flipping the three differing flags one at a time, off a single fixed config-B stack. The result is clean.

**Table 3 - Single-variable knob ablation (CoEdit, 3 seeds {42,1,7}; one flag flipped per arm off fixed config B).**

| Arm | flag flipped | ind-AP | Δ vs. B | trans-AP |
|---|---|---|---|---|
| **B (baseline)** | - | **0.9899 ± 0.0016** | - | 0.9986 |
| **K1** | `enable_main_predictor` False→**True** | **0.7788 ± 0.0193** | **−21.11 pp** | 0.9597 |
| K2 | `lfg_mode` soft→hard | 0.9868 ± 0.0014 | −0.31 pp | 0.9985 |
| K3 | `compliance_floor` 0.05→0.0 | 0.9889 ± 0.0018 | −0.10 pp | 0.9986 |
| C (all three) | `design=correct` | 0.7798 ± 0.0221 | −21.01 pp | 0.9612 |

**K1 alone is the whole gap.** Enabling the end-to-end head - replacing the detached existence decoder (reading $s^{\text{pos}}_{t+1}$) with a non-detached 2-layer MLP that carries the link gradient into the backbone - costs **−21.11 pp** inductive AP, indistinguishable from the full three-flag C arm (−21.01 pp), while the two co-varying gates are **inert** (K2 −0.31, K3 −0.10 pp, both within seed noise). So decoupling is a **single-variable, isolated** effect worth +21.1 pp inductive, not a confounded configuration delta - converting the earlier "+22.1 is a configuration effect" caveat into a measured attribution. K1's coupled head is exactly an unconstrained MLP given backbone access, and it is the arm that collapses.

**Detach-path decomposition (secondary).** The main-predictor knob (K1) carries the bulk of the mechanism; a separate three-seed probe isolates the *additional* `edge_h.detach()` on the score path: detach on 0.9871 ± 0.0037 vs off 0.9777 ± 0.0037, a further **+0.94 pp**. So the honest decomposition is: main-prediction-head OFF (+21.1 pp, primary) plus score-path detach (+0.94 pp, secondary).

**Identical-head control: the effect is the gradient, not the head.** K1 changes the scoring head (existence decoder → MLP) at the same time as the gradient coupling, so the drop could in principle be read as the MLP destroying the hand-crafted point-process features rather than as gradient contamination. We separate the two by holding the head **identical** - the same 2-layer MLP scoring head, bit-identical initialization per seed - and toggling **only** the `.detach()` on the backbone input. The detached arm feeds zero link-prediction gradient to the backbone, the coupled arm feeds it (CPU-verified: the head modules are `torch.equal`, and the backbone parameters receive zero vs nonzero gradient).

**Table 7 - Identical-head detach toggle (same MLP head, only the backbone detach flipped; ind-AP, 3 seeds {42,1,7}).**

| Dataset | DETACHED-MLP | COUPLED-MLP | Δ(detach − couple) |
|---|---|---|---|
| CoEdit | 0.9357 ± 0.0032 | 0.7969 ± 0.0367 | **+13.9 pp** |
| Wikipedia | 0.9784 ± 0.0019 | 0.9061 ± 0.0003 | **+7.2 pp** |

With the head architecture held constant, detaching the backbone wins by **+13.9 pp** (CoEdit) and **+7.2 pp** (Wikipedia) - the same direction and comparable magnitude as the K1 knob. The inductive damage is therefore attributable to the **link gradient reaching the backbone**, not to the head architecture; the "a coupled MLP merely destroys hand-crafted features" reading is excluded. (The detached-MLP level, 0.936 on CoEdit, is below config B's 0.989 because config B's scored head is the structured existence decoder, not a plain MLP; this comparison is strictly within the identical MLP head.)

**Backbone-removed control: the learnable backbone adds real work, but the deterministic features carry the bulk.** To characterize what the KL-trained learnable backbone (the CSN event encoder + DRGC node memory) contributes, we froze it at initialization and zeroed its signal to the scored head, leaving only the deterministic point-process channels (Hawkes intensity, Welford gap statistics, EWMA rate, recurrence) feeding config B's detached head (CPU-verified: 56 backbone parameters frozen, zero backbone gradient).

**Table 8 - Backbone-removed ablation (learnable CSN/DRGC removed; deterministic statistics + detached head only; ind-AP, 3 seeds {42,1,7}).**

| Dataset | FULL-B | DETERM-ONLY | Δ(FULL − DETERM) |
|---|---|---|---|
| CoEdit | 0.9883 ± 0.0023 | 0.9205 ± 0.0017 | +6.8 pp |
| Wikipedia | 0.9963 ± 0.0009 | 0.8965 ± 0.0000 | +10.0 pp |

Removing the learnable backbone costs +6.8 pp (CoEdit) and +10.0 pp (Wikipedia) inductive AP - the KL-trained backbone is doing real work - but the deterministic-only model is already strong (0.92 / 0.90), so the hand-crafted point-process features carry most of the performance. We report this plainly: the contribution is split, not all-or-nothing. A reading that the deterministic features largely suffice is partially correct, with the learnable backbone adding a real but minority margin; this is orthogonal to the decoupling claim, which concerns the *gradient* to whichever backbone is present (Tables 3, 7).

**The decisive novelty control: decoupling-by-construction vs. freeze-then-probe.** The obvious objection is that "decoupling" is just classic *freeze-then-probe* relocated to temporal graphs - pretrain end-to-end, freeze, fit a fresh head. We ran exactly that on RS-GNN's *own* backbone (CoEdit, three seeds {42, 1, 7}, B-protocol). ARM1 is decoupling-by-construction (config B; backbone never sees link-pred gradient). ARM2 is freeze-then-probe: a Phase-1 end-to-end pretrain shapes the backbone with the link loss, then we freeze it (weights byte-identical after the fresh head's optimizer step) and re-fit a fresh existence head.

**Table 4 - Decoupling-by-construction vs. freeze-then-probe vs. coupled, on two datasets.** CoEdit at three seeds {42, 1, 7}; Wikipedia at the available seeds (decoupling and coupled three-seed {42, 1, 7}; freeze-then-probe two-seed {42, 1}). The coupled column is the K1 end-to-end arm from the same knob driver (§6.2/§6.5).

| Dataset | decouple-by-construction (ARM1) | freeze-then-probe (ARM2) | coupled / end-to-end (K1) |
|---|---|---|---|
| **CoEdit** (ind-AP) | **0.9883 ± 0.0023** | 0.7684 ± 0.0052 | 0.7593 ± 0.0140 (5-seed C) |
| **Wikipedia** (ind-AP) | **0.9961 ± 0.0011** (n=3) | 0.8975 ± 0.0152 (n=2) | 0.9093 ± 0.0061 (n=3) |

On CoEdit the result is decisive - **+21.99 pp** (0.9883 → 0.7684) - and is the load-bearing insight of the paper: freeze-then-probe (0.768) lands on the *coupled* number (C at five seeds, 0.759), not the decoupled one. *Once the backbone has been contaminated by the link loss, freezing and re-probing does not recover inductive transfer; the damage is irreversible.* The contribution is therefore not "freezing" - freeze-then-probe freezes and still fails - but **preventing link-prediction contamination by construction**, which the standard linear-probing recipe does not realize. **Wikipedia confirms the same direction on a dataset we did not build:** freeze-then-probe (0.897, n=2) sits beside the coupled end-to-end arm (0.909, n=3) and far below decoupling-by-construction (0.996, n=3) - FtP ≈ coupled $\ll$ decoupled. The Wikipedia gap is smaller than CoEdit's (8.6 pp coupled-vs-decoupled, tracking how much node identity the coupled head can exploit, §6.5), but its *ordering* is identical, so irreversibility is a two-dataset finding, not a CoEdit peculiarity. (Wikipedia freeze-then-probe is at two seeds {42, 1} and CoEdit at three; we report both seed counts plainly.)

*Statistical posture.* No multiple-comparisons correction is applied. The **primary** confirmatory deltas are K1 (−21.1, three-seed), B−C (+22.8) and B−TGAT (+14.1) - both promoted to **five seeds** (§6.1) - and the freeze-then-probe control (+22.0, Table 4); all others (`causal_batch` +5.7, `hier_causal_policy` neutrality, the score-path detach +0.94, the dose-response signs and kill decomposition in §4) are **exploratory**, reported with per-seed values. Three-seed paired-$t$ CIs are at $df=2$ ($t_{0.975}=4.303$): valid but wide; the CoEdit headline and B−C no longer rest on them since the five-seed promotion landed (§6.1). The remaining seed-budget item is promoting the Wikipedia/MOOC cross-dataset knob and the frontier baselines to five seeds (§8.2).

| Arm (CoEdit, 3 seeds {1, 7, 42}) | design | detach? | ind-AP | trans-AP |
|---|---|---|---|---|
| **B - decoupled (full model)** | full readout | yes | **0.9885 ± 0.0035** | 0.9985 ± 0.0004 |
| C - end-to-end | full readout | **no** | 0.7672 ± 0.0107 | 0.9609 ± 0.0034 |
| A - no-lifecycle, no-tune | flat readout | yes | 0.928 ± 0.0043 | 0.9912 ± 0.0009 |

All cells are mean ± sample std over the three seeds {1, 7, 42}.

![](../figs/fig6_decoupling_ablation.png)

*Figure 8. Decoupling ablation, CoEdit inductive AP. Bars: seed-mean ind-AP, whiskers sample std. Takeaway: the end-to-end configuration (C) collapses inductive AP relative to the decoupled full model (B); at five seeds B 0.9876 / C 0.7593, B−C = +22.8 pts (three-seed B 0.9885 / C 0.767 / A 0.928 for cross-arm comparability). Arm A (0.928, no-lifecycle ablation) already clears every baseline.*

**The B-vs-C configuration contrast.** The decoupled configuration is worth **+22.8 points** inductive AP at five seeds (B 0.9876 ± 0.0030 vs C 0.7593 ± 0.0140; three-seed B−C = +22.1 ± 1.4 pp, paired 95% CI [18.6, 25.6], $df=2$). The collapse is split-localized - K1 costs −21.1 pp inductively but only −3.9 pp transductively (trans 0.9986 → 0.9597), the identity-shortcut signature: the coupled head reshapes the backbone toward training-node identity, dead weight on unseen nodes.

**Arm A is the canonical-design floor.** Arm A (detached, flat readout, no de-collapse CE / hierarchical decode / `causal_batch` / tuning) differs from C by both the detach and the readout config, so it is not a clean detach-alone contrast. It establishes a floor: even stripped, RS-GNN reaches **0.928 ± 0.0043** inductive AP, +7.5 over TGAT - baseline-beating before any lifecycle machinery; the further +6 to B is what lifecycle supervision adds.

**Two further ablations.**
- **`causal_batch` (read-before-write, §3.2):** config B ON 0.9885 vs OFF 0.9312 inductive (**+5.7 pp**, three seeds; trans +0.65 pp), confirming the collapsed statistics were load-bearing. The same sign holds in the stripped no-lifecycle setting (single-seed A/B, non-evidentiary; Appendix A).
- **`hier_causal_policy` (soft causal mask, §3.5):** three seeds, train ON vs. OFF, per-seed inductive Δ = +1.0e-4 / +9.3e-4 / −1.5e-3 (max $|\Delta|=1.5\mathrm{e}{-3}$, sign-mixed) vs. ±3.5e-3 seed std. **AP-neutral within seed noise** (§3.4 training-time statement): the residual is RNG jitter, not bit-identical. The policy buys interpretability for statistically free.

### 6.3 Integrity audit

We independently audited the headline: (1) Arm A (0.928) is real - old vs. new baselines match within GPU noise at matched seeds, and 0.871 → 0.928 is a *config* difference (v3 operator on a flat readout), not an eval shift; (2) the pre-update evaluation is leak-free (an anti-leak re-gate pulls test AP off 1.0 into the v2 band); (3) AP is a model-agnostic sklearn routine over the same negative pool. Provenance in Appendix A.

### 6.4 Lifecycle faithfulness

On the hierarchical readout DECAY carries real argmax mass, **three-seed and serialized**: DECAY is the interpretable argmax for **47.7% ± 0.4%** of pairs vs. **0.18% ± 0.28%** under the interpolating flat readout - the empirical confirmation of §3.3. Falling-but-active streams win DECAY while alive, sustained-silent pairs go to DEATH, new pairs to BIRTH, and decoded DECAY tracks each pair's own cadence (Spearman $\rho = -0.61 \pm 0.02$, recurring $n=9157$, $p \approx 0$). This is faithful to its *own supervised objective* (the plotted $s^{\text{cal}}_{t+1}$ is what the de-collapse CE optimizes), **not** to the scored $s^{\text{pos}}_{t+1}$; the scored path's invariance is the separate §3.4 guarantee.

### 6.5 The decoupling mechanism is not a CoEdit artifact (cross-dataset knob)

CoEdit is introduced here, so we are careful not to let a single self-built dataset carry the mechanistic claim. We therefore ran the *same single-variable knob ablation* - flip `enable_main_predictor` False→True, every other config-B flag fixed - on two **standard** benchmarks we did not construct, Wikipedia and MOOC (three seeds {42, 1, 7}, B-protocol).

**Table 5 - Cross-dataset single-flag knob ablation (B vs K1, ind-AP, 3 seeds).**

| Dataset | B (decoupled) | K1 (`enable_main_predictor`=True) | Δind (B − K1) |
|---|---|---|---|
| CoEdit (built here) | 0.9899 ± 0.0016 | 0.7788 ± 0.0193 | **+21.11 pp** |
| Wikipedia (standard) | 0.9957 ± 0.0010 | 0.9093 ± 0.0061 | **+8.64 pp** |
| MOOC (standard) | 0.9978 ± 0.0016 | 0.9894 ± 0.0043 | **+0.85 pp** |

**Decoupling wins everywhere; magnitude is dataset-dependent, honestly.** On all three datasets B > K1 - the decoupling direction is robust *including on two graphs we did not build*, so the effect is **not a CoEdit-construction artifact**. The magnitude measures *how much training-node identity the coupled head can exploit*: large where unseen-node generalization is hard (CoEdit +21 pp), small where the dataset's own features already transfer (MOOC +0.9 pp). This is exactly the identity-shortcut prediction - the end-to-end head only hurts to the extent it *can* overfit identity - and we report the small MOOC delta plainly. The between-model TGAT split-asymmetry on Wikipedia (inductive 0.998, transductive collapsed to 0.658, vs detached RS-GNN's 0.996/0.999) corroborates the same-model knob. (The cross-dataset lifecycle-*shape* check, Table A1, is single-seed and non-evidentiary.)

**The advantage survives - and widens under - both hard-negative regimes.** Random negatives are the easy regime, and rankings can invert under harder negatives [Poursafaei et al., 2022], so we re-evaluated the same B-vs-K1 knob under **both** Poursafaei hard-negative strategies: **historical** (negative destinations drawn from edges seen in training but absent at the current step - punishing memorization that a pair has interacted before) and **inductive** (destinations drawn from edges that appear only in the test phase). On the same trained models, B and K1 are scored on bit-identical paired negative sets per seed.

**Table 6 - B vs K1 inductive AP under random and hard negatives (3 seeds {42, 1, 7}).**

| Dataset | Negative regime | B (decoupled) | K1 (coupled) | Δ(B − K1) |
|---|---|---|---|---|
| Wikipedia | random | 0.9968 ± 0.0005 | 0.9011 ± 0.0106 | +9.6 pp |
| Wikipedia | historical (hard) | 0.9744 ± 0.0047 | 0.5585 ± 0.0147 | **+41.6 pp** |
| Wikipedia | inductive (hard) | 0.9682 ± 0.0035 | 0.5423 ± 0.0032 | **+42.6 pp** |
| CoEdit | random | 0.9898 ± 0.0012 | 0.7899 ± 0.0545 | +20.0 pp |
| CoEdit | historical (hard) | 0.9573 ± 0.0031 | 0.5294 ± 0.0276 | **+42.8 pp** |
| CoEdit | inductive (hard) | 0.9625 ± 0.0024 | 0.5579 ± 0.0215 | **+40.5 pp** |

Under **both** hard-negative regimes the coupled arm **collapses toward chance** (K1 ≈ 0.53-0.56) while the decoupled arm holds (B ≈ 0.96-0.97), so the gap does not shrink - it **widens to +40-43 pp on both graphs under both hard regimes**. This makes the identity-overfitting mechanism visible: the coupled head, having encoded which pairs interacted during training, scores plausible-but-currently-absent edges as positive and is punished exactly where hard negatives probe, whereas the decoupled backbone - never exposed to the link gradient - has no such shortcut. The ordering B > K1 does **not** invert under the harder regimes the evaluation literature prescribes; it strengthens on both. (Inductive negatives use the Poursafaei test-phase-edge pool. Sources: `experiments/results/hardneg/hardneg_B_vs_K1_{wikipedia,coedit}_3seed_v2.json`.)

### 6.6 Why per-pair, why CoEdit-scoped: a deliberate trade-off, not a gap

The paper makes **two separable contributions**. **C1 - decoupling for inductive AP** is a *general* principle, shown directionally on three datasets (§6.5) with no reference to CoEdit's domain. **C2 - the per-pair causal lifecycle FSM** (§4) is *deliberately dataset-scoped*; we argue that scoping is the correct design, not a shortfall.

**The lifecycle drivers are different *concepts* across domains, not the same concept at a different scale.** What pushes a pair through born→reinforced→decay→die is domain-specific and not inter-convertible: a banking graph is driven by {transaction *frequency*, *amount* per transaction (a money-weight)}; a preference graph by {*frequency*, number of distinct users (a *degree*)}; CoEdit by {edit-*rate*, gap-versus-habit, recurrence}. A money-weight, a counterparty-degree, and an interaction-rate are categorically different quantities - different *concepts*, not different units. A single global FSM would have to know which driver concept governs which domain; discarding that knowledge to force one global readout would forfeit the causal faithfulness §4 measures. So the part that *generalizes* - decoupling (C1) - is shown cross-dataset, while the causal readout (C2) is scoped by design; asking C2 to run domain-agnostically is asking it to discard the per-domain knowledge that makes it faithful. Globalization (a normalized-dynamics space with a domain-adaptive driver basis, plus cross-pair spillover the current per-pair $T_{uv}$ omits) is design-staged as future work.

---

## 7. Analysis

**Why decoupling helps inductively, and where the tax lands.** A backbone with gradient access to the link loss is rewarded for identity features that help transductively but are dead weight on unseen nodes; RS-GNN's backbone never sees the link gradient, so it cannot learn that shortcut. The B−C gap measures this tax, and its split decomposition is the localization claim: removing the detach costs **−22.8 pp inductively** (five-seed) but only **−3.8 pp transductively**. The tax thus lands almost entirely on the inductive split.

**Three independent cuts agree, and they rule out the generic frozen-probe reading.** The split-localized signature is not the uniform transfer gain a generic linear-probe account predicts, and the irreversibility control (§6.2, Table 4) confirms that freezing after the fact recovers nothing - only never exposing the backbone keeps the inductive transfer. So the mechanism is triangulated by (a) a single-flag knob on three datasets (§6.5), (b) the irreversibility control (§6.2), and (c) the split-localized tax above - not by one self-built benchmark. The finer-grained *advantage-grows-with-inductive-novelty* version - binning inductive test edges by node-novelty - is not yet measured (§8.2); we state within-split monotonicity as a prediction, not a result.

---

## 8. Limitations

The body cross-references this section rather than re-stating.

**8.1 Scope of the win and the causal claim.**
- **CoEdit-tuned headline; mechanism shown cross-dataset.** Config B was tuned on CoEdit; Wikipedia/MOOC run it untuned, so the +13.5 *AP* headline is CoEdit-specific. The decoupling *mechanism*, however, is now shown on standard ground: the single-flag knob ablation (B vs K1) was run on Wikipedia (+8.6 pp) and MOOC (+0.9 pp) as well as CoEdit (§6.5), and corroborated by the TGAT split-asymmetry. What remains CoEdit-only is the full B-vs-C *configuration* contrast and the five-seed promotion of the cross-dataset knob (§8.2).
- **The per-pair causal readout (C2) is dataset-scoped by design**, because lifecycle drivers are different concepts across domains; the argument and the design-staged globalization are in §6.6.
- **The causal structure is designer-imposed, and its directionality is forced by construction.** The gates are hand-specified analytic priors with ~150 residual parameters and $C$ fixed. The `do(state)` numbers read off a sorted softplus-weight vector whose sign structure is hand-set ($w_{\text{DEATH}}$ minimal, $w_{\text{BIRTH/REINFORCE}}$ maximal), so "do(DEATH) down for 100%" is entailed by that ordering, not learned (§4). The trained per-seed $\theta$ are serialized and Figure 2 plots them (the partial order survives training and the init REINFORCE = BIRTH tie is broken), which rebuts the "weights hardcoded" reading; but the *directionality* itself is still designer-imposed, not a discovered causal fact, and the ±0.001 population stability is a numerical reconstruction property, not causal content. The intervene-ability is also not separable from the decoupling: the same detach causes both.
- **The scored path is non-Markov via `ever_alive`, so it does not realize the §3.6 typed Markov operator.** The scored $s^{\text{pos}}_{t+1}$ applies `ever_alive` after the soft mask; the typed-transition SCM formalism is therefore bounded to the **interpretable** path, and we do not describe the scored path as realizing the same operator (§3.6). This is orthogonal to Property 1, which concerns only the flat↔hier and causal-policy toggles on $s^{\text{cal}}$.
- **The [G] name-partition is descriptive, not load-bearing.** The backbone/head split by name is a heuristic; the sound certificate is the graph-ancestry assert (every parameter with nonzero scored-loss gradient lies in the head set), which does not depend on the partition being correct (§3.4).
- **Causal axes exercised only on the alive axis.** On real CoEdit the rising axis (slope) is degenerate (essentially always negative) and staleness is synthetic-only; those axes are *wired* (synthetic flip 0.9999) but not validated as real-data counterfactuals (§4).
- **The model is per-pair; cross-pair spillover is out of scope.** The edge-state operator $T_{uv}$ runs independently per ordered pair and does **not** model cross-pair or cross-region spillover (one pair's reinforcement raising a neighbor's intensity). Globalizing the per-pair readout into a region-coupled model is design-staged but unbuilt; we scope it to future work.

**8.2 Statistical scope and remaining evidence.**
- **Confidence is self-consistency, not error prediction.** $c_t$ stably predicts the model's *own* rule violations (AUC 0.9985) but not actual errors (AUC 0.405 ± 0.484 across seeds). We retract the single-seed error-prediction claim (§5).
- **Seed-locking.** Three-seed-locked on CoEdit: every headline ablation delta (K1 −21.1, B−C, score-path detach +0.94, `causal_batch` +5.7), the cross-dataset knob (Table 5) and freeze-then-probe control (Table 4), `hier_causal_policy` neutrality, and the full counterfactual battery (§4) - all serialized (Appendix A). The CoEdit B/C/TGAT headline is **five**-seed (§6.1). Only the cross-dataset lifecycle *shape* (Table A1) and the stripped no-lifecycle `causal_batch` A/B remain **single-seed (seed 42) and explicitly non-evidentiary** - sanity checks, no significance claim drawn.
- **Remaining evidence we flag honestly (and do not claim as run).** (i) A frozen *standard*-GNN backbone (TGAT/TGN) freeze-then-probe to complement the RS-GNN-backbone control of Table 4. (ii) Promote the **Wikipedia/MOOC knob** (Table 5) and the **full B-vs-C configuration contrast** to five seeds on the standard datasets (the CoEdit headline is already five-seed; the cross-dataset knob is three-seed). (iii) Finalize **MOOC DyGFormer** and add **TCL/NAT** on all datasets, or keep the frontier comparison downscoped (§2, §6.1). (iv) A **per-novelty-bin identity probe** within the inductive split (advantage rising with node-novelty), beyond the split-level −22.8 vs −3.8 decomposition (§7). (v) Promote the single-seed cross-dataset lifecycle-shape check (Table A1) to ≥3 seeds. (vi) **[Future work - design-staged]** Globalize the per-pair causal readout via a normalized-dynamics space with a domain-adaptive driver basis, and a region-coupled operator that captures cross-pair spillover. None of (i)-(vi) is fabricated or claimed as completed here.

**8.3 Other retractions and negative results.**
- **Regime-switch hypothesis falsified.** A change-point synthetic showed RS-GNN's per-pair adaptation does *not* beat CAWN on post-change-point slices; the validated edge is the inductive readout, not faster regime adaptation.
- **Echo memory and a learned transition-CE are retracted** - the backbone regularizer is a VAE/parsimony KL, and the transition supervision is the de-collapse CE on the interpretable next-state, not a separate transition-matrix CE term.
- **MOOC is near-saturated**, so its #1 result is weak evidence; the meaningful separation is CoEdit.

**8.4 Evaluation regime and baseline fidelity (the two largest threats).**
- **Hard negatives: both regimes done (ordering widens), full table not re-run.** The main Tables 1-2 use random negative sampling - the standard but easy regime (EdgeBank floors at ≈0.59). We additionally re-evaluated the B-vs-K1 knob under **both** Poursafaei hard-negative regimes - historical and inductive [Poursafaei et al., 2022] - on Wikipedia and CoEdit (Table 6): the decoupling ordering does not invert under either, it widens to +40-43 pp as the coupled arm collapses toward chance. We have not re-run the full nine-model Table 1 under hard negatives - only the load-bearing B-vs-K1 contrast.
- **CoEdit is not source-independent from Wikipedia, and is recurrence-heavy.** CoEdit is derived from the Wikipedia edit stream, so "two standard graphs we did not build" reduces to **one** genuinely independent source (MOOC), where the knob effect is only +0.85 pp - comparable to seed noise. The cross-dataset evidence for the mechanism is correspondingly weaker than a two-independent-source reading suggests. The CoEdit inductive split is also materially more recurrence-dominated than Wikipedia/MOOC (per-pair repeat median 2 vs 1; 55.6% of inductive pairs recur vs 36-39%; 535 unseen nodes over 5,336 inductive edges - Appendix A). An alternative reading we cannot yet exclude is that hand-crafted point-process features simply suffice on recency-dominated co-edit data while a coupled MLP destroys them; the identical-head K1 variant and a backbone-removed ablation (§8.2.i) are the controls that would separate these.
- **Baselines are simplified re-implementations, not the official library models.** Our DyGFormer/TGN/TGAT are single-hop re-implementations run through our shared harness, and their absolute inductive APs sit below DyGLib's published figures (e.g. DyGFormer Wikipedia inductive ≈0.79 here vs ≈0.98 published). Table 1 is therefore an **internally-consistent within-harness** comparison - every model, including RS-GNN, uses the same simplified backbones, the same split, and the same protocol - and is **not** a claim against published state of the art. A reconciliation run under DyGLib's published training budget (epochs/early-stopping/learning rate) is in progress; until it lands, the +14.1 pp over TGAT should be read as a within-harness margin, and any comparison to published SOTA requires the official library models.

---

## 9. Conclusion

RS-GNN reframes a design decision usually treated as obvious - train the representation on the task loss - and shows the opposite is better for inductive temporal link prediction. The primitives are prior art; the result is not. We establish that end-to-end coupling is the wrong default for inductive temporal link prediction and that the damage is irreversible. The mechanism is triangulated: a single-flag knob attributes the gap to `enable_main_predictor` (−21.1 pp) and holds on two standard graphs we did not build (§6.5); a freeze-then-probe control shows the contamination is irreversible on two datasets (CoEdit and Wikipedia, FtP ≈ coupled $\ll$ decoupled), so the contribution is preventing it by construction (§6.2); and the CoEdit headline is five-seed (§6.1). The *same* detach frees a faithful, intervene-able lifecycle readout - a detached temporal concept bottleneck, not a learned SCM - at zero scored-path cost (§3.4), a clean interpretability add-on, deliberately dataset-scoped because lifecycle drivers are different concepts across domains (§6.6). We see decoupling-by-construction as a reusable principle for temporal graph models that must generalize to unseen entities while remaining interrogable. What remains - and which we do not present as done - is five-seed promotion of the cross-dataset knob, a standard-GNN-backbone freeze-then-probe, and the per-novelty-bin probe (§8.2).

---

## References

Alain, G., & Bengio, Y. (2017). Understanding Intermediate Layers Using Linear Classifier Probes. *International Conference on Learning Representations (ICLR) Workshop*. arXiv:1610.01644.

Alemi, A. A., Fischer, I., Dillon, J. V., & Murphy, K. (2017). Deep Variational Information Bottleneck. *International Conference on Learning Representations (ICLR)*. arXiv:1612.00410.

Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). Emerging Properties in Self-Supervised Vision Transformers. *IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 9650-9660. [DINO; self-distillation with stop-gradient]

Chen, X., & He, K. (2021). Exploring Simple Siamese Representation Learning. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 15750-15758.

Cong, W., Zhang, S., Kang, J., Yuan, B., Wu, H., Zhou, X., Tong, H., & Mahdavi, M. (2023). Do We Really Need Complicated Model Architectures for Temporal Networks? *International Conference on Learning Representations (ICLR)*. [GraphMixer]

Hawkes, A. G. (1971). Spectra of Some Self-Exciting and Mutually Exciting Point Processes. *Biometrika*, 58(1), 83-90.

Huang, S., Poursafaei, F., Danovitch, J., Fey, M., Hu, W., Rossi, E., Leskovec, J., Bronstein, M., Rabusseau, G., & Rabbany, R. (2023). Temporal Graph Benchmark for Machine Learning on Temporal Graphs. *Advances in Neural Information Processing Systems (NeurIPS)*. [TGB; fair-negative protocol]

Jacovi, A., & Goldberg, Y. (2020). Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness? *Annual Meeting of the Association for Computational Linguistics (ACL)*, pp. 4198-4205.

Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *International Conference on Learning Representations (ICLR)*. [VAE]

Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., Kim, B., & Liang, P. (2020). Concept Bottleneck Models. *International Conference on Machine Learning (ICML)*, pp. 5338-5348. [CBM; concept-override intervention]

Kumar, A., Raghunathan, A., Jones, R., Ma, T., & Liang, P. (2022). Fine-Tuning Can Distort Pretrained Features and Underperform Out-of-Distribution. *International Conference on Learning Representations (ICLR)*. [linear-probe vs. fine-tuning]

Kumar, S., Zhang, X., & Leskovec, J. (2019). Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, pp. 1269-1278. [JODIE]

Luo, Y., & Li, P. (2022). Neighborhood-Aware Scalable Temporal Network Representation Learning. *Learning on Graphs Conference (LoG)*. [NAT]

Mei, H., & Eisner, J. (2017). The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process. *Advances in Neural Information Processing Systems (NeurIPS) 30*, pp. 6754-6764.

Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.

Poursafaei, F., Huang, S., Pelrine, K., & Rabbany, R. (2022). Towards Better Evaluation for Dynamic Link Prediction. *Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track*. [EdgeBank; memorization floor, harder negatives]

Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., & Bronstein, M. (2020). Temporal Graph Networks for Deep Learning on Dynamic Graphs. *ICML Workshop on Graph Representation Learning and Beyond (GRL+)*. [TGN]

Rudin, C. (2019). Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead. *Nature Machine Intelligence*, 1(5), 206-215.

Tishby, N., Pereira, F. C., & Bialek, W. (2000). The Information Bottleneck Method. *arXiv:physics/0004057*. [orig. Proc. 37th Allerton Conf. on Communication, Control and Computing, 1999]

Trivedi, R., Farajtabar, M., Biswal, P., & Zha, H. (2019). DyRep: Learning Representations over Dynamic Graphs. *International Conference on Learning Representations (ICLR)*. [DyRep]

Wang, L., Chang, X., Li, S., Chu, Y., Li, H., Zhang, W., He, X., Song, L., Zhou, J., & Yang, H. (2021b). TCL: Transformer-based Dynamic Graph Modelling via Contrastive Learning. *arXiv:2105.07944*. [TCL]

Wang, Y., Chang, Y.-Y., Liu, Y., Leskovec, J., & Li, P. (2021). Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks. *International Conference on Learning Representations (ICLR)*. [CAWN]

Welford, B. P. (1962). Note on a Method for Calculating Corrected Sums of Squares and Products. *Technometrics*, 4(3), 419-420.

Xu, D., Ruan, C., Korpeoglu, E., Kumar, S., & Achan, K. (2020). Inductive Representation Learning on Temporal Graphs. *International Conference on Learning Representations (ICLR)*. [TGAT]

Ying, R., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019). GNNExplainer: Generating Explanations for Graph Neural Networks. *Advances in Neural Information Processing Systems (NeurIPS) 32*, pp. 9240-9251.

Yu, L., Sun, L., Du, B., & Lv, W. (2023). Towards Better Dynamic Graph Learning: New Architecture and Unified Library. *Advances in Neural Information Processing Systems (NeurIPS)*. [DyGFormer; DyGLib]

---

## Appendix A - Evidence provenance

Paths are given **explicitly per artifact** because they live in two trees: top-level run JSONs under `SR-GNN/experiments/results/` (and ``), and the v3.3 LAB artifacts (faithfulness `.npz`, counterfactual JSONs, the decoupling-invariants test + its JSON, the WC-conf summaries) under `SR-GNN/experiments/LAB/v3_3/` (with the model at `experiments/LAB/v3_3/models/sr_gnn_v3_3.py`). All cited files exist and verify at the paths below. Three-seed = {1, 7, 42} for all models.

- **Inductive-split characterization.** The inductive split is built on-the-fly (`experiments/train.py` L311-315: inductive nodes = test nodes − train∪val nodes; inductive edges = test edges touching ≥1 such node) on a chronological 70/15/15 split (`data/download.py` L124). *CoEdit* (`experiments/data/coedit.npz`, 80,000 events, 4,131 occurring nodes): **535 unseen** inductive-test nodes, **5,336 inductive-test edges**; directed per-pair repeat over 1,383 test pairs - min 1 / median 2 / mean 3.86 / p90 7 / max 156, **44.4% singletons, 55.6% repeated**. *Wikipedia* (`wikipedia.npz`, 157,469 events): 900 unseen nodes, 2,579 inductive edges; per-pair median 1 / mean 2.65 / max 134, **64.2% singletons**. *MOOC* (`mooc.npz`, 411,749 events): 217 unseen nodes, 5,151 inductive edges; per-pair median 1 / mean 1.73 / max 19, **60.7% singletons**. CoEdit is materially more recurrence-dominated than the two reference graphs (median 2 vs 1; 55.6% vs 36-39% repeated), flagged as a threat in §8.4. The directed-pair convention matches the AP code; undirected shifts CoEdit repeated to 59.0%.
- **RS-GNN config B, 3-seed:** `v3_3_coedit_ARM_B_publishable_3seed.json` (ind 0.9885, trans 0.9985); `v3_3_wikipedia_ARM_B_publishable_3seed.json` (ind 0.9959, trans 0.9993); `v3_3_mooc_ARM_B_publishable_3seed_rerun.json` (ind 0.9978, trans 0.9988).
- **Std convention (stated once, for all artifacts).** Some JSONs store a legacy `*_std` field as population std (÷n). **Every ± in this paper is the sample std (n−1)**, computed from the per-seed values in Table A2; where a stored field disagrees (e.g. DyGFormer's `summary` 0.0177 vs. the n−1 value 0.0217) the n−1 value is used. The body and tables carry the n−1 value directly.
- **Config-B value reconciliation (stated once).** Config B's CoEdit inductive AP appears at three slightly different values, each from a *self-consistent* run; we use each only within its own contrast and never mix them across a single delta. (1) **0.9885 ± 0.0035** - the publishable three-seed config-B run (`v3_3_coedit_ARM_B_publishable_3seed.json`); used in Tables 1-2 and the B-vs-C contrast. (2) **0.9899 ± 0.0016** - the knob-ablation fixed-stack baseline (`v3_3_coedit_knob_ablation_3seed.json`), which differs from (1) only in run-internal stack bookkeeping; all Table 3/Table 5 knob Δ's are computed *within* this run, so the +21.1 pp etc. are unaffected by the 0.0014 baseline offset. (3) **0.9876 ± 0.0030** - the five-seed promotion (§6.1); the headline magnitude. These differ by <0.25 pp (seed-set and run bookkeeping), and each delta in the paper is taken inside one run, so no comparison crosses two B baselines.

  **Table A2 - Per-seed AP for every cell of Tables 1-2 (seeds {1, 7, 42}).** Each entry is `ind-AP / trans-AP`; the table mean ± n−1 std is recomputed from these three numbers, so any reader can reproduce both without trusting the override note.

  | Model | CoEdit (s1 / s7 / s42) | Wikipedia (s1 / s7 / s42) | MOOC (s1 / s7 / s42) |
  |---|---|---|---|
  | RS-GNN (B) | 0.9887/0.9985 · 0.9920/0.9989 · 0.9850/0.9981 | 0.9947/0.9994 · 0.9974/0.9995 · 0.9955/0.9991 | 0.9983/0.9990 · 0.9963/0.9985 · 0.9988/0.9989 |
  | JODIE | 0.7429/0.9450 · 0.9213/0.9882 · 0.7797/0.9640 | 0.9835/0.9954 · 0.9853/0.9945 · 0.9892/0.9965 | 0.9931/0.9943 · 0.9912/0.9911 · 0.9860/0.9897 |
  | TGAT | 0.8516/0.8632 · 0.8536/0.8749 · 0.8538/0.8688 | 0.9974/0.6331 · 0.9973/0.6712 · 0.9995/0.6691 | 0.9808/0.6767 · 0.9701/0.5904 · 0.9701/0.6408 |
  | CAWN | 0.8331/0.8680 · 0.7687/0.8936 · 0.7459/0.8792 | 0.9805/0.9841 · 0.9906/0.9872 · 0.9920/0.9870 | 0.5431/0.8696 · 0.9795/0.9863 · 0.9076/0.9747 |
  | TGN | 0.6413/0.9465 · 0.6283/0.9326 · 0.6351/0.9467 | 0.8859/0.9142 · 0.8942/0.9157 · 0.8109/0.9077 | 0.9760/0.9881 · 0.9828/0.9879 · 0.9866/0.9867 |
  | DyRep | 0.6124/0.9296 · 0.6351/0.9299 · 0.6179/0.9286 | 0.5936/0.8763 · 0.6946/0.8892 · 0.6061/0.8858 | 0.4705/0.8878 · 0.8901/0.9615 · 0.9844/0.9833 |
  | GraphMixer | 0.5986/0.7206 · 0.6231/0.7622 · 0.6479/0.7594 | 0.7566/0.8743 · 0.8040/0.8740 · 0.6535/0.7430 | 0.9797/0.9684 · 0.9496/0.9778 · 0.9913/0.9763 |
  | DyGFormer | 0.6161/0.8065 · 0.6314/0.8104 · 0.5886/0.8069 | 0.7619/0.8933 · 0.7705/0.8925 · 0.8253/0.9067 | -ᵃ |
  | EdgeBank-∞ | 0.5896/0.6229 · 0.5901/0.6228 · 0.5901/0.6231 | 0.6541/0.8172 · 0.6541/0.8169 · 0.6541/0.8170 | 0.5529/0.6292 · 0.5537/0.6293 · 0.5535/0.6288 |
  | EdgeBank-tw | 0.5890/0.5553 · 0.5896/0.5553 · 0.5897/0.5552 | 0.6535/0.6532 · 0.6535/0.6531 · 0.6535/0.6531 | 0.5529/0.5879 · 0.5537/0.5880 · 0.5535/0.5877 |

  ᵃ MOOC DyGFormer has only a single serialized seed (s42 ind 0.9770 / trans 0.9801); it is omitted from Tables 1-2 rather than shown as a one-seed cell (§8.2).
- **CoEdit benchmark (introduced here).** Built by `experiments/data/build_coedit.py` → `coedit.npz`. *Provenance:* derived from the public Wikipedia edit stream; an edge $(u_1,u_2,t)$ is emitted when users $u_1,u_2$ edit the *same* page within a 60-minute window (a standard collaboration-network construction), yielding a **non-bipartite user-user** temporal graph (both endpoints carry a lifecycle). *Size:* **80,000 co-edit events, 8,227 nodes, 172-dim edge features**, chronological 70/15/15 split, same leak-audited negative pool as Wikipedia/MOOC. The stream + splits ship with the code.
- **Baselines, B-protocol, 3-seed {1,7,42} (used in Tables 1-2):** `baselines_coedit_Bprotocol.json`, `baselines_wikipedia_Bprotocol.json`, `baselines_mooc_Bprotocol.json`. The MOOC baselines were switched from `baselines_mooc.json` (seeds {7,42,123}) to the matched `baselines_mooc_Bprotocol.json` (seeds {1,7,42}) so every model in Tables 1-2 uses the *same* seed set as RS-GNN (resolving the prior TGAT-MOOC mismatch: trans 0.6360 vs 0.6174, ind 0.9737 vs 0.9763).
- **Frontier + EdgeBank floor, 3-seed {1,7,42} (used in Tables 1-2):** `baselines_extra_coedit_Bprotocol.json`, `..._wikipedia_Bprotocol.json`, `..._mooc_Bprotocol.json`. **DyGFormer** (same harness): CoEdit ind **0.6120 ± 0.0217** (per-seed 0.5886/0.6161/0.6314), trans 0.8079 ± 0.0017; **Wikipedia** ind **0.7859 ± 0.0344** (per-seed 0.8253/0.7619/0.7705), trans 0.8975 ± 0.0080 - the harness-fairness control (§6.1). `edgebank_inf`: CoEdit ind 0.5899 ± 0.0003, trans 0.6229; Wikipedia ind 0.6541, trans 0.8170; MOOC ind 0.5534, trans 0.6291. `edgebank_tw`: CoEdit ind 0.5894, trans 0.5553; Wikipedia ind 0.6535, trans 0.6531; MOOC ind 0.5534, trans 0.5879 (EdgeBank stds round to 0.0000, deterministic floor). MOOC DyGFormer and TCL/NAT remain (§8.2).
- **Paired significance (n=3, paired $t$, $df=2$, $t_{0.975}=4.303$):** decoupling B−C mean +0.2213, std 0.0142 → 95% CI **[18.6, 25.6] pp** ($t=27.1$); CoEdit headline B−TGAT mean +0.1356, std 0.0038 → 95% CI **[12.6, 14.5] pp**. Per-seed diffs recomputed from the per-seed `ind_ap` fields of the cited JSONs.
- **Single-variable knob ablation (Table 3, three seeds {42,1,7}):** `experiments/results/v3_3_coedit_knob_ablation_3seed.json` - off one fixed config-B stack (`fsm_arch=v3`, `fsm_decode=hier`, `decol_hier_v2`, `causal_batch`, `hier_causal_policy`, `lambda_edge_trans=0.5`, `edge_h_detach_scorepath` true), one flag flipped per arm. `base_ind_ap_mean` = 0.9899. K1 `enable_main_predictor` False→True (with its required `p0_fix`): ind **0.7788 ± 0.0193** (Δ −21.11 pp), trans 0.9597. K2 `lfg_mode` soft→hard: 0.9868 ± 0.0014 (Δ −0.31 pp). K3 `compliance_floor` 0.05→0.0: 0.9889 ± 0.0018 (Δ −0.10 pp). C `design=correct` (all three): 0.7798 ± 0.0221 (Δ −21.01 pp). K1 alone ≈ full C-arm gap; K2/K3 inert. (Note: the knob-stack B baseline is 0.9899, separate from the publishable-config B 0.9885 used in Tables 1-2; the knob deltas are internal to this self-consistent run.)
- **Score-path detach probe (secondary, three seeds {42,1,7}):** `experiments/results/v3_3_coedit_detach_probe_{ON,OFF}_3seed.json` - toggling only `edge_h_detach_scorepath` off fixed config B: ON ind 0.9871 ± 0.0037 (per-seed 0.9832/0.9876/0.9905) vs OFF 0.9777 ± 0.0037 (0.9746/0.9767/0.9818) = +0.94 pp. The bulk of the mechanism is the main-prediction head (K1, +21.1 pp); this score-path detach is the secondary +0.94 pp.
- **Cross-dataset knob ablation (Table 5, three seeds {42,1,7}):** `experiments/results/v3_3_knob_ablation_wikipedia_3seed.json`, `..._mooc_3seed.json` - same `_knob_ablation_stddata_3seed.py` driver, config-B stack, arms B (`enable_main_predictor`=False, detached) vs K1 (=True, `p0_fix` on). Wikipedia B ind 0.9957 ± 0.0010 / trans 0.9993 vs K1 ind 0.9093 ± 0.0061 / trans 0.9430 → **Δind +8.64 pp**. MOOC B ind 0.9978 ± 0.0016 / trans 0.9988 vs K1 ind 0.9894 ± 0.0043 / trans 0.9883 → **Δind +0.85 pp**. CoEdit reference (same driver) +21.11 pp. B > K1 on all three datasets (mechanism not a CoEdit artifact; §6.5).
- **Hard-negative robustness (Table 6, B vs K1, three seeds {42,1,7}):** `experiments/results/hardneg/hardneg_B_vs_K1_wikipedia_3seed_v2.json`, `..._coedit_3seed_v2.json` - driver `_hardneg_B_vs_K1_3seed.py`; B and K1 scored on bit-identical paired negative sets per seed (fixed `hardneg_eval_seed`); all three strategies run on the same trained models. **Random:** Wikipedia Δ +9.6 pp (B 0.9968 vs K1 0.9011), CoEdit Δ +20.0 pp (0.9898 vs 0.7899). **Historical:** Wikipedia Δ **+41.6 pp** (B 0.9744 ± 0.0047 vs K1 0.5585 ± 0.0147), CoEdit Δ **+42.8 pp** (B 0.9573 ± 0.0031 vs K1 0.5294 ± 0.0276). **Inductive** (corrected test-phase-edge pool, non-degenerate): Wikipedia Δ **+42.6 pp** (B 0.9682 ± 0.0035 vs K1 0.5423 ± 0.0032), CoEdit Δ **+40.5 pp** (B 0.9625 ± 0.0024 vs K1 0.5579 ± 0.0215). Under both hard regimes the coupled arm collapses toward chance, the gap widens, ordering does not invert (§6.5). (The `_v2` JSONs supersede the first run, whose inductive-NS pool degenerated by restricting to unseen nodes.)
- **Identical-head detach toggle (Table 7, three seeds {42,1,7}):** `experiments/results/identhead/identhead_K1_{coedit,wikipedia}_3seed.json` - driver `_identhead_K1_3seed.py`; both arms set `enable_main_predictor=True` with the **same** 2-layer MLP scoring head (`torch.equal` on the head params, bit-identical init per seed), toggling only `main_predictor_detach` (the `.detach()` on the backbone→head path; CPU-verified zero vs nonzero backbone gradient). DETACHED-MLP vs COUPLED-MLP ind-AP: CoEdit 0.9357 ± 0.0032 vs 0.7969 ± 0.0367 (**Δ +13.9 pp**); Wikipedia 0.9784 ± 0.0019 vs 0.9061 ± 0.0003 (**Δ +7.2 pp**). Head held constant, so the inductive gap is the gradient flow, not the head architecture (§6.2).
- **Backbone-removed ablation (Table 8, three seeds {42,1,7}):** `experiments/results/backbone_removed/backbone_removed_{coedit,wikipedia}_3seed.json` - driver `_backbone_removed_3seed.py`, ctor flag `determ_only_backbone` (CSN/DRGC/ECTG frozen at init, `edge_h` zeroed to the scored head; CPU-verified 56 backbone params frozen, zero backbone gradient). FULL-B vs DETERM-ONLY ind-AP: CoEdit 0.9883 ± 0.0023 vs 0.9205 ± 0.0017 (**Δ +6.8 pp**); Wikipedia 0.9963 ± 0.0009 vs 0.8965 ± 0.0000 (**Δ +10.0 pp**). The learnable KL-trained backbone adds a real but minority margin; deterministic point-process features carry the bulk (§6.2).
- **Freeze-then-probe control (Table 4, CoEdit, three seeds {42,1,7}):** `v3_3_frozen_probe_ARM1_decoupling.json` (decouple-by-construction = config B) vs `v3_3_frozen_probe_ARM2_ftp.json` (freeze-then-probe). ARM1 CoEdit ind **0.9883 ± 0.0023** (per-seed 0.9855/0.9881/0.9912) / trans 0.9985; ARM2 ind **0.7684 ± 0.0052** (0.7758/0.7649/0.7646) / trans 0.9604 → **Δ +21.99 pp**. ARM2 phase-1 pretrains end-to-end, then freezes the backbone (weights byte-identical after the fresh head's `opt.step`, CPU-verified by ML) and re-probes. **Wikipedia is in the same JSONs:** ARM1 (decoupling) ind **0.9961 ± 0.0011** (per-seed 0.9949/0.9959/0.9976, n=3) / trans 0.9993; ARM2 (freeze-then-probe) ind **0.8975 ± 0.0152** (per-seed 0.8822/0.9127, n=2 seeds {42,1}) / trans 0.9373. The coupled end-to-end Wikipedia arm (K1, `enable_main_predictor`=True) is 0.9093 ± 0.0061 from `v3_3_knob_ablation_wikipedia_3seed.json` (Table 5). So on Wikipedia FtP 0.897 ≈ coupled 0.909 $\ll$ decoupled 0.996 - the same FtP ≈ coupled $\ll$ decoupled ordering as CoEdit (irreversibility confirmed on two datasets; Wikipedia gap 8.6 pp, CoEdit gap 21.1 pp). Aggregator `aggregate_frozen_probe_control.py` (mean ± n−1). The 0.768 CoEdit freeze-then-probe number ≈ the coupled five-seed C (0.7593): freezing after link-loss contamination does not recover inductive transfer (§6.2).
- **CoEdit five-seed headline (§6.1, seeds {1,7,42,2,3}):** `experiments/results/v3_3_coedit_{B,C}_5seed.json`, `baselines_coedit_TGAT_5seed.json` - B ind **0.9876 ± 0.0030** (per-seed 0.9850/0.9887/0.9920/0.9878/0.9846), C ind **0.7593 ± 0.0140** (0.7792/0.7639/0.7585/0.7539/0.7409), TGAT ind **0.8466 ± 0.0133** (0.8538/0.8516/0.8536/0.8230/0.8510). B−C = +22.8 pp, B−TGAT = +14.1 pp at n=5; idempotent driver `_promote_5seed_coedit.py` reuses the matched three-seed per-seed records and adds seeds {2,3}.
- **Trained existence-decoder weights (§4 ladder, three seeds {42,1,7}):** `experiments/results/cf_trained_theta_3seed.json` - per-seed $w=\mathrm{softplus}(\theta)$ after BCE training. Seed 42: DEATH 7.8e-5 / IDLE 0.060 / DECAY 0.310 / BIRTH 1.077 / REINFORCE 1.270. The partial order DEATH<IDLE<DECAY<{REINFORCE,BIRTH} holds on all three seeds (`ladder_detail_trained`: DEATH<IDLE, IDLE<DECAY, DECAY<REINFORCE, DECAY<BIRTH all True/seed); the init tie REINFORCE≈BIRTH is **broken** by training (REINFORCE largest; $|w_{\text{REINF}}-w_{\text{BIRTH}}|$ = 0.193/0.356/… per seed, `REINFORCE~BIRTH`=False). This serializes the trained ordering, answering the "existence weights hardcoded" objection (§4).
- **Decoupling ablation (B vs C, three seeds {42,1,7}):** B from `v3_3_coedit_ARM_B_publishable_3seed.json` (ind 0.9885 ± 0.0035); C from `v3_3_coedit_ARM_C_correct_3seed.json` (ind 0.7672 ± 0.0107, per-seed [0.7792, 0.7639, 0.7585], trans 0.9609; job 5503786). Δ(B−C) = +22.1 ± 1.4 pp, per-seed [0.206, 0.225, 0.234]. The single-variable decomposition of this gap is the knob ablation above (K1 alone = −21.1 pp). (Earlier seed-42 dumps: `v3_3_3arm_coedit_B_decoupled_s42.json` 0.9871, `v3_3_3arm_coedit_C_correct_s42.json` 0.7655.)
- **causal_batch A/B (full config B, three seeds):** ON from `v3_3_coedit_ARM_B_publishable_3seed.json` (ind 0.9885 ± 0.0035, trans 0.9985); OFF from `v3_3_coedit_B_causalOFF_3seed.json` (ind 0.9312 ± 0.0027, trans 0.9920; job 5503786). Δ = +5.7 ± 0.2 pp ind / +0.65 pp trans. (Earlier stripped single-seed A/B: `v3_3_causal_ab_coedit_cbON.json` 0.7907, `v3_3_causal_ab_coedit_cbOFF.json` 0.7462, job 5467100.)
- **hier_causal_policy A/B (three seeds {1,7,42}, job 5511229):** ON from `v3_3_coedit_ARM_B_publishable_3seed.json`, OFF from `v3_3_coedit_B_hcpOFF_3seed.json`. Per-seed inductive Δ(ON−OFF) = +1.04e-4 (s1) / +9.29e-4 (s7) / −1.53e-3 (s42); max $|\Delta_{\text{ind}}| = 1.5\mathrm{e}{-3}$, mean ≈ −1.7e-4, sign-mixed; max $|\Delta_{\text{trans}}| = 1.6\mathrm{e}{-4}$; vs. ±3.5e-3 ind seed std. AP-neutral within seed noise (not bit-identical: training-time RNG jitter). (Earlier seed-42 dump: `v3_3_hcp_coedit_ON_s42.json` 0.9871 vs `_OFF_s42.json` 0.9872, job 5471271.)
- **Counterfactual battery (three seeds {42,1,7}, config B / cbON):** `experiments/LAB/v3_3/fsm_intervene.py` on `faithfulness_coedit_v3_hier_hv2_let0.5_s{42,1,7}_cbON.npz` (N=12000 each; s1/s7 trained fresh as config-B cbON, job 5506704). The offline engine reconstructs the per-event SCM from dumped pre-update drivers and pushes forced states through the existence readout (no GPU/re-train/leak). Serialized in three JSONs:
  - `cf_trajectory_reversibility_3seed.json` (`dump_cf_trajectory_3seed.py`): `do_DEATH_frac_pedge_down`=**1.0** every seed; do(DEATH) mean Δ over full N=12000 = **−0.522 ± 0.001** (−0.5224/−0.5209/−0.5213), recurring subset −0.384 ± 0.003; `do_noop_delta`=0.0; `reversibility_max_abs_delta_pedge_after_undo`=0.000e+00 every seed; `decay_to_reinforce_flip_frac`=0.9999 ± 0.0002 (1.0/1.0/0.9996).
  - **`cf_updir_dose_faith_3seed.json` (`dump_cf_updir_dose_3seed.py`):** the **up-direction** fractions `do_REINFORCE_frac_pedge_up`=`do_BIRTH_frac_pedge_up`=**1.0 every seed**. *Provenance note:* the **down-direction** `do_DEATH_frac_pedge_down`=1.0 is cited from `cf_trajectory_reversibility_3seed.json` (its canonical home); this file also carries a `do_DEATH_frac_pedge_down` copy but §4 points to the trajectory file for the down value and to this file only for the up values, for clean provenance. the **dose-response regression slopes** rate→REINFORCE +0.017, rate→DEATH −0.057, slope→REINFORCE +0.284, true_occ→BIRTH −0.004 (sign-stable across all three seeds; claim is the sign); the existence-readout one-hot ladder `w=[IDLE 0.1, BIRTH 1.0, REINFORCE 1.0, DECAY 0.3, DEATH 0.0]`; and the faithfulness counts (hier DECAY-argmax 47.7% ± 0.4%, flat 0.18% ± 0.28%, Spearman ρ −0.61 ± 0.02, n_recurring 9157).
  - `cf_kill_REINFORCE_3seed.json`: REINFORCE→DEATH kill decomposition - isolated do(rate=dead)→DEATH 0.597 ± 0.028 (0.574/0.588/0.629), isolated do(staleness=high)→DEATH 0.482 ± 0.007 (0.477/0.479/0.489), all-drivers-dead 0.999 ± 0.002 (1.0/1.0/0.997).
  - **Existence-ladder note (effective init and trained weights).** Config B is *invoked* with `fix_existence_init=False` (the value logged in the run JSON), but the `correct_decoupled` preset forces it to `True` (`models/sr_gnn_v3_3.py` L692-693), so the **effective** init is the clean softplus-inverse $w=[0.1,1,1,0.3,0]$, then BCE-trained. The **trained** per-state weights $w=\mathrm{softplus}(\theta)$ are serialized per seed (`..._s{42,1,7}_theta.json`, aggregated in `cf_trained_theta_3seed.json`); Figure 2 plots them (3-seed mean$\pm$std) against this init prior. Since $\mathrm{do}(\text{state})$ moves $p_{\text{edge}}$ monotonically in $w_s$, the do(state) ordering is the sorted order of $w$ for **any** $\theta$ (the monotonicity argument in §4); the trained ordering and the 100% up/down fractions are exact/measured, and the init REINFORCE = BIRTH tie is broken by training ($|\Delta w|$ = 0.19/0.36/0.43 per seed).
  - **Forward-graph line provenance (Property 1 premise (i)).** $s^{\text{pos}}_{t+1}$ is finalized at `sr_gnn_v3_3.py` L1219-1222 (logits + soft log-mask + `ever_alive` gate + softmax) before the hierarchical branch (L1240+) writes `s_t1_cal`; the existence decoder reads `s_t1_pos` only (L1903/L1954). This descriptive line argument is *superseded* as the load-bearing evidence by the graph-level ancestry assert [A] below.
- **Cross-dataset lifecycle shape - single-seed, NON-EVIDENTIARY (Table A1).** *Seed-42 only, no error bars; no conclusion is drawn from it beyond "the readout is not a degenerate CoEdit clone." It is not part of any quantitative or significance claim; promote to ≥3 seeds before any evidentiary use (§8.2).* Sources (job 5506705): `faithfulness_coedit_v3_hier_hv2_let0.5_s42_cbON.npz`, `faithfulness_wikipedia_v3_hier_hv2_cb_let0.5_s42.npz`, `faithfulness_mooc_v3_hier_hv2_cb_let0.5_s42.npz`. Recurring-subset (`true_occ`≥2) argmax of `s_t1_cal`:

  **Table A1 - Lifecycle argmax distribution over active states (seed 42 only; single-seed sanity check).**

  | Dataset | REINFORCE | DECAY | DEATH | Shape |
  |---|---|---|---|---|
  | CoEdit | 0.35 | 0.62 | 0.02 | DECAY-heavy - slow fade, rare hard death |
  | Wikipedia | 0.42 | 0.43 | 0.16 | balanced full cycle through to DEATH (entropy 1.31) |
  | MOOC | 0.62 | 0.08 | 0.30 | BIRTH-heavy, transient - born, active, then dies |

  All three are non-degenerate and dataset-appropriate; the readout adapts to each domain's dynamics. This supports only the qualitative "not a degenerate, CoEdit-cloned readout" point (§6.5); the quantitative cross-dataset signal is the Wikipedia protocol-split asymmetry in Tables 1-2.
- **Confidence (WC-CONF grounded-init, 3-seed):** `wc_grnd/wc_conf_calib_grnd_coedit_s{42,1,7}_summary.json` - self-consistency AUC 0.9985±0.0015, external posMiss10 AUC 0.405±0.484 (sample std, n−1). Jobs 5503466/5503467.
- **Decoupling invariants (committed assert-based test, three asserts on one model instance):** `experiments/LAB/v3_3/verify_decoupling_invariants.py` → `decoupling_invariants_verify.json` (exists/verifies; all asserts PASS). On a config-B model over a 1500-edge CoEdit subsample:
  - **[G] (graph-ancestry, name-independent)** Under `pred_loss.backward()`, the set of parameters receiving nonzero gradient (`n_grad_recipients_under_pred_loss`) is asserted to be a **subset of the head set**: `max_backbone_grad_under_pred_loss`=**0.000e+00**, `nonzero_backbone_tensors=[]` (no non-head tensor receives scored-loss gradient). The assert now lets autograd decide which tensors are reached and only uses the name partition to define the permitted head destinations - so a true backbone tensor named like a head (e.g. a coupled-GRU `*gate*`) cannot be silently skipped (the earlier version iterated only over the heuristically-labeled backbone subset; corrected in `verify_decoupling_invariants.py`).
  - **[A] (new)** `max_cal_grad_under_scored_logit`=**0.000e+00** over `n_cal_only_tensors=12` cal-path heads (`hier_birth/alive/rising_head`), `nonzero_cal_tensors=[]` - backpropagating *from the scored logit* deposits zero gradient on every cal-path-only parameter, certifying at the graph level (single model instance) that $s^{\text{cal}}_{t+1}$ is not an ancestor of the scored logit (Property 1 premise (i)). This replaces the descriptive L1219/L1240 order-of-operations argument and is robust to refactoring; the assert exits non-zero if any cal head ever enters the scored graph.
  - **[S]** `max_abs_delta_pos_score_flat_vs_hier`=`..._neg_..`=**0.000e+00** on 225 scored positives - a *consistency check only* (it syncs two instances via `load_state_dict(strict=False)`, so [A], not [S], carries the claim).
  - `n_backbone_tensors=44` is a name-heuristic partition used only for descriptive reporting (an earlier in-code comment said "56 backbone tensors"; corrected to the emitted 44). The [G]/[A] graph-ancestry asserts, not this count, carry the guarantee. `all_pass=true`; the test exits non-zero on any violation (Property 1, §3.4).
- **Integrity audit:** 2026-06-06. **Anti-leak re-gate:** job 5450095.

**Figure list.** Figure 1 - decoded lifecycle trajectory of a real CoEdit pair (§3.3); Figure 2 - existence-counterfactual ordering ladder (§4); Figure 3 - faithful, falsifiable lifecycle rule, three views (§4); Figure 4 - intervene-able lifecycle readout, bidirectional control (§4); Figure 5 - causal-coherence signal by outcome (§5); Figure 6 - CoEdit inductive AP vs. baselines (§6.1); Figure 7 - cross-dataset result summary (§6.1); Figure 8 - decoupling ablation, CoEdit inductive AP (§6.2). Figures A1-A5 are the architecture schematics (Appendix C). Figures 6 and 8 are rendered from the three-seed values in Tables 1-2 and §6.2 (B 0.9885 / C 0.767 / A 0.928); Figures 3-4 from the counterfactual battery and trained existence-decoder weights (§4).

---

## Appendix B - Notation

Where the body uses code identifiers, they map to math symbols as follows; we use the math symbol in running text and reserve `monospace` for the actual flag/field name.

| Code identifier | Symbol / meaning |
|---|---|
| `edge_h` | $h_{uv}$ - detached per-pair backbone representation (input to Stream B) |
| `s_t1_pos` | $s^{\text{pos}}_{t+1}$ - *scored* next-state distribution (sole input to existence decoder; sets AP) |
| `s_t1_cal` | $s^{\text{cal}}_{t+1}$ - *interpretable* next-state distribution (de-collapse CE, faithfulness, counterfactuals; never scored) |
| `p_decay_cal` | $P(\text{DECAY})$ from the calibrated hierarchical tree |
| `slope_rel` | per-pair relative edit-rate slope (rising-vs-falling cadence driver) |
| `p_birth, p_alive, p_rising` | $p_{\text{birth}}, p_{\text{alive}}, p_{\text{rising}}\in[0,1]$ - hierarchical decode gates (§3.3) |
| `C_BAND_5` | $C_{\text{BAND-5}}$ - strict band-diagonal admissibility matrix ($|i-j|\le1$) |
| `causal_batch` (cbON/cbOFF) | read-before-write batched estimator on/off (§3.2) |
| `hier_causal_policy` (hcp) | soft causal-admissibility policy on $s^{\text{cal}}_{t+1}$ (§3.5) |
| `decol_hier_v2` | the de-collapse hierarchical decode refinement (§3.3) |
| `let0.5` | de-collapse target temperature 0.5 (config-B setting) |
| `true_occ` | per-pair observed occurrence count; "recurring subset" = `true_occ` ≥ 2 |
| `posMiss10` | external miss flag (positive ranked outside top-10) used to test $c_t$ (§5) |
| `fix_existence_init` | passed False but forced True by the `correct_decoupled` preset; effective init $w=[0.1,1,1,0.3,0]$, then BCE-trained (§4) |

**The interpolating flat readout.** The flat head referenced in §3.3/§6.4 does not expose five free logits; it scores the five ordered classes by interpolating a single cadence statistic across the BIRTH→REINFORCE→DECAY→DEATH axis, which is why DECAY (the interior class) is pinned out of argmax empirically. This is a property of *that* head, not of softmax in general (§3.3).

---

## Appendix C - Architecture schematics

These five schematics illustrate the architecture described in §3 and §5; they are diagrams, not results, and carry no numbers.

![](../figs/A1_two_stream_detach.png)

*Figure A1. Two-stream architecture. Backbone representation `edge_h` crosses a stop-gradient (detach wall) before the symbolic Stream B; no link-prediction gradient reaches the backbone (§3.1, §3.7).*

![](../figs/A2_gradient_decoupling.png)

*Figure A2. Three disjoint gradient routes: KL → backbone; BCE → scored head ($s^{\text{pos}}_{t+1}$); de-collapse CE → interpretable head ($s^{\text{cal}}_{t+1}$). The wall stops every Stream-B gradient at `edge_h.detach()` - zero backbone gradient (§3.7, verified §3.4).*

![](../figs/A3_hier_decode_tree.png)

*Figure A3. Decode tree: gates $p_{\text{birth}}/p_{\text{alive}}/p_{\text{rising}}$ factor the five states (§3.3).*

![](../figs/A4_lifecycle_fsm_band5.png)

*Figure A4. Admissibility band $C_{\text{BAND-5}}$: only adjacent transitions along the IDLE-BIRTH-REINFORCE-DECAY-DEATH axis are permitted; IDLE→DEATH is band-blocked (§3.5).*

![](../figs/A5_causal_confidence_overlay.png)

*Figure A5. The free next-state prediction overlaid against the walked belief $b_t$; their agreement is $c_t$. Off by default, byte-identical when off (§5).*
