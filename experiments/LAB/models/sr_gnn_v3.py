"""
RS-GNN v3 — Multi-signal Edge State + (future) REACT.

This file implements STEP 1 of Master_ML.md roadmap:
  - EdgeStateStoreV3: 12-dim state vector with Welford stats + Hawkes λ + recurrence EWMA
  - ECTGv3: heuristic targets driven by z-score, Hawkes intensity, recurrence
  - SRGNN_v3: drop-in for srgnn_v2 in train.py

Future steps (REACT) will add:
  - EchoMemory with time-decay
  - Adaptive router (state-gated update)
  - Periodic Hopfield pass (post-batch)
  - JointProfile (pair affinity)

State vector layout (12 dims):
  [0:5]  state_logits  (IDLE, BIRTH, REINFORCE, DECAY, DEATH)
  [5]    recur          EWMA recurrence with time decay
  [6]    hawkes_lambda  self-exciting intensity
  [7]    mean_dt        Welford running mean of inter-arrival
  [8]    var_dt         Welford running variance
  [9]    n_obs          observation count (for Welford)
  [10]   vol            volatility (state-flip rate)
  [11]   life           cumulative log-time
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Tuple

from models.sr_gnn import (
    IDLE, BIRTH, REINFORCE, DECAY, DEATH,
    VALID_TRANSITIONS,
    TimeEncoder,
    NodeMemoryStore,
)
from models.sr_gnn_v2 import ResidualCSN, DRGC_v2, NSCP_v2

STATE_DIM = 12

# Multi-signal hyperparameters
RECUR_ALPHA   = 0.3      # EWMA weight for new event
RECUR_LAMBDA  = 0.001    # decay rate for forgetting
HAWKES_ALPHA  = 1.0      # jump magnitude on event
HAWKES_BETA   = 0.01     # decay rate of hawkes intensity
HAWKES_MU     = 0.1      # baseline rate


class EdgeStateStoreV3:
    """
    Sparse 12-dim edge state store with Welford online stats + Hawkes recursion.

    On each event (u,v,t,Δt):
      1. Update Welford (n, mean_dt, M2 for var)
      2. Update recurrence EWMA: r ← α·1 + (1-α)·r_prev·exp(-λ·Δt)
      3. Update hawkes_λ: λ ← α + (λ_prev - μ)·exp(-β·Δt) + μ
      4. (state_logits updated outside by ECTGv3)
    """
    def __init__(self, num_nodes: int, hidden: int, device: torch.device):
        self.N      = num_nodes
        self.device = device
        self._key_to_idx: Dict[int, int] = {}
        self._state_table: list = []   # list of 12-dim tensors

    def _make_init_state(self) -> Tensor:
        s = torch.zeros(STATE_DIM, device=self.device)
        s[IDLE]       = 1.0   # start IDLE
        s[6]          = HAWKES_MU  # hawkes_λ baseline
        return s

    def get_batch(self, src: Tensor, dst: Tensor) -> Tensor:
        B = src.size(0)
        states = torch.zeros(B, STATE_DIM, device=self.device)
        src_c = src.tolist()
        dst_c = dst.tolist()
        for i, (u, v) in enumerate(zip(src_c, dst_c)):
            key = u * self.N + v
            if key not in self._key_to_idx:
                idx = len(self._state_table)
                self._key_to_idx[key] = idx
                self._state_table.append(self._make_init_state())
            states[i] = self._state_table[self._key_to_idx[key]]
        return states

    def update_batch(self, src: Tensor, dst: Tensor, new_states: Tensor):
        src_c = src.tolist()
        dst_c = dst.tolist()
        ns = new_states.detach()
        for i, (u, v) in enumerate(zip(src_c, dst_c)):
            key = u * self.N + v
            if key in self._key_to_idx:
                self._state_table[self._key_to_idx[key]] = ns[i]

    def reset(self):
        self._key_to_idx.clear()
        self._state_table.clear()


def update_multisignal(prev_state: Tensor, dt: Tensor) -> Tensor:
    """
    Vectorised update of recur, hawkes_λ, Welford stats given Δt.
    Input: prev_state (B, 12), dt (B,)
    Output: updated state (B, 12) with [5:10] refreshed; [0:5], [10:12] kept as-is.
    """
    s = prev_state.clone()
    dt_f = dt.float().clamp(min=0.0)

    # (a) Recurrence EWMA with forgetting
    r_prev = s[:, 5]
    r_new  = RECUR_ALPHA + (1 - RECUR_ALPHA) * r_prev * torch.exp(-RECUR_LAMBDA * dt_f)
    s[:, 5] = r_new

    # (b) Hawkes self-exciting intensity (recursive form)
    lam_prev = s[:, 6]
    lam_new  = HAWKES_ALPHA + (lam_prev - HAWKES_MU) * torch.exp(-HAWKES_BETA * dt_f) + HAWKES_MU
    s[:, 6] = lam_new

    # (c) Welford online: mean_dt, var_dt, n
    n_prev    = s[:, 9]
    mean_prev = s[:, 7]
    M2_prev   = s[:, 8] * n_prev.clamp(min=1.0)   # store var; reconstruct M2

    n_new    = n_prev + 1.0
    delta    = dt_f - mean_prev
    mean_new = mean_prev + delta / n_new
    delta2   = dt_f - mean_new
    M2_new   = M2_prev + delta * delta2
    var_new  = M2_new / n_new.clamp(min=1.0)

    s[:, 7] = mean_new
    s[:, 8] = var_new
    s[:, 9] = n_new

    return s


class ECTGv3(nn.Module):
    """
    Multi-signal ECTG with z-score / Hawkes / recurrence-driven heuristic targets.
    """
    def __init__(self, feat_dim: int, hidden: int):
        super().__init__()
        in_dim = feat_dim + 1 + 5 + 3   # feat | log_dt | cur_dist | (recur, hawkes_λ, z)
        self.trans_net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 5)
        )
        self.intensity_fc = nn.Linear(1, 1)
        self.ctx_encoder  = nn.Linear(STATE_DIM, hidden)

    def forward(self, feat: Tensor, salience: Tensor,
                delta_t: Tensor, edge_states: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        edge_states: (B, 12)  raw state BEFORE this event update
        Returns:
          new_states (B, 12)
          edge_ctx (B, hidden)
          student_logits (B, 5)
          heuristic_target_dist (B, 5)
        """
        B      = feat.size(0)
        cur_dist = torch.softmax(edge_states[:, :5], dim=-1)
        log_dt   = torch.log1p(delta_t.float()).unsqueeze(-1)

        # 1) Update multi-signal stats (recur, hawkes, mean_dt, var_dt, n)
        s_updated = update_multisignal(edge_states, delta_t)
        recur      = s_updated[:, 5:6]
        hawkes_lam = s_updated[:, 6:7]
        mean_dt    = s_updated[:, 7:8]
        var_dt     = s_updated[:, 8:9]

        # 2) z-score (use updated mean/var)
        std_dt = torch.sqrt(var_dt.clamp(min=1e-6))
        z      = (delta_t.float().unsqueeze(-1) - mean_dt) / (std_dt + 1e-6)

        # 3) Transition network input
        trans_in = torch.cat([
            feat, log_dt, cur_dist,
            recur, torch.log1p(hawkes_lam), z.clamp(-5, 5),
        ], dim=-1)
        student_logits = self.trans_net(trans_in)

        # 4) Causal mask
        cur_idx = cur_dist.argmax(-1)
        mask    = VALID_TRANSITIONS[cur_idx.cpu()].to(feat.device)
        masked_logits = student_logits.masked_fill(~mask, -1e9)
        new_dist = torch.softmax(masked_logits, dim=-1)

        # 5) Update intensity (cumulative salience), volatility, life
        intensity_old = edge_states[:, 5:6]   # NOTE: dim 5 is recur in v3
        # We keep "intensity"-like behavior via hawkes_lam → not duplicating
        new_vol_in   = (new_dist.argmax(-1) != cur_idx).float().unsqueeze(-1)
        new_vol      = edge_states[:, 10:11] * 0.9 + new_vol_in * 0.1
        new_life     = edge_states[:, 11:12] + log_dt * 0.01

        new_states = torch.cat([
            new_dist,                  # [0:5]
            recur,                     # [5]
            hawkes_lam,                # [6]
            mean_dt, var_dt,           # [7], [8]
            s_updated[:, 9:10],        # [9] n_obs
            new_vol, new_life,         # [10], [11]
        ], dim=-1)
        edge_ctx = self.ctx_encoder(new_states)

        # 6) Heuristic target — multi-signal
        is_first  = (recur.squeeze(-1) < 0.05).float()
        is_active = torch.sigmoid(hawkes_lam.squeeze(-1) - 1.0)
        is_late   = torch.sigmoid(z.squeeze(-1) - 1.5)
        is_dead   = torch.sigmoid(z.squeeze(-1) - 3.0)

        target_logits = torch.zeros(B, 5, device=feat.device)
        target_logits[:, IDLE]      = (1 - is_active - is_first).clamp(min=-2, max=2)
        target_logits[:, BIRTH]     = is_first * 2.0
        target_logits[:, REINFORCE] = is_active * (1 - is_late) * 1.5
        target_logits[:, DECAY]     = is_late * (1 - is_dead) * 1.5
        target_logits[:, DEATH]     = is_dead * 2.0

        target_masked = target_logits.masked_fill(~mask, -1e9)
        heuristic_target_dist = torch.softmax(target_masked, dim=-1).detach()

        return new_states, edge_ctx, student_logits, heuristic_target_dist


class SRGNN_v3(nn.Module):
    """
    RS-GNN v3 (Step 1): multi-signal edge state.
    Drops in for srgnn_v2: same forward signature, same return dict.
    """
    def __init__(self, num_nodes: int, feat_dim: int, hidden: int = 128,
                 tip_beta: float = 0.001,
                 lambda_tip: float = 0.01, lambda_causal: float = 0.1,
                 lambda_trans: float = 0.05, lambda_distill: float = 0.0,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.num_nodes      = num_nodes
        self.feat_dim       = feat_dim
        self.hidden         = hidden
        self.lambda_tip     = lambda_tip
        self.lambda_causal  = lambda_causal
        self.lambda_trans   = lambda_trans
        self.lambda_distill = lambda_distill
        self.device         = device
        self._feat_in       = max(feat_dim, 1)

        self.csn  = ResidualCSN(self._feat_in, hidden)
        self.ectg = ECTGv3(self._feat_in, hidden)
        self.drgc = DRGC_v2(self._feat_in, hidden, tip_beta)
        self.nscp = NSCP_v2(hidden)

        # NSCP_v2 expects 8-dim edge_state; we'll pass [logits(5) | hawkes_lam | recur | vol] = 8 dims
        # state_oracle: predict 5-dim state from [h_src, h_dst]
        self.state_oracle = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 5),
        ).to(device)

        self.node_mem = NodeMemoryStore(num_nodes, hidden, device)
        self.edge_mem = EdgeStateStoreV3(num_nodes, hidden, device)

    def reset(self):
        self.node_mem.reset()
        self.edge_mem.reset()

    def _v3_to_v2_state(self, s12: Tensor) -> Tensor:
        """Adapt 12-dim v3 state to 8-dim format expected by NSCP_v2.
        NSCP_v2 uses dims [:5] as state_logits, [5] as intensity, [6] as vol, [7] as life.
        We map: hawkes_lam → intensity, vol → vol, life → life."""
        return torch.cat([
            s12[:, :5],         # state_logits
            s12[:, 6:7],        # hawkes_lam (proxy for intensity)
            s12[:, 10:11],      # vol
            s12[:, 11:12],      # life
        ], dim=-1)

    def forward(self, src: Tensor, dst: Tensor, t: Tensor,
                feat: Tensor, neg_dst: Tensor,
                rel_type: Optional[Tensor] = None) -> Dict[str, Tensor]:

        device = self.device
        B = src.size(0)

        if feat.shape[-1] == 0:
            feat = torch.zeros(B, 1, device=device)
        elif feat.shape[-1] < self._feat_in:
            feat = F.pad(feat, (0, self._feat_in - feat.shape[-1]))

        # L1 — Residual CSN
        dt_src = self.node_mem.delta_t(src, t)
        feat_g, sal = self.csn(feat, dt_src)

        # L2 — ECTG v3 (multi-signal)
        edge_st = self.edge_mem.get_batch(src, dst)
        new_est, edge_ctx, student_logits, heuristic_target_dist = self.ectg(
            feat_g, sal, dt_src, edge_st
        )
        self.edge_mem.update_batch(src, dst, new_est)

        # Adapt for NSCP_v2 (still expects 8-dim)
        edge_state_v2 = self._v3_to_v2_state(new_est)
        intensity = edge_state_v2[:, 5]   # hawkes_lam used as intensity proxy

        # L3 — DRGC v2 + TIP
        h_src = self.node_mem.get(src)
        h_dst = self.node_mem.get(dst)
        dt_dst = self.node_mem.delta_t(dst, t)
        all_idx = torch.unique(torch.cat([src, dst]))
        all_h = self.node_mem.get(all_idx)
        all_staleness = (t.max().float() - self.node_mem.last_t[all_idx]).clamp(0)

        new_h_src, new_h_dst, parsed_h, kl = self.drgc(
            h_src, h_dst, feat_g, edge_ctx, dt_src, dt_dst,
            intensity, all_h, all_staleness
        )

        # L4 — StateOracle + NSCP_v2
        teacher_logits = self.state_oracle(torch.cat([new_h_src, new_h_dst], dim=-1))
        neg_emb = self.node_mem.get(neg_dst)
        pos_sc, neg_sc, c_loss, ccs = self.nscp(
            new_h_src, new_h_dst, neg_emb, edge_state_v2, teacher_logits
        )

        # Memory update (after scoring)
        self.node_mem.set(all_idx, parsed_h)
        self.node_mem.set(src, new_h_src)
        self.node_mem.set(dst, new_h_dst)
        self.node_mem.update_time(torch.cat([src, dst]), torch.cat([t, t]))

        # Loss
        pred_loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_sc, neg_sc]),
            torch.cat([torch.ones(B, device=device), torch.zeros(B, device=device)])
        )

        # Hybrid trans loss (heuristic + optional distill)
        log_s_student = F.log_softmax(student_logits, dim=-1)
        loss_heuristic = F.kl_div(log_s_student, heuristic_target_dist, reduction='batchmean')
        if self.lambda_distill > 0:
            s_teacher = F.softmax(teacher_logits, dim=-1).detach()
            loss_predictive = F.kl_div(log_s_student, s_teacher, reduction='batchmean')
            trans_loss = (1 - self.lambda_distill) * loss_heuristic + self.lambda_distill * loss_predictive
        else:
            trans_loss = loss_heuristic

        total = (pred_loss
                 + self.lambda_tip * kl
                 + self.lambda_causal * c_loss
                 + self.lambda_trans * trans_loss)

        return {
            "pos_score": pos_sc, "neg_score": neg_sc,
            "loss": total, "pred_loss": pred_loss,
            "tip_loss": kl, "causal_loss": c_loss,
            "trans_loss": trans_loss,
            "ccs": ccs.mean(), "salience": sal.mean(),
        }
