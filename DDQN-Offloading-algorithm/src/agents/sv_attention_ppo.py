"""
SVAttentionPPOAgent — Cross-Attention PPO for IoV Task Offloading.

Architecture:
  Task features (query) attend over the joint set of all 8 candidate nodes
  [RSU_0, RSU_1, RSU_2, SV_0..SV_4] (keys/values). This is principled because:
    • Task features are pure demand (CPU cycles, deadline, QoS) — a clean query.
    • All 8 nodes are pure supply candidates — both RSUs and SVs are valid actions.
    • The token order matches the action index layout exactly, so attention weights
      α_i over node i directly approximate "relevance of action i for this task".
    • Permutation-invariant: SV re-sorting by distance has no effect.

Why NOT task+RSU as query (original design that was corrected):
  If RSU features appear in both the query AND the KV set, RSU tokens attend to
  their own features in the query (self-referential loop → noisy weights). Using
  task-only as query avoids this and is validated by Chen et al. (IEEE TVT 2023).

Padding mask:
  The full action mask (shape [ACTION_DIM=8]) is stored per transition in the
  AttentionRolloutBuffer as node_pad_mask = (mask == 0). This is passed to MHA
  as key_padding_mask during both inference and all PPO training epochs, ensuring
  absent SVs / offline RSUs are never attended to.

References:
  Chen et al., "Multi-Head Attention-Based DRL for Dynamic Task Offloading in
                Vehicular Networks", IEEE TVT, 2023 — task-query over all nodes.
  GAPO: "Graph Attention-Based PPO for VEC", Sensors, 2025.
  Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.config import Config
from src.agents.ppo import (
    PPO_GAMMA, PPO_GAE_LAMBDA, PPO_CLIP_EPS,
    PPO_VALUE_COEF, PPO_ENTROPY_INIT, PPO_ENTROPY_MIN,
    PPO_MAX_GRAD_NORM, PPO_EPOCHS, PPO_MINI_BATCH,
)

# Phase 3 uses a slightly longer rollout for more stable attention gradients
ATTN_PPO_ROLLOUT_LEN   = 256
ATTN_PPO_ACTOR_LR      = 3e-4
ATTN_PPO_CRITIC_LR     = 1e-3
ATTN_PPO_ATTN_DIM      = 64
ATTN_PPO_NUM_HEADS      = 4   # 64 / 4 = 16 head_dim


# ---------------------------------------------------------------------------
# Attention Rollout Buffer (stores node_pad_mask per transition)
# ---------------------------------------------------------------------------

class AttentionRolloutBuffer:
    """
    On-policy rollout buffer that additionally stores node_pad_mask per step.

    node_pad_mask[t] has shape [ACTION_DIM] bool, where True = action/node is
    absent or invalid.  This is used as key_padding_mask for MHA during PPO
    training epochs so that stale historical masks are preserved correctly.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity   = capacity
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self._ptr       = 0
        self._full      = False

        self.states        = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.actions       = np.zeros(capacity,               dtype=np.int64)
        self.log_probs     = np.zeros(capacity,               dtype=np.float32)
        self.rewards       = np.zeros(capacity,               dtype=np.float32)
        self.values        = np.zeros(capacity,               dtype=np.float32)
        self.dones         = np.zeros(capacity,               dtype=np.float32)
        self.node_pad_masks= np.zeros((capacity, action_dim), dtype=bool)

    def push(self, state: np.ndarray, action: int, log_prob: float,
             reward: float, value: float, done: bool,
             node_pad_mask: np.ndarray) -> None:
        idx = self._ptr % self.capacity
        self.states[idx]         = state
        self.actions[idx]        = action
        self.log_probs[idx]      = log_prob
        self.rewards[idx]        = reward
        self.values[idx]         = value
        self.dones[idx]          = float(done)
        self.node_pad_masks[idx] = node_pad_mask
        self._ptr += 1
        if self._ptr >= self.capacity:
            self._full = True

    def is_full(self) -> bool:
        return self._full

    def get(self, gamma: float = PPO_GAMMA,
            gae_lambda: float = PPO_GAE_LAMBDA) -> dict:
        T         = self.capacity
        values    = self.values.copy()
        rewards   = self.rewards
        dones     = self.dones

        advantages = np.zeros(T, dtype=np.float32)
        last_gae   = 0.0
        for t in reversed(range(T)):
            next_val  = 0.0 if bool(dones[t]) else (values[t + 1] if t + 1 < T else 0.0)
            delta     = rewards[t] + gamma * next_val - values[t]
            last_gae  = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values

        adv_mean = advantages.mean()
        adv_std  = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        return {
            'states':          self.states.copy(),
            'actions':         self.actions.copy(),
            'log_probs':       self.log_probs.copy(),
            'advantages':      advantages,
            'returns':         returns,
            'node_pad_masks':  self.node_pad_masks.copy(),
        }

    def clear(self) -> None:
        self._ptr  = 0
        self._full = False


# ---------------------------------------------------------------------------
# Cross-Attention Encoder
# ---------------------------------------------------------------------------

class CrossAttentionEncoder(nn.Module):
    """
    Task-conditioned cross-attention over all candidate nodes.

    State [B, 48] is split into:
      task_block  [B, 4]   → task_embed [B, attn_dim] (query)
      rsu_block   [B, 9]   → rsu_tokens [B, 3, attn_dim] (KV tokens 0-2)
      sv_block    [B, 35]  → sv_tokens  [B, 5, attn_dim] (KV tokens 3-7)

    Token order matches action indices:
      [RSU_0, RSU_1, RSU_2, SV_0, SV_1, SV_2, SV_3, SV_4]

    Cross-attention:
      query     = task_embed.unsqueeze(1)          [B, 1, D]
      all_nodes = cat([rsu_tokens, sv_tokens], 1)  [B, 8, D]
      MHA output [B, 1, D]

    Context = cat([task_embed, attn_out]) → shared MLP → [B, 128]
    """

    def __init__(self,
                 task_dim:     int = 4,
                 rsu_feat_dim: int = 3,
                 num_rsus:     int = 3,
                 sv_feat_dim:  int = 7,
                 num_svs:      int = 5,
                 attn_dim:     int = ATTN_PPO_ATTN_DIM,
                 num_heads:    int = ATTN_PPO_NUM_HEADS):
        super().__init__()
        assert attn_dim % num_heads == 0

        self._task_end = task_dim
        self._rsu_end  = task_dim + rsu_feat_dim * num_rsus   # 4 + 9 = 13
        self._num_rsus = num_rsus
        self._num_svs  = num_svs
        self._rsu_feat = rsu_feat_dim
        self._sv_feat  = sv_feat_dim

        # Task query projection
        self.task_proj = nn.Sequential(nn.Linear(task_dim, attn_dim), nn.ReLU())

        # Node token projections (separate for RSU vs SV due to different feat dims)
        self.rsu_proj = nn.Linear(rsu_feat_dim, attn_dim, bias=False)
        self.sv_proj  = nn.Linear(sv_feat_dim,  attn_dim, bias=False)

        # Cross-attention: task queries all 8 nodes
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = attn_dim,
            num_heads   = num_heads,
            batch_first = True,
            dropout     = 0.0,
        )

        # Shared MLP after context concatenation
        self.shared = nn.Sequential(
            nn.Linear(attn_dim * 2, 128),   # task_embed(D) + attn_out(D) = 2D
            nn.LayerNorm(128),
            nn.ReLU(),
        )

    def forward(self, state: torch.Tensor,
                node_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            state:         [B, STATE_DIM]
            node_pad_mask: [B, ACTION_DIM] bool, True = node absent/invalid

        Returns:
            context [B, 128]
        """
        B = state.shape[0]

        task_block = state[:, :self._task_end]              # [B, 4]
        rsu_block  = state[:, self._task_end:self._rsu_end] # [B, 9]
        sv_block   = state[:, self._rsu_end:]               # [B, 35]

        task_embed  = self.task_proj(task_block)             # [B, D]

        # Tokenise nodes: project each node's features to attn_dim
        rsu_tokens  = self.rsu_proj(
            rsu_block.view(B, self._num_rsus, self._rsu_feat)
        )                                                    # [B, 3, D]
        sv_tokens   = self.sv_proj(
            sv_block.view(B, self._num_svs, self._sv_feat)
        )                                                    # [B, 5, D]

        # Concatenate in action-index order: [RSU_0..2, SV_0..4]
        all_nodes   = torch.cat([rsu_tokens, sv_tokens], dim=1)  # [B, 8, D]

        query = task_embed.unsqueeze(1)                      # [B, 1, D]

        # Guard: if all nodes are masked, skip cross-attention
        if node_pad_mask is not None and bool(node_pad_mask.all()):
            attn_out = torch.zeros_like(query)
        else:
            attn_out, _ = self.cross_attn(
                query,
                all_nodes,
                all_nodes,
                key_padding_mask = node_pad_mask,
                need_weights     = False,
            )                                                # [B, 1, D]

        context = torch.cat(
            [task_embed, attn_out.squeeze(1)], dim=1
        )                                                    # [B, 2D=128]
        return self.shared(context)                          # [B, 128]


# ---------------------------------------------------------------------------
# Actor and Critic with Cross-Attention Encoder
# ---------------------------------------------------------------------------

class AttentionActor(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.encoder = CrossAttentionEncoder()
        self.head    = nn.Linear(128, action_dim)

    def forward(self, state: torch.Tensor,
                node_pad_mask: Optional[torch.Tensor] = None,
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        ctx    = self.encoder(state, node_pad_mask=node_pad_mask)  # [B, 128]
        logits = self.head(ctx)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)
        return logits


class AttentionCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CrossAttentionEncoder()
        self.head    = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor,
                node_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        ctx = self.encoder(state, node_pad_mask=node_pad_mask)   # [B, 128]
        return self.head(ctx)


# ---------------------------------------------------------------------------
# SVAttentionPPOAgent
# ---------------------------------------------------------------------------

class SVAttentionPPOAgent:
    """
    PPO with cross-attention encoder. Drop-in replacement for PPOAgent / DDQNAgent.
    """

    def __init__(self):
        self.actor  = AttentionActor(Config.ACTION_DIM).to(Config.DEVICE)
        self.critic = AttentionCritic().to(Config.DEVICE)

        self.actor_optimizer  = optim.Adam(
            self.actor.parameters(),  lr=ATTN_PPO_ACTOR_LR
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=ATTN_PPO_CRITIC_LR
        )

        self.rollout = AttentionRolloutBuffer(
            ATTN_PPO_ROLLOUT_LEN, Config.STATE_DIM, Config.ACTION_DIM
        )

        self.entropy_coef  = PPO_ENTROPY_INIT
        self._entropy_decay = (PPO_ENTROPY_INIT - PPO_ENTROPY_MIN) / (
            200 * ATTN_PPO_ROLLOUT_LEN
        )

        # Compatibility attributes
        self.epsilon     = 0.0
        self.beta        = 1.0
        self.global_step = 0

        # Temporaries set during select_action
        self._last_log_prob     = 0.0
        self._last_value        = 0.0
        self._last_node_pad_mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def _build_node_pad_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert action mask to attention key_padding_mask.
        Returns bool [ACTION_DIM], True = node is absent/invalid.
        """
        return (mask == 0).astype(bool)

    def _to_node_pad_tensor(self, mask_np: np.ndarray) -> torch.Tensor:
        """[ACTION_DIM] numpy bool → [1, ACTION_DIM] bool tensor on device."""
        return torch.BoolTensor(mask_np).unsqueeze(0).to(Config.DEVICE)

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray,
                      mask: Optional[np.ndarray] = None,
                      eval_mode: bool = False) -> int:
        """
        Sample (or greedily select) an action.
        Stores log_prob, value, and node_pad_mask for store_transition.
        """
        node_pad_np = (self._build_node_pad_mask(mask)
                       if mask is not None
                       else np.zeros(Config.ACTION_DIM, dtype=bool))
        self._last_node_pad_mask = node_pad_np

        with torch.no_grad():
            state_t    = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
            npm_t      = self._to_node_pad_tensor(node_pad_np)   # [1, 8]
            mask_t     = (torch.FloatTensor(mask).unsqueeze(0).to(Config.DEVICE)
                          if mask is not None else None)

            logits     = self.actor(state_t, node_pad_mask=npm_t, action_mask=mask_t)
            probs      = F.softmax(logits, dim=-1)
            dist       = torch.distributions.Categorical(probs)

            action     = logits.argmax(dim=-1) if eval_mode else dist.sample()

            self._last_log_prob = dist.log_prob(action).item()
            self._last_value    = self.critic(state_t, node_pad_mask=npm_t).item()

        return int(action.item())

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool) -> None:
        node_pad = (self._last_node_pad_mask
                    if self._last_node_pad_mask is not None
                    else np.zeros(Config.ACTION_DIM, dtype=bool))
        self.rollout.push(
            state, action,
            self._last_log_prob, reward, self._last_value, done,
            node_pad,
        )

    def train(self) -> Optional[float]:
        if not self.rollout.is_full():
            return None

        batch = self.rollout.get(gamma=PPO_GAMMA, gae_lambda=PPO_GAE_LAMBDA)

        states_np    = batch['states']
        actions_np   = batch['actions']
        old_lp_np    = batch['log_probs']
        advantages_np= batch['advantages']
        returns_np   = batch['returns']
        masks_np     = batch['node_pad_masks']   # [T, ACTION_DIM] bool

        T = len(states_np)
        last_loss = 0.0

        for _ in range(PPO_EPOCHS):
            idx = np.random.permutation(T)

            for start in range(0, T, PPO_MINI_BATCH):
                mb_idx = idx[start: start + PPO_MINI_BATCH]
                if len(mb_idx) < 2:
                    continue

                mb_states  = torch.FloatTensor(states_np[mb_idx]).to(Config.DEVICE)
                mb_actions = torch.LongTensor(actions_np[mb_idx]).to(Config.DEVICE)
                mb_old_lp  = torch.FloatTensor(old_lp_np[mb_idx]).to(Config.DEVICE)
                mb_advs    = torch.FloatTensor(advantages_np[mb_idx]).to(Config.DEVICE)
                mb_returns = torch.FloatTensor(returns_np[mb_idx]).to(Config.DEVICE)
                mb_npm     = torch.BoolTensor(masks_np[mb_idx]).to(Config.DEVICE)
                # mb_npm: [B, ACTION_DIM] — key_padding_mask for cross-attention

                # --- Actor loss ---
                logits  = self.actor(mb_states, node_pad_mask=mb_npm)
                probs   = F.softmax(logits, dim=-1)
                dist    = torch.distributions.Categorical(probs)
                new_lp  = dist.log_prob(mb_actions)
                entropy = dist.entropy()

                ratio  = torch.exp(new_lp - mb_old_lp)
                surr1  = ratio * mb_advs
                surr2  = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS,
                                     1.0 + PPO_CLIP_EPS) * mb_advs
                actor_loss = -torch.min(surr1, surr2).mean()

                # --- Critic loss ---
                values     = self.critic(mb_states, node_pad_mask=mb_npm).squeeze(-1)
                value_loss = PPO_VALUE_COEF * F.mse_loss(values, mb_returns)

                # --- Entropy bonus ---
                entropy_loss = -self.entropy_coef * entropy.mean()

                total_loss = actor_loss + value_loss + entropy_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    PPO_MAX_GRAD_NORM,
                )
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                last_loss = total_loss.item()

        self.entropy_coef = max(
            PPO_ENTROPY_MIN,
            self.entropy_coef - self._entropy_decay * ATTN_PPO_ROLLOUT_LEN,
        )

        self.rollout.clear()
        return last_loss

    def update_target_network_soft(self) -> None:
        """No-op — PPO has no target network."""
        pass

    def save_model(self, path: str, global_step: int = 0) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_state_dict':  self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'entropy_coef':      self.entropy_coef,
            'global_step':       global_step,
        }, path)

    def load_model(self, path: str) -> int:
        if not os.path.exists(path):
            print(f"[SVAttentionPPOAgent] Model not found at {path}, starting fresh.")
            return 0
        checkpoint = torch.load(path, map_location=Config.DEVICE)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.entropy_coef = checkpoint.get('entropy_coef', self.entropy_coef)
        step = checkpoint.get('global_step', 0)
        print(f"[SVAttentionPPOAgent] Loaded from {path} (step={step})")
        return step
