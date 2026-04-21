"""
DDQNAttentionAgent — DDQN + Multi-Head Self-Attention over Service Vehicle tokens.

Adds SVAttentionModule between the raw state vector and the dueling Q-streams.
The SV block (dims [task+rsu:end]) is reshaped into [B, num_svs, sv_feat_dim],
refined by MHSA with residual + LayerNorm, then flattened back to its original
size — so STATE_DIM is unchanged and the PER replay buffer requires no modification.

Design rationale for MHSA (not cross-attention) at this phase:
  SVs are re-sorted by distance every step, so the flat MLP sees different feature
  positions for the same physical vehicle across timesteps (permutation-sensitivity).
  MHSA is permutation-invariant and resolves this without altering the replay buffer.
  Cross-attention (task queries over all nodes) is reserved for Phase 3 (PPO) where
  the on-policy rollout buffer can store the padding mask per transition.

Reference:
  ATOQN: "Attention-Based Twin-Q Network for Vehicular Task Offloading",
          arXiv:2401.09134, 2024.
  Zhou et al.: "Attention-Augmented Double DQN in Mobile Edge Computing",
               IEEE Access, 2022.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from src.config import Config
from src.agents.ddqn import PrioritizedReplayBuffer  # identical buffer, no changes needed


# ---------------------------------------------------------------------------
# SV Attention Module
# ---------------------------------------------------------------------------

class SVAttentionModule(nn.Module):
    """
    Multi-head self-attention over the set of service vehicle feature tokens.

    Input SV block [B, num_svs * sv_feat_dim] is reshaped to [B, num_svs, sv_feat_dim],
    projected to attn_dim, refined by MHSA with pre/post LayerNorm and residual
    connection, then projected back to sv_feat_dim and flattened.

    The output shape equals the input shape, keeping STATE_DIM unchanged.
    Absent SV slots (zero-padded) are masked from attention via key_padding_mask.
    During training (no mask), zero-padded tokens produce near-zero projections
    (bias=False in_proj) and are naturally downweighted after convergence.
    """

    def __init__(self, num_svs: int, sv_feat_dim: int,
                 attn_dim: int = 32, num_heads: int = 4):
        super().__init__()
        assert attn_dim % num_heads == 0, (
            f"attn_dim ({attn_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_svs     = num_svs
        self.sv_feat_dim = sv_feat_dim

        # bias=False: zero-feature inputs (absent SVs) project to zero vectors
        self.in_proj   = nn.Linear(sv_feat_dim, attn_dim, bias=False)
        self.pre_norm  = nn.LayerNorm(attn_dim)
        self.attn      = nn.MultiheadAttention(
            embed_dim   = attn_dim,
            num_heads   = num_heads,
            batch_first = True,
            dropout     = 0.0,
        )
        self.post_norm = nn.LayerNorm(attn_dim)
        self.out_proj  = nn.Linear(attn_dim, sv_feat_dim, bias=False)

    def forward(self, sv_block: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            sv_block: [B, num_svs * sv_feat_dim]  raw SV features
            key_padding_mask: [B, num_svs] bool, True = absent/padded SV token

        Returns:
            [B, num_svs * sv_feat_dim] attention-refined SV features
        """
        B  = sv_block.shape[0]
        x  = sv_block.view(B, self.num_svs, self.sv_feat_dim)  # [B, S, F]
        xp = self.pre_norm(self.in_proj(x))                    # [B, S, D]

        # Guard: if every SV slot is masked (no SVs in range at all), skip MHA
        # to avoid NaN from softmax over fully-masked keys.
        if key_padding_mask is not None and bool(key_padding_mask.all()):
            return self.out_proj(xp).view(B, -1)

        attn_out, _ = self.attn(
            xp, xp, xp,
            key_padding_mask = key_padding_mask,
            need_weights     = False,
        )                                                       # [B, S, D]

        out = self.post_norm(attn_out + xp)                    # residual + norm [B, S, D]
        out = self.out_proj(out)                               # [B, S, F]
        return out.view(B, -1)                                 # [B, S*F]


# ---------------------------------------------------------------------------
# Dueling Q-Network with SV Attention
# ---------------------------------------------------------------------------

class DuelingAttentionQNetwork(nn.Module):
    """
    Dueling Q-network with SV-attention pre-processing.

    State is split into task / RSU / SV blocks using Config dimensions.
    SVAttentionModule refines the SV block in-place (same shape).
    The reconstructed 48-dim processed state feeds into standard dueling streams.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        # Block boundaries (frozen at construction; Config is loaded before agent init)
        self._task_end = Config.TASK_FEAT_DIM
        self._rsu_end  = Config.TASK_FEAT_DIM + Config.RSU_FEAT_DIM * Config.NUM_RSUS

        self.sv_attention = SVAttentionModule(
            num_svs     = Config.MAX_NEIGHBORS,
            sv_feat_dim = Config.VEHICLE_FEAT_DIM,
            attn_dim    = 32,
            num_heads   = 4,
        )

        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor,
                sv_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:           [B, STATE_DIM]
            sv_pad_mask: [B, num_svs] bool (optional; only passed at inference)

        Returns:
            Q-values [B, ACTION_DIM]
        """
        task_block = x[:, :self._task_end]            # [B, 4]
        rsu_block  = x[:, self._task_end:self._rsu_end]  # [B, 9]
        sv_block   = x[:, self._rsu_end:]              # [B, 35]

        sv_refined = self.sv_attention(sv_block, key_padding_mask=sv_pad_mask)  # [B, 35]
        state_in   = torch.cat([task_block, rsu_block, sv_refined], dim=1)      # [B, 48]

        features   = self.feature_layer(state_in)
        values     = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))


# ---------------------------------------------------------------------------
# DDQNAttentionAgent
# ---------------------------------------------------------------------------

class DDQNAttentionAgent:
    """
    DDQN with SV-attention. Identical interface to DDQNAgent.

    Only the network class differs. PER, Double DQN target updates, soft
    target sync, epsilon-greedy, and save/load are all identical to DDQNAgent.
    """

    def __init__(self):
        self.policy_net = DuelingAttentionQNetwork(
            Config.STATE_DIM, Config.ACTION_DIM
        ).to(Config.DEVICE)
        self.target_net = DuelingAttentionQNetwork(
            Config.STATE_DIM, Config.ACTION_DIM
        ).to(Config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Config.LR)
        self.memory    = PrioritizedReplayBuffer(Config.MEMORY_SIZE, Config.PER_ALPHA)
        self.epsilon   = Config.EPSILON_START
        self.beta      = Config.PER_BETA
        self.criterion = nn.SmoothL1Loss(reduction='none')

        self._num_rsus = Config.NUM_RSUS   # offset to extract SV slice from action mask

    # ------------------------------------------------------------------
    def _sv_pad_mask(self, mask: np.ndarray) -> torch.Tensor:
        """Build [1, num_svs] bool mask. True = SV absent / padded."""
        sv_valid = mask[self._num_rsus:]                              # [num_svs]
        return torch.BoolTensor(sv_valid == 0).unsqueeze(0).to(Config.DEVICE)

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray,
                      mask: Optional[np.ndarray] = None,
                      eval_mode: bool = False) -> int:
        if not eval_mode and random.random() < self.epsilon:
            if mask is not None:
                valid = np.where(mask == 1)[0]
                return int(np.random.choice(valid))
            return random.randint(0, Config.ACTION_DIM - 1)

        with torch.no_grad():
            state_t     = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
            sv_pad      = self._sv_pad_mask(mask) if mask is not None else None
            q_values    = self.policy_net(state_t, sv_pad_mask=sv_pad)
            if mask is not None:
                mask_t  = torch.FloatTensor(mask).unsqueeze(0).to(Config.DEVICE)
                q_values = q_values.masked_fill(mask_t == 0, -1e9)
            return int(q_values.argmax().item())

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool) -> None:
        self.memory.push(state, action, reward, next_state, done)

    def train(self) -> Optional[float]:
        if len(self.memory.buffer) < Config.BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample(Config.BATCH_SIZE, self.beta)

        states      = torch.FloatTensor(states).to(Config.DEVICE)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(Config.DEVICE)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(Config.DEVICE)
        next_states = torch.FloatTensor(next_states).to(Config.DEVICE)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(Config.DEVICE)
        weights     = torch.FloatTensor(weights).unsqueeze(1).to(Config.DEVICE)

        # Double DQN: policy selects action, target evaluates it.
        # sv_pad_mask=None during training (mask not stored in replay buffer).
        best_actions    = self.policy_net(next_states).argmax(1, keepdim=True)
        target_q_values = self.target_net(next_states).gather(1, best_actions)
        expected_q      = rewards + (Config.GAMMA * target_q_values * (1 - dones))
        current_q       = self.policy_net(states).gather(1, actions)

        loss_elementwise = self.criterion(current_q, expected_q)
        loss = (loss_elementwise * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        td_errors = loss_elementwise.detach().cpu().numpy().flatten()
        self.memory.update_priorities(indices, td_errors)

        self.epsilon = max(Config.EPSILON_END, self.epsilon * Config.EPSILON_DECAY)
        self.beta    = min(1.0, self.beta + 0.0001)
        return loss.item()

    def update_target_network_soft(self) -> None:
        for target_p, local_p in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_p.data.copy_(
                Config.TAU * local_p.data + (1.0 - Config.TAU) * target_p.data
            )

    def save_model(self, path: str, global_step: int = 0) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'epsilon':          self.epsilon,
            'global_step':      global_step,
        }, path)

    def load_model(self, path: str) -> int:
        if not os.path.exists(path):
            print(f"[DDQNAttentionAgent] Model not found at {path}, starting fresh.")
            return 0
        checkpoint = torch.load(path, map_location=Config.DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            step = checkpoint.get('global_step', 0)
            print(f"[DDQNAttentionAgent] Loaded from {path} "
                  f"(step={step}, ε={self.epsilon:.3f})")
            return step
        # Legacy: raw state dict
        self.policy_net.load_state_dict(checkpoint)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"[DDQNAttentionAgent] Loaded legacy model from {path}")
        return 0
