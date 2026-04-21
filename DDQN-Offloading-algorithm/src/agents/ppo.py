"""
PPOAgent — Proximal Policy Optimization with discrete Categorical policy.

Uses separate Actor and Critic networks (no weight sharing) with LayerNorm for
stability. On-policy rollout buffer accumulates ROLLOUT_LENGTH transitions before
performing K epochs of PPO updates, then clears the buffer.

Compatibility with the existing async Redis training loop in main.py:
  - select_action(state, mask) → int  (stores log_prob + value internally)
  - store_transition(s, a, r, s', done) → pushes to rollout buffer
  - train() → returns None until buffer full, then performs PPO update
  - update_target_network_soft() → no-op (PPO has no target network)
  - epsilon = 0.0, beta = 1.0  (compatibility attributes, not used)

Discrete action handling:
  Actor outputs raw logits → invalid actions masked to -1e9 → softmax →
  Categorical distribution → sample action. Log-prob and entropy are exact.

GAE note: in the Redis environment every transition has done=True (each task is
a single-step episode). GAE degenerates to A_t = r_t - V(s_t), and the critic
target equals the raw reward. Full GAE is implemented for generality.

References:
  Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347.
  "Delay and Battery Degradation Optimization based on PPO for Task Offloading
   in RSU-assisted IoV", IEEE WCNC 2024 — +11.24% latency over DDQN.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.config import Config


# ---------------------------------------------------------------------------
# Hyper-parameters (PPO-specific; separate from Config.DDQN values)
# ---------------------------------------------------------------------------

PPO_ACTOR_LR     = 3e-4
PPO_CRITIC_LR    = 1e-3
PPO_GAMMA        = 0.99
PPO_GAE_LAMBDA   = 0.95
PPO_CLIP_EPS     = 0.2
PPO_VALUE_COEF   = 0.5
PPO_ENTROPY_INIT = 0.01    # anneals toward PPO_ENTROPY_MIN
PPO_ENTROPY_MIN  = 0.001
PPO_MAX_GRAD_NORM = 0.5
PPO_ROLLOUT_LEN  = 128     # transitions between updates
PPO_EPOCHS       = 4       # SGD epochs per rollout
PPO_MINI_BATCH   = 32      # mini-batch size within each epoch


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Fixed-capacity on-policy rollout buffer.
    Stores (state, action, log_prob, reward, value, done) per step.
    is_full() returns True once ROLLOUT_LEN transitions have been pushed.
    Calling get() computes GAE advantages and returns numpy arrays.
    clear() must be called after each PPO update cycle.
    """

    def __init__(self, capacity: int, state_dim: int):
        self.capacity  = capacity
        self.state_dim = state_dim
        self._ptr      = 0
        self._full     = False

        self.states    = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions   = np.zeros(capacity,              dtype=np.int64)
        self.log_probs = np.zeros(capacity,              dtype=np.float32)
        self.rewards   = np.zeros(capacity,              dtype=np.float32)
        self.values    = np.zeros(capacity,              dtype=np.float32)
        self.dones     = np.zeros(capacity,              dtype=np.float32)

    def push(self, state: np.ndarray, action: int, log_prob: float,
             reward: float, value: float, done: bool) -> None:
        idx = self._ptr % self.capacity
        self.states[idx]    = state
        self.actions[idx]   = action
        self.log_probs[idx] = log_prob
        self.rewards[idx]   = reward
        self.values[idx]    = value
        self.dones[idx]     = float(done)
        self._ptr += 1
        if self._ptr >= self.capacity:
            self._full = True

    def is_full(self) -> bool:
        return self._full

    def get(self, gamma: float = PPO_GAMMA,
            gae_lambda: float = PPO_GAE_LAMBDA) -> dict:
        """
        Compute GAE advantages and returns over the stored rollout.
        Returns all arrays needed for PPO update.
        """
        T         = self.capacity
        values    = self.values.copy()    # [T]
        rewards   = self.rewards          # [T]
        dones     = self.dones            # [T]

        advantages = np.zeros(T, dtype=np.float32)
        last_gae   = 0.0
        for t in reversed(range(T)):
            # next value: 0.0 if terminal (done=1), else values[t+1] if available
            next_val  = 0.0 if bool(dones[t]) else (values[t + 1] if t + 1 < T else 0.0)
            delta     = rewards[t] + gamma * next_val - values[t]
            last_gae  = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values

        # Normalise advantages (zero-mean, unit-variance over the rollout)
        adv_mean = advantages.mean()
        adv_std  = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        return {
            'states':     self.states.copy(),
            'actions':    self.actions.copy(),
            'log_probs':  self.log_probs.copy(),
            'advantages': advantages,
            'returns':    returns,
        }

    def clear(self) -> None:
        self._ptr  = 0
        self._full = False


# ---------------------------------------------------------------------------
# Actor and Critic networks
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Discrete policy network: state → action logits."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns logits [B, action_dim] with invalid actions masked to -1e9."""
        logits = self.net(x)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        return logits


class Critic(nn.Module):
    """Value network: state → V(s)."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    PPO agent with Categorical policy for discrete IoV task offloading.
    Drop-in replacement for DDQNAgent in the main.py training loop.
    """

    def __init__(self):
        self.actor  = Actor(Config.STATE_DIM, Config.ACTION_DIM).to(Config.DEVICE)
        self.critic = Critic(Config.STATE_DIM).to(Config.DEVICE)

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=PPO_ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=PPO_CRITIC_LR)

        self.rollout = RolloutBuffer(PPO_ROLLOUT_LEN, Config.STATE_DIM)

        self.entropy_coef = PPO_ENTROPY_INIT
        self._entropy_decay = (PPO_ENTROPY_INIT - PPO_ENTROPY_MIN) / (
            # decay over ~200 rollout updates
            200 * PPO_ROLLOUT_LEN
        )

        # Compatibility attributes (main.py reads these for logging)
        self.epsilon     = 0.0
        self.beta        = 1.0
        self.global_step = 0

        # Temporaries: set during select_action, consumed in store_transition
        self._last_log_prob = 0.0
        self._last_value    = 0.0

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray,
                      mask: Optional[np.ndarray] = None,
                      eval_mode: bool = False) -> int:
        """
        Sample action from the policy (or take greedy action in eval mode).
        Stores log_prob and value estimate for the upcoming store_transition call.
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
            mask_t  = (torch.FloatTensor(mask).unsqueeze(0).to(Config.DEVICE)
                       if mask is not None else None)

            logits = self.actor(state_t, mask=mask_t)               # [1, A]
            probs  = F.softmax(logits, dim=-1)
            dist   = torch.distributions.Categorical(probs)

            if eval_mode:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()

            self._last_log_prob = dist.log_prob(action).item()
            self._last_value    = self.critic(state_t).item()

        return int(action.item())

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool) -> None:
        """Push one transition into the rollout buffer."""
        self.rollout.push(
            state, action,
            self._last_log_prob, reward, self._last_value, done,
        )

    def train(self) -> Optional[float]:
        """
        Returns None until the rollout buffer is full.
        When full: compute GAE, run PPO_EPOCHS × mini-batch updates, clear buffer.
        Returns mean total loss of the last epoch.
        """
        if not self.rollout.is_full():
            return None

        batch = self.rollout.get(gamma=PPO_GAMMA, gae_lambda=PPO_GAE_LAMBDA)

        states_np    = batch['states']     # [T, STATE_DIM]
        actions_np   = batch['actions']    # [T]
        old_lp_np    = batch['log_probs']  # [T]
        advantages_np= batch['advantages'] # [T]
        returns_np   = batch['returns']    # [T]

        T = len(states_np)
        last_loss = 0.0

        for _ in range(PPO_EPOCHS):
            # Shuffle indices for mini-batch sampling
            idx = np.random.permutation(T)

            for start in range(0, T, PPO_MINI_BATCH):
                mb_idx = idx[start: start + PPO_MINI_BATCH]
                if len(mb_idx) < 2:   # skip incomplete final mini-batch
                    continue

                mb_states  = torch.FloatTensor(states_np[mb_idx]).to(Config.DEVICE)
                mb_actions = torch.LongTensor(actions_np[mb_idx]).to(Config.DEVICE)
                mb_old_lp  = torch.FloatTensor(old_lp_np[mb_idx]).to(Config.DEVICE)
                mb_advs    = torch.FloatTensor(advantages_np[mb_idx]).to(Config.DEVICE)
                mb_returns = torch.FloatTensor(returns_np[mb_idx]).to(Config.DEVICE)

                # --- Actor loss (PPO clipped surrogate) ---
                logits  = self.actor(mb_states)                      # [B, A]
                probs   = F.softmax(logits, dim=-1)
                dist    = torch.distributions.Categorical(probs)
                new_lp  = dist.log_prob(mb_actions)                  # [B]
                entropy = dist.entropy()                             # [B]

                ratio  = torch.exp(new_lp - mb_old_lp)              # [B]
                surr1  = ratio * mb_advs
                surr2  = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS,
                                     1.0 + PPO_CLIP_EPS) * mb_advs
                actor_loss = -torch.min(surr1, surr2).mean()

                # --- Critic loss ---
                values     = self.critic(mb_states).squeeze(-1)      # [B]
                value_loss = PPO_VALUE_COEF * F.mse_loss(values, mb_returns)

                # --- Entropy bonus (encourages exploration) ---
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

        # Anneal entropy coefficient
        self.entropy_coef = max(
            PPO_ENTROPY_MIN,
            self.entropy_coef - self._entropy_decay * PPO_ROLLOUT_LEN,
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
            print(f"[PPOAgent] Model not found at {path}, starting fresh.")
            return 0
        checkpoint = torch.load(path, map_location=Config.DEVICE)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.entropy_coef = checkpoint.get('entropy_coef', self.entropy_coef)
        step = checkpoint.get('global_step', 0)
        print(f"[PPOAgent] Loaded from {path} (step={step})")
        return step
