"""
VanillaDQNAgent — Standard DQN baseline for ablation comparison.

Used to justify the architectural contributions of the full Dueling DDQN + PER agent.
Differences from DDQNAgent:
  - Plain Q-Network (no dueling value/advantage streams)
  - Uniform replay buffer (no Prioritized Experience Replay)
  - Hard target network update every TARGET_UPDATE_FREQ steps (no soft update)

This is the standard DQN as described in:
  Mnih et al., "Human-level control through deep reinforcement learning",
  Nature, 2015.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from src.config import Config


class UniformReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        batch = list(zip(*samples))
        return (
            np.array(batch[0]),
            np.array(batch[1]),
            np.array(batch[2]),
            np.array(batch[3]),
            np.array(batch[4]),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Standard flat Q-Network (no dueling streams)."""

    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class VanillaDQNAgent:
    """
    Standard DQN agent (no dueling, no PER, hard target update).

    Exposes the same interface as DDQNAgent so it can be used as a
    drop-in replacement in main.py:
        select_action(state, mask=None, eval_mode=False)
        store_transition(state, action, reward, next_state, done)
        train()
        update_target_network_soft()   ← no-op here; update done internally
        save_model(path)
    """

    TARGET_UPDATE_FREQ = 100  # hard update every N training steps

    def __init__(self):
        self.policy_net = QNetwork(Config.STATE_DIM, Config.ACTION_DIM).to(Config.DEVICE)
        self.target_net = QNetwork(Config.STATE_DIM, Config.ACTION_DIM).to(Config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Config.LR)
        self.memory = UniformReplayBuffer(Config.MEMORY_SIZE)
        self.criterion = nn.SmoothL1Loss()

        self.epsilon = Config.EPSILON_START
        self._train_steps = 0

        # Expose beta as a no-op attribute so any code that checks it doesn't crash
        self.beta = 1.0

    def select_action(self, state, mask=None, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            if mask is not None:
                valid = np.where(mask == 1)[0]
                return int(np.random.choice(valid)) if len(valid) > 0 else 0
            return random.randint(0, Config.ACTION_DIM - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
            q_values = self.policy_net(state_t)
            if mask is not None:
                mask_t = torch.FloatTensor(mask).unsqueeze(0).to(Config.DEVICE)
                q_values = q_values.masked_fill(mask_t == 0, -1e9)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train(self):
        if len(self.memory) < Config.BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(Config.BATCH_SIZE)

        states      = torch.FloatTensor(states).to(Config.DEVICE)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(Config.DEVICE)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(Config.DEVICE)
        next_states = torch.FloatTensor(next_states).to(Config.DEVICE)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(Config.DEVICE)

        # Standard DQN target (greedy action from target net, not double-DQN)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
        expected_q = rewards + Config.GAMMA * max_next_q * (1 - dones)
        current_q  = self.policy_net(states).gather(1, actions)

        loss = self.criterion(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self._train_steps += 1
        # Hard target update
        if self._train_steps % self.TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(Config.EPSILON_END, self.epsilon * Config.EPSILON_DECAY)
        return loss.item()

    def update_target_network_soft(self):
        # No-op: VanillaDQN uses hard updates on a fixed schedule inside train()
        pass

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
