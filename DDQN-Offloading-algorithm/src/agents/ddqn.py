import os
"""
DDQNAgent — Deep Dueling Double Q-Network with Prioritized Experience Replay.

Architecture:
  - Dueling Q-Network: separate value and advantage streams
  - Double DQN: policy net selects action, target net evaluates it
  - Prioritized Experience Replay (PER): alpha=0.6, beta annealed 0.4→1.0

References:
  Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", ICML 2016.
  van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning", AAAI 2016.
  Schaul et al., "Prioritized Experience Replay", ICLR 2016.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from src.config import Config


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return None

        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = list(zip(*samples))
        states      = np.array(batch[0])
        actions     = np.array(batch[1])
        rewards     = np.array(batch[2])
        next_states = np.array(batch[3])
        dones       = np.array(batch[4])

        return states, actions, rewards, next_states, dones, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio + 1e-5


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingQNetwork, self).__init__()
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

    def forward(self, x):
        features   = self.feature_layer(x)
        values     = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))


class DDQNAgent:
    def __init__(self):
        self.policy_net = DuelingQNetwork(Config.STATE_DIM, Config.ACTION_DIM).to(Config.DEVICE)
        self.target_net = DuelingQNetwork(Config.STATE_DIM, Config.ACTION_DIM).to(Config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Config.LR)
        self.memory    = PrioritizedReplayBuffer(Config.MEMORY_SIZE, Config.PER_ALPHA)
        self.epsilon   = Config.EPSILON_START
        self.beta      = Config.PER_BETA
        self.criterion = nn.SmoothL1Loss(reduction='none')  # 'none' to handle IS weights manually

    def select_action(self, state, mask=None, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            if mask is not None:
                valid_indices = np.where(mask == 1)[0]
                return np.random.choice(valid_indices)
            return random.randint(0, Config.ACTION_DIM - 1)

        with torch.no_grad():
            state_t  = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
            q_values = self.policy_net(state_t)
            if mask is not None:
                mask_t   = torch.FloatTensor(mask).unsqueeze(0).to(Config.DEVICE)
                q_values = q_values.masked_fill(mask_t == 0, -1e9)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train(self):
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

        # Double DQN: policy net selects action, target net evaluates it
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

    def update_target_network_soft(self):
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                Config.TAU * local_param.data + (1.0 - Config.TAU) * target_param.data
            )

    def save_model(self, path, global_step=0):
        checkpoint = {
            'model_state_dict': self.policy_net.state_dict(),
            'epsilon': self.epsilon,
            'global_step': global_step
        }
        torch.save(checkpoint, path)

    def load_model(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=Config.DEVICE)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                step = checkpoint.get('global_step', 0)
                print(f"[DDQNAgent] Loaded checkpoint from {path} (Step: {step}, Epsilon: {self.epsilon:.3f})")
                return step
            else:
                self.policy_net.load_state_dict(checkpoint)
                self.target_net.load_state_dict(self.policy_net.state_dict())
                print(f"[DDQNAgent] Loaded legacy model from {path}")
                return 0
        else:
            print(f"[DDQNAgent] Model file not found at {path}, starting fresh.")
            return 0
