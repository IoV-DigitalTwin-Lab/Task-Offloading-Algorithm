import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from src.config import Config

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingQNetwork, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

class DDQNAgent:
    def __init__(self):
        self.policy_net = DuelingQNetwork(Config.STATE_DIM, Config.ACTION_DIM).to(Config.DEVICE)
        self.target_net = DuelingQNetwork(Config.STATE_DIM, Config.ACTION_DIM).to(Config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Config.LR)
        self.memory = deque(maxlen=Config.MEMORY_SIZE)
        self.epsilon = Config.EPSILON_START
        
    def select_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, Config.ACTION_DIM - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < Config.BATCH_SIZE:
            return None
        
        batch = random.sample(self.memory, Config.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(Config.DEVICE)
        actions = torch.LongTensor(actions).unsqueeze(1).to(Config.DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(Config.DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(Config.DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(Config.DEVICE)
        
        best_actions = self.policy_net(next_states).argmax(1, keepdim=True)
        target_q_values = self.target_net(next_states).gather(1, best_actions)
        
        expected_q = rewards + (Config.GAMMA * target_q_values * (1 - dones))
        current_q = self.policy_net(states).gather(1, actions)
        
        loss = nn.MSELoss()(current_q, expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(Config.EPSILON_END, self.epsilon * Config.EPSILON_DECAY)
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)