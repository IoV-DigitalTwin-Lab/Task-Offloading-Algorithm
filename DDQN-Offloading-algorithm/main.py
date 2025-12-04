import matplotlib.pyplot as plt
import numpy as np
import os
import random
from torch.utils.tensorboard import SummaryWriter # TENSORBOARD

from src.environment import IoVDummyEnv
from src.agent import DDQNAgent
from src.baselines import GreedyAgent
from src.config import Config

def run():
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(Config.PLOT_SAVE_PATH), exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    writer = SummaryWriter(log_dir=Config.LOG_DIR)
    env = IoVDummyEnv()
    ddqn_agent = DDQNAgent()
    greedy_agent = GreedyAgent()
    
    history = {
        "ddqn_rewards": [],
        "greedy_rewards": [],
        "ddqn_success": [],
        "greedy_success": [],
        "ddqn_latency": [],
        "greedy_latency": []
    }
    
    print(f"Starting Training on {Config.DEVICE}...")
    episodes = 500
    
    for episode in range(episodes):
        # --- A. RUN DDQN ---
        seed = Config.SEED + episode
        random.seed(seed)
        np.random.seed(seed)
        
        state = env.reset() # Env uses the seed internally now
        action = ddqn_agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        # Train DDQN
        ddqn_agent.store_transition(state, action, reward, next_state, done)
        loss = ddqn_agent.train()
        
        history["ddqn_rewards"].append(reward)
        history["ddqn_success"].append(info['success'])
        history["ddqn_latency"].append(info['latency'])
        
        # Soft Update
        ddqn_agent.update_target_network_soft()

        # --- B. RUN GREEDY (Baseline) ---
        # Reset Env with SAME seed to ensure fair comparison
        random.seed(seed)
        np.random.seed(seed)
        _ = env.reset() # State is same as DDQN saw
        
        greedy_action = greedy_agent.select_action(env.candidates, env.rsu)
        _, g_reward, _, g_info = env.step(greedy_action)
        
        # --- LOGGING ---
        history["greedy_rewards"].append(g_reward)
        history["greedy_success"].append(g_info['success'])
        history["greedy_latency"].append(g_info['latency'])

        writer.add_scalar("Reward/DDQN", reward, episode)
        writer.add_scalar("Reward/Greedy", g_reward, episode)
        writer.add_scalar("Success_Rate/DDQN", info['success'], episode)
        writer.add_scalar("Success_Rate/Greedy", g_info['success'], episode)
        writer.add_scalar("Latency/DDQN", info['latency'], episode)
        writer.add_scalar("Latency/Greedy", g_info['latency'], episode)
        writer.add_scalar("Epsilon", ddqn_agent.epsilon, episode)
        if loss is not None:
            writer.add_scalar("Loss", loss, episode)
        
        if episode % 20 == 0:
            avg_ddqn = np.mean(history["ddqn_rewards"][-20:])
            avg_greedy = np.mean(history["greedy_rewards"][-20:])
            print(f"Ep {episode} | DDQN Rew: {avg_ddqn:.2f} | Greedy Rew: {avg_greedy:.2f} | Loss: {loss if loss else 0:.4f} | Epsilon: {ddqn_agent.epsilon:.2f}")

    ddqn_agent.save_model(Config.MODEL_SAVE_PATH)
    writer.close()
    print(f"Training Complete. Logs saved to {Config.LOG_DIR}")
    print(f"DDQN Model saved to {Config.MODEL_SAVE_PATH}")

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    def smooth(data, window=20):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # Plot 1: Rewards
    ax[0].plot(smooth(history["ddqn_rewards"]), label='DDQN (Ours)', color='blue')
    ax[0].plot(smooth(history["greedy_rewards"]), label='Greedy (Baseline)', color='orange', linestyle='--')
    ax[0].set_title("Average Reward")
    ax[0].set_xlabel("Episode")
    ax[0].legend()
    
    # Plot 2: Success Rate
    ax[1].plot(smooth(history["ddqn_success"], 50), label='DDQN', color='blue')
    ax[1].plot(smooth(history["greedy_success"], 50), label='Greedy', color='orange', linestyle='--')
    ax[1].set_title("Task Completion Rate")
    ax[1].set_ylim(0, 1.1)
    
    # Plot 3: Latency
    ax[2].plot(smooth(history["ddqn_latency"], 50), label='DDQN', color='blue')
    ax[2].plot(smooth(history["greedy_latency"], 50), label='Greedy', color='orange', linestyle='--')
    ax[2].set_title("Average Latency (Lower is Better)")
    
    plt.tight_layout()
    plt.savefig(Config.PLOT_SAVE_PATH)
    print(f"Comparison plots saved to {Config.PLOT_SAVE_PATH}")
    plt.show()

if __name__ == "__main__":
    run()