import matplotlib.pyplot as plt
import numpy as np
import os
import random
import argparse
import threading
from torch.utils.tensorboard import SummaryWriter

from src.environment import IoVDummyEnv, IoVRedisEnv
from src.agent import DDQNAgent
from src.baselines import RandomAgent, GreedyComputeAgent, MinLatencyAgent, LeastQueueAgent, GreedyDistanceAgent, LocalAgent
from src.config import Config

# Agent names used in both redis and dummy training loops
REDIS_AGENTS  = ['ddqn', 'random', 'greedy_comp', 'min_latency', 'least_queue']
DUMMY_AGENTS  = ['ddqn', 'random', 'greedy_comp', 'greedy_dist']


def _run_redis_instance(instance_cfg):
    """Training loop for one RSU's DRL instance."""
    iid      = instance_cfg['instance_id']
    redis_db = instance_cfg['redis_db']
    rsu_id   = instance_cfg['rsu_id']

    log_dir    = os.path.join(Config.LOG_DIR, f"instance_{iid}")
    model_path = os.path.join(Config.BASE_DIR, "models", f"ddqn_rsu{iid}.pth")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    env    = IoVRedisEnv(redis_db=redis_db, instance_id=iid)

    ddqn_agent        = DDQNAgent()
    random_agent      = RandomAgent()
    greedy_comp_agent = GreedyComputeAgent()
    min_latency_agent = MinLatencyAgent()
    least_queue_agent = LeastQueueAgent()

    metrics         = {a: {"reward": [], "success": [], "latency": []} for a in REDIS_AGENTS}
    best_avg_reward = -float('inf')

    print(f"[DRL-{iid}] Starting training for {rsu_id} on Redis DB {redis_db} ...")

    for episode in range(Config.EPISODS):
        state = env.reset()
        mask  = env.get_action_mask()

        actions = {
            'ddqn':        ddqn_agent.select_action(state, mask=mask),
            'random':      random_agent.select_action(mask),
            'greedy_comp': greedy_comp_agent.select_action(env.candidates, env.rsus, mask),
            'min_latency': min_latency_agent.select_action(
                               env.candidates, env.rsus, mask, task_info=env.task_request),
            'least_queue': least_queue_agent.select_action(env.candidates, env.rsus, mask),
        }

        next_state, results = env.step_multi(actions)

        ddqn_reward, _ = results['ddqn']
        ddqn_agent.store_transition(state, actions['ddqn'], ddqn_reward, next_state, done=True)
        loss = ddqn_agent.train()
        ddqn_agent.update_target_network_soft()

        for agent_name, (reward, info) in results.items():
            metrics[agent_name]["reward"].append(reward)
            metrics[agent_name]["success"].append(info['success'])
            metrics[agent_name]["latency"].append(info['latency'])

        writer.add_scalars("Rewards",      {a: results[a][0]           for a in REDIS_AGENTS}, episode)
        writer.add_scalars("Success_Rate", {a: results[a][1]['success'] for a in REDIS_AGENTS}, episode)
        writer.add_scalars("Latency",      {a: results[a][1]['latency'] for a in REDIS_AGENTS}, episode)
        writer.add_scalar("Epsilon", ddqn_agent.epsilon, episode)
        if loss is not None:
            writer.add_scalar("Loss", loss, episode)

        if episode % 100 == 0:
            avg_ddqn = np.mean(metrics["ddqn"]["reward"][-100:])
            print(f"[DRL-{iid}] Ep {episode:5d} | DDQN: {avg_ddqn:.2f} | Eps: {ddqn_agent.epsilon:.3f}")
            if avg_ddqn > best_avg_reward and episode > 1000:
                best_avg_reward = avg_ddqn
                ddqn_agent.save_model(model_path)
                print(f"  >> [DRL-{iid}] New Best Model Saved! Avg Reward: {best_avg_reward:.2f}")

    writer.close()
    print(f"[DRL-{iid}] Training Complete. Logs: {log_dir}  Model: {model_path}")


def run():
    parser = argparse.ArgumentParser(description="Train Task Offloading DRL Agent")
    parser.add_argument('--env', type=str, default='dummy', choices=['dummy', 'redis'],
                        help='Environment type: dummy or redis')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(Config.PLOT_SAVE_PATH), exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    if args.env == 'redis':
        Config.load_config("redis_config.json")
        print("Initializing Redis Environment(s) ...")
        active = [i for i in Config.DRL_INSTANCES if i['active']]
        if not active:
            print("[ERROR] No active DRL instances in redis_config.json. Set at least one 'active': true.")
            return
        if len(active) == 1:
            _run_redis_instance(active[0])
        else:
            threads = [
                threading.Thread(target=_run_redis_instance, args=(i,), daemon=True)
                for i in active
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        print("All Redis DRL instances finished.")
        return

    # --- Dummy path ---
    Config.load_config("dummy_config.json")
    print("Initializing Dummy Environment...")
    env = IoVDummyEnv()

    writer      = SummaryWriter(log_dir=Config.LOG_DIR)
    agent_names = DUMMY_AGENTS

    ddqn_agent        = DDQNAgent()
    random_agent      = RandomAgent()
    greedy_comp_agent = GreedyComputeAgent()
    greedy_dist_agent = GreedyDistanceAgent()

    metrics         = {a: {"reward": [], "success": [], "latency": []} for a in agent_names}
    best_avg_reward = -float('inf')

    print(f"Starting Training on {Config.DEVICE}...")
    episodes = Config.EPISODS

    for episode in range(episodes):
        seed = Config.SEED + episode

        random.seed(seed); np.random.seed(seed)
        state = env.reset(); mask = env.get_action_mask()
        action = ddqn_agent.select_action(state, mask=mask)
        next_state, reward, done, info = env.step(action)
        ddqn_agent.store_transition(state, action, reward, next_state, done)
        loss = ddqn_agent.train()
        ddqn_agent.update_target_network_soft()
        metrics["ddqn"]["reward"].append(reward)
        metrics["ddqn"]["success"].append(info['success'])
        metrics["ddqn"]["latency"].append(info['latency'])

        random.seed(seed); np.random.seed(seed)
        _ = env.reset(); mask = env.get_action_mask()
        action = random_agent.select_action(mask)
        _, r_reward, _, r_info = env.step(action)
        metrics["random"]["reward"].append(r_reward)
        metrics["random"]["success"].append(r_info['success'])
        metrics["random"]["latency"].append(r_info['latency'])

        random.seed(seed); np.random.seed(seed)
        _ = env.reset(); mask = env.get_action_mask()
        action = greedy_comp_agent.select_action(env.candidates, env.active_rsu, mask)
        _, gc_reward, _, gc_info = env.step(action)
        metrics["greedy_comp"]["reward"].append(gc_reward)
        metrics["greedy_comp"]["success"].append(gc_info['success'])
        metrics["greedy_comp"]["latency"].append(gc_info['latency'])

        random.seed(seed); np.random.seed(seed)
        _ = env.reset(); mask = env.get_action_mask()
        action = greedy_dist_agent.select_action(env.candidates, env.active_rsu, mask)
        _, gd_reward, _, gd_info = env.step(action)
        metrics["greedy_dist"]["reward"].append(gd_reward)
        metrics["greedy_dist"]["success"].append(gd_info['success'])
        metrics["greedy_dist"]["latency"].append(gd_info['latency'])

        writer.add_scalars("Rewards", {
            "DDQN": reward, "Random": r_reward, "Greedy_Comp": gc_reward, "Greedy_Dist": gd_reward
        }, episode)
        writer.add_scalars("Success_Rate", {
            "DDQN": info['success'], "Random": r_info['success'],
            "Greedy_Comp": gc_info['success'], "Greedy_Dist": gd_info['success']
        }, episode)
        writer.add_scalars("Latency", {
            "DDQN": info['latency'], "Random": r_info['latency'],
            "Greedy_Comp": gc_info['latency'], "Greedy_Dist": gd_info['latency']
        }, episode)
        writer.add_scalar("Epsilon", ddqn_agent.epsilon, episode)
        if loss is not None:
            writer.add_scalar("Loss", loss, episode)

        if episode % 100 == 0:
            avg_ddqn = np.mean(metrics["ddqn"]["reward"][-100:])
            print(f"Ep {episode} | DDQN: {avg_ddqn:.2f} | Eps: {ddqn_agent.epsilon:.2f}")
            if avg_ddqn > best_avg_reward and episode > 1000:
                best_avg_reward = avg_ddqn
                ddqn_agent.save_model(Config.MODEL_SAVE_PATH)
                print(f"  >> New Best Model Saved! Avg Reward: {best_avg_reward:.2f}")

    writer.close()
    print(f"Training Complete. Logs saved to {Config.LOG_DIR}")
    print(f"DDQN Model saved to {Config.MODEL_SAVE_PATH}")

    # --- Plotting ---
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    def smooth(data, window=100):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    def plot_all(ax_idx, metric_key, title):
        ax[ax_idx].plot(smooth(metrics["ddqn"][metric_key]), label='DDQN (Ours)', color='blue', linewidth=2)
        ax[ax_idx].plot(smooth(metrics["greedy_comp"][metric_key]), label='Greedy-Comp', color='orange', linestyle='--')
        ax[ax_idx].plot(smooth(metrics["greedy_dist"][metric_key]), label='Greedy-Dist', color='green', linestyle=':')
        ax[ax_idx].plot(smooth(metrics["random"][metric_key]), label='Random', color='grey', alpha=0.3)
        ax[ax_idx].set_title(title)
        ax[ax_idx].legend()

    plot_all(0, "reward", "Average Reward")
    plot_all(1, "success", "Task Completion Rate")
    ax[1].set_ylim(0, 1.1)
    plot_all(2, "latency", "Latency (s)")

    plt.tight_layout()
    plt.savefig(Config.PLOT_SAVE_PATH)
    print(f"Plots saved to {Config.PLOT_SAVE_PATH}")
    plt.show()


if __name__ == "__main__":
    run()
