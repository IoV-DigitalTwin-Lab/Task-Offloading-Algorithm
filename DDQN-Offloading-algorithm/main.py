import matplotlib.pyplot as plt
import numpy as np
import os
import random
import argparse
import threading
import time
from torch.utils.tensorboard import SummaryWriter

from src.environment import IoVDummyEnv, IoVRedisEnv
from src.agent import DDQNAgent
from src.baselines import RandomAgent, GreedyComputeAgent, MinLatencyAgent, LeastQueueAgent, GreedyDistanceAgent
from src.config import Config

# Agent names used in both redis and dummy training loops
REDIS_AGENTS  = ['ddqn', 'random', 'greedy_comp', 'min_latency', 'least_queue']
DUMMY_AGENTS  = ['ddqn', 'random', 'greedy_comp', 'greedy_dist']


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _running_mean(data, window=100):
    """Running mean over the most recent `window` entries."""
    window_data = data[-window:] if len(data) >= window else data
    return float(np.mean(window_data)) if window_data else 0.0


def _decision_pct(decisions, dtype, window=50):
    """Fraction of decisions matching `dtype` over the most recent `window` entries."""
    recent = decisions[-window:] if len(decisions) >= window else decisions
    if not recent:
        return 0.0
    return sum(1 for d in recent if d == dtype) / len(recent)


def _action_to_dtype(action):
    """Map a dummy-env action index to a decision-type string."""
    if action == Config.MAX_NEIGHBORS:
        return "RSU"
    elif action < Config.MAX_NEIGHBORS:
        return "SERVICE_VEHICLE"
    return "LOCAL"


def _fail_reason_pct(reasons, target_reason, window=50):
    """Fraction of entries matching target_reason in the most recent `window` entries."""
    recent = reasons[-window:] if len(reasons) >= window else reasons
    if not recent:
        return 0.0
    return sum(1 for r in recent if r == target_reason) / len(recent)


def _qos_success_pct(qos_list, success_list, target_qos, window=100):
    """Success rate for a specific QoS level (1, 2, or 3) over recent `window` entries."""
    recent = list(zip(qos_list, success_list))[-window:]
    filtered = [s for q, s in recent if int(round(q)) == target_qos]
    return (sum(filtered) / len(filtered)) if filtered else 0.0


def _run_redis_instance(instance_cfg):
    """
    Async training loop for one RSU's DRL instance.

    Design:
      - Accept a task → infer all agents → write decisions → add to pending_dict
      - On every poll cycle, check ALL pending tasks for results (non-blocking)
      - When results arrive, compute reward and add to replay buffer
      - Batch-train periodically regardless of pending state

    This avoids blocking on each task's result before processing the next one,
    increasing throughput from ~0.5 tasks/s (serial) to many tasks in parallel.
    Result ordering does not affect training quality because DDQN uses a replay buffer
    (each (s, a, r, s') tuple is stored and sampled independently).
    """
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

    metrics         = {a: {"reward": [], "success": [], "latency": [], "energy": [],
                           "decision_type": [], "fail_reason": [], "qos": []}
                       for a in REDIS_AGENTS}
    best_avg_reward = -float('inf')

    # pending[task_id] = {state, actions, agent_decisions, task_request, timestamp}
    pending              = {}
    TIMEOUT              = Config.REDIS_RESULT_TIMEOUT  # seconds before discarding a pending task
    TRAIN_EVERY          = 4                            # batch-train after every N completions
    MAX_DRAIN_PER_CYCLE  = 20                           # max new tasks to ingest per poll cycle
    completions_since_train = 0
    timeout_count        = 0
    episode              = 0

    print(f"[DRL-{iid}] Starting ASYNC training for {rsu_id} on Redis DB {redis_db} ...")

    # ── Startup: wait until vehicle state keys appear in Redis ──────────────
    # Vehicles send heartbeats every ~1s; we need at least one state before training.
    # Without this wait, every task is skipped (state=None) and the simulator stalls.
    _wait_start = time.time()
    _MAX_STARTUP_WAIT = 40  # seconds
    print(f"[DRL-{iid}] Waiting for vehicle states in Redis (up to {_MAX_STARTUP_WAIT}s)...")
    while time.time() - _wait_start < _MAX_STARTUP_WAIT:
        keys = env.r.keys('vehicle:*:state')
        if keys:
            print(f"[DRL-{iid}] Found {len(keys)} vehicle state(s). Starting training loop.")
            break
        time.sleep(1.0)
    else:
        print(f"[DRL-{iid}] WARNING: no vehicle states after {_MAX_STARTUP_WAIT}s — starting anyway.")

    while episode < Config.EPISODS:

        # ── 1. Drain incoming tasks (non-blocking, up to MAX_DRAIN_PER_CYCLE) ──
        for _ in range(MAX_DRAIN_PER_CYCLE):
            request = env._poll_request_nonblocking()
            if request is None:
                break  # queue empty

            state = env.setup_from_request(request)
            if state is None:
                # Vehicle state still unavailable — write fallback RSU decision so the
                # simulator is not left waiting indefinitely for a decision that never arrives.
                pipe = env.r.pipeline()
                fallback_map = {'agents': ','.join(REDIS_AGENTS)}
                for _a in REDIS_AGENTS:
                    fallback_map[f'{_a}_type']   = 'RSU'
                    fallback_map[f'{_a}_target'] = request.get('rsu_id', env.rsu_ids[0])
                pipe.hset(f"task:{request['task_id']}:decisions", mapping=fallback_map)
                pipe.expire(f"task:{request['task_id']}:decisions", 300)
                pipe.execute()
                print(f"[DRL-{iid}] Task {request['task_id']}: vehicle state missing — fallback RSU decision written")
                continue

            mask = env.get_action_mask()
            actions = {
                'ddqn':        ddqn_agent.select_action(state, mask=mask),
                'random':      random_agent.select_action(mask),
                'greedy_comp': greedy_comp_agent.select_action(env.candidates, env.rsus, mask),
                'min_latency': min_latency_agent.select_action(
                                   env.candidates, env.rsus, mask, task_info=request),
                'least_queue': least_queue_agent.select_action(env.candidates, env.rsus, mask),
            }

            agent_decisions = env.write_decisions(request['task_id'], actions)
            pending[request['task_id']] = {
                'state':           state,
                'actions':         actions,
                'agent_decisions': agent_decisions,
                'task_request':    request,
                'timestamp':       time.time(),
            }
            print(f"[DRL-{iid}] Task {request['task_id']} dispatched "
                  f"({len(pending)} pending, ep={episode})")

        # ── 2. Check ALL pending tasks for results (non-blocking) ──
        for task_id in list(pending.keys()):
            entry = pending[task_id]

            # Discard timed-out entries
            if time.time() - entry['timestamp'] > TIMEOUT:
                print(f"[DRL-{iid}] Task {task_id} timed out — discarding")
                pending.pop(task_id)
                timeout_count += 1
                continue

            agent_names = list(entry['actions'].keys())
            results_raw = env.check_results_nonblocking(task_id, agent_names)
            if results_raw is None:
                continue  # not ready yet — check again next cycle

            pending.pop(task_id)

            # done=True for every task-episode: next_state is unused in the target computation
            next_state = entry['state']

            for agent_name in agent_names:
                result = results_raw[agent_name]
                dtype, tid = entry['agent_decisions'][agent_name]
                reward, info = env.compute_reward_for(entry['task_request'], result, dtype, tid)

                metrics[agent_name]["reward"].append(reward)
                metrics[agent_name]["success"].append(info['success'])
                metrics[agent_name]["latency"].append(info['latency'])
                metrics[agent_name]["energy"].append(info.get('energy', 0.0))
                metrics[agent_name]["decision_type"].append(info.get('decision_type', 'RSU'))
                metrics[agent_name]["fail_reason"].append(info.get('fail_reason', 'NONE'))
                metrics[agent_name]["qos"].append(entry['task_request'].get('qos', 1.0))

                if agent_name == 'ddqn':
                    ddqn_agent.store_transition(
                        entry['state'], entry['actions']['ddqn'],
                        reward, next_state, done=True
                    )

            completions_since_train += 1
            episode += 1

            # TensorBoard — log only agents that have at least one data point
            active = [a for a in REDIS_AGENTS if metrics[a]["reward"]]
            if active:
                writer.add_scalars("Rewards",              {a: metrics[a]["reward"][-1]                                        for a in active}, episode)
                writer.add_scalars("Rewards_Smoothed",     {a: _running_mean(metrics[a]["reward"])                             for a in active}, episode)
                writer.add_scalars("Success_Rate",         {a: metrics[a]["success"][-1]                                       for a in active}, episode)
                writer.add_scalars("Latency",              {a: metrics[a]["latency"][-1]                                       for a in active}, episode)
                writer.add_scalars("Energy",               {a: metrics[a]["energy"][-1]                                        for a in active}, episode)
                writer.add_scalars("Decision_RSU_Pct",     {a: _decision_pct(metrics[a]["decision_type"], "RSU")               for a in active}, episode)
                writer.add_scalars("Decision_SV_Pct",      {a: _decision_pct(metrics[a]["decision_type"], "SERVICE_VEHICLE")   for a in active}, episode)
                writer.add_scalars("Fail_Deadline_Rate",   {a: _fail_reason_pct(metrics[a]["fail_reason"], "DEADLINE_MISSED")  for a in active}, episode)
                writer.add_scalars("Fail_Queue_Full_Rate", {a: _fail_reason_pct(metrics[a]["fail_reason"], "RSU_QUEUE_FULL")   for a in active}, episode)
                writer.add_scalars("Fail_Routing_Rate",    {a: _fail_reason_pct(metrics[a]["fail_reason"], "SV_MAC_UNKNOWN") +
                                                               _fail_reason_pct(metrics[a]["fail_reason"], "NEIGHBOR_RSU_UNKNOWN")
                                                                                                                               for a in active}, episode)
                writer.add_scalars("QoS_Success_Rate", {
                    f"{a}_qos1": _qos_success_pct(metrics[a]["qos"], metrics[a]["success"], 1)
                    for a in active
                }, episode)
                writer.add_scalars("QoS_Success_Rate", {
                    f"{a}_qos2": _qos_success_pct(metrics[a]["qos"], metrics[a]["success"], 2)
                    for a in active
                }, episode)
                writer.add_scalars("QoS_Success_Rate", {
                    f"{a}_qos3": _qos_success_pct(metrics[a]["qos"], metrics[a]["success"], 3)
                    for a in active
                }, episode)
            writer.add_scalar("Epsilon", ddqn_agent.epsilon, episode)
            writer.add_scalar("Task_Timeout_Rate", timeout_count / max(episode, 1), episode)

            if episode % 100 == 0:
                window = metrics["ddqn"]["reward"][-100:] if len(metrics["ddqn"]["reward"]) >= 100 \
                         else metrics["ddqn"]["reward"]
                avg_ddqn = np.mean(window) if window else 0.0
                print(f"[DRL-{iid}] Ep {episode:5d} | DDQN: {avg_ddqn:.2f} "
                      f"| Eps: {ddqn_agent.epsilon:.3f} | Pending: {len(pending)}")
                if avg_ddqn > best_avg_reward and episode > 1000:
                    best_avg_reward = avg_ddqn
                    ddqn_agent.save_model(model_path)
                    print(f"  >> [DRL-{iid}] New Best Model Saved! Avg Reward: {best_avg_reward:.2f}")

        # ── 3. Batch-train periodically (every TRAIN_EVERY completions) ──
        if completions_since_train >= TRAIN_EVERY:
            loss = ddqn_agent.train()
            ddqn_agent.update_target_network_soft()
            if loss is not None:
                writer.add_scalar("Loss", loss, episode)
            completions_since_train = 0

        # Brief sleep when idle to avoid busy-wait CPU spin
        if not pending:
            time.sleep(Config.REDIS_POLL_INTERVAL)

    writer.close()
    print(f"[DRL-{iid}] Training Complete. Logs: {log_dir}  Model: {model_path}")


def run():
    parser = argparse.ArgumentParser(description="Train Task Offloading DRL Agent")
    parser.add_argument('--env', type=str, default='dummy', choices=['dummy', 'redis'],
                        help='Environment type: dummy or redis')
    args = parser.parse_args()

    if args.env == 'redis':
        Config.load_config("redis_config.json")   # load FIRST so paths are correct
        os.makedirs(Config.LOG_DIR, exist_ok=True)
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
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(Config.PLOT_SAVE_PATH), exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    print("Initializing Dummy Environment...")
    env = IoVDummyEnv()

    writer      = SummaryWriter(log_dir=Config.LOG_DIR)
    agent_names = DUMMY_AGENTS

    ddqn_agent        = DDQNAgent()
    random_agent      = RandomAgent()
    greedy_comp_agent = GreedyComputeAgent()
    greedy_dist_agent = GreedyDistanceAgent()

    metrics         = {a: {"reward": [], "success": [], "latency": [], "energy": [],
                           "decision_type": [], "fail_reason": [], "qos": []}
                       for a in agent_names}
    best_avg_reward = -float('inf')

    print(f"Starting Training on {Config.DEVICE}...")
    episodes = Config.EPISODS

    for episode in range(episodes):
        seed = Config.SEED + episode

        random.seed(seed); np.random.seed(seed)
        state = env.reset(); mask = env.get_action_mask()
        ep_qos = env.current_task.qos  # same for all agents (same seed)
        action = ddqn_agent.select_action(state, mask=mask)
        next_state, reward, done, info = env.step(action)
        ddqn_agent.store_transition(state, action, reward, next_state, done)
        loss = ddqn_agent.train()
        ddqn_agent.update_target_network_soft()
        metrics["ddqn"]["reward"].append(reward)
        metrics["ddqn"]["success"].append(info['success'])
        metrics["ddqn"]["latency"].append(info['latency'])
        metrics["ddqn"]["energy"].append(info.get('energy', 0.0))
        metrics["ddqn"]["decision_type"].append(_action_to_dtype(action))
        metrics["ddqn"]["fail_reason"].append('NONE' if info['success'] else info.get('reason', 'DEADLINE_MISSED'))
        metrics["ddqn"]["qos"].append(ep_qos)

        random.seed(seed); np.random.seed(seed)
        _ = env.reset(); mask = env.get_action_mask()
        action = random_agent.select_action(mask)
        _, r_reward, _, r_info = env.step(action)
        metrics["random"]["reward"].append(r_reward)
        metrics["random"]["success"].append(r_info['success'])
        metrics["random"]["latency"].append(r_info['latency'])
        metrics["random"]["energy"].append(r_info.get('energy', 0.0))
        metrics["random"]["decision_type"].append(_action_to_dtype(action))
        metrics["random"]["fail_reason"].append('NONE' if r_info['success'] else r_info.get('reason', 'DEADLINE_MISSED'))
        metrics["random"]["qos"].append(ep_qos)

        random.seed(seed); np.random.seed(seed)
        _ = env.reset(); mask = env.get_action_mask()
        action = greedy_comp_agent.select_action(env.candidates, env.active_rsu, mask)
        _, gc_reward, _, gc_info = env.step(action)
        metrics["greedy_comp"]["reward"].append(gc_reward)
        metrics["greedy_comp"]["success"].append(gc_info['success'])
        metrics["greedy_comp"]["latency"].append(gc_info['latency'])
        metrics["greedy_comp"]["energy"].append(gc_info.get('energy', 0.0))
        metrics["greedy_comp"]["decision_type"].append(_action_to_dtype(action))
        metrics["greedy_comp"]["fail_reason"].append('NONE' if gc_info['success'] else gc_info.get('reason', 'DEADLINE_MISSED'))
        metrics["greedy_comp"]["qos"].append(ep_qos)

        random.seed(seed); np.random.seed(seed)
        _ = env.reset(); mask = env.get_action_mask()
        action = greedy_dist_agent.select_action(env.candidates, env.active_rsu, mask)
        _, gd_reward, _, gd_info = env.step(action)
        metrics["greedy_dist"]["reward"].append(gd_reward)
        metrics["greedy_dist"]["success"].append(gd_info['success'])
        metrics["greedy_dist"]["latency"].append(gd_info['latency'])
        metrics["greedy_dist"]["energy"].append(gd_info.get('energy', 0.0))
        metrics["greedy_dist"]["decision_type"].append(_action_to_dtype(action))
        metrics["greedy_dist"]["fail_reason"].append('NONE' if gd_info['success'] else gd_info.get('reason', 'DEADLINE_MISSED'))
        metrics["greedy_dist"]["qos"].append(ep_qos)

        writer.add_scalars("Rewards", {
            "DDQN": reward, "Random": r_reward, "Greedy_Comp": gc_reward, "Greedy_Dist": gd_reward
        }, episode)
        writer.add_scalars("Rewards_Smoothed", {
            "DDQN":        _running_mean(metrics["ddqn"]["reward"]),
            "Random":      _running_mean(metrics["random"]["reward"]),
            "Greedy_Comp": _running_mean(metrics["greedy_comp"]["reward"]),
            "Greedy_Dist": _running_mean(metrics["greedy_dist"]["reward"]),
        }, episode)
        writer.add_scalars("Success_Rate", {
            "DDQN": info['success'], "Random": r_info['success'],
            "Greedy_Comp": gc_info['success'], "Greedy_Dist": gd_info['success']
        }, episode)
        writer.add_scalars("Latency", {
            "DDQN": info['latency'], "Random": r_info['latency'],
            "Greedy_Comp": gc_info['latency'], "Greedy_Dist": gd_info['latency']
        }, episode)
        writer.add_scalars("Energy", {
            "DDQN":        info.get('energy', 0.0),
            "Random":      r_info.get('energy', 0.0),
            "Greedy_Comp": gc_info.get('energy', 0.0),
            "Greedy_Dist": gd_info.get('energy', 0.0),
        }, episode)
        writer.add_scalars("Decision_RSU_Pct", {
            "DDQN":        _decision_pct(metrics["ddqn"]["decision_type"],        "RSU"),
            "Random":      _decision_pct(metrics["random"]["decision_type"],      "RSU"),
            "Greedy_Comp": _decision_pct(metrics["greedy_comp"]["decision_type"], "RSU"),
            "Greedy_Dist": _decision_pct(metrics["greedy_dist"]["decision_type"], "RSU"),
        }, episode)
        writer.add_scalars("Decision_SV_Pct", {
            "DDQN":        _decision_pct(metrics["ddqn"]["decision_type"],        "SERVICE_VEHICLE"),
            "Random":      _decision_pct(metrics["random"]["decision_type"],      "SERVICE_VEHICLE"),
            "Greedy_Comp": _decision_pct(metrics["greedy_comp"]["decision_type"], "SERVICE_VEHICLE"),
            "Greedy_Dist": _decision_pct(metrics["greedy_dist"]["decision_type"], "SERVICE_VEHICLE"),
        }, episode)
        writer.add_scalars("Fail_Deadline_Rate", {
            "DDQN":        _fail_reason_pct(metrics["ddqn"]["fail_reason"],        "DEADLINE_MISSED"),
            "Random":      _fail_reason_pct(metrics["random"]["fail_reason"],      "DEADLINE_MISSED"),
            "Greedy_Comp": _fail_reason_pct(metrics["greedy_comp"]["fail_reason"], "DEADLINE_MISSED"),
            "Greedy_Dist": _fail_reason_pct(metrics["greedy_dist"]["fail_reason"], "DEADLINE_MISSED"),
        }, episode)
        writer.add_scalars("Fail_Handover_Rate", {
            "DDQN":        _fail_reason_pct(metrics["ddqn"]["fail_reason"],        "Handover_Fail"),
            "Random":      _fail_reason_pct(metrics["random"]["fail_reason"],      "Handover_Fail"),
            "Greedy_Comp": _fail_reason_pct(metrics["greedy_comp"]["fail_reason"], "Handover_Fail"),
            "Greedy_Dist": _fail_reason_pct(metrics["greedy_dist"]["fail_reason"], "Handover_Fail"),
        }, episode)
        writer.add_scalars("QoS_Success_Rate", dict(
            **{f"DDQN_qos{q}":        _qos_success_pct(metrics["ddqn"]["qos"],        metrics["ddqn"]["success"],        q) for q in (1, 2, 3)},
            **{f"Random_qos{q}":      _qos_success_pct(metrics["random"]["qos"],      metrics["random"]["success"],      q) for q in (1, 2, 3)},
            **{f"Greedy_Comp_qos{q}": _qos_success_pct(metrics["greedy_comp"]["qos"], metrics["greedy_comp"]["success"], q) for q in (1, 2, 3)},
            **{f"Greedy_Dist_qos{q}": _qos_success_pct(metrics["greedy_dist"]["qos"], metrics["greedy_dist"]["success"], q) for q in (1, 2, 3)},
        ), episode)
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
