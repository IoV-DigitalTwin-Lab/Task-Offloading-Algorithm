import matplotlib.pyplot as plt
import numpy as np
import os
import random
import argparse
import threading
import time
import json
import datetime
import signal
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from src.environment import IoVDummyEnv, IoVRedisEnv
from src.agents import (
    DDQNAgent, VanillaDQNAgent,
    RandomAgent, GreedyComputeAgent, MinLatencyAgent,
    LeastQueueAgent, GreedyDistanceAgent, LocalAgent,
)
from src.config import Config

# Dummy-env agent names (unchanged)
DUMMY_AGENTS = ['ddqn', 'random', 'greedy_comp', 'greedy_dist']

# 6 canonical task types from the simulator taxonomy
TASK_TYPES = [
    "LOCAL_OBJECT_DETECTION",
    "COOPERATIVE_PERCEPTION",
    "ROUTE_OPTIMIZATION",
    "FLEET_TRAFFIC_FORECAST",
    "VOICE_COMMAND_PROCESSING",
    "SENSOR_HEALTH_CHECK",
]


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _running_mean(data, window=100):
    window_data = data[-window:] if len(data) >= window else data
    return float(np.mean(window_data)) if window_data else 0.0


def _decision_pct(decisions, dtype, window=50):
    recent = decisions[-window:] if len(decisions) >= window else decisions
    if not recent:
        return 0.0
    return sum(1 for d in recent if d == dtype) / len(recent)


def _action_to_dtype(action):
    if action == Config.MAX_NEIGHBORS:
        return "RSU"
    elif action < Config.MAX_NEIGHBORS:
        return "SERVICE_VEHICLE"
    return "LOCAL"


def _fail_reason_pct(reasons, target_reason, window=50):
    recent = reasons[-window:] if len(reasons) >= window else reasons
    if not recent:
        return 0.0
    return sum(1 for r in recent if r == target_reason) / len(recent)


def _qos_success_pct(qos_list, success_list, target_qos, window=100):
    recent = list(zip(qos_list, success_list))[-window:]
    filtered = [s for q, s in recent if int(round(q)) == target_qos]
    return (sum(filtered) / len(filtered)) if filtered else 0.0


def _success_rate(successes, window=100):
    """Rolling success rate as percentage (0-100)."""
    recent = successes[-window:] if len(successes) >= window else successes
    if not recent:
        return 0.0
    return sum(recent) / len(recent) * 100.0


def _rolling_mean(data, key, window=50):
    """Rolling mean of the last `window` entries from a list of dicts."""
    recent = [d[key] for d in data[-window:]] if len(data) >= window else [d[key] for d in data]
    return float(np.mean(recent)) if recent else 0.0


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _create_agent(agent_name, load_path=None):
    if agent_name == 'ddqn':
        agent = DDQNAgent()
        initial_ep = 0
        if load_path:
            initial_ep = agent.load_model(load_path) or 0
        agent.global_step = initial_ep
        return agent
    elif agent_name == 'vanilla_dqn':
        return VanillaDQNAgent()
    elif agent_name == 'random':
        return RandomAgent()
    elif agent_name == 'greedy_compute':
        return GreedyComputeAgent()
    elif agent_name == 'min_latency':
        return MinLatencyAgent()
    elif agent_name == 'least_queue':
        return LeastQueueAgent()
    elif agent_name == 'greedy_distance':
        return GreedyDistanceAgent()
    elif agent_name == 'local':
        return LocalAgent()
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


def _select_action(agent, agent_name, state, mask, env, request):
    """Dispatch select_action for different agent interfaces."""
    if agent_name in ('ddqn', 'vanilla_dqn'):
        return agent.select_action(state, mask=mask)
    elif agent_name == 'random':
        return agent.select_action(mask)
    elif agent_name == 'min_latency':
        return agent.select_action(env.candidates, env.rsus, mask, task_info=request)
    else:  # greedy_compute, least_queue, greedy_distance
        return agent.select_action(env.candidates, env.rsus, mask)


# ---------------------------------------------------------------------------
# Single-agent Redis training loop
# ---------------------------------------------------------------------------

def _run_single_agent_instance(instance_cfg, agent_name, offload_mode, stop_event=None, load_path=None, resume_training=False):
    """
    Single-agent training/evaluation loop for one RSU instance.

    Offload flow (all agents except 'local'):
      1. Drain offloading_requests:queue (non-blocking, up to MAX_DRAIN_PER_CYCLE)
      2. For each request: build state → agent selects action → write_decision
      3. Batch-check task:{id}:result for all pending tasks
      4. Drain local_results:queue (heuristic/allLocal modes)
      5. Train DDQN every TRAIN_EVERY completions
      6. Log to TensorBoard; save JSON results on completion

    For 'local' agent: skip steps 1-3, only drain local_results:queue.
    """
    if stop_event is None:
        stop_event = threading.Event()

    iid      = instance_cfg['instance_id']
    redis_db = instance_cfg['redis_db']
    rsu_id   = instance_cfg['rsu_id']

    log_dir    = os.path.join(Config.LOG_DIR, f"{agent_name}_{offload_mode}", f"instance_{iid}")
    model_path = os.path.join(Config.BASE_DIR, "models", f"ddqn_rsu{iid}.pth")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.join(Config.BASE_DIR, "results"), exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    env    = IoVRedisEnv(redis_db=redis_db, instance_id=iid)
    if resume_training and not load_path:
        default_path = os.path.join(Config.BASE_DIR, "models", f"{agent_name}_rsu{iid}.pth")
        if os.path.exists(default_path):
            load_path = default_path
            print(f"[MAIN-{iid}] Auto-resuming from {load_path}")
    
    agent  = _create_agent(agent_name, load_path=load_path)

    # DRL-agent reference for training (None for heuristic baselines)
    ddqn_agent = agent if agent_name in ('ddqn', 'vanilla_dqn') else None

    TIMEOUT             = Config.REDIS_RESULT_TIMEOUT
    TRAIN_EVERY         = 4
    MAX_DRAIN_PER_CYCLE = 20

    last_logged_episode = -1   # guard against log spam: only print once per milestone

    # ── Accumulators ─────────────────────────────────────────────────────────
    # Per-task-type rolling data (all tasks combined)
    lat_by_type  = defaultdict(list)   # task_type → [latency, ...]
    ene_by_type  = defaultdict(list)   # task_type → [energy, ...]

    offload_rewards   = []   # only for ddqn
    offload_successes = []
    local_successes   = []
    all_successes     = []   # combined; used for Success_Rate

    decision_types    = []   # "RSU" | "SERVICE_VEHICLE"
    all_fail_reasons  = []   # raw reason strings
    qos_values        = []   # raw qos floats (for QoS_Success_Rate)
    qos_successes     = []   # bool, aligned with qos_values

    fail_counts = defaultdict(int)  # cumulative; used in JSON results

    pending            = {}   # task_id → {state, action, decision_type, target, task_request, task_type, timestamp}
    episode            = getattr(agent, 'global_step', 0)    # total tasks processed
    completions_since_train = 0
    timeout_count      = 0
    total_tasks        = 0
    offload_tasks      = 0
    local_tasks        = 0

    print(f"[{agent_name}-{iid}] Starting | mode={offload_mode} | RSU={rsu_id} | DB={redis_db}")

    # ── Wait for vehicle states ───────────────────────────────────────────────
    if agent_name != 'local':
        _wait_start = time.time()
        _MAX_WAIT = 15
        print(f"[{agent_name}-{iid}] Waiting for vehicle states (up to {_MAX_WAIT}s)...")
        while time.time() - _wait_start < _MAX_WAIT:
            if stop_event.is_set():
                print(f"[{agent_name}-{iid}] Shutdown requested during vehicle wait — exiting.")
                return
            if env.r.keys('vehicle:*:state'):
                print(f"[{agent_name}-{iid}] Vehicle states found. Starting loop.")
                break
            time.sleep(1.0)
        else:
            if stop_event.is_set():
                print(f"[{agent_name}-{iid}] Shutdown requested during vehicle wait — exiting.")
                return
            print(f"[{agent_name}-{iid}] WARNING: no vehicle states — starting anyway.")

    try:
        while not stop_event.is_set():

            # ── 1. Drain offloading requests ──────────────────────────────────
            _offload_drained = 0
            if agent_name != 'local':
                for _ in range(MAX_DRAIN_PER_CYCLE):
                    request = env._poll_request_nonblocking()
                    if request is None:
                        break
                    _offload_drained += 1
                    dequeued_wall_s = time.time()

                    state = env.setup_from_request(request)
                    if state is None:
                        # Vehicle state missing: fall back to the RSU that received the task.
                        now_wall_s = time.time()
                        pipe = env.r.pipeline()
                        req_rsu = request.get('rsu_id', env.rsu_ids[0])
                        fallback_action = (env.rsu_ids.index(req_rsu)
                                           if req_rsu in env.rsu_ids else 0)
                        env.write_decision(request['task_id'], fallback_action, agent_name)
                        pipe.hset(
                            f"task:{request['task_id']}:decision",
                            mapping={
                                'agent': agent_name,
                                'type': 'RSU',
                                'target': req_rsu,
                                'drl_instance_id': iid,
                                'rsu_id': rsu_id,
                                'drl_dequeued_wall_s': dequeued_wall_s,
                                'drl_state_ready_wall_s': now_wall_s,
                                'drl_decision_written_wall_s': now_wall_s,
                            },
                        )
                        pipe.expire(f"task:{request['task_id']}:decision", 300)
                        pipe.execute()
                        print(f"[{agent_name}-{iid}] Task {request['task_id']}: "
                              f"vehicle state missing — fallback to RSU {req_rsu} "
                              f"(action {fallback_action})")
                        continue

                    mask   = env.get_action_mask()
                    action = _select_action(agent, agent_name, state, mask, env, request)
                    dec    = env.write_decision(request['task_id'], action, agent_name)
                    task_type = getattr(env, 'task_type', 'UNKNOWN')
                    decision_written_wall_s = time.time()
                    trace_metadata = {
                        'drl_instance_id': iid,
                        'rsu_id': rsu_id,
                        'drl_dequeued_wall_s': dequeued_wall_s,
                        'drl_state_ready_wall_s': decision_written_wall_s,
                        'drl_decision_written_wall_s': decision_written_wall_s,
                    }
                    pending[request['task_id']] = {
                        'state':         state,
                        'action':        action,
                        'decision_type': dec['type'],
                        'target':        dec['target'],
                        'task_request':  request,
                        'task_type':     task_type,
                        'timestamp':     time.time(),
                    }

            # ── 2. Expire timed-out pending entries ───────────────────────────
            _now = time.time()
            for tid in list(pending.keys()):
                if _now - pending[tid]['timestamp'] > TIMEOUT:
                    env.r.hset(
                        f"task:{tid}:decision",
                        mapping={
                            'drl_result_timeout_discard_wall_s': _now,
                            'drl_result_timeout_s': TIMEOUT,
                        },
                    )
                    env.r.expire(f"task:{tid}:decision", 300)
                    print(f"[DRL-{iid}] Task {tid} timed out — discarding")
                    pending.pop(tid)
                    timeout_count += 1

            # ── 3. Batch-check offloaded results ──────────────────────────────
            ready = env.batch_check_single_results(pending)
            for task_id, result in ready.items():
                entry   = pending.pop(task_id)
                success = result['status'] == 'COMPLETED_ON_TIME'
                latency = result['latency']
                energy  = result['energy']
                reason  = result.get('reason', 'NONE')
                ttype   = entry['task_type']
                qos     = float(entry['task_request'].get('qos', 1.0))

                lat_by_type[ttype].append(latency)
                ene_by_type[ttype].append(energy)
                offload_successes.append(1 if success else 0)
                all_successes.append(1 if success else 0)
                decision_types.append(entry['decision_type'])
                all_fail_reasons.append(reason)
                fail_counts[reason] += 1
                qos_values.append(qos)
                qos_successes.append(1 if success else 0)

                offload_tasks += 1
                total_tasks   += 1
                episode       += 1
                completions_since_train += 1

                if ddqn_agent is not None:
                    reward, _ = env.compute_reward_for(
                        entry['task_request'], result, entry['decision_type'], entry['target']
                    )
                    offload_rewards.append(reward)
                    ddqn_agent.store_transition(
                        entry['state'], entry['action'], reward, entry['state'], done=True
                    )

                # TensorBoard (offloaded tasks drive the episode counter)
                _log_episode(writer, episode, success, latency, energy, reason,
                             entry['decision_type'], ttype, qos,
                             all_successes, lat_by_type, ene_by_type,
                             offload_successes, local_successes,
                             decision_types, all_fail_reasons, qos_values, qos_successes,
                             offload_rewards, timeout_count, total_tasks,
                             agent_name, ddqn_agent)

            # ── 4. Drain local results ────────────────────────────────────────
            _local_drained = 0
            for _ in range(MAX_DRAIN_PER_CYCLE):
                local = env.poll_local_result()
                if local is None:
                    break
                task_id, result = local
                if not result:
                    continue
                ttype   = result.get('task_type', 'UNKNOWN')
                success = result.get('status', 'FAILED') == 'COMPLETED_ON_TIME'
                latency = result.get('latency', 999.0)
                energy  = result.get('energy',  0.0)
                reason  = result.get('reason',  'NONE')
                qos     = float(result.get('qos_value', 1.0))

                lat_by_type[ttype].append(latency)
                ene_by_type[ttype].append(energy)
                local_successes.append(1 if success else 0)
                all_successes.append(1 if success else 0)
                all_fail_reasons.append(reason)
                fail_counts[reason] += 1
                qos_values.append(qos)
                qos_successes.append(1 if success else 0)

                local_tasks  += 1
                total_tasks  += 1
                _local_drained += 1

                # For local-only agent: TensorBoard updates driven by local completions
                if agent_name == 'local':
                    episode += 1
                    _log_episode(writer, episode, success, latency, energy, reason,
                                 'LOCAL', ttype, qos,
                                 all_successes, lat_by_type, ene_by_type,
                                 offload_successes, local_successes,
                                 decision_types, all_fail_reasons, qos_values, qos_successes,
                                 offload_rewards, timeout_count, total_tasks,
                                 agent_name, ddqn_agent)

            # ── 5. Train DDQN ─────────────────────────────────────────────────
            if ddqn_agent is not None and completions_since_train >= TRAIN_EVERY:
                loss = ddqn_agent.train()
                ddqn_agent.update_target_network_soft()
                if loss is not None:
                    writer.add_scalar("Loss", loss, episode)
                completions_since_train = 0

            # ── 6. Periodic console log + model save ──────────────────────────
            if episode > 0 and episode % 50 == 0 and episode != last_logged_episode:
                last_logged_episode = episode
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                sr = _success_rate(all_successes)
                print(f"[{agent_name}-{iid}] [{ts}] ep={episode} | "
                      f"SR={sr:.1f}% | offload={offload_tasks} local={local_tasks} "
                      f"pending={len(pending)} timeout={timeout_count}")
                if ddqn_agent is not None and offload_rewards and episode > 500:
                    avg_r = _running_mean(offload_rewards)
                    ddqn_agent.save_model(model_path, global_step=episode)
                    print(f"  >> [{agent_name}-{iid}] Model saved | avg_reward={avg_r:.3f}")

            # ── Idle sleep ────────────────────────────────────────────────────
            # Sleep whenever nothing happened this cycle (no new requests, no new results,
            # no local results).  This avoids busy-looping while waiting for simulator
            # results even when the pending dict is non-empty.
            nothing_happened = _offload_drained == 0 and len(ready) == 0 and _local_drained == 0
            if nothing_happened:
                time.sleep(Config.REDIS_POLL_INTERVAL)

    except KeyboardInterrupt:
        print(f"[{agent_name}-{iid}] Interrupted — saving results...")
        stop_event.set()

    if stop_event.is_set():
        print(f"[{agent_name}-{iid}] Stop requested — finalizing artifacts...")

    # ── Save final model (ddqn) ───────────────────────────────────────────────
    if ddqn_agent is not None and offload_rewards:
        ddqn_agent.save_model(model_path, global_step=episode)
        print(f"[{agent_name}-{iid}] Final model saved to {model_path}")

    # ── Save results JSON ─────────────────────────────────────────────────────
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(
        Config.BASE_DIR, "results",
        f"{agent_name}_{offload_mode}_{ts_str}_inst{iid}.json"
    )
    results = {
        "agent":        agent_name,
        "offload_mode": offload_mode,
        "rsu_id":       rsu_id,
        "instance_id":  iid,
        "timestamp":    datetime.datetime.now().isoformat(),
        "metrics": {
            "rewards":            offload_rewards,
            "success_rates":      _build_rolling_series(all_successes, window=100,
                                                        transform=lambda w: sum(w)/len(w)*100),
            "latencies":          {t: lat_by_type[t] for t in TASK_TYPES},
            "energies":           {t: ene_by_type[t] for t in TASK_TYPES},
            "failure_reasons":    dict(fail_counts),
            "qos_success_rates": {
                f"qos{q}": _build_rolling_series(
                    [(s if int(round(v)) == q else None)
                     for v, s in zip(qos_values, qos_successes)],
                    window=100, skip_none=True,
                    transform=lambda w: sum(w)/len(w)*100
                )
                for q in (1, 2, 3)
            },
            "offload_success_rates": _build_rolling_series(offload_successes, window=100,
                                                            transform=lambda w: sum(w)/len(w)*100),
            "local_success_rates":   _build_rolling_series(local_successes, window=100,
                                                            transform=lambda w: sum(w)/len(w)*100),
            "total_tasks":   total_tasks,
            "offload_tasks": offload_tasks,
            "local_tasks":   local_tasks,
        }
    }
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[{agent_name}-{iid}] Results saved to {result_path}")

    writer.close()


def _build_rolling_series(data, window=100, transform=None, skip_none=False):
    """Build a rolling-window series from a list."""
    if skip_none:
        data = [x for x in data if x is not None]
    if not data or transform is None:
        return []
    series = []
    for i in range(len(data)):
        w = data[max(0, i - window + 1): i + 1]
        series.append(transform(w))
    return series


def _log_episode(writer, episode, success, latency, energy, reason, decision_type, task_type,
                 qos, all_successes, lat_by_type, ene_by_type,
                 offload_successes, local_successes,
                 decision_types, all_fail_reasons, qos_values, qos_successes,
                 offload_rewards, timeout_count, total_tasks,
                 agent_name, ddqn_agent):
    """Log metrics to TensorBoard for a single completed task."""
    writer.add_scalar("Success_Rate", _success_rate(all_successes), episode)

    if offload_rewards:
        writer.add_scalar("Rewards",         offload_rewards[-1],                episode)
        writer.add_scalar("Rewards_Smoothed", _running_mean(offload_rewards),    episode)

    # Per-task-type latency and energy (rolling mean per type)
    for ttype in TASK_TYPES:
        if lat_by_type[ttype]:
            writer.add_scalar(f"Latency/{ttype}", _running_mean(lat_by_type[ttype], 50), episode)
        if ene_by_type[ttype]:
            writer.add_scalar(f"Energy/{ttype}",  _running_mean(ene_by_type[ttype], 50), episode)

    # Decision distribution (offloaded tasks only)
    if decision_types:
        writer.add_scalar("Decision_RSU_Pct", _decision_pct(decision_types, "RSU"),              episode)
        writer.add_scalar("Decision_SV_Pct",  _decision_pct(decision_types, "SERVICE_VEHICLE"),  episode)

    # Fail reasons
    writer.add_scalar("Fail_Deadline_Rate",   _fail_reason_pct(all_fail_reasons, "DEADLINE_MISSED"), episode)
    writer.add_scalar("Fail_Queue_Full_Rate", _fail_reason_pct(all_fail_reasons, "RSU_QUEUE_FULL"),  episode)
    writer.add_scalar("Fail_SV_OOR_Rate",     _fail_reason_pct(all_fail_reasons, "SV_OUT_OF_RANGE"), episode)
    writer.add_scalar("Fail_Handover_Rate",   _fail_reason_pct(all_fail_reasons, "HANDOVER_FAIL"),   episode)

    # QoS success rates
    for q in (1, 2, 3):
        filtered = [(s) for v, s in zip(qos_values, qos_successes) if int(round(v)) == q]
        if filtered:
            rate = sum(filtered[-100:]) / len(filtered[-100:]) * 100.0
            writer.add_scalar(f"QoS_Success_Rate/qos{q}", rate, episode)

    # DDQN-specific
    if ddqn_agent is not None:
        writer.add_scalar("Epsilon", ddqn_agent.epsilon, episode)

    # Task timeout rate (cumulative)
    writer.add_scalar("Task_Timeout_Rate", timeout_count / max(total_tasks, 1), episode)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run():
    parser = argparse.ArgumentParser(description="Train Task Offloading DRL Agent")
    parser.add_argument('--env', type=str, default='dummy', choices=['dummy', 'redis'],
                        help='Environment type: dummy or redis')
    parser.add_argument('--agent', type=str, default=None,
                        choices=['ddqn', 'vanilla_dqn', 'random', 'greedy_compute',
                                 'min_latency', 'least_queue', 'greedy_distance', 'local'],
                        help='Agent type (required for --env redis)')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to a saved model to resume training from.')
    parser.add_argument('--resume_training', action='store_true',
                        help='Resume training automatically checking models folder.')
    args = parser.parse_args()

    if args.env == 'redis':
        if args.agent is None:
            print("[ERROR] --agent is required when using --env redis")
            print("  choices: ddqn, vanilla_dqn, random, greedy_compute, "
                  "min_latency, least_queue, greedy_distance, local")
            return

        Config.load_config("redis_config.json")
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        print(f"Initializing Redis Environment(s) for agent={args.agent} ...")

        active = [i for i in Config.DRL_INSTANCES if i['active']]
        if not active:
            print("[ERROR] No active agent_instances in redis_config.json.")
            return

        # Read offload_mode from Redis (written by simulator at startup)
        import redis as _redis_lib
        _r0 = _redis_lib.Redis(host=Config.REDIS_HOST, port=Config.REDIS_PORT,
                               db=active[0]['redis_db'], decode_responses=True)
        offload_mode = _r0.get("sim:offload_mode") or "unknown"
        _r0.close()
        print(f"Simulator offload mode: {offload_mode}")

        if len(active) == 1:
            stop_event = threading.Event()
            def _request_shutdown_single(signum, _frame):
                if not stop_event.is_set():
                    print(f"\n[MAIN] Received signal {signum}; stopping agent loop...")
                    stop_event.set()
            signal.signal(signal.SIGINT, _request_shutdown_single)
            signal.signal(signal.SIGTERM, _request_shutdown_single)
            _run_single_agent_instance(active[0], args.agent, offload_mode, stop_event=stop_event, load_path=args.load_model, resume_training=args.resume_training)
        else:
            stop_event = threading.Event()
            def _request_shutdown(signum, _frame):
                if not stop_event.is_set():
                    print(f"\n[MAIN] Received signal {signum}; stopping all agent instances...")
                    stop_event.set()
            signal.signal(signal.SIGINT, _request_shutdown)
            signal.signal(signal.SIGTERM, _request_shutdown)
            threads = [
                threading.Thread(
                    target=_run_single_agent_instance,
                    args=(inst, args.agent, offload_mode, stop_event, args.load_model, args.resume_training),
                    daemon=False
                )
                for inst in active
            ]
            for t in threads:
                t.start()
            try:
                while any(t.is_alive() for t in threads):
                    for t in threads:
                        t.join(timeout=0.5)
            except KeyboardInterrupt:
                print("\n[MAIN] KeyboardInterrupt received; requesting shutdown...")
                stop_event.set()
            finally:
                stop_event.set()
                for t in threads:
                    t.join()
        print("All agent instances finished.")
        return

    # ── Dummy env path (unchanged) ────────────────────────────────────────────
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
        ep_qos = env.current_task.qos
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
        writer.add_scalar("Epsilon", ddqn_agent.epsilon, episode)
        if loss is not None:
            writer.add_scalar("Loss", loss, episode)

        if episode % 10 == 0:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            avg_ddqn = np.mean(metrics["ddqn"]["reward"][-100:])
            print(f"[{ts}] Ep {episode} | DDQN: {avg_ddqn:.2f} | Eps: {ddqn_agent.epsilon:.2f}")
            if avg_ddqn > best_avg_reward and episode > 1000:
                best_avg_reward = avg_ddqn
                ddqn_agent.save_model(Config.MODEL_SAVE_PATH, global_step=episode)
                print(f"  >> New Best Model Saved! Avg Reward: {best_avg_reward:.2f}")

    writer.close()
    print(f"Training Complete. Logs saved to {Config.LOG_DIR}")
    print(f"DDQN Model saved to {Config.MODEL_SAVE_PATH}")

    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    def smooth(data, window=100):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    def plot_all(ax_idx, metric_key, title):
        ax[ax_idx].plot(smooth(metrics["ddqn"][metric_key]), label='DDQN (Ours)', color='blue', linewidth=2)
        ax[ax_idx].plot(smooth(metrics["greedy_comp"][metric_key]), label='Greedy-Comp', color='orange', linestyle='--')
        ax[ax_idx].plot(smooth(metrics["greedy_dist"][metric_key]), label='Greedy-Dist', color='green', linestyle=':')
        ax[ax_idx].plot(smooth(metrics["random"][metric_key]), label='Random', color='grey', alpha=0.3)
        ax[ax_idx].set_title(title); ax[ax_idx].legend()
    plot_all(0, "reward", "Average Reward")
    plot_all(1, "success", "Task Completion Rate"); ax[1].set_ylim(0, 1.1)
    plot_all(2, "latency", "Latency (s)")
    plt.tight_layout(); plt.savefig(Config.PLOT_SAVE_PATH)
    print(f"Plots saved to {Config.PLOT_SAVE_PATH}")


if __name__ == "__main__":
    run()
