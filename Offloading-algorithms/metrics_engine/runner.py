"""
Metrics Engine Runner — standalone process alongside simulation and agents.

Usage:
    python metrics_engine/runner.py [--host 127.0.0.1] [--port 6379] [--db 0]
                                     [--experiment balanced_optimal] [--k 6]

The runner:
  1. Connects to Redis and verifies engine_active flag
  2. Polls engine_requests:queue for task IDs (pushed by environment.py)
  3. For each task: MetricsEngine.process_task(task_id) → writes result
  4. Logs processed count and per-agent stats at regular intervals
  5. Exits cleanly on SIGINT/SIGTERM

Run BEFORE starting the DRL agents so the engine is ready when the first
task decision arrives.
"""

import argparse
import signal
import sys
import time
from collections import defaultdict
from typing import Optional

from metrics_engine.config import ENGINE_POLL_INTERVAL_S, ENGINE_ACTIVE_KEY
from metrics_engine.engine import MetricsEngine
from metrics_engine.redis_interface import RedisInterface


def _setup_signal(stop_flag: list) -> None:
    """Install SIGINT / SIGTERM handler that sets stop_flag[0] = True."""
    def _handler(signum, _frame):
        print(f"\n[Engine] Signal {signum} received — shutting down...", flush=True)
        stop_flag[0] = True
    signal.signal(signal.SIGINT,  _handler)
    signal.signal(signal.SIGTERM, _handler)


def _print_stats(processed: int, per_agent: dict, failed: int, elapsed_s: float) -> None:
    rate = processed / max(elapsed_s, 1)
    print(f"[Engine] Processed={processed}  Failed={failed}  "
          f"Rate={rate:.1f} tasks/s  |  Per-agent: {dict(per_agent)}", flush=True)


def run(host: str = "127.0.0.1",
        port: int = 6379,
        db: int = 0,
        log_interval: int = 50,
        poll_timeout: float = 0.5) -> None:
    """
    Main runner loop.

    Args:
        host: Redis host
        port: Redis port
        db: Redis logical database (must match simulation and agents)
        log_interval: print stats every N processed tasks
        poll_timeout: blocking pop timeout in seconds (lower = more CPU; higher = more latency)
    """
    print(f"[Engine] Starting Realistic Metrics Engine  host={host} port={port} db={db}",
          flush=True)

    ri = RedisInterface(host=host, port=port, db=db)

    if not ri.ping():
        print("[Engine] ERROR: Cannot connect to Redis. Is redis-server running?", flush=True)
        sys.exit(1)

    # Check engine_active flag
    if not ri.engine_is_active():
        print(f"[Engine] WARNING: Redis key '{ENGINE_ACTIVE_KEY}' is not set to 1.")
        print("[Engine] The engine will still run but the simulation may also write results.")
        print("[Engine] Set engine_active=1 in Redis or use run_all_agents.sh to avoid conflicts.",
              flush=True)

    engine = MetricsEngine(ri)
    stop_flag = [False]
    _setup_signal(stop_flag)

    processed     = 0
    failed        = 0
    per_agent     = defaultdict(int)
    start_time    = time.time()
    last_log_time = start_time

    print("[Engine] Ready. Waiting for task decisions...", flush=True)

    while not stop_flag[0]:
        task_id = ri.pop_engine_request(timeout=poll_timeout)
        if task_id is None:
            continue  # timeout — loop back and check stop_flag

        # Read agent name before processing (for stats)
        dec = ri.get_decision(task_id)
        agent_name = dec.get("agent", "unknown") if dec else "unknown"

        ok = engine.process_task(task_id)
        if ok:
            processed += 1
            per_agent[agent_name] += 1
        else:
            failed += 1
            print(f"[Engine] WARN: failed to process task {task_id} "
                  f"(missing decision or request in Redis)", flush=True)

        now = time.time()
        if processed > 0 and processed % log_interval == 0:
            _print_stats(processed, per_agent, failed, now - start_time)

    elapsed = time.time() - start_time
    print(f"\n[Engine] Shutdown complete. "
          f"Processed={processed} Failed={failed} Elapsed={elapsed:.1f}s", flush=True)
    _print_stats(processed, per_agent, failed, elapsed)
    ri.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IoV MEC Realistic Metrics Engine — computes physically grounded task metrics"
    )
    parser.add_argument("--host",       default="127.0.0.1", help="Redis host")
    parser.add_argument("--port",       default=6379,  type=int, help="Redis port")
    parser.add_argument("--db",         default=0,     type=int, help="Redis DB index")
    parser.add_argument("--log-interval", default=50,  type=int,
                        help="Print stats every N processed tasks")
    parser.add_argument("--poll-timeout", default=0.5, type=float,
                        help="Blocking pop timeout in seconds")
    parser.add_argument("--experiment", default=None,
                        help="(Informational) experiment name from experiments/configs.py")
    parser.add_argument("--k",          default=None,  type=int,
                        help="(Informational) action mask k value being evaluated")
    args = parser.parse_args()

    if args.experiment:
        print(f"[Engine] Experiment: {args.experiment}", flush=True)
    if args.k is not None:
        print(f"[Engine] Action mask k: {args.k}", flush=True)

    run(
        host         = args.host,
        port         = args.port,
        db           = args.db,
        log_interval = args.log_interval,
        poll_timeout = args.poll_timeout,
    )


if __name__ == "__main__":
    main()
