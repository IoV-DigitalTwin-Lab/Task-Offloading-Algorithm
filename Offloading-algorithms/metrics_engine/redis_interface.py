"""
Redis read/write interface for the metrics engine.

Reads:
  - task:{id}:request     → task metadata (size, cycles, deadline, type)
  - task:{id}:decision    → agent decision (type: RSU/SV, target node id)
  - rsu:{id}:resources    → RSU state (cpu_available, queue_length, pos)
  - vehicle:{id}:state    → vehicle state (cpu_available, pos, speed)
  - engine:agent:{name}:task_count → training step counter
  - engine_active          → flag set by run_all_agents.sh

Writes:
  - task:{id}:result      → status, latency, energy, reason
    (same key that MyRSUApp.cc's writeSingleResult() uses — engine takes priority)
  - engine:agent:{name}:task_count → incremented after each processed task

All key names are derived directly from RedisDigitalTwin.cc and environment.py
to guarantee schema consistency.
"""

import time
from typing import Optional, Dict, Any
import redis as _redis
from metrics_engine.config import (
    ENGINE_RESULT_TTL_S,
    ENGINE_ACTIVE_KEY,
    ENGINE_REQUEST_QUEUE,
    ENGINE_TASK_COUNT_PREFIX,
    ENGINE_TASK_COUNT_SUFFIX,
    MAX_TRAINING_TASKS,
)


class RedisInterface:
    """Thin Redis wrapper for the metrics engine."""

    def __init__(self, host: str = "127.0.0.1", port: int = 6379, db: int = 0):
        self.r = _redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def ping(self) -> bool:
        try:
            return self.r.ping()
        except _redis.RedisError:
            return False

    # ── Reading ───────────────────────────────────────────────────────────────

    def get_task_request(self, task_id: str) -> Optional[Dict[str, str]]:
        """Read task:{id}:request hash (written by MyRSUApp.cc pushOffloadingRequest)."""
        data = self.r.hgetall(f"task:{task_id}:request")
        return data if data else None

    def get_decision(self, task_id: str) -> Optional[Dict[str, str]]:
        """Read task:{id}:decision hash (written by environment.py write_decision)."""
        data = self.r.hgetall(f"task:{task_id}:decision")
        return data if data else None

    def get_rsu_state(self, rsu_id: str) -> Optional[Dict[str, str]]:
        """Read rsu:{id}:resources hash (written by MyRSUApp.cc updateRSUResources)."""
        data = self.r.hgetall(f"rsu:{rsu_id}:resources")
        return data if data else None

    def get_vehicle_state(self, vehicle_id: str) -> Optional[Dict[str, str]]:
        """Read vehicle:{id}:state hash (written by MyRSUApp.cc updateVehicleState)."""
        data = self.r.hgetall(f"vehicle:{vehicle_id}:state")
        return data if data else None

    def get_task_count(self, agent_name: str) -> int:
        """Read per-agent task counter (engine-internal key)."""
        key = f"{ENGINE_TASK_COUNT_PREFIX}{agent_name}{ENGINE_TASK_COUNT_SUFFIX}"
        val = self.r.get(key)
        try:
            return min(int(val), MAX_TRAINING_TASKS) if val else 0
        except (ValueError, TypeError):
            return 0

    def engine_is_active(self) -> bool:
        """True if run_all_agents.sh has set engine_active = 1 in Redis."""
        val = self.r.get(ENGINE_ACTIVE_KEY)
        return val in ("1", "true", "True", "yes")

    def pop_engine_request(self, timeout: float = 0.5) -> Optional[str]:
        """
        Blocking pop from engine_requests:queue.

        Returns task_id string or None on timeout.
        """
        result = self.r.blpop(ENGINE_REQUEST_QUEUE, timeout=timeout)
        if result:
            _, task_id = result
            return task_id
        return None

    def pop_engine_request_nonblocking(self) -> Optional[str]:
        """Non-blocking pop from engine_requests:queue."""
        return self.r.lpop(ENGINE_REQUEST_QUEUE)

    # ── Writing ───────────────────────────────────────────────────────────────

    def write_result(self, task_id: str, status: str, latency_s: float,
                     energy_j: float, reason: str) -> None:
        """
        Write to task:{id}:result — same key that writeSingleResult() targets.

        Fields written:
          status  → "COMPLETED_ON_TIME" or "FAILED"
          latency → total latency in seconds
          energy  → energy in Joules
          reason  → "NONE" or failure code

        The engine writes this BEFORE the C++ simulation would (which has to
        simulate execution first), so the Python training loop reads the engine
        value. When engine_active=1, C++ skips its own writeSingleResult call.
        """
        effective_reason = reason if reason else ("NONE" if status == "COMPLETED_ON_TIME" else "UNKNOWN")
        pipe = self.r.pipeline()
        pipe.hset(f"task:{task_id}:result", mapping={
            "status":  status,
            "latency": str(latency_s),
            "energy":  str(energy_j),
            "reason":  effective_reason,
        })
        pipe.expire(f"task:{task_id}:result", ENGINE_RESULT_TTL_S)
        pipe.execute()

    def increment_task_count(self, agent_name: str) -> int:
        """
        Increment per-agent task counter and return new value.
        Capped at MAX_TRAINING_TASKS to avoid unbounded growth.
        """
        key = f"{ENGINE_TASK_COUNT_PREFIX}{agent_name}{ENGINE_TASK_COUNT_SUFFIX}"
        new_val = self.r.incr(key)
        if new_val > MAX_TRAINING_TASKS:
            self.r.set(key, MAX_TRAINING_TASKS)
            return MAX_TRAINING_TASKS
        return int(new_val)

    def reset_task_count(self, agent_name: str) -> None:
        """Reset the task counter for an agent (called at run start by run_all_agents.sh)."""
        key = f"{ENGINE_TASK_COUNT_PREFIX}{agent_name}{ENGINE_TASK_COUNT_SUFFIX}"
        self.r.set(key, 0)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def parse_task_request(self, data: Dict[str, str]) -> Dict[str, Any]:
        """Convert raw Redis task request hash to typed Python dict."""
        mem_bytes = float(data.get("mem_footprint_bytes", 0) or 0)
        return {
            "task_id":           data.get("task_id", ""),
            "vehicle_id":        data.get("vehicle_id", ""),
            "rsu_id":            data.get("rsu_id", "RSU_0"),
            "mem_footprint_bytes": mem_bytes,
            "input_size_bytes":  float(data.get("input_size_bytes", mem_bytes) or mem_bytes),
            "cpu_cycles":        float(data.get("cpu_cycles", 1e8) or 1e8),
            "deadline_s":        float(data.get("deadline_seconds", 1.0) or 1.0),
            "qos_value":         float(data.get("qos_value", 1.0) or 1.0),
            "task_type":         data.get("task_type", "UNKNOWN"),
            "is_offloadable":    int(data.get("is_offloadable", 1) or 1),
            "priority_level":    int(data.get("priority_level", 1) or 1),
        }

    def parse_rsu_state(self, data: Dict[str, str]) -> Dict[str, Any]:
        """Convert raw Redis RSU resources hash to typed dict."""
        return {
            "cpu_available_ghz": float(data.get("cpu_available", 0) or 0),
            "memory_available":  float(data.get("memory_available", 0) or 0),
            "queue_length":      int(float(data.get("queue_length", 0) or 0)),
            "pos_x":             float(data.get("pos_x", 0) or 0),
            "pos_y":             float(data.get("pos_y", 0) or 0),
            "cpu_utilization":   float(data.get("cpu_utilization", 0) or 0),
        }

    def parse_vehicle_state(self, data: Dict[str, str]) -> Dict[str, Any]:
        """Convert raw Redis vehicle state hash to typed dict."""
        return {
            "cpu_available_ghz": float(data.get("cpu_available", 0) or 0),
            "mem_available":     float(data.get("mem_available", 0) or 0),
            "pos_x":             float(data.get("pos_x", 0) or 0),
            "pos_y":             float(data.get("pos_y", 0) or 0),
            "speed":             float(data.get("speed", 0) or 0),
            "queue_length":      int(float(data.get("queue_length", 0) or 0)),
            "cpu_utilization":   float(data.get("cpu_utilization", 0) or 0),
        }

    def close(self) -> None:
        self.r.close()
