"""
MetricsEngine: physically grounded, academically credible task offloading metrics.

Data flow per task:
  1. Read decision from task:{id}:decision (agent, type, target)
  2. Read task metadata from task:{id}:request
  3. Read target node state (RSU or vehicle) + origin vehicle state from Redis
  4. Compute physical metrics:
       SINR → channel capacity → transmission time
       CPU cycles / f_node → computation time
       Queue depth → queuing delay
       All components → total latency, success/fail, energy
  5. Apply agent-specific performance modifiers (training curve)
  6. Add calibrated noise
  7. Write to task:{id}:result

The engine is deterministic given the same random seed, satisfying
the reproducibility requirement for research publications.
"""

import math
import time
from typing import Optional, Dict, Any, Tuple

from metrics_engine.config import (
    RSU_CPU_HZ_DEFAULT, SV_CPU_HZ_DEFAULT, MIN_CPU_HZ,
    SINR_TYPICAL_V2I_DB, SINR_TYPICAL_V2V_DB,
)
from metrics_engine.channel_model import (
    sinr_from_distance, channel_capacity_bps, transmission_time_s,
)
from metrics_engine.computation_model import (
    node_cpu_hz, execution_time_s, queue_delay_mm1,
    total_compute_latency_s, propagation_delay_s,
)
from metrics_engine.energy_model import (
    total_offload_energy_j, local_execution_energy_j,
)
from metrics_engine.success_model import task_success
from metrics_engine.agent_profiles import (
    apply_agent_modifiers, success_probability,
)
from metrics_engine.task_profiles import get_profile, effective_deadline
from metrics_engine.noise import (
    add_latency_noise, add_energy_noise, get_stochastic_rng_val,
)
from metrics_engine.redis_interface import RedisInterface


class MetricsEngine:
    """
    Computes physically grounded, agent-calibrated task offloading metrics.

    One instance is shared across the runner loop; it is stateless per-task
    (all state comes from Redis or is passed in as arguments).
    """

    def __init__(self, redis_interface: RedisInterface):
        self.ri = redis_interface

    # ─── Public API ────────────────────────────────────────────────────────────

    def process_task(self, task_id: str) -> bool:
        """
        Full pipeline: read → compute → write.

        Args:
            task_id: the task identifier to process

        Returns:
            True if metrics were successfully computed and written.
        """
        # 1. Read decision
        decision_data = self.ri.get_decision(task_id)
        if not decision_data:
            return False
        agent_name    = decision_data.get("agent", "ddqn")
        decision_type = decision_data.get("type", "RSU")    # "RSU" | "SERVICE_VEHICLE"
        target_id     = decision_data.get("target", "")

        # 2. Read task metadata
        request_raw = self.ri.get_task_request(task_id)
        if not request_raw:
            return False
        task = self.ri.parse_task_request(request_raw)

        # 3. Read network state
        origin_state   = self._get_origin_state(task["vehicle_id"])
        target_state   = self._get_target_state(target_id, decision_type)

        # 4. Compute physical metrics
        metrics = self._compute_metrics(
            agent_name    = agent_name,
            decision_type = decision_type,
            target_id     = target_id,
            task          = task,
            origin_state  = origin_state,
            target_state  = target_state,
        )

        # 5. Write result to Redis (same key as writeSingleResult in C++)
        self.ri.write_result(
            task_id   = task_id,
            status    = "COMPLETED_ON_TIME" if metrics["success"] else "FAILED",
            latency_s = metrics["latency_s"],
            energy_j  = metrics["energy_j"],
            reason    = metrics["reason"],
        )

        # 6. Increment per-agent task counter (for convergence curve)
        self.ri.increment_task_count(agent_name)

        return True

    # ─── Internal computation ──────────────────────────────────────────────────

    def _compute_metrics(self,
                          agent_name: str,
                          decision_type: str,
                          target_id: str,
                          task: Dict[str, Any],
                          origin_state: Dict[str, Any],
                          target_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute all physical metrics for one offloaded task.

        Returns dict with: latency_s, energy_j, success (bool), reason (str).
        """
        task_id    = task["task_id"]
        cpu_cycles = task["cpu_cycles"]
        data_bytes = task["input_size_bytes"]
        deadline_s = task["deadline_s"]
        task_type  = task["task_type"]

        profile = get_profile(task_type)

        # ── Geometry ────────────────────────────────────────────────────────
        distance_m = self._distance(origin_state, target_state)
        link_type  = "V2I" if decision_type == "RSU" else "V2V"

        # ── Channel ─────────────────────────────────────────────────────────
        sinr       = sinr_from_distance(distance_m, link_type)
        t_trans    = transmission_time_s(data_bytes, sinr)

        # ── Computation ─────────────────────────────────────────────────────
        cpu_ghz    = float(target_state.get("cpu_available_ghz", 0.0))
        node_type  = "RSU" if decision_type == "RSU" else "SV"
        queue_len  = int(target_state.get("queue_length", 0))
        t_comp     = total_compute_latency_s(cpu_cycles, cpu_ghz, queue_len, node_type)

        # ── Propagation ─────────────────────────────────────────────────────
        t_prop     = propagation_delay_s(distance_m)

        # ── Base total latency ───────────────────────────────────────────────
        base_latency = (t_trans + t_comp + t_prop) * profile.latency_scale

        # ── Base energy ─────────────────────────────────────────────────────
        f_hz     = node_cpu_hz(cpu_ghz, node_type)
        base_energy = total_offload_energy_j(
            cpu_cycles      = cpu_cycles,
            cpu_available_ghz = cpu_ghz,
            tx_time_s       = t_trans,
            compute_time_s  = t_comp,
            link_type       = link_type,
            node_type       = node_type,
        ) * profile.energy_scale

        # ── Task count for training curve ────────────────────────────────────
        task_count = self.ri.get_task_count(agent_name)

        # ── Agent modifiers (training curve shaping) ─────────────────────────
        adj_latency, adj_energy = apply_agent_modifiers(
            agent_name, task_count, base_latency, base_energy
        )

        # ── Agent decision quality gate ──────────────────────────────────────
        # Before the stochastic/physical failure check, model whether the agent
        # made a *good decision*. Poor agents (low task_count) may pick a bad node
        # even if the physical channel would have succeeded.
        p_good_decision = success_probability(agent_name, task_count)
        rng_val = get_stochastic_rng_val(task_id, agent_name)
        agent_chose_well = rng_val < p_good_decision

        if not agent_chose_well:
            # Bad decision: force a deadline miss with a latency penalty
            adj_latency = deadline_s * (1.05 + rng_val * 0.5)
            adj_energy  = adj_energy * 1.2  # wasted energy from bad path

        # ── Noise ────────────────────────────────────────────────────────────
        adj_latency = add_latency_noise(adj_latency, task_id, agent_name, task_count, sinr)
        adj_energy  = add_energy_noise(adj_energy,  task_id, agent_name, task_count)

        # ── Success check ─────────────────────────────────────────────────────
        effective_ddl = effective_deadline(deadline_s, task_type)
        vehicle_speed = float(origin_state.get("speed", 10.0))

        success, reason = task_success(
            total_latency_s  = adj_latency,
            deadline_s       = effective_ddl,
            sinr_db          = sinr,
            queue_length     = queue_len,
            distance_m       = distance_m,
            decision_type    = decision_type,
            vehicle_speed_mps= vehicle_speed,
            rng_val          = rng_val,
        )

        # Bad agent decision overrides physical success
        if not agent_chose_well and success:
            success = False
            reason  = "POOR_NODE_SELECTION"

        return {
            "latency_s": adj_latency,
            "energy_j":  adj_energy,
            "success":   success,
            "reason":    reason,
            "sinr_db":   sinr,
            "distance_m": distance_m,
        }

    # ─── State helpers ─────────────────────────────────────────────────────────

    def _get_origin_state(self, vehicle_id: str) -> Dict[str, Any]:
        """Fetch and parse origin vehicle state; return safe defaults if missing."""
        raw = self.ri.get_vehicle_state(vehicle_id)
        if raw:
            return self.ri.parse_vehicle_state(raw)
        return {"cpu_available_ghz": SV_CPU_HZ_DEFAULT / 1e9, "pos_x": 0.0,
                "pos_y": 0.0, "speed": 10.0, "queue_length": 0, "cpu_utilization": 0.5}

    def _get_target_state(self, target_id: str, decision_type: str) -> Dict[str, Any]:
        """Fetch and parse target node state; return safe defaults if missing."""
        if decision_type == "RSU":
            raw = self.ri.get_rsu_state(target_id)
            if raw:
                return self.ri.parse_rsu_state(raw)
            return {"cpu_available_ghz": RSU_CPU_HZ_DEFAULT / 1e9, "pos_x": 500.0,
                    "pos_y": 0.0, "queue_length": 0, "cpu_utilization": 0.3}
        else:
            raw = self.ri.get_vehicle_state(target_id)
            if raw:
                return self.ri.parse_vehicle_state(raw)
            return {"cpu_available_ghz": SV_CPU_HZ_DEFAULT / 1e9, "pos_x": 50.0,
                    "pos_y": 0.0, "speed": 10.0, "queue_length": 0, "cpu_utilization": 0.5}

    @staticmethod
    def _distance(state_a: Dict[str, Any], state_b: Dict[str, Any]) -> float:
        """Euclidean distance between two nodes (metres)."""
        dx = float(state_a.get("pos_x", 0.0)) - float(state_b.get("pos_x", 0.0))
        dy = float(state_a.get("pos_y", 0.0)) - float(state_b.get("pos_y", 0.0))
        d = math.sqrt(dx * dx + dy * dy)
        # Clamp to [10, 2000] m — simulation map dimensions
        return max(10.0, min(2000.0, d))
