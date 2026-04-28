"""
Task success/failure model.

Failure modes considered:
  1. Deadline miss:     t_total > deadline_s
  2. Channel outage:    SINR < SINR_MIN_RELIABLE_DB (link not reliable)
  3. V2V link break:    mobility-induced interruption for SV offloading
  4. Node overload:     RSU queue above saturation threshold

Success probability combines these failure modes multiplicatively.

References:
  Togou et al., "DDRL-based Task Offloading and Resource Allocation in MEC",
      IEEE IoTJ, 2021.
  Liu et al., "Cooperative Edge Computing with Channel Uncertainty",
      IEEE INFOCOM, 2021.
"""

import math
from metrics_engine.config import (
    SINR_MIN_RELIABLE_DB, V2V_LINK_BREAK_PROB_PER_100M,
    RSU_QUEUE_OVERFLOW_THRESHOLD,
)
from metrics_engine.channel_model import is_link_reliable


def deadline_met(total_latency_s: float, deadline_s: float) -> bool:
    """True if the task completes before its deadline."""
    return total_latency_s <= deadline_s


def channel_success_prob(sinr_db: float) -> float:
    """
    Probability of a successful transmission given SINR (dB).

    Models BER-based packet loss:
      - SINR ≥ threshold → P_success = 1.0
      - SINR < threshold → degraded probability decaying with SINR deficit
    The degradation follows a sigmoid centred at SINR_MIN_RELIABLE_DB.

    Returns:
        probability in [0.0, 1.0]
    """
    if sinr_db >= SINR_MIN_RELIABLE_DB + 5.0:
        return 1.0
    if sinr_db < SINR_MIN_RELIABLE_DB - 10.0:
        return 0.01  # essentially unreliable
    # Smooth degradation across ±10 dB around the threshold
    x = sinr_db - SINR_MIN_RELIABLE_DB
    return max(0.0, min(1.0, 0.5 + x / 20.0))


def v2v_link_stability_prob(distance_m: float, vehicle_speed_mps: float = 10.0) -> float:
    """
    Probability that a V2V link remains stable for the duration of a task.

    Higher speed and longer distance increase the probability of link break.
    Model: P_stable = exp(-λ × d × v) where λ = V2V_LINK_BREAK_PROB_PER_100M/100

    Args:
        distance_m: V2V separation
        vehicle_speed_mps: average speed of the faster vehicle (m/s)

    Returns:
        probability in [0.0, 1.0]
    """
    lambda_per_m = V2V_LINK_BREAK_PROB_PER_100M / 100.0
    # Speed normalisation: at 20 m/s, mobility doubles the instability rate
    speed_factor = max(1.0, vehicle_speed_mps / 10.0)
    rate = lambda_per_m * distance_m * speed_factor
    return math.exp(-rate)


def node_availability_prob(queue_length: int, node_type: str = "RSU") -> float:
    """
    Probability that the node accepts the task without overflow.

    For RSU: sigmoid drop at RSU_QUEUE_OVERFLOW_THRESHOLD.
    For SV: more aggressive drop at smaller queue depth (limited memory).

    Returns:
        probability in [0.0, 1.0]
    """
    if node_type == "RSU":
        threshold = RSU_QUEUE_OVERFLOW_THRESHOLD
    else:
        threshold = 8  # SVs saturate faster

    if queue_length <= threshold * 0.5:
        return 1.0
    x = (queue_length - threshold * 0.5) / (threshold * 0.5)
    return max(0.05, 1.0 - 0.9 * min(1.0, x))


def task_success(total_latency_s: float,
                 deadline_s: float,
                 sinr_db: float,
                 queue_length: int,
                 distance_m: float,
                 decision_type: str,
                 vehicle_speed_mps: float = 10.0,
                 rng_val: float = 0.5) -> tuple:
    """
    Determine task success or failure.

    Evaluates all failure modes in priority order:
    1. Channel outage (immediate physical impossibility)
    2. Node overload (refused at ingress)
    3. Deadline miss (most common failure mode)
    4. V2V link break during execution (SV only)

    Args:
        total_latency_s: computed total task latency
        deadline_s: task deadline in seconds
        sinr_db: link SINR at time of offloading
        queue_length: queue depth at destination node
        distance_m: distance to destination
        decision_type: "RSU" or "SERVICE_VEHICLE"
        vehicle_speed_mps: origin vehicle speed (for V2V stability)
        rng_val: uniform [0,1] for stochastic outcomes

    Returns:
        (success: bool, reason: str)
    """
    # 1. Channel reliability check
    p_channel = channel_success_prob(sinr_db)
    if rng_val > p_channel:
        return False, "CHANNEL_OUTAGE"

    # 2. Node availability (queue saturation)
    node_type = "RSU" if decision_type == "RSU" else "SV"
    p_node = node_availability_prob(queue_length, node_type)
    if rng_val > p_node:
        return False, "NODE_OVERLOADED"

    # 3. Deadline check (deterministic given the latency model)
    if not deadline_met(total_latency_s, deadline_s):
        return False, "DEADLINE_MISSED"

    # 4. V2V link break (SV only, stochastic)
    if decision_type == "SERVICE_VEHICLE":
        p_v2v = v2v_link_stability_prob(distance_m, vehicle_speed_mps)
        # Use a different bucket of rng_val to avoid correlation with channel check
        if (1.0 - rng_val) > p_v2v:
            return False, "SV_LINK_BREAK"

    return True, "NONE"
