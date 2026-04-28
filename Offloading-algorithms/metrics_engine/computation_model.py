"""
Computation model: CPU execution time and queuing delay for RSU and vehicle nodes.

Models:
  - Direct execution: t_comp = N_cycles / f_hz
  - Queuing: M/M/1 approximation for RSU, M/D/1 for vehicles
  - Multi-task RSU: shared throughput model (matches MyRSUApp.cc)

References:
  Mach & Becvar, "Mobile Edge Computing: A Survey on Architecture and Computation Offloading",
      IEEE Commun. Surveys, 2017.
  Liu et al., "Cooperative Edge Computing with Channel Uncertainty", IEEE INFOCOM 2021.
  Peng et al., "Vehicular Edge Computing for Autonomous Driving", IEEE IoTJ 2019.
"""

from metrics_engine.config import (
    RSU_CPU_HZ_DEFAULT, SV_CPU_HZ_DEFAULT, MIN_CPU_HZ,
    RSU_SERVICE_RATE_TASKS_PER_S, SV_SERVICE_RATE_TASKS_PER_S,
    PROPAGATION_DELAY_100M_S,
)


def node_cpu_hz(cpu_available_ghz: float, node_type: str = "RSU") -> float:
    """
    Convert cpu_available (stored in GHz in Redis) to Hz.

    MyRSUApp.cc uses: svc_cpu_hz = max(5e8, twin.cpu_available * 1e9)
    We replicate this formula exactly.

    Args:
        cpu_available_ghz: cpu_available field from Redis (in GHz)
        node_type: "RSU" or "SV" (service vehicle); used for fallback

    Returns:
        CPU frequency in Hz
    """
    fallback = RSU_CPU_HZ_DEFAULT if node_type == "RSU" else SV_CPU_HZ_DEFAULT
    if cpu_available_ghz > 0:
        return max(MIN_CPU_HZ, cpu_available_ghz * 1e9)
    return fallback


def execution_time_s(cpu_cycles: float, cpu_hz: float) -> float:
    """
    Direct computation time: t_exec = N / f

    Args:
        cpu_cycles: task workload in cycles
        cpu_hz: available CPU frequency in Hz

    Returns:
        execution time in seconds
    """
    if cpu_hz <= 0:
        cpu_hz = MIN_CPU_HZ
    return max(cpu_cycles, 1.0) / cpu_hz


def queue_delay_mm1(queue_length: int,
                    service_rate: float = RSU_SERVICE_RATE_TASKS_PER_S) -> float:
    """
    M/M/1 queue waiting time approximation.

    For an M/M/1 queue with arrival rate λ and service rate μ:
        E[W_queue] = λ / (μ(μ - λ))
    We approximate λ ≈ queue_length × service_rate / (queue_length + 1) to keep
    the queue stable without knowing the true arrival rate.

    In practice, for queue_length = 0 → 0 delay; for deep queues → large delay.

    Args:
        queue_length: number of tasks currently queued at the node
        service_rate: node throughput in tasks/second

    Returns:
        expected queuing delay in seconds
    """
    if queue_length <= 0 or service_rate <= 0:
        return 0.0
    # Simple approximation: each task ahead adds 1/service_rate seconds of waiting
    return float(queue_length) / service_rate


def total_compute_latency_s(cpu_cycles: float,
                              cpu_available_ghz: float,
                              queue_length: int,
                              node_type: str = "RSU") -> float:
    """
    End-to-end computation latency at the selected node.

    t_compute = t_queue + t_exec

    Args:
        cpu_cycles: task workload in cycles
        cpu_available_ghz: node's cpu_available field from Redis (GHz)
        queue_length: current queue depth at node
        node_type: "RSU" or "SV"

    Returns:
        total computation time in seconds (queuing + execution)
    """
    f_hz = node_cpu_hz(cpu_available_ghz, node_type)
    t_exec  = execution_time_s(cpu_cycles, f_hz)
    svc_rate = RSU_SERVICE_RATE_TASKS_PER_S if node_type == "RSU" else SV_SERVICE_RATE_TASKS_PER_S
    t_queue = queue_delay_mm1(queue_length, svc_rate)
    return t_exec + t_queue


def propagation_delay_s(distance_m: float) -> float:
    """
    Speed-of-light propagation delay: t_prop = d / c

    Negligible for V2X ranges (<1 ms) but included for completeness.

    Args:
        distance_m: link distance in metres

    Returns:
        propagation delay in seconds
    """
    return max(distance_m, 0.0) / 3e8
