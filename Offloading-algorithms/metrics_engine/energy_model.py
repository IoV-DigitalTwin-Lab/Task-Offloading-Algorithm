"""
Energy model for IoV MEC task offloading.

Models:
  1. Computation energy: E_comp = κ × f² × N_cycles   (dynamic CMOS power)
  2. Transmission energy: E_tx = P_tx × t_trans
  3. Idle reception energy: E_idle = P_circuit × t_wait

The computation energy formula matches MyRSUApp.cc (lines 825, 860, 3909, 3987).

References:
  Kumar et al., "A Survey of Computation Offloading for Mobile Systems",
      ACM Mobile Networks and Applications, 2013.
  Mach & Becvar, "Mobile Edge Computing: A Survey", IEEE Commun. Surveys, 2017.
  ETSI EN 302 663 for 802.11p transmit power specifications.
"""

from metrics_engine.config import (
    ENERGY_KAPPA_RSU, ENERGY_KAPPA_SV,
    TX_POWER_V2I_WATTS, TX_POWER_V2V_WATTS,
    CIRCUIT_POWER_IDLE_W, MIN_CPU_HZ,
)
from metrics_engine.computation_model import node_cpu_hz


def computation_energy_j(cpu_cycles: float,
                          cpu_hz: float,
                          node_type: str = "RSU") -> float:
    """
    Dynamic CMOS computation energy: E = κ × f² × N

    This formula matches MyRSUApp.cc's energy calculation:
        energy_j = 2e-27 * f_hz * f_hz * cpu_cycles

    Args:
        cpu_cycles: workload in CPU cycles
        cpu_hz: clock frequency in Hz
        node_type: "RSU" (κ=2e-27) or "SV" (κ=5e-27)

    Returns:
        computation energy in Joules
    """
    kappa = ENERGY_KAPPA_RSU if node_type == "RSU" else ENERGY_KAPPA_SV
    f = max(cpu_hz, MIN_CPU_HZ)
    return kappa * f * f * max(cpu_cycles, 0.0)


def transmission_energy_j(tx_time_s: float, link_type: str = "V2I") -> float:
    """
    Transmission energy at the originating vehicle.
    E_tx = P_tx × t_trans

    Args:
        tx_time_s: transmission duration in seconds
        link_type: "V2I" (23 dBm / 0.2 W) or "V2V" (20 dBm / 0.1 W)

    Returns:
        transmission energy in Joules
    """
    p_watts = TX_POWER_V2I_WATTS if link_type == "V2I" else TX_POWER_V2V_WATTS
    return p_watts * max(tx_time_s, 0.0)


def idle_energy_j(wait_time_s: float) -> float:
    """
    Idle circuit energy while waiting for remote computation result.
    E_idle = P_circuit × t_wait

    This models the vehicle's transceiver idle power during the offloading wait.

    Args:
        wait_time_s: wait duration (t_comp + t_queue at the remote node)

    Returns:
        idle energy in Joules
    """
    return CIRCUIT_POWER_IDLE_W * max(wait_time_s, 0.0)


def total_offload_energy_j(cpu_cycles: float,
                            cpu_available_ghz: float,
                            tx_time_s: float,
                            compute_time_s: float,
                            link_type: str = "V2I",
                            node_type: str = "RSU") -> float:
    """
    Total energy expended by the originating vehicle to offload one task.

    E_total = E_tx + E_idle
    (The computation energy is borne by the remote node, not the originating vehicle.)

    Args:
        cpu_cycles: task workload in cycles
        cpu_available_ghz: remote node CPU (GHz) from Redis
        tx_time_s: transmission time (seconds)
        compute_time_s: remote computation time (seconds)
        link_type: "V2I" or "V2V"
        node_type: "RSU" or "SV"

    Returns:
        total energy at the originating vehicle in Joules
    """
    e_tx   = transmission_energy_j(tx_time_s, link_type)
    e_idle = idle_energy_j(compute_time_s)
    return e_tx + e_idle


def local_execution_energy_j(cpu_cycles: float, local_cpu_hz: float) -> float:
    """
    Energy for executing a task locally on the originating vehicle.
    Uses vehicle κ coefficient (higher than RSU).

    Args:
        cpu_cycles: task workload in cycles
        local_cpu_hz: vehicle's own CPU frequency in Hz

    Returns:
        local execution energy in Joules
    """
    return computation_energy_j(cpu_cycles, local_cpu_hz, node_type="SV")
