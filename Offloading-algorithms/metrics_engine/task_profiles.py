"""
Per-task-type performance characteristics for IoV MEC.

The six task types from the simulation taxonomy each have distinct:
  - Compute intensity (cycles per byte)
  - Latency sensitivity (deadline tightness class)
  - QoS category: LATENCY_SENSITIVE | COMPUTE_INTENSIVE | BALANCED

QoS categories map to success criteria:
  - LATENCY_SENSITIVE: tight deadline; latency dominates success
  - COMPUTE_INTENSIVE: loose deadline; compute quality dominates
  - BALANCED: moderate deadline; both matter

References:
  Al-Turjman et al., "Intelligence and Security in Big 5G-Oriented IoNT",
      Future Generation Computer Systems, 2020.
  Zeng et al., "Boomerang: On-Demand Cooperative Deep Neural Network Inference
      for Edge Intelligence", ACM HotNets, 2019.
  Liu et al., "When Mobile Blockchain Meets Edge Computing", IEEE Commun. Mag., 2018.
"""

from dataclasses import dataclass
from typing import Dict

LATENCY_SENSITIVE  = "LATENCY_SENSITIVE"
COMPUTE_INTENSIVE  = "COMPUTE_INTENSIVE"
BALANCED           = "BALANCED"


@dataclass(frozen=True)
class TaskProfile:
    """Characteristics of one IoV task type."""
    name: str
    qos_category: str       # LATENCY_SENSITIVE | COMPUTE_INTENSIVE | BALANCED
    # Fraction of the raw deadline to treat as the "tight" success threshold.
    # (1.0 = must finish by deadline, 0.8 = must finish at 80% of deadline)
    deadline_tightness: float
    # Scale factor on computed latency for tasks of this type.
    # Object detection is latency-critical (SINR matters most);
    # Forecast tasks are compute-heavy (CPU quality matters most).
    latency_scale: float
    # Scale factor on computed energy for tasks of this type.
    energy_scale: float
    # Description for RESULTS_GUIDE.md
    description: str


TASK_PROFILES: Dict[str, TaskProfile] = {
    "LOCAL_OBJECT_DETECTION": TaskProfile(
        name="LOCAL_OBJECT_DETECTION",
        qos_category=LATENCY_SENSITIVE,
        deadline_tightness=0.85,
        latency_scale=0.90,    # typically small data, fast path
        energy_scale=0.80,
        description="Safety-critical obstacle detection; tight deadline, GPU-accelerated at RSU",
    ),
    "COOPERATIVE_PERCEPTION": TaskProfile(
        name="COOPERATIVE_PERCEPTION",
        qos_category=LATENCY_SENSITIVE,
        deadline_tightness=0.90,
        latency_scale=1.10,    # larger data (sensor fusion)
        energy_scale=1.10,
        description="Multi-vehicle sensor fusion; moderate data, tight deadline",
    ),
    "ROUTE_OPTIMIZATION": TaskProfile(
        name="ROUTE_OPTIMIZATION",
        qos_category=BALANCED,
        deadline_tightness=1.00,
        latency_scale=0.85,    # lightweight compute, loose deadline
        energy_scale=0.75,
        description="Graph-based routing; lightweight, benefits from RSU map data",
    ),
    "FLEET_TRAFFIC_FORECAST": TaskProfile(
        name="FLEET_TRAFFIC_FORECAST",
        qos_category=COMPUTE_INTENSIVE,
        deadline_tightness=1.00,
        latency_scale=1.20,    # heavy ML inference
        energy_scale=1.30,
        description="Traffic prediction ML model; compute-heavy, loose latency requirements",
    ),
    "VOICE_COMMAND_PROCESSING": TaskProfile(
        name="VOICE_COMMAND_PROCESSING",
        qos_category=LATENCY_SENSITIVE,
        deadline_tightness=0.80,
        latency_scale=0.75,    # small audio payload, fast NLP inference
        energy_scale=0.65,
        description="NLP inference on short audio; very tight latency, small payload",
    ),
    "SENSOR_HEALTH_CHECK": TaskProfile(
        name="SENSOR_HEALTH_CHECK",
        qos_category=COMPUTE_INTENSIVE,
        deadline_tightness=1.00,
        latency_scale=0.60,    # low-priority, tiny compute
        energy_scale=0.50,
        description="Diagnostic sensor scan; non-critical, very lightweight",
    ),
    "UNKNOWN": TaskProfile(
        name="UNKNOWN",
        qos_category=BALANCED,
        deadline_tightness=1.00,
        latency_scale=1.00,
        energy_scale=1.00,
        description="Unclassified task; use nominal parameters",
    ),
}


def get_profile(task_type: str) -> TaskProfile:
    """Return the TaskProfile for the given task type name (defaults to UNKNOWN)."""
    return TASK_PROFILES.get(task_type, TASK_PROFILES["UNKNOWN"])


def effective_deadline(deadline_s: float, task_type: str) -> float:
    """
    Return the effective success deadline after applying deadline_tightness.
    A LATENCY_SENSITIVE task must finish before 80-90% of the raw deadline;
    other tasks can use the full deadline.
    """
    profile = get_profile(task_type)
    return deadline_s * profile.deadline_tightness
