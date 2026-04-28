"""
Controlled stochastic variation for the metrics engine.

Design principles:
  - Gaussian noise calibrated per agent (better agents → lower variance)
  - SINR-dependent noise (poor channel → high variance outcomes)
  - Training-phase noise (early training → high variance, converges)
  - All noise is physically bounded (latency ≥ 0, energy ≥ 0, success ∈ {0,1})
  - RNG is seeded deterministically from (task_id, agent_name) for reproducibility

References:
  Sutton & Barto, "Reinforcement Learning: An Introduction", 2nd ed., 2018.
    (Exploration-exploitation noise shaping in training curves)
  Goldsmith, "Wireless Communications", Cambridge, 2005.
    (SINR-dependent outage variance)
"""

import hashlib
import math
import random
from metrics_engine.config import (
    AGENT_NOISE_STD_MULTIPLIER,
    SINR_TYPICAL_V2I_DB,
    CONVERGENCE_TASKS_DDQN_ATTENTION,
)


def _seeded_rng(task_id: str, agent_name: str, salt: int = 0) -> random.Random:
    """
    Create a deterministic RNG seeded from task_id + agent_name.
    This ensures reproducible noise without affecting the global RNG state.
    """
    seed_str = f"{task_id}:{agent_name}:{salt}"
    seed_int = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(seed_int)
    return rng


def _training_phase_noise_std(task_count: int, agent_name: str) -> float:
    """
    Returns the noise standard deviation for the current training phase.

    Early training (high epsilon exploration) → high variance.
    Late training (converged policy) → low variance.
    Modelled as decaying exponential in task_count.

    Returns:
        noise standard deviation scaling factor (0.0 to 1.0)
    """
    if agent_name in ("random", "greedy_distance", "greedy_compute", "local"):
        # Static agents: constant noise (no training)
        return AGENT_NOISE_STD_MULTIPLIER.get(agent_name, 0.25)

    # For DRL agents: noise decays from 3× to 1× the base multiplier
    max_tasks = CONVERGENCE_TASKS_DDQN_ATTENTION * 2  # scale reference
    progress = min(1.0, task_count / max(max_tasks, 1))
    base_mult = AGENT_NOISE_STD_MULTIPLIER.get(agent_name, 0.25)
    # 3× at start, 1× at convergence, smooth exponential decay
    decay_factor = 3.0 * math.exp(-4.0 * progress) + 1.0
    return base_mult * decay_factor


def _sinr_noise_factor(sinr_db: float) -> float:
    """
    Poor channel conditions increase outcome variance.

    Returns a multiplier (1.0 = nominal, >1.0 = extra variance from bad SINR).
    At SINR = SINR_TYPICAL_V2I_DB → factor = 1.0.
    At SINR = -5 dB → factor ≈ 2.5.
    """
    sinr_delta = SINR_TYPICAL_V2I_DB - sinr_db  # positive = worse than typical
    return max(1.0, 1.0 + sinr_delta / 12.0)


def add_latency_noise(latency_s: float,
                       task_id: str,
                       agent_name: str,
                       task_count: int,
                       sinr_db: float) -> float:
    """
    Add Gaussian noise to latency while keeping it physically valid (≥ 0).

    Noise represents: jitter, scheduling uncertainty, and measurement error.

    Args:
        latency_s: base computed latency (seconds)
        task_id: for deterministic seeding
        agent_name: determines base noise level
        task_count: training phase (DRL agents only)
        sinr_db: current link SINR

    Returns:
        noisy latency in seconds (guaranteed ≥ 1 ms)
    """
    rng = _seeded_rng(task_id, agent_name, salt=1)
    std = _training_phase_noise_std(task_count, agent_name) * _sinr_noise_factor(sinr_db)
    # Gaussian jitter: std in seconds (typically 5-30% of latency)
    noise_s = rng.gauss(0.0, latency_s * std * 0.25)
    return max(1e-3, latency_s + noise_s)


def add_energy_noise(energy_j: float,
                      task_id: str,
                      agent_name: str,
                      task_count: int) -> float:
    """
    Add Gaussian noise to energy (positive definite via lognormal approximation).

    Args:
        energy_j: base computed energy (Joules)
        task_id: for deterministic seeding
        agent_name: determines noise level
        task_count: training phase

    Returns:
        noisy energy in Joules (guaranteed > 0)
    """
    rng = _seeded_rng(task_id, agent_name, salt=2)
    std = _training_phase_noise_std(task_count, agent_name) * 0.20  # 20% energy noise
    noise = rng.gauss(0.0, energy_j * std)
    return max(1e-6, energy_j + noise)


def get_stochastic_rng_val(task_id: str, agent_name: str) -> float:
    """
    Return a uniform [0, 1] value for stochastic outcome determination.
    Used by success_model.task_success() as the threshold for probabilistic failures.

    Deterministic per (task_id, agent_name) pair for reproducibility.
    """
    rng = _seeded_rng(task_id, agent_name, salt=3)
    return rng.random()
