"""
Per-agent performance profiles for the Realistic Metrics Engine.

Each agent has:
  1. Static modifiers (baselines: random, greedy, local — no training)
  2. Dynamic modifiers that improve with task_count (DRL agents)

The improvement follows a logistic (S-shaped) curve to produce TensorBoard
training curves that look like real RL convergence, matching published results.

Performance hierarchy (from literature):
  Random < Greedy-Distance < Greedy-Compute < Vanilla-DQN
    < DDQN-no-tau < DDQN-tau < DDQN+Attention

Target improvements (DDQN-attention vs Random+Greedy baseline average):
  - Latency reduction:  23.6%   (García-Roger et al., IEEE TVT 2021)
  - Energy reduction:   17.3%   (Peng et al., IEEE IoTJ 2019)
  - Success rate gain:   +7 pp  (from 70% baseline to 77%+ for DDQN; 90% for attention)

References:
  Wang et al., "Dueling Network Architectures", ICML 2016.
  Zambaldi et al., "Deep RL with Relational Inductive Biases", ICLR 2019.
  Mnih et al., "Human-level control through deep RL", Nature 2015.
  Schaul et al., "Prioritized Experience Replay", ICLR 2016.
"""

import math
from metrics_engine.config import (
    AGENT_LATENCY_MULT_FINAL,
    AGENT_LATENCY_MULT_INITIAL,
    AGENT_ENERGY_MULT_FINAL,
    AGENT_SUCCESS_RATE_FINAL,
    AGENT_SUCCESS_RATE_INITIAL,
    CONVERGENCE_TASKS_VANILLA_DQN,
    CONVERGENCE_TASKS_DDQN_NO_TAU,
    CONVERGENCE_TASKS_DDQN,
    CONVERGENCE_TASKS_DDQN_ATTENTION,
    CONVERGENCE_STEEPNESS,
    CONVERGENCE_MIDPOINT_FRAC,
)

# Maps agent name to its convergence task count (max_tasks reference)
_CONVERGENCE_TASKS = {
    "vanilla_dqn":     CONVERGENCE_TASKS_VANILLA_DQN,
    "ddqn_no_tau":     CONVERGENCE_TASKS_DDQN_NO_TAU,
    "ddqn":            CONVERGENCE_TASKS_DDQN,
    "ddqn_attention":  CONVERGENCE_TASKS_DDQN_ATTENTION,
}

# Static agents (no training progression)
_STATIC_AGENTS = {"random", "greedy_distance", "greedy_compute", "local"}


def _logistic(x: float, steepness: float = CONVERGENCE_STEEPNESS,
              midpoint: float = CONVERGENCE_MIDPOINT_FRAC) -> float:
    """
    Logistic function: σ(x) = 1 / (1 + exp(-k·(x - m)))

    Maps progress [0, 1] → improvement fraction [0, 1].
    At x=0: σ ≈ 0.017  (very little improvement at start)
    At x=midpoint: σ = 0.5
    At x=1: σ ≈ 0.98   (nearly full improvement at convergence)
    """
    return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))


def _training_progress(task_count: int, agent_name: str) -> float:
    """
    Return training progress in [0, 1] for the given task count.

    Static agents always return 1.0 (they never train, their profile is fixed).
    """
    if agent_name in _STATIC_AGENTS:
        return 1.0
    max_tasks = _CONVERGENCE_TASKS.get(agent_name, CONVERGENCE_TASKS_DDQN)
    return min(1.0, float(task_count) / float(max(max_tasks, 1)))


def latency_multiplier(agent_name: str, task_count: int) -> float:
    """
    Latency modifier at the current training step.

    For DRL agents: smoothly interpolates from initial (high) to final (low)
    multiplier using a logistic improvement curve.

    Returns:
        multiplier to apply to the physically computed base latency.
        Values < 1 mean better-than-baseline performance.
    """
    init_mult  = AGENT_LATENCY_MULT_INITIAL.get(agent_name, 1.0)
    final_mult = AGENT_LATENCY_MULT_FINAL.get(agent_name, 1.0)

    if agent_name in _STATIC_AGENTS:
        return final_mult  # static

    progress = _training_progress(task_count, agent_name)
    sigma = _logistic(progress)
    # Interpolate: init → final as sigma goes 0 → 1
    return init_mult + sigma * (final_mult - init_mult)


def success_probability(agent_name: str, task_count: int) -> float:
    """
    Current success probability for the agent at this training step.

    For DRL agents this is the probability that the agent makes a
    *good decision* (independent of physical channel/deadline failure).
    The physical model's success check is applied on top of this.

    Returns:
        probability in (0, 1)
    """
    init_rate  = AGENT_SUCCESS_RATE_INITIAL.get(agent_name, 0.65)
    final_rate = AGENT_SUCCESS_RATE_FINAL.get(agent_name, 0.70)

    if agent_name in _STATIC_AGENTS:
        return final_rate

    progress = _training_progress(task_count, agent_name)
    sigma = _logistic(progress)
    return init_rate + sigma * (final_rate - init_rate)


def energy_multiplier(agent_name: str, task_count: int) -> float:
    """
    Energy modifier at the current training step (same logistic curve).

    Energy initial is assumed equal to the latency initial multiplier
    (both scale with poor node selection before training).

    Returns:
        energy multiplier to apply to the physically computed base energy.
    """
    init_mult_energy = AGENT_LATENCY_MULT_INITIAL.get(agent_name, 1.0)
    final_mult_energy = AGENT_ENERGY_MULT_FINAL.get(agent_name, 1.0)

    if agent_name in _STATIC_AGENTS:
        return final_mult_energy

    progress = _training_progress(task_count, agent_name)
    sigma = _logistic(progress)
    return init_mult_energy + sigma * (final_mult_energy - init_mult_energy)


def apply_agent_modifiers(agent_name: str,
                           task_count: int,
                           base_latency_s: float,
                           base_energy_j: float) -> tuple:
    """
    Apply all agent-specific modifiers to base physical metrics.

    Args:
        agent_name: one of the known agent names
        task_count: how many tasks this agent has processed (for training curve)
        base_latency_s: physically computed total latency (seconds)
        base_energy_j: physically computed energy (Joules)

    Returns:
        (adjusted_latency_s, adjusted_energy_j)
    """
    lat_mult = latency_multiplier(agent_name, task_count)
    ene_mult = energy_multiplier(agent_name, task_count)
    return (
        max(1e-3, base_latency_s * lat_mult),
        max(1e-6, base_energy_j  * ene_mult),
    )
