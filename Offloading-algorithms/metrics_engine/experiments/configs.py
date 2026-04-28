"""
Experiment configurations for IoV MEC task offloading research.

Three experiment dimensions:
  1. Reward Weight Tuning   — which objective mix produces best DDQN convergence?
  2. Action Mask Sensitivity — how does k-nearest vehicle count affect performance?
  3. Agent Comparison       — full hierarchy from Random to DDQN+Attention

Each EXPERIMENT entry overrides the defaults in src/config.py for a targeted run.
Pass --experiment <name> to runner.py and run_all_agents.sh to tag TensorBoard logs.

Reward formula (environment.py compute_reward_for):
    r = W_LATENCY  * latency_reward
      + W_ENERGY   * energy_reward
      + W_DEADLINE * deadline_bonus
where each component is normalised to [-1, 1].

References:
  Liu et al., "Multi-Objective Reward Shaping for DRL Task Offloading",
      IEEE Trans. Veh. Technol. 2023.
  Wang et al., "Edge Intelligence: The Confluence of Edge Computing and AI",
      IEEE Network 2020.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Complete description of one experimental condition.

    Attributes:
        name:           Unique identifier (used in TensorBoard run tag)
        description:    One-line human-readable purpose
        w_latency:      Reward weight for latency component (default 0.6)
        w_energy:       Reward weight for energy component (default 0.2)
        w_deadline:     Reward weight for deadline success component (default 0.2)
        k_neighbors:    Action mask: number of k-nearest service vehicles + RSUs
        agents:         Which agent names to run under this config
        episodes:       Training episodes (None → use agent default)
        notes:          Free-form notes for RESULTS_GUIDE.md
    """
    name:        str
    description: str
    w_latency:   float
    w_energy:    float
    w_deadline:  float
    k_neighbors: int
    agents:      List[str]
    episodes:    Optional[int]
    notes:       str


# ── Experiment 1: Reward Weight Tuning ──────────────────────────────────────
# Fixed k=6 (baseline), vary the reward objective mix.
# Research question: which weight combination maximises DDQN+Attention reward?
# Expected finding: balanced_optimal (0.5/0.3/0.2) converges fastest due to
# sufficient latency signal without ignoring energy/deadline, matching
# García-Roger et al. TVT 2021 findings for heterogeneous V2X environments.

EXPERIMENTS: Dict[str, ExperimentConfig] = {

    "latency_priority": ExperimentConfig(
        name        = "latency_priority",
        description = "Heavy latency weight (0.7/0.15/0.15) — safety-critical workloads",
        w_latency   = 0.70,
        w_energy    = 0.15,
        w_deadline  = 0.15,
        k_neighbors = 6,
        agents      = ["random", "greedy_distance", "greedy_compute",
                       "vanilla_dqn", "ddqn_no_tau", "ddqn", "ddqn_attention"],
        episodes    = None,
        notes       = (
            "Maximises latency reduction at cost of energy efficiency. "
            "Expected: DDQN+Attention achieves ~24% latency improvement vs baseline; "
            "energy savings may drop to ~12% (below 17.3% nominal)."
        ),
    ),

    "energy_priority": ExperimentConfig(
        name        = "energy_priority",
        description = "Heavy energy weight (0.3/0.5/0.2) — battery-constrained IoV nodes",
        w_latency   = 0.30,
        w_energy    = 0.50,
        w_deadline  = 0.20,
        k_neighbors = 6,
        agents      = ["random", "greedy_distance", "greedy_compute",
                       "vanilla_dqn", "ddqn_no_tau", "ddqn", "ddqn_attention"],
        episodes    = None,
        notes       = (
            "Optimises for energy-constrained nodes (EVs, pedestrian units). "
            "Expected: DDQN agents converge to RSU offloading more than V2V, "
            "trading ~8% latency for ~22% energy reduction vs Greedy baselines."
        ),
    ),

    "balanced_optimal": ExperimentConfig(
        name        = "balanced_optimal",
        description = "Balanced weights (0.5/0.3/0.2) — standard research comparison baseline",
        w_latency   = 0.50,
        w_energy    = 0.30,
        w_deadline  = 0.20,
        k_neighbors = 6,
        agents      = ["random", "greedy_distance", "greedy_compute",
                       "vanilla_dqn", "ddqn_no_tau", "ddqn", "ddqn_attention"],
        episodes    = None,
        notes       = (
            "PRIMARY EXPERIMENT. Replicates García-Roger et al. TVT 2021 weight setup. "
            "DDQN+Attention target: 23.6% latency reduction, 17.3% energy reduction, "
            "+7pp success rate vs (Random + Greedy) average baseline. "
            "Use this for the main TensorBoard comparison figure."
        ),
    ),

    "success_priority": ExperimentConfig(
        name        = "success_priority",
        description = "Deadline-heavy (0.4/0.2/0.4) — strict QoS SLA enforcement",
        w_latency   = 0.40,
        w_energy    = 0.20,
        w_deadline  = 0.40,
        k_neighbors = 6,
        agents      = ["random", "greedy_distance", "greedy_compute",
                       "vanilla_dqn", "ddqn_no_tau", "ddqn", "ddqn_attention"],
        episodes    = None,
        notes       = (
            "High deadline bonus drives agents toward conservative, reliable nodes. "
            "Expected: DDQN+Attention success rate >90%; latency/energy improvement "
            "moderate (~18%/~14%) due to risk-averse node selection."
        ),
    ),
}

# ── Experiment 2: Action Mask k-Sensitivity ──────────────────────────────────
# Fixed: balanced_optimal weights. Vary k (number of candidate neighbours).
# Research question: is more choice always better? Hypothesis: k=12 is optimal;
# k<8 over-restricts good nodes; k>14 adds noise that slows convergence.
#
# Run via: for k in K_VALUES; do python runner.py --experiment balanced_optimal --k $k; done

K_VALUES: List[int] = [6, 10, 12, 15, 18]

K_EXPERIMENT: ExperimentConfig = ExperimentConfig(
    name        = "k_sensitivity",
    description = "Action mask k-nearest sweep over [6, 10, 12, 15, 18]",
    w_latency   = 0.50,
    w_energy    = 0.30,
    w_deadline  = 0.20,
    k_neighbors = 12,          # default; overridden per sweep iteration
    agents      = ["ddqn", "ddqn_attention"],   # DRL only — baselines are k-independent
    episodes    = None,
    notes       = (
        "Sweep k ∈ {6, 10, 12, 15, 18}. Plot final-episode average reward vs k. "
        "Expected: unimodal with peak near k=12, matching MAX_NEIGHBORS=12 in redis_config.json. "
        "k=6 is the original project default; k=12 is our proposed optimum."
    ),
)

# ── Experiment 3: Full Agent Comparison ─────────────────────────────────────
# All 7 agents, balanced weights, k=12.
# Produces the primary bar chart (final mean reward / latency / energy / success).

AGENT_COMPARISON: ExperimentConfig = ExperimentConfig(
    name        = "agent_comparison",
    description = "Full 7-agent comparison: Random → DDQN+Attention at k=12",
    w_latency   = 0.50,
    w_energy    = 0.30,
    w_deadline  = 0.20,
    k_neighbors = 12,
    agents      = [
        "random",
        "greedy_distance",
        "greedy_compute",
        "vanilla_dqn",
        "ddqn_no_tau",
        "ddqn",
        "ddqn_attention",
    ],
    episodes    = None,
    notes       = (
        "Main comparison figure. Expected performance order: "
        "random < greedy_distance < greedy_compute < vanilla_dqn "
        "< ddqn_no_tau < ddqn < ddqn_attention. "
        "DDQN+Attention improvement over Random baseline: "
        "latency −23.6%, energy −17.3%, success +7pp."
    ),
)

# Register k_sensitivity and agent_comparison in EXPERIMENTS for CLI lookup
EXPERIMENTS["k_sensitivity"]   = K_EXPERIMENT
EXPERIMENTS["agent_comparison"] = AGENT_COMPARISON


def get_experiment(name: str) -> ExperimentConfig:
    """
    Return ExperimentConfig by name.

    Raises KeyError with a helpful message listing valid names.
    """
    if name not in EXPERIMENTS:
        valid = ", ".join(sorted(EXPERIMENTS.keys()))
        raise KeyError(f"Unknown experiment '{name}'. Valid options: {valid}")
    return EXPERIMENTS[name]
