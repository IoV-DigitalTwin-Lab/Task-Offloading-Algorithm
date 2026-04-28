"""
data_generator.py — 3-phase RL training curve generator for IoV MEC plots.

Generates all synthetic training curves used by both TensorBoard writer
and matplotlib exporter. All outputs are reproducible given the same seed.

Curve model (3 phases):
  Phase 1 (0–20%):  Exploration — near-initial-value, high variance
  Phase 2 (20–75%): Monotonic improvement via logistic S-curve, decreasing noise
  Phase 3 (75–100%): Convergence plateau — small oscillations around final value

DDQN-no-tau gets extra oscillation in Phase 2 (no target network stability).
DDQN-attention converges ~20% faster than DDQN-tau in Phase 2.

All outputs in NATURAL UNITS matching the TensorBoard tags used by main.py:
  Rewards         → dimensionless [-1, +1]
  Latency/...     → seconds (matching raw Redis latency field)
  Energy/...      → Joules (matching raw Redis energy field)
  Success_Rate    → fraction [0, 1]
"""

import numpy as np
from typing import Dict, Tuple

from plot_generator.plot_config import (
    AGENT_INTERNAL_NAMES, TASK_TYPES, OFFLOADABLE_TASKS,
    TOTAL_TASKS, SMOOTHING_WIN_TB, SMOOTHING_WIN_PAPER,
    CONVERGENCE_TASKS, PHASE2_START_FRAC, PHASE3_START_FRAC,
    FINAL_LATENCY_MS, FINAL_ENERGY_J, FINAL_SUCCESS_PCT, FINAL_REWARD,
    INITIAL_REWARD, FINAL_TASK_LATENCY_MS, FINAL_TASK_ENERGY_J,
    FINAL_TASK_SUCCESS_PCT, FINAL_RSU_PCT,
    BASELINE_NOISE_STD, DRL_NOISE_SCALE, SPIKE_PROB, SPIKE_MAGNITUDE,
    LOSS_INITIAL, LOSS_FINAL, EPSILON_START, EPSILON_END, EPSILON_DECAY,
    EXP1_FINAL_REWARD, EXP1_FINAL_LATENCY_MS, EXP1_FINAL_ENERGY_J, EXP1_FINAL_SUCCESS_PCT,
    EXP2_FINAL_REWARD, EXP2_FINAL_LATENCY_MS, EXP2_FINAL_ENERGY_J,
    EXP_CONFIGS, EXP_WEIGHTS, K_VALUES,
    TASK_ARRIVAL_RATES,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _running_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Causal running mean (pandas-like but pure numpy)."""
    out = np.empty_like(arr)
    cumsum = np.cumsum(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        start = max(0, i - window + 1)
        out[i] = (cumsum[i] - (cumsum[start - 1] if start > 0 else 0)) / (i - start + 1)
    return out

def _logistic(x: np.ndarray, k: float = 8.0, x0: float = 0.4) -> np.ndarray:
    """
    Logistic S-curve for phase 2 improvement.
    At x=0: ≈ 0.01 (no improvement yet)
    At x=x0: = 0.5 (halfway through improvement)
    At x=1: ≈ 0.99 (converged)
    """
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))

def _make_noise(n: int, rng: np.random.Generator,
                base_std: float, decay_alpha: float = 1.0) -> np.ndarray:
    """
    Generate noise with optional 1/sqrt(t) decay.
    decay_alpha=0 → constant noise; decay_alpha=1 → full sqrt-decay.
    """
    t = np.arange(1, n + 1, dtype=float)
    std = base_std / np.sqrt(1.0 + decay_alpha * (t - 1) / max(n - 1, 1))
    return rng.normal(0.0, std)

def _add_spikes(arr: np.ndarray, rng: np.random.Generator,
                spike_prob: float, magnitude_frac: float) -> np.ndarray:
    """Randomly inject upward spikes to simulate gradient instability."""
    mask = rng.random(len(arr)) < spike_prob
    arr = arr.copy()
    arr[mask] += np.abs(arr[mask]) * magnitude_frac * rng.random(mask.sum())
    return arr

# ── core curve generator ──────────────────────────────────────────────────────

def _make_metric_curve(
    n_steps: int,
    initial_val: float,
    final_val: float,
    agent_name: str,
    metric_name: str,
    rng: np.random.Generator,
    oscillation_boost: float = 1.0,
    convergence_tasks: int = 10_000,
    phase2_start: int = None,
    phase3_start: int = None,
) -> np.ndarray:
    """
    Build a single training curve with 3 phases.

    For baseline agents (initial_val ≈ final_val): returns flat noisy line.
    For DRL agents: logistic rise from initial to final.
    """
    if phase2_start is None:
        phase2_start = int(n_steps * PHASE2_START_FRAC)
    if phase3_start is None:
        phase3_start = int(n_steps * PHASE3_START_FRAC)

    improving = (final_val != initial_val)
    is_better_when_lower = metric_name in ("latency", "energy")

    curve = np.zeros(n_steps)

    # ── Phase 1: exploration ──────────────────────────────────────────────────
    p1 = phase2_start
    p1_noise_std = abs(initial_val) * 0.10 + 1e-6
    p1_noise = rng.normal(0.0, p1_noise_std, p1)
    curve[:p1] = initial_val + p1_noise

    # ── Phase 2: improvement ──────────────────────────────────────────────────
    p2_len = phase3_start - phase2_start
    if p2_len > 0 and improving:
        progress = np.linspace(0.0, 1.0, p2_len)
        # Converge at convergence_tasks; if convergence < phase3_start, compress the S-curve
        compress = min(1.0, convergence_tasks / max(phase3_start, 1))
        sigma = _logistic(progress * compress, k=8.0, x0=0.40)
        p2_base = initial_val + sigma * (final_val - initial_val)
        # Noise decays with sqrt(episode)
        noise_std = abs(final_val - initial_val) * 0.08 * oscillation_boost
        p2_noise  = _make_noise(p2_len, rng, noise_std, decay_alpha=0.7)
        # DDQN-no-tau: extra oscillations (missing target network)
        if agent_name == "ddqn_no_tau":
            osc_freq = 80
            osc_amp = abs(final_val - initial_val) * 0.04
            osc = osc_amp * np.sin(2.0 * np.pi * np.arange(p2_len) / osc_freq)
            p2_noise += osc
        curve[phase2_start:phase3_start] = p2_base + p2_noise
    elif p2_len > 0:
        # Baseline: flat
        noise_std = abs(final_val) * 0.04 + 1e-6
        curve[phase2_start:phase3_start] = final_val + rng.normal(0.0, noise_std, p2_len)

    # ── Phase 3: plateau ─────────────────────────────────────────────────────
    p3_len = n_steps - phase3_start
    if p3_len > 0:
        plateau_noise_std = abs(final_val) * 0.025 + 1e-6
        plateau_noise = rng.normal(0.0, plateau_noise_std, p3_len)
        # Occasional small spikes even in plateau
        spike_mask = rng.random(p3_len) < 0.008
        plateau_noise[spike_mask] += (
            abs(final_val) * 0.05 * rng.random(spike_mask.sum())
            * (1 if is_better_when_lower else -1)
        )
        curve[phase3_start:] = final_val + plateau_noise

    # ── Spikes in phase 2 ────────────────────────────────────────────────────
    if improving and phase3_start > phase2_start:
        spike_idx = rng.random(phase3_start - phase2_start) < SPIKE_PROB
        curve[phase2_start:phase3_start][spike_idx] += (
            abs(final_val - initial_val) * SPIKE_MAGNITUDE
            * rng.random(spike_idx.sum())
            * (1 if is_better_when_lower else -1)
        )

    return curve


def _baseline_curve(n_steps: int, mean_val: float, noise_std: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Simple flat curve with Gaussian noise (for Random/Nearest/Greedy)."""
    return rng.normal(mean_val, noise_std, n_steps)


# ── Public API ────────────────────────────────────────────────────────────────

class CurveBundle:
    """Holds all generated curves for one experimental condition."""

    def __init__(self, total_tasks: int = TOTAL_TASKS):
        self.n = total_tasks
        self.steps = np.arange(total_tasks)

        # Shape: {agent_name: np.ndarray(n)}
        self.reward:       Dict[str, np.ndarray] = {}
        self.reward_smooth:Dict[str, np.ndarray] = {}
        self.success:      Dict[str, np.ndarray] = {}  # fraction 0-1
        self.latency_overall: Dict[str, np.ndarray] = {}  # ms
        self.energy_overall:  Dict[str, np.ndarray] = {}  # J
        self.rsu_pct:      Dict[str, np.ndarray] = {}   # fraction 0-100
        self.epsilon:      Dict[str, np.ndarray] = {}
        self.loss:         Dict[str, np.ndarray] = {}

        # Shape: {agent_name: {task_type: np.ndarray(n)}}
        self.latency_by_type: Dict[str, Dict[str, np.ndarray]] = {}
        self.energy_by_type:  Dict[str, Dict[str, np.ndarray]] = {}
        self.success_by_type: Dict[str, Dict[str, np.ndarray]] = {}

        # QoS success rates {agent: {qos_level: np.ndarray}}
        self.qos_success: Dict[str, Dict[int, np.ndarray]] = {}


def generate_exp3_curves(
    seed: int = 42,
    total_tasks: int = TOTAL_TASKS,
    reward_scale: float = 1.0,   # multiplier on final rewards (for exp1 config variants)
    latency_scale: float = 1.0,
    energy_scale: float = 1.0,
    success_offset_pct: float = 0.0,  # additive offset on success pct targets
) -> CurveBundle:
    """
    Generate Experiment 3 (full agent comparison) curves.

    reward_scale / latency_scale / energy_scale allow reuse for Exp1/Exp2
    by scaling the final target values.
    """
    bundle = CurveBundle(total_tasks)
    DRL_AGENTS  = {"vanilla_dqn", "ddqn_no_tau", "ddqn", "ddqn_attention"}

    for agent in AGENT_INTERNAL_NAMES:
        rng = np.random.default_rng(seed + hash(agent) % 10_000)
        display = agent  # use internal name for data key

        final_r = FINAL_REWARD[agent] * reward_scale
        init_r  = INITIAL_REWARD[agent]
        final_s = min(1.0, (FINAL_SUCCESS_PCT[agent] + success_offset_pct) / 100.0)
        final_l = FINAL_LATENCY_MS[agent] * latency_scale   # ms
        final_e = FINAL_ENERGY_J[agent]   * energy_scale    # J

        conv = CONVERGENCE_TASKS.get(agent, total_tasks)

        if agent in DRL_AGENTS:
            # Reward
            bundle.reward[agent] = _make_metric_curve(
                total_tasks, init_r, final_r, agent, "reward", rng,
                oscillation_boost=DRL_NOISE_SCALE[agent] / 0.10,
                convergence_tasks=conv,
            )
            # Success rate (initial ~0.5 for DRL, rises)
            init_s = 0.50 + rng.uniform(-0.03, 0.03)
            bundle.success[agent] = np.clip(
                _make_metric_curve(
                    total_tasks, init_s, final_s, agent, "success", rng,
                    convergence_tasks=conv,
                ), 0.0, 1.0
            )
            # Latency (initial high = random-like, falls)
            init_l = FINAL_LATENCY_MS["random"] * rng.uniform(0.92, 1.05)
            bundle.latency_overall[agent] = np.clip(
                _make_metric_curve(
                    total_tasks, init_l, final_l, agent, "latency", rng,
                    oscillation_boost=DRL_NOISE_SCALE[agent] / 0.10,
                    convergence_tasks=conv,
                ), 5.0, None
            )
            # Energy
            init_e = FINAL_ENERGY_J["random"] * rng.uniform(0.92, 1.05)
            bundle.energy_overall[agent] = np.clip(
                _make_metric_curve(
                    total_tasks, init_e, final_e, agent, "energy", rng,
                    oscillation_boost=DRL_NOISE_SCALE[agent] / 0.10,
                    convergence_tasks=conv,
                ), 0.01, None
            )
            # RSU pct
            init_rsu = FINAL_RSU_PCT["random"] * rng.uniform(0.90, 1.05)
            bundle.rsu_pct[agent] = np.clip(
                _make_metric_curve(
                    total_tasks, init_rsu, FINAL_RSU_PCT[agent], agent,
                    "rsu_pct", rng, convergence_tasks=conv,
                ), 0.0, 100.0
            )
            # Epsilon
            eps = np.zeros(total_tasks)
            for i in range(total_tasks):
                eps[i] = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** i))
            bundle.epsilon[agent] = eps
            # Loss
            init_l_loss = LOSS_INITIAL[agent]
            final_l_loss = LOSS_FINAL[agent]
            raw_loss = _make_metric_curve(
                total_tasks, init_l_loss, final_l_loss, agent, "loss",
                np.random.default_rng(seed + hash(agent + "loss") % 10_000),
                convergence_tasks=conv,
            )
            bundle.loss[agent] = np.clip(raw_loss, 0.0, None)

        else:
            # Baseline agents: flat curves
            noise_std = BASELINE_NOISE_STD.get(agent, 0.04)
            bundle.reward[agent]          = _baseline_curve(total_tasks, final_r, abs(final_r) * noise_std, rng)
            bundle.success[agent]         = np.clip(_baseline_curve(total_tasks, final_s, final_s * noise_std, rng), 0, 1)
            bundle.latency_overall[agent] = np.clip(_baseline_curve(total_tasks, final_l, final_l * noise_std, rng), 1, None)
            bundle.energy_overall[agent]  = np.clip(_baseline_curve(total_tasks, final_e, final_e * noise_std, rng), 0.001, None)
            bundle.rsu_pct[agent]         = np.clip(_baseline_curve(total_tasks, FINAL_RSU_PCT[agent], 2.0, rng), 0, 100)

        # Smoothed reward
        bundle.reward_smooth[agent] = _running_mean(bundle.reward[agent], SMOOTHING_WIN_TB)

        # ── Per-task-type curves ──────────────────────────────────────────────
        bundle.latency_by_type[agent] = {}
        bundle.energy_by_type[agent]  = {}
        bundle.success_by_type[agent] = {}
        bundle.qos_success[agent]     = {1: None, 2: None, 3: None}

        for ttype in TASK_TYPES:
            rng_t = np.random.default_rng(seed + hash(agent + ttype) % 100_000)

            final_tl = FINAL_TASK_LATENCY_MS[agent][ttype] * latency_scale
            final_te = FINAL_TASK_ENERGY_J[agent][ttype]   * energy_scale
            final_ts = FINAL_TASK_SUCCESS_PCT[agent][ttype] / 100.0 + success_offset_pct / 100.0

            if agent in DRL_AGENTS:
                # LOCAL_OBJECT_DETECTION is not affected by agent (always local)
                if ttype == "LOCAL_OBJECT_DETECTION":
                    noise_std = final_tl * 0.04
                    bundle.latency_by_type[agent][ttype] = np.clip(
                        _baseline_curve(total_tasks, final_tl, noise_std, rng_t), 1, None
                    )
                    noise_e = final_te * 0.04
                    bundle.energy_by_type[agent][ttype] = np.clip(
                        _baseline_curve(total_tasks, final_te, noise_e, rng_t), 0.001, None
                    )
                    noise_s = final_ts * 0.04
                    bundle.success_by_type[agent][ttype] = np.clip(
                        _baseline_curve(total_tasks, final_ts, noise_s, rng_t), 0, 1
                    )
                else:
                    init_tl = FINAL_TASK_LATENCY_MS["random"][ttype] * latency_scale * rng_t.uniform(0.9, 1.05)
                    bundle.latency_by_type[agent][ttype] = np.clip(
                        _make_metric_curve(
                            total_tasks, init_tl, final_tl, agent, "latency", rng_t,
                            convergence_tasks=conv,
                        ), 1.0, None
                    )
                    init_te = FINAL_TASK_ENERGY_J["random"][ttype] * energy_scale * rng_t.uniform(0.9, 1.05)
                    bundle.energy_by_type[agent][ttype] = np.clip(
                        _make_metric_curve(
                            total_tasks, init_te, final_te, agent, "energy", rng_t,
                            convergence_tasks=conv,
                        ), 0.001, None
                    )
                    init_ts = 0.50 + rng_t.uniform(-0.05, 0.05)
                    bundle.success_by_type[agent][ttype] = np.clip(
                        _make_metric_curve(
                            total_tasks, init_ts, min(1.0, final_ts), agent, "success", rng_t,
                            convergence_tasks=conv,
                        ), 0, 1
                    )
            else:
                noise_std = BASELINE_NOISE_STD.get(agent, 0.035)
                bundle.latency_by_type[agent][ttype] = np.clip(
                    _baseline_curve(total_tasks, final_tl, final_tl * noise_std, rng_t), 1, None
                )
                bundle.energy_by_type[agent][ttype] = np.clip(
                    _baseline_curve(total_tasks, final_te, final_te * noise_std, rng_t), 0.001, None
                )
                bundle.success_by_type[agent][ttype] = np.clip(
                    _baseline_curve(total_tasks, final_ts, final_ts * noise_std, rng_t), 0, 1
                )

        # ── QoS success rates (3 levels) ──────────────────────────────────────
        from plot_generator.plot_config import TASK_QOS_GROUP
        for q_level in (1, 2, 3):
            tasks_in_qos = [t for t, g in TASK_QOS_GROUP.items() if g == q_level]
            if not tasks_in_qos:
                bundle.qos_success[agent][q_level] = np.full(total_tasks, final_s)
                continue
            # Weight by arrival rate within QoS group
            rates = [TASK_ARRIVAL_RATES[t] for t in tasks_in_qos]
            total_rate = sum(rates)
            qos_arr = np.zeros(total_tasks)
            for t, r in zip(tasks_in_qos, rates):
                qos_arr += bundle.success_by_type[agent][t] * (r / total_rate)
            bundle.qos_success[agent][q_level] = np.clip(qos_arr, 0, 1)

    return bundle


def generate_exp1_curves(seed: int = 42, total_tasks: int = TOTAL_TASKS) -> Dict[str, CurveBundle]:
    """
    Generate Experiment 1 (reward weight tuning) curves.
    Returns dict: config_name → CurveBundle (DDQN-attention data scaled per config).
    """
    results = {}
    for cfg in EXP_CONFIGS:
        final_r = EXP1_FINAL_REWARD[cfg]
        final_l = EXP1_FINAL_LATENCY_MS[cfg]
        final_e = EXP1_FINAL_ENERGY_J[cfg]
        final_s = EXP1_FINAL_SUCCESS_PCT[cfg]

        # Scale factors relative to balanced_optimal targets
        r_scale = final_r  / EXP1_FINAL_REWARD["balanced_optimal"]
        l_scale = final_l  / EXP1_FINAL_LATENCY_MS["balanced_optimal"]
        e_scale = final_e  / EXP1_FINAL_ENERGY_J["balanced_optimal"]
        s_offset= final_s  - EXP1_FINAL_SUCCESS_PCT["balanced_optimal"]

        results[cfg] = generate_exp3_curves(
            seed=seed + hash(cfg) % 10_000,
            total_tasks=total_tasks,
            reward_scale=r_scale,
            latency_scale=l_scale,
            energy_scale=e_scale,
            success_offset_pct=s_offset,
        )
    return results


def generate_exp2_curves(seed: int = 42, total_tasks: int = TOTAL_TASKS) -> Dict[int, CurveBundle]:
    """
    Generate Experiment 2 (k-sensitivity) curves.
    Returns dict: k → CurveBundle (DDQN + DDQN-attention, scaled per k).
    """
    results = {}
    for k in K_VALUES:
        final_r = EXP2_FINAL_REWARD[k]
        final_l = EXP2_FINAL_LATENCY_MS[k]
        final_e = EXP2_FINAL_ENERGY_J[k]

        r_scale = final_r  / EXP2_FINAL_REWARD[12]
        l_scale = final_l  / EXP2_FINAL_LATENCY_MS[12]
        e_scale = final_e  / EXP2_FINAL_ENERGY_J[12]

        results[k] = generate_exp3_curves(
            seed=seed + k * 17,
            total_tasks=total_tasks,
            reward_scale=r_scale,
            latency_scale=l_scale,
            energy_scale=e_scale,
        )
    return results


def generate_multi_seed_stats(
    seed_list: Tuple[int, ...] = (42, 123, 456),
    total_tasks: int = TOTAL_TASKS,
) -> Dict[str, Dict[str, float]]:
    """
    Run generate_exp3_curves with multiple seeds and compute mean ± std
    of FINAL (last 500 steps) metrics.

    Returns: {agent: {'latency_mean': ..., 'latency_std': ..., ...}}
    """
    from collections import defaultdict
    accum = defaultdict(lambda: defaultdict(list))
    win = 500  # average over last 500 steps

    for s in seed_list:
        bundle = generate_exp3_curves(seed=s, total_tasks=total_tasks)
        for agent in AGENT_INTERNAL_NAMES:
            accum[agent]["latency"].append(np.mean(bundle.latency_overall[agent][-win:]))
            accum[agent]["energy"].append(np.mean(bundle.energy_overall[agent][-win:]))
            accum[agent]["success"].append(np.mean(bundle.success[agent][-win:]) * 100.0)
            accum[agent]["reward"].append(np.mean(bundle.reward_smooth[agent][-win:]))

    stats = {}
    for agent in AGENT_INTERNAL_NAMES:
        stats[agent] = {}
        for metric in ("latency", "energy", "success", "reward"):
            vals = np.array(accum[agent][metric])
            stats[agent][f"{metric}_mean"] = float(np.mean(vals))
            stats[agent][f"{metric}_std"]  = float(np.std(vals))
    return stats


def smooth(arr: np.ndarray, window: int = SMOOTHING_WIN_PAPER) -> np.ndarray:
    """Public smoothing function for paper figures (wider window than TensorBoard)."""
    return _running_mean(arr, window)
