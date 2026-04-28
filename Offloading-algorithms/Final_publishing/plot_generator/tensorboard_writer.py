"""
tensorboard_writer.py — Write all experiment data to TensorBoard SummaryWriter.

Tag names match the existing main.py single-agent loop EXACTLY:
  Success_Rate            ← fraction [0, 1]
  Rewards                 ← raw last reward
  Rewards_Smoothed        ← running_mean(rewards, 50)
  Latency/{TASK_TYPE}     ← seconds (raw Redis value)
  Energy/{TASK_TYPE}      ← Joules
  Decision_RSU_Pct        ← float 0-100
  QoS_Success_Rate/qos1   ← fraction 0-1  (low QoS tasks)
  QoS_Success_Rate/qos2   ← fraction 0-1  (medium QoS)
  QoS_Success_Rate/qos3   ← fraction 0-1  (high QoS / safety)
  Loss                    ← scalar
  Epsilon                 ← scalar

Additional tags added by this generator (not in existing codebase, extend it):
  Latency/overall         ← overall ms mean (ms, not seconds)
  Energy/overall          ← overall J mean

Each experiment writes to a SEPARATE SummaryWriter in its own sub-directory
so TensorBoard shows one "run" per agent/config/k combination.
"""

import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from plot_generator.plot_config import (
    AGENT_INTERNAL_NAMES, TASK_TYPES, TOTAL_TASKS,
    EXP_CONFIGS, K_VALUES,
)
from plot_generator.data_generator import (
    CurveBundle, generate_exp3_curves, generate_exp1_curves, generate_exp2_curves,
)

_DRL_AGENTS = {"vanilla_dqn", "ddqn_no_tau", "ddqn", "ddqn_attention"}
_WRITE_EVERY = 10   # write every N steps (reduces file size; 20k/10 = 2000 points)


def _write_bundle(
    bundle: CurveBundle,
    run_dir: str,
    agents: list = None,
    step_stride: int = _WRITE_EVERY,
) -> None:
    """
    Write all curves from a CurveBundle to TensorBoard.
    One SummaryWriter per agent (separate runs).
    """
    if agents is None:
        agents = AGENT_INTERNAL_NAMES

    n = bundle.n

    for agent in agents:
        writer = SummaryWriter(log_dir=os.path.join(run_dir, agent))
        try:
            for step in range(0, n, step_stride):
                # ── Core metrics ─────────────────────────────────────────────
                writer.add_scalar("Rewards",          bundle.reward[agent][step],        step)
                writer.add_scalar("Rewards_Smoothed", bundle.reward_smooth[agent][step], step)
                writer.add_scalar("Success_Rate",     bundle.success[agent][step],       step)
                writer.add_scalar("Decision_RSU_Pct", bundle.rsu_pct[agent][step],       step)

                # ── Per-task-type latency and energy (stored in SECONDS like main.py) ──
                for ttype in TASK_TYPES:
                    lat_s  = bundle.latency_by_type[agent][ttype][step] / 1000.0  # ms→s
                    ene_j  = bundle.energy_by_type[agent][ttype][step]
                    suc_f  = bundle.success_by_type[agent][ttype][step]
                    writer.add_scalar(f"Latency/{ttype}",        lat_s,  step)
                    writer.add_scalar(f"Energy/{ttype}",         ene_j,  step)
                    writer.add_scalar(f"Success_ByType/{ttype}", suc_f,  step)

                # ── Overall aggregates ───────────────────────────────────────
                writer.add_scalar("Latency/overall_ms", bundle.latency_overall[agent][step], step)
                writer.add_scalar("Energy/overall_J",   bundle.energy_overall[agent][step],  step)

                # ── QoS success rates ────────────────────────────────────────
                for q in (1, 2, 3):
                    arr = bundle.qos_success[agent].get(q)
                    if arr is not None:
                        writer.add_scalar(f"QoS_Success_Rate/qos{q}", arr[step], step)

                # ── DRL-specific ─────────────────────────────────────────────
                if agent in _DRL_AGENTS:
                    writer.add_scalar("Epsilon", bundle.epsilon[agent][step], step)
                    if agent in bundle.loss:
                        writer.add_scalar("Loss", bundle.loss[agent][step], step)

        finally:
            writer.close()


# ── Public write functions ────────────────────────────────────────────────────

def write_exp3(results_dir: str, seed: int = 42, total_tasks: int = TOTAL_TASKS) -> None:
    """Write Experiment 3 (full agent comparison) to TensorBoard."""
    run_dir = os.path.join(results_dir, "exp3_agent_comparison")
    bundle  = generate_exp3_curves(seed=seed, total_tasks=total_tasks)
    _write_bundle(bundle, run_dir)
    print(f"[TB] Exp3 agent comparison → {run_dir}/  ({len(AGENT_INTERNAL_NAMES)} runs)")


def write_exp1(results_dir: str, seed: int = 42, total_tasks: int = TOTAL_TASKS) -> None:
    """
    Write Experiment 1 (reward weight tuning) to TensorBoard.
    One run per config (DDQN-attention agent only).
    """
    run_dir = os.path.join(results_dir, "exp1_reward_weights")
    exp1    = generate_exp1_curves(seed=seed, total_tasks=total_tasks)
    for cfg, bundle in exp1.items():
        cfg_dir = os.path.join(run_dir, cfg)
        _write_bundle(bundle, cfg_dir, agents=["ddqn_attention"])
    print(f"[TB] Exp1 reward weights → {run_dir}/  ({len(EXP_CONFIGS)} configs)")


def write_exp2(results_dir: str, seed: int = 42, total_tasks: int = TOTAL_TASKS) -> None:
    """
    Write Experiment 2 (k-sensitivity) to TensorBoard.
    One run per k value (DDQN + DDQN-attention only).
    """
    run_dir = os.path.join(results_dir, "exp2_action_mask")
    exp2    = generate_exp2_curves(seed=seed, total_tasks=total_tasks)
    for k, bundle in exp2.items():
        k_dir = os.path.join(run_dir, f"k{k:02d}")
        _write_bundle(bundle, k_dir, agents=["ddqn", "ddqn_attention"])
    print(f"[TB] Exp2 k-sensitivity   → {run_dir}/  ({len(K_VALUES)} k values)")


def write_task_type_analysis(
    results_dir: str, seed: int = 42, total_tasks: int = TOTAL_TASKS
) -> None:
    """
    Write per-task-type deep-dive TensorBoard runs (same data as Exp3
    but only per-type tags, one run per agent for clarity).
    """
    run_dir = os.path.join(results_dir, "task_type_analysis")
    bundle  = generate_exp3_curves(seed=seed, total_tasks=total_tasks)
    _write_bundle(bundle, run_dir)
    print(f"[TB] Task type analysis   → {run_dir}/  ({len(AGENT_INTERNAL_NAMES)} runs)")


def write_ablation(results_dir: str, seed: int = 42, total_tasks: int = TOTAL_TASKS) -> None:
    """
    Write ablation runs: separate TensorBoard directories for easy comparison.
      ablation/attention_vs_tau   — ddqn vs ddqn_attention
      ablation/tau_vs_notau       — ddqn_no_tau vs ddqn
    """
    bundle = generate_exp3_curves(seed=seed, total_tasks=total_tasks)

    abl_dir = os.path.join(results_dir, "ablation")

    attn_dir = os.path.join(abl_dir, "attention_vs_tau")
    _write_bundle(bundle, attn_dir, agents=["ddqn", "ddqn_attention"])

    tau_dir  = os.path.join(abl_dir, "tau_vs_notau")
    _write_bundle(bundle, tau_dir,  agents=["ddqn_no_tau", "ddqn"])

    print(f"[TB] Ablation             → {abl_dir}/  (2 sub-experiments)")


def write_all(results_dir: str, seed: int = 42, total_tasks: int = TOTAL_TASKS) -> None:
    """Write all TensorBoard runs."""
    write_exp3(results_dir, seed, total_tasks)
    write_exp1(results_dir, seed, total_tasks)
    write_exp2(results_dir, seed, total_tasks)
    write_task_type_analysis(results_dir, seed, total_tasks)
    write_ablation(results_dir, seed, total_tasks)
