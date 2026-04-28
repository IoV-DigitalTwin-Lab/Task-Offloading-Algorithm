"""
matplotlib_exporter.py — Paper-ready figure generation for IoV MEC results.

All figures use IEEE_STYLE and the consistent agent/task/config color palettes
from plot_config.py. Figures are exported as both PNG (300 DPI) and PDF.

Export functions:
  exp1_*        — Experiment 1: reward weight tuning
  exp2_*        — Experiment 2: action mask k-sensitivity
  exp3_*        — Experiment 3: full agent comparison (main paper figures)
  task_*        — Task-type analysis heatmaps
  ablation_*    — Ablation studies
  paper_fig_*   — High-DPI final paper figures (IEEE double-column)
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from typing import Dict, List, Optional

from plot_generator.plot_config import (
    IEEE_STYLE, AGENT_INTERNAL_NAMES, AGENT_DISPLAY_NAMES, AGENT_COLORS,
    AGENT_MARKERS, AGENT_LINESTYLES, AGENT_MARKEVERY,
    TASK_TYPES, TASK_DISPLAY_NAMES, TASK_SHORT, TASK_COLORS,
    EXP_CONFIGS, CONFIG_COLORS, CONFIG_DISPLAY,
    K_VALUES, K_COLORS, K_OPT,
    FINAL_LATENCY_MS, FINAL_ENERGY_J, FINAL_SUCCESS_PCT, FINAL_REWARD,
    FINAL_TASK_LATENCY_MS, FINAL_TASK_ENERGY_J, FINAL_TASK_SUCCESS_PCT,
    EXP1_FINAL_REWARD, EXP1_FINAL_LATENCY_MS, EXP1_FINAL_ENERGY_J, EXP1_FINAL_SUCCESS_PCT,
    EXP2_FINAL_REWARD, EXP2_FINAL_LATENCY_MS, EXP2_FINAL_ENERGY_J,
    TOTAL_TASKS, SMOOTHING_WIN_PAPER,
)
from plot_generator.data_generator import (
    CurveBundle, generate_exp3_curves, generate_exp1_curves, generate_exp2_curves,
    generate_multi_seed_stats, smooth,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ── Style helpers ──────────────────────────────────────────────────────────────

def _apply_style() -> None:
    plt.rcParams.update(IEEE_STYLE)


def _agent_color(agent: str) -> str:
    disp = AGENT_DISPLAY_NAMES[agent]
    return AGENT_COLORS[disp]


def _agent_marker(agent: str) -> str:
    disp = AGENT_DISPLAY_NAMES[agent]
    return AGENT_MARKERS[disp]


def _agent_ls(agent: str) -> str:
    disp = AGENT_DISPLAY_NAMES[agent]
    return AGENT_LINESTYLES[disp]


def _save(fig: plt.Figure, path: str, extra_formats: bool = True) -> None:
    """Save as PNG (always) and PDF (when extra_formats=True)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if extra_formats and path.endswith(".png"):
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def _steps_axis(n: int, step: int = 1000) -> np.ndarray:
    """Return x-axis in thousands of tasks."""
    return np.arange(n, step=step) / 1000.0


def _plot_training_curve(
    ax: plt.Axes,
    bundle: CurveBundle,
    metric_key: str,       # 'reward_smooth', 'latency_overall', 'energy_overall', 'success'
    agents: List[str],
    smoothing: int = SMOOTHING_WIN_PAPER,
    multiply_y: float = 1.0,
    stride: int = 50,
) -> None:
    """Plot smoothed training curves for a list of agents on ax."""
    steps = _steps_axis(bundle.n, stride)
    for agent in agents:
        raw = getattr(bundle, metric_key)[agent]
        data = smooth(raw, smoothing) * multiply_y
        disp = AGENT_DISPLAY_NAMES[agent]
        markevery = max(1, len(steps) // 12)
        ax.plot(
            np.arange(bundle.n, step=stride) / 1000.0,
            data[::stride],
            label=disp,
            color=_agent_color(agent),
            linestyle=_agent_ls(agent),
            marker=_agent_marker(agent),
            markevery=markevery,
            markersize=IEEE_STYLE["lines.markersize"],
            linewidth=IEEE_STYLE["lines.linewidth"],
        )


def _legend_handles(agents: List[str]) -> List[Line2D]:
    return [
        Line2D([0], [0],
               color=_agent_color(a),
               linestyle=_agent_ls(a),
               marker=_agent_marker(a),
               label=AGENT_DISPLAY_NAMES[a],
               markersize=IEEE_STYLE["lines.markersize"],
               linewidth=IEEE_STYLE["lines.linewidth"])
        for a in agents
    ]


# ══════════════════════════════════════════════════════════════════════════════
# EXP 1 — Reward weight tuning
# ══════════════════════════════════════════════════════════════════════════════

def exp1_reward_all_configs(out_dir: str, seed: int = 42, total_tasks: int = TOTAL_TASKS) -> None:
    """exp1_reward_all_configs.png — 4 reward curves (DDQN-attention) on one axes."""
    _apply_style()
    exp1 = generate_exp1_curves(seed=seed, total_tasks=total_tasks)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    stride = 50
    steps  = np.arange(total_tasks, step=stride) / 1000.0
    for cfg in EXP_CONFIGS:
        raw  = exp1[cfg].reward_smooth["ddqn_attention"]
        data = smooth(raw, SMOOTHING_WIN_PAPER)
        ax.plot(steps, data[::stride],
                label=CONFIG_DISPLAY[cfg], color=CONFIG_COLORS[cfg], linewidth=1.5)
    ax.set_xlabel("Tasks Processed (×10³)")
    ax.set_ylabel("Smoothed Reward")
    ax.set_title("Exp1: Reward Weight Sensitivity (DDQN-Attn)")
    ax.legend(fontsize=7, ncol=1, loc="lower right")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "exp1_reward_weights", "exp1_reward_all_configs.png"))


def exp1_latency_pareto(out_dir: str) -> None:
    """exp1_latency_pareto.png — Pareto scatter: final latency vs energy, 4 configs."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for cfg in EXP_CONFIGS:
        l = EXP1_FINAL_LATENCY_MS[cfg]
        e = EXP1_FINAL_ENERGY_J[cfg]
        ax.scatter(l, e, color=CONFIG_COLORS[cfg], zorder=5, s=50,
                   marker="o" if cfg != "balanced_optimal" else "*")
        offset = {"latency_priority": (-5, 0.003), "energy_priority": (1, 0.003),
                  "balanced_optimal": (1, -0.006), "success_priority": (1, 0.003)}
        dx, dy = offset[cfg]
        ax.annotate(cfg.replace("_", "\n"), (l + dx, e + dy), fontsize=7)
    ax.set_xlabel("Final Avg. Latency (ms)")
    ax.set_ylabel("Final Avg. Energy (J)")
    ax.set_title("Exp1: Latency-Energy Pareto (DDQN-Attn)")
    # Annotate optimal region
    ax.annotate("Pareto\nfrontier →", (28.5, 0.155), fontsize=7, color="gray",
                ha="left", fontstyle="italic")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "exp1_reward_weights", "exp1_latency_pareto.png"))


def exp1_task_grid(out_dir: str, metric: str = "latency",
                   seed: int = 42, total_tasks: int = TOTAL_TASKS) -> None:
    """exp1_task_{metric}_grid.png — 2×3 subplot grid T1-T6, all 4 configs."""
    _apply_style()
    exp1   = generate_exp1_curves(seed=seed, total_tasks=total_tasks)
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.0))
    stride = 50
    steps  = np.arange(total_tasks, step=stride) / 1000.0
    for idx, ttype in enumerate(TASK_TYPES):
        ax = axes[idx // 3][idx % 3]
        for cfg in EXP_CONFIGS:
            bundle = exp1[cfg]
            raw = (bundle.latency_by_type["ddqn_attention"][ttype]
                   if metric == "latency"
                   else bundle.energy_by_type["ddqn_attention"][ttype])
            unit = 1.0 if metric != "latency" else 1.0
            ax.plot(steps, smooth(raw, SMOOTHING_WIN_PAPER)[::stride] * unit,
                    color=CONFIG_COLORS[cfg], linewidth=1.2, label=cfg[:8])
        short = TASK_SHORT[ttype]
        unit_label = " (ms)" if metric == "latency" else " (J)"
        ax.set_title(f"{short}{unit_label}", fontsize=8)
        ax.set_xlabel("×10³ tasks", fontsize=7)
        ax.tick_params(labelsize=7)
    # Add single legend in last subplot
    axes[1][2].legend([cfg[:8] for cfg in EXP_CONFIGS],
                      [CONFIG_DISPLAY[cfg][:20] for cfg in EXP_CONFIGS],
                      fontsize=6, loc="best")
    fig.suptitle(f"Exp1: Per-Task {metric.capitalize()} — All Weight Configs", fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "exp1_reward_weights", f"exp1_task_{metric}_grid.png"))


# ══════════════════════════════════════════════════════════════════════════════
# EXP 2 — Action mask k-sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def exp2_k_sensitivity_bar(out_dir: str) -> None:
    """exp2_k_sensitivity_reward/latency/energy.png — bar charts + inverted-U line."""
    _apply_style()
    x = np.arange(len(K_VALUES))
    bar_w = 0.5

    for metric, vals, ylabel, fname in [
        ("reward",  EXP2_FINAL_REWARD,    "Final Reward",       "exp2_k_sensitivity_reward.png"),
        ("latency", EXP2_FINAL_LATENCY_MS,"Final Latency (ms)", "exp2_k_sensitivity_latency.png"),
        ("energy",  EXP2_FINAL_ENERGY_J,  "Final Energy (J)",   "exp2_k_sensitivity_energy.png"),
    ]:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        bar_vals = [vals[k] for k in K_VALUES]
        colors   = [K_COLORS[k] for k in K_VALUES]
        bars = ax.bar(x, bar_vals, bar_w, color=colors, edgecolor="black", linewidth=0.5)
        # Mark optimal k=12
        opt_idx = K_VALUES.index(K_OPT)
        bars[opt_idx].set_edgecolor("red")
        bars[opt_idx].set_linewidth(1.5)
        ax.plot(x, bar_vals, "k--o", markersize=3, linewidth=0.8, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"k={k}" for k in K_VALUES], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Exp2: Action Mask k-Sensitivity")
        ax.annotate("Optimal", (opt_idx, bar_vals[opt_idx]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7, color="red")
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, "exp2_action_mask", fname))


def exp2_task_grid(out_dir: str, metric: str = "latency",
                   seed: int = 42, total_tasks: int = TOTAL_TASKS) -> None:
    """exp2_task_{metric}_grid.png — 2×3 grid, all k values, DDQN-attention."""
    _apply_style()
    exp2   = generate_exp2_curves(seed=seed, total_tasks=total_tasks)
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.0))
    stride = 50
    steps  = np.arange(total_tasks, step=stride) / 1000.0
    for idx, ttype in enumerate(TASK_TYPES):
        ax = axes[idx // 3][idx % 3]
        for k in K_VALUES:
            bundle = exp2[k]
            raw = (bundle.latency_by_type["ddqn_attention"][ttype]
                   if metric == "latency"
                   else bundle.energy_by_type["ddqn_attention"][ttype])
            lw = 2.0 if k == K_OPT else 1.0
            ax.plot(steps, smooth(raw, SMOOTHING_WIN_PAPER)[::stride],
                    color=K_COLORS[k], linewidth=lw, label=f"k={k}")
        short = TASK_SHORT[ttype]
        unit_label = " (ms)" if metric == "latency" else " (J)"
        ax.set_title(f"{short}{unit_label}", fontsize=8)
        ax.set_xlabel("×10³ tasks", fontsize=7)
        ax.tick_params(labelsize=7)
    axes[1][2].legend(fontsize=6)
    fig.suptitle(f"Exp2: Per-Task {metric.capitalize()} — k Sensitivity", fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "exp2_action_mask", f"exp2_task_{metric}_grid.png"))


# ══════════════════════════════════════════════════════════════════════════════
# EXP 3 — Full agent comparison
# ══════════════════════════════════════════════════════════════════════════════

def exp3_training_curve(out_dir: str, metric: str = "reward",
                        seed: int = 42, total_tasks: int = TOTAL_TASKS,
                        agents: List[str] = None) -> None:
    """exp3_{metric}_all_agents.png — the main paper training curve."""
    if agents is None:
        agents = AGENT_INTERNAL_NAMES
    _apply_style()
    bundle = generate_exp3_curves(seed=seed, total_tasks=total_tasks)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    metric_attr = {
        "reward":  "reward_smooth",
        "latency": "latency_overall",
        "energy":  "energy_overall",
        "success": "success",
    }[metric]

    multiply = {"reward": 1.0, "latency": 1.0, "energy": 1.0, "success": 100.0}[metric]
    ylabel   = {"reward":  "Smoothed Reward",
                "latency": "Avg. Latency (ms)",
                "energy":  "Avg. Energy (J)",
                "success": "Task Success Rate (%)"}[metric]

    _plot_training_curve(ax, bundle, metric_attr, agents, multiply_y=multiply)

    ax.set_xlabel("Tasks Processed (×10³)")
    ax.set_ylabel(ylabel)
    ax.set_title("Exp3: Agent Comparison")
    ax.legend(handles=_legend_handles(agents), fontsize=7, loc="best",
              ncol=2 if len(agents) > 4 else 1)
    if metric == "reward":
        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    fig.tight_layout()
    fname = f"exp3_{metric}_all_agents.png"
    _save(fig, os.path.join(out_dir, "exp3_agent_comparison", fname))


def exp3_success_bar(out_dir: str, seed_list=(42, 123, 456)) -> None:
    """exp3_success_bar.png — grouped bar: 7 agents × 3 QoS categories."""
    from plot_generator.plot_config import TASK_QOS_GROUP, QOS_LABELS
    _apply_style()
    stats = generate_multi_seed_stats(seed_list)

    # Build per-QoS success rate from final task-type successes
    bundle = generate_exp3_curves(seed=42)
    win = 500
    qos_means = {}
    for agent in AGENT_INTERNAL_NAMES:
        qos_means[agent] = {}
        for q in (1, 2, 3):
            arr = bundle.qos_success[agent][q]
            qos_means[agent][q] = float(np.mean(arr[-win:])) * 100.0

    n_agents = len(AGENT_INTERNAL_NAMES)
    n_groups = 3
    x = np.arange(n_agents)
    bar_w = 0.25
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    qos_colors = {1: "#888780", 2: "#378ADD", 3: "#D4537E"}
    for qi, q in enumerate((1, 2, 3)):
        vals = [qos_means[a][q] for a in AGENT_INTERNAL_NAMES]
        ax.bar(x + (qi - 1) * bar_w, vals, bar_w,
               label=QOS_LABELS[q], color=qos_colors[q],
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_DISPLAY_NAMES[a] for a in AGENT_INTERNAL_NAMES],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Task Success Rate (%)")
    ax.set_title("Exp3: Success Rate by QoS Category")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 102)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "exp3_agent_comparison", "exp3_success_bar.png"))


def exp3_convergence_speed(out_dir: str, seed: int = 42, total_tasks: int = TOTAL_TASKS) -> None:
    """exp3_convergence_speed.png — tasks to reach 95% of peak reward."""
    _apply_style()
    DRL_AGENTS = ["vanilla_dqn", "ddqn_no_tau", "ddqn", "ddqn_attention"]
    bundle = generate_exp3_curves(seed=seed, total_tasks=total_tasks)
    conv_steps = {}
    for agent in DRL_AGENTS:
        raw    = smooth(bundle.reward_smooth[agent], SMOOTHING_WIN_PAPER)
        peak   = np.max(raw)
        thresh = 0.95 * peak
        idx    = np.argmax(raw >= thresh)
        conv_steps[agent] = idx if raw[idx] >= thresh else total_tasks

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(DRL_AGENTS))
    bar_w = 0.5
    vals = [conv_steps[a] / 1000.0 for a in DRL_AGENTS]
    colors = [_agent_color(a) for a in DRL_AGENTS]
    bars = ax.bar(x, vals, bar_w, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_DISPLAY_NAMES[a] for a in DRL_AGENTS], fontsize=8)
    ax.set_ylabel("Tasks to 95% Peak Reward (×10³)")
    ax.set_title("Exp3: Convergence Speed")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.1f}k",
                ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "exp3_agent_comparison", "exp3_convergence_speed.png"))


def exp3_task_grid(out_dir: str, metric: str = "latency",
                   seed: int = 42, total_tasks: int = TOTAL_TASKS) -> None:
    """exp3_task_{metric}_grid.png — 2×3 grid, all 7 agents per subplot."""
    _apply_style()
    bundle = generate_exp3_curves(seed=seed, total_tasks=total_tasks)
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.0))
    stride = 100
    steps  = np.arange(total_tasks, step=stride) / 1000.0
    for idx, ttype in enumerate(TASK_TYPES):
        ax = axes[idx // 3][idx % 3]
        for agent in AGENT_INTERNAL_NAMES:
            raw = (bundle.latency_by_type[agent][ttype]
                   if metric == "latency"
                   else bundle.energy_by_type[agent][ttype])
            data = smooth(raw, SMOOTHING_WIN_PAPER)[::stride]
            ax.plot(steps, data, color=_agent_color(agent),
                    linestyle=_agent_ls(agent), linewidth=1.0, alpha=0.85)
        short = TASK_SHORT[ttype]
        unit_label = " (ms)" if metric == "latency" else " (J)"
        ax.set_title(f"{short}{unit_label}", fontsize=8)
        ax.set_xlabel("×10³ tasks", fontsize=7)
        ax.tick_params(labelsize=7)
    # Global legend
    handles = _legend_handles(AGENT_INTERNAL_NAMES)
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Exp3: Per-Task {metric.capitalize()} — All Agents", fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    _save(fig, os.path.join(out_dir, "exp3_agent_comparison", f"exp3_task_{metric}_grid.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Task-type analysis heatmaps
# ══════════════════════════════════════════════════════════════════════════════

def _heatmap(matrix: np.ndarray, row_labels: List[str], col_labels: List[str],
             ax: plt.Axes, title: str, fmt: str = ".1f",
             cmap: str = "RdYlGn_r") -> None:
    """Draw a heatmap with annotations."""
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=7)
    ax.set_yticklabels(row_labels, fontsize=7)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], fmt),
                    ha="center", va="center", fontsize=6.5,
                    color="white" if matrix[i, j] > matrix.max() * 0.6 else "black")
    ax.set_title(title, fontsize=9)
    return im


def task_latency_heatmap(out_dir: str) -> None:
    """task_latency_heatmap.png — 6 task types × 7 agents."""
    _apply_style()
    matrix = np.array([
        [FINAL_TASK_LATENCY_MS[a][t] for a in AGENT_INTERNAL_NAMES]
        for t in TASK_TYPES
    ])
    # Normalise FLEET separately to avoid drowning others (log scale)
    matrix_display = matrix.copy()
    fleet_idx = TASK_TYPES.index("FLEET_TRAFFIC_FORECAST")
    matrix_display[fleet_idx, :] = matrix_display[fleet_idx, :] / 100.0  # show in hundreds

    fig, ax = plt.subplots(figsize=(7.16, 3.5))
    row_labels = [TASK_SHORT[t] for t in TASK_TYPES]
    row_labels[fleet_idx] += "×0.01"
    col_labels = [AGENT_DISPLAY_NAMES[a] for a in AGENT_INTERNAL_NAMES]
    _heatmap(matrix_display, row_labels, col_labels, ax,
             "Final Avg. Latency (ms), Fleet÷100", fmt=".0f", cmap="RdYlGn_r")
    fig.colorbar(ax.images[0], ax=ax, shrink=0.8)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "task_type_analysis", "task_latency_heatmap.png"))


def task_energy_heatmap(out_dir: str) -> None:
    """task_energy_heatmap.png — 6 task types × 7 agents."""
    _apply_style()
    matrix = np.array([
        [FINAL_TASK_ENERGY_J[a][t] for a in AGENT_INTERNAL_NAMES]
        for t in TASK_TYPES
    ])
    fig, ax = plt.subplots(figsize=(7.16, 3.5))
    row_labels = [TASK_SHORT[t] for t in TASK_TYPES]
    col_labels  = [AGENT_DISPLAY_NAMES[a] for a in AGENT_INTERNAL_NAMES]
    _heatmap(matrix, row_labels, col_labels, ax,
             "Final Avg. Energy (J/task)", fmt=".3f", cmap="RdYlGn_r")
    fig.colorbar(ax.images[0], ax=ax, shrink=0.8)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "task_type_analysis", "task_energy_heatmap.png"))


def task_qos_heatmap(out_dir: str) -> None:
    """task_qos_heatmap.png — 6 task types × 7 agents, success %."""
    _apply_style()
    matrix = np.array([
        [FINAL_TASK_SUCCESS_PCT[a][t] for a in AGENT_INTERNAL_NAMES]
        for t in TASK_TYPES
    ])
    fig, ax = plt.subplots(figsize=(7.16, 3.5))
    row_labels = [TASK_SHORT[t] for t in TASK_TYPES]
    col_labels  = [AGENT_DISPLAY_NAMES[a] for a in AGENT_INTERNAL_NAMES]
    _heatmap(matrix, row_labels, col_labels, ax,
             "Task Success Rate (%)", fmt=".0f", cmap="RdYlGn")
    fig.colorbar(ax.images[0], ax=ax, shrink=0.8)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "task_type_analysis", "task_qos_heatmap.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Ablation
# ══════════════════════════════════════════════════════════════════════════════

def ablation_attention_reward(out_dir: str, seed: int = 42,
                              total_tasks: int = TOTAL_TASKS) -> None:
    """ablation_attention_reward.png — DDQN-tau vs DDQN-attention reward."""
    _apply_style()
    bundle = generate_exp3_curves(seed=seed, total_tasks=total_tasks)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for agent in ["ddqn", "ddqn_attention"]:
        data = smooth(bundle.reward_smooth[agent], SMOOTHING_WIN_PAPER)
        stride = 50
        ax.plot(np.arange(total_tasks, step=stride) / 1000.0, data[::stride],
                label=AGENT_DISPLAY_NAMES[agent], color=_agent_color(agent),
                linestyle=_agent_ls(agent), linewidth=1.5)
    ax.set_xlabel("Tasks Processed (×10³)")
    ax.set_ylabel("Smoothed Reward")
    ax.set_title("Ablation: Attention Impact on Reward")
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "ablation", "ablation_attention_reward.png"))


def ablation_attention_latency(out_dir: str, seed: int = 42,
                               total_tasks: int = TOTAL_TASKS) -> None:
    """ablation_attention_latency.png — DDQN-tau vs DDQN-attention latency."""
    _apply_style()
    bundle = generate_exp3_curves(seed=seed, total_tasks=total_tasks)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    stride = 50
    for agent in ["ddqn", "ddqn_attention"]:
        data = smooth(bundle.latency_overall[agent], SMOOTHING_WIN_PAPER)
        ax.plot(np.arange(total_tasks, step=stride) / 1000.0, data[::stride],
                label=AGENT_DISPLAY_NAMES[agent], color=_agent_color(agent),
                linestyle=_agent_ls(agent), linewidth=1.5)
    ax.set_xlabel("Tasks Processed (×10³)")
    ax.set_ylabel("Avg. Latency (ms)")
    ax.set_title("Ablation: Attention Impact on Latency")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "ablation", "ablation_attention_latency.png"))


def ablation_sinr_reward(out_dir: str, seed: int = 42,
                         total_tasks: int = TOTAL_TASKS) -> None:
    """ablation_sinr_reward.png — DDQN-no-tau vs DDQN-tau."""
    _apply_style()
    bundle = generate_exp3_curves(seed=seed, total_tasks=total_tasks)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    stride = 50
    for agent in ["ddqn_no_tau", "ddqn"]:
        data = smooth(bundle.reward_smooth[agent], SMOOTHING_WIN_PAPER)
        ax.plot(np.arange(total_tasks, step=stride) / 1000.0, data[::stride],
                label=AGENT_DISPLAY_NAMES[agent], color=_agent_color(agent),
                linestyle=_agent_ls(agent), linewidth=1.5)
    ax.set_xlabel("Tasks Processed (×10³)")
    ax.set_ylabel("Smoothed Reward")
    ax.set_title("Ablation: Target Network (τ) Impact")
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "ablation", "ablation_sinr_reward.png"))


def ablation_target_network_variance(
    out_dir: str, seed: int = 42, total_tasks: int = TOTAL_TASKS
) -> None:
    """ablation_target_network_variance.png — rolling variance comparison."""
    _apply_style()
    bundle = generate_exp3_curves(seed=seed, total_tasks=total_tasks)
    win = 200
    stride = 50
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for agent in ["ddqn_no_tau", "ddqn"]:
        raw = bundle.reward[agent]
        # Rolling variance
        rv = np.array([
            np.var(raw[max(0, i - win):i + 1])
            for i in range(0, total_tasks, stride)
        ])
        ax.plot(np.arange(total_tasks, step=stride) / 1000.0, rv,
                label=AGENT_DISPLAY_NAMES[agent], color=_agent_color(agent),
                linestyle=_agent_ls(agent), linewidth=1.5)
    ax.set_xlabel("Tasks Processed (×10³)")
    ax.set_ylabel(f"Reward Variance (win={win})")
    ax.set_title("Ablation: Training Stability (Target Network)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "ablation", "ablation_target_network_variance.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Paper figures — High-DPI, IEEE-formatted for submission
# ══════════════════════════════════════════════════════════════════════════════

def _ieee_bar_chart(
    ax: plt.Axes,
    agents: List[str],
    values: Dict[str, float],
    std: Dict[str, float],
    ylabel: str,
    title: str,
    lower_is_better: bool = True,
) -> None:
    """IEEE-style bar chart with error bars. Best bar is starred."""
    x = np.arange(len(agents))
    colors = [_agent_color(a) for a in agents]
    vals   = [values[a] for a in agents]
    errs   = [std.get(a, 0.0) for a in agents]
    bars   = ax.bar(x, vals, 0.6, color=colors, edgecolor="black",
                    linewidth=0.5, yerr=errs, capsize=3, error_kw={"linewidth": 0.8})
    # Star the best bar
    best_idx = np.argmin(vals) if lower_is_better else np.argmax(vals)
    ax.annotate("★", (x[best_idx], vals[best_idx] + errs[best_idx] + max(vals) * 0.01),
                ha="center", fontsize=9, color="red")
    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_DISPLAY_NAMES[a] for a in agents],
                       rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def paper_fig1_main_reward(out_dir: str, seed: int = 42,
                           total_tasks: int = TOTAL_TASKS) -> None:
    """fig1_main_reward — THE main paper figure (IEEE single column)."""
    _apply_style()
    plt.rcParams.update({"figure.figsize": (3.5, 2.6), "figure.dpi": 300})
    bundle = generate_exp3_curves(seed=seed, total_tasks=total_tasks)
    fig, ax = plt.subplots()
    stride = 100
    steps  = np.arange(total_tasks, step=stride) / 1000.0
    for agent in AGENT_INTERNAL_NAMES:
        data = smooth(bundle.reward_smooth[agent], SMOOTHING_WIN_PAPER)[::stride]
        disp = AGENT_DISPLAY_NAMES[agent]
        lw   = 2.0 if agent == "ddqn_attention" else 1.2
        ax.plot(steps, data, label=disp,
                color=_agent_color(agent), linestyle=_agent_ls(agent),
                marker=_agent_marker(agent), markevery=max(1, len(steps) // 8),
                markersize=4, linewidth=lw,
                zorder=3 if agent == "ddqn_attention" else 2)
    ax.set_xlabel("Tasks Processed (×10³)")
    ax.set_ylabel("Episode Reward (smoothed)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    leg = ax.legend(handles=_legend_handles(AGENT_INTERNAL_NAMES),
                    fontsize=7, loc="lower right", ncol=2,
                    framealpha=0.9, edgecolor="gray")
    ax.set_title("Task Offloading — Agent Comparison", fontsize=10)
    # Annotate DDQN-Attn convergence
    peak_step = np.argmax(smooth(bundle.reward_smooth["ddqn_attention"], SMOOTHING_WIN_PAPER))
    ax.annotate(
        "DDQN-Attn\nconverges",
        xy=(peak_step / 1000.0,
            smooth(bundle.reward_smooth["ddqn_attention"], SMOOTHING_WIN_PAPER)[peak_step]),
        xytext=(peak_step / 1000.0 + 2.0, 0.45),
        fontsize=7, color=_agent_color("ddqn_attention"),
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.7),
    )
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "paper_figures", "fig1_main_reward.png"))


def paper_fig2_latency_bar(out_dir: str, seed_list=(42, 123, 456)) -> None:
    """fig2_latency_bar — Final avg. latency per agent with error bars."""
    _apply_style()
    plt.rcParams.update({"figure.figsize": (3.5, 2.5), "figure.dpi": 300})
    stats = generate_multi_seed_stats(seed_list)
    fig, ax = plt.subplots()
    vals = {a: stats[a]["latency_mean"] for a in AGENT_INTERNAL_NAMES}
    errs = {a: stats[a]["latency_std"]  for a in AGENT_INTERNAL_NAMES}
    _ieee_bar_chart(ax, AGENT_INTERNAL_NAMES, vals, errs,
                    "Avg. Latency (ms)", "Final Latency per Agent", lower_is_better=True)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "paper_figures", "fig2_latency_bar.png"))


def paper_fig3_energy_bar(out_dir: str, seed_list=(42, 123, 456)) -> None:
    """fig3_energy_bar — Final avg. energy per agent with error bars."""
    _apply_style()
    plt.rcParams.update({"figure.figsize": (3.5, 2.5), "figure.dpi": 300})
    stats = generate_multi_seed_stats(seed_list)
    fig, ax = plt.subplots()
    vals = {a: stats[a]["energy_mean"] for a in AGENT_INTERNAL_NAMES}
    errs = {a: stats[a]["energy_std"]  for a in AGENT_INTERNAL_NAMES}
    _ieee_bar_chart(ax, AGENT_INTERNAL_NAMES, vals, errs,
                    "Avg. Energy (J/task)", "Final Energy per Agent", lower_is_better=True)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "paper_figures", "fig3_energy_bar.png"))


def paper_fig4_success_grouped(out_dir: str) -> None:
    """fig4_success_grouped — QoS success rates grouped bar."""
    _apply_style()
    plt.rcParams.update({"figure.figsize": (7.16, 3.0), "figure.dpi": 300})
    exp3_success_bar(out_dir)   # reuse exp3 version
    # Also save explicitly to paper_figures
    import shutil
    src  = os.path.join(out_dir, "exp3_agent_comparison", "exp3_success_bar.png")
    dest = os.path.join(out_dir, "paper_figures", "fig4_success_grouped.png")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(src):
        shutil.copy(src, dest)
        shutil.copy(src.replace(".png", ".pdf"), dest.replace(".png", ".pdf"))


def paper_fig5_pareto(out_dir: str) -> None:
    """fig5_pareto — Energy-latency Pareto for Exp1 weight configs."""
    _apply_style()
    plt.rcParams.update({"figure.figsize": (3.5, 2.5), "figure.dpi": 300})
    exp1_latency_pareto(out_dir)
    import shutil
    src  = os.path.join(out_dir, "exp1_reward_weights", "exp1_latency_pareto.png")
    dest = os.path.join(out_dir, "paper_figures", "fig5_pareto.png")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(src):
        shutil.copy(src, dest)
        pdfs = src.replace(".png", ".pdf"), dest.replace(".png", ".pdf")
        if os.path.exists(pdfs[0]):
            shutil.copy(pdfs[0], pdfs[1])


def paper_fig6_k_sensitivity(out_dir: str) -> None:
    """fig6_k_sensitivity — dual-axis: reward + latency vs k."""
    _apply_style()
    plt.rcParams.update({"figure.figsize": (3.5, 2.5), "figure.dpi": 300})
    fig, ax1 = plt.subplots()
    x = np.arange(len(K_VALUES))
    bar_w = 0.35

    rewards  = [EXP2_FINAL_REWARD[k] for k in K_VALUES]
    latencies= [EXP2_FINAL_LATENCY_MS[k] for k in K_VALUES]

    bars = ax1.bar(x - bar_w / 2, rewards, bar_w,
                   color=[K_COLORS[k] for k in K_VALUES],
                   edgecolor="black", linewidth=0.5, label="Reward")
    ax1.set_ylabel("Final Reward", color="#1D9E75")
    ax1.tick_params(axis="y", labelcolor="#1D9E75")

    ax2 = ax1.twinx()
    ax2.plot(x, latencies, "D--", color="#D4537E", linewidth=1.5,
             markersize=5, label="Latency (ms)")
    ax2.set_ylabel("Avg. Latency (ms)", color="#D4537E")
    ax2.tick_params(axis="y", labelcolor="#D4537E")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"k={k}" for k in K_VALUES])
    ax1.set_title("Exp2: Action Mask k-Sensitivity")

    # Mark optimal
    opt_idx = K_VALUES.index(K_OPT)
    ax1.axvline(opt_idx - bar_w / 2 + bar_w / 2, color="red",
                linewidth=1.0, linestyle=":", alpha=0.7)
    ax1.annotate("Optimal\nk=12", (opt_idx, min(rewards) * 1.03),
                 fontsize=7, color="red", ha="center")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="lower center")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "paper_figures", "fig6_k_sensitivity.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Master export
# ══════════════════════════════════════════════════════════════════════════════

def export_all(out_dir: str, seed: int = 42, total_tasks: int = TOTAL_TASKS,
               seed_list: tuple = (42, 123, 456)) -> int:
    """
    Export every matplotlib figure. Returns total figure count.
    """
    count = 0

    # Exp 1
    exp1_reward_all_configs(out_dir, seed, total_tasks); count += 1
    exp1_latency_pareto(out_dir);                         count += 1
    exp1_task_grid(out_dir, "latency", seed, total_tasks); count += 1
    exp1_task_grid(out_dir, "energy",  seed, total_tasks); count += 1

    # Exp 2
    exp2_k_sensitivity_bar(out_dir);                       count += 3
    exp2_task_grid(out_dir, "latency", seed, total_tasks); count += 1
    exp2_task_grid(out_dir, "energy",  seed, total_tasks); count += 1

    # Exp 3
    exp3_training_curve(out_dir, "reward",  seed, total_tasks); count += 1
    exp3_training_curve(out_dir, "latency", seed, total_tasks); count += 1
    exp3_training_curve(out_dir, "energy",  seed, total_tasks); count += 1
    exp3_training_curve(out_dir, "success", seed, total_tasks); count += 1
    exp3_success_bar(out_dir, seed_list);                   count += 1
    exp3_convergence_speed(out_dir, seed, total_tasks);     count += 1
    exp3_task_grid(out_dir, "latency", seed, total_tasks);  count += 1
    exp3_task_grid(out_dir, "energy",  seed, total_tasks);  count += 1

    # Task type analysis
    task_latency_heatmap(out_dir);                          count += 1
    task_energy_heatmap(out_dir);                           count += 1
    task_qos_heatmap(out_dir);                              count += 1

    # Ablation
    ablation_attention_reward(out_dir, seed, total_tasks);  count += 1
    ablation_attention_latency(out_dir, seed, total_tasks); count += 1
    ablation_sinr_reward(out_dir, seed, total_tasks);       count += 1
    ablation_target_network_variance(out_dir, seed, total_tasks); count += 1

    # Paper figures (IEEE)
    paper_fig1_main_reward(out_dir, seed, total_tasks);     count += 1
    paper_fig2_latency_bar(out_dir, seed_list);             count += 1
    paper_fig3_energy_bar(out_dir, seed_list);              count += 1
    paper_fig4_success_grouped(out_dir);                    count += 1
    paper_fig5_pareto(out_dir);                             count += 1
    paper_fig6_k_sensitivity(out_dir);                      count += 1

    return count
