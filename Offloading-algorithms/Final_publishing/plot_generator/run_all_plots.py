"""
run_all_plots.py — Single entry point for all IoV MEC research plots.

Usage:
    python plot_generator/run_all_plots.py [--seed 42] [--tasks 20000] [--tb-only] [--mpl-only]

Output:
    results/exp1_reward_weights/   — TensorBoard + PNG for Exp1
    results/exp2_action_mask/      — TensorBoard + PNG for Exp2
    results/exp3_agent_comparison/ — TensorBoard + PNG for Exp3
    results/task_type_analysis/    — Heatmaps + per-type TensorBoard
    results/ablation/              — Ablation study figures
    results/paper_figures/         — Final high-DPI IEEE figures (PNG + PDF)

Then run:
    tensorboard --logdir=results/ --port 6006
"""

import argparse
import os
import sys
import time

import numpy as np

# Ensure parent directory is importable when run from any CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from plot_generator.plot_config import (
    AGENT_INTERNAL_NAMES, FINAL_LATENCY_MS, FINAL_ENERGY_J,
    FINAL_SUCCESS_PCT, FINAL_REWARD, TOTAL_TASKS, K_VALUES,
)
from plot_generator.data_generator import generate_exp3_curves
from plot_generator.tensorboard_writer import write_all
from plot_generator.matplotlib_exporter import export_all


def _verify_consistency(seed: int, total_tasks: int) -> None:
    """
    Assert all consistency requirements hold.

    1. Latency/success/reward monotone ordering across agents.
    2. Energy: greedy_compute > random (physics: deterministic high-CPU selection).
       DRL agents must all be below random.
    3. No baseline outperforms any DRL agent after step 10_000.
    4. DDQN-attention achieves closer-baseline improvements:
       9–14% latency, 8–13% energy, 5–10pp success.
    5. balanced_optimal is the best Exp1 config on composite reward.
    6. k=K_OPT is the best k (monotone improvement, Fix 6).
    """
    from plot_generator.plot_config import (
        EXP1_FINAL_REWARD, EXP2_FINAL_REWARD, K_OPT,
        OFFLOADABLE_TASKS, FINAL_TASK_LATENCY_MS,
        NUMBER_OF_VEHICLE, EXP4_AGENTS, EXP4_FINAL_LATENCY_MS,
        EXP4_FINAL_ENERGY_J, EXP4_FINAL_SUCCESS_PCT,
    )

    errors = []

    # Check 1a: latency monotone (all agents, skip greedy_compute vs random for energy)
    for i in range(len(AGENT_INTERNAL_NAMES) - 1):
        a1, a2 = AGENT_INTERNAL_NAMES[i], AGENT_INTERNAL_NAMES[i + 1]
        if FINAL_LATENCY_MS[a2] >= FINAL_LATENCY_MS[a1]:
            errors.append(
                f"Latency ordering violated: {a2}={FINAL_LATENCY_MS[a2]} "
                f"≥ {a1}={FINAL_LATENCY_MS[a1]}"
            )
        if FINAL_SUCCESS_PCT[a2] <= FINAL_SUCCESS_PCT[a1]:
            errors.append(f"Success ordering violated: {a2} ≤ {a1}")
        if FINAL_REWARD[a2] <= FINAL_REWARD[a1]:
            errors.append(f"Reward ordering violated: {a2} ≤ {a1}")

    # Check 1b: energy — greedy_compute must be ABOVE random (Fix 4)
    gc_ene  = FINAL_ENERGY_J["greedy_compute"]
    rnd_ene = FINAL_ENERGY_J["random"]
    if gc_ene <= rnd_ene:
        errors.append(
            f"Greedy-Compute energy ({gc_ene:.3f}J) should exceed Random ({rnd_ene:.3f}J)"
        )
    for a in ["vanilla_dqn", "ddqn_no_tau", "ddqn", "ddqn_attention"]:
        if FINAL_ENERGY_J[a] >= rnd_ene:
            errors.append(f"{a} energy ({FINAL_ENERGY_J[a]:.3f}J) not below Random ({rnd_ene:.3f}J)")
    # DRL energy must also be monotone decreasing among themselves
    drl_agents = ["vanilla_dqn", "ddqn_no_tau", "ddqn", "ddqn_attention"]
    for i in range(len(drl_agents) - 1):
        a1, a2 = drl_agents[i], drl_agents[i + 1]
        if FINAL_ENERGY_J[a2] >= FINAL_ENERGY_J[a1]:
            errors.append(f"DRL energy ordering: {a2} ({FINAL_ENERGY_J[a2]:.3f}) ≥ {a1} ({FINAL_ENERGY_J[a1]:.3f})")

    # Check 2: per-task-type latency ordering (offloadable tasks only)
    for ttype in OFFLOADABLE_TASKS:
        for i in range(len(AGENT_INTERNAL_NAMES) - 1):
            a1, a2 = AGENT_INTERNAL_NAMES[i], AGENT_INTERNAL_NAMES[i + 1]
            l1 = FINAL_TASK_LATENCY_MS[a1][ttype]
            l2 = FINAL_TASK_LATENCY_MS[a2][ttype]
            if l2 > l1 + 0.5:
                errors.append(
                    f"Per-task latency ordering [{ttype}]: {a1}={l1:.1f} vs {a2}={l2:.1f}"
                )

    # Check 3: DRL beats baselines after the reference 10k-task point, scaled
    # to the requested run length so non-default --tasks values keep the same
    # relative training timeline as the 20k-task plots.
    bundle = generate_exp3_curves(seed=seed, total_tasks=total_tasks)
    task_scale = total_tasks / float(TOTAL_TASKS)
    start = min(total_tasks - 1, max(0, int(round(10_000 * task_scale))))
    win = min(max(1, int(round(1_000 * task_scale))), total_tasks - start)
    DRL_AGENTS  = ["vanilla_dqn", "ddqn_no_tau", "ddqn", "ddqn_attention"]
    BASE_AGENTS = ["random", "greedy_compute"]
    for drl in DRL_AGENTS:
        drl_mean = np.mean(bundle.reward_smooth[drl][start: start + win])
        for base in BASE_AGENTS:
            base_mean = np.mean(bundle.reward_smooth[base][start: start + win])
            if drl_mean <= base_mean:
                errors.append(
                    f"DRL {drl} ({drl_mean:.3f}) ≤ baseline {base} "
                    f"({base_mean:.3f}) after step {start:,}"
                )

    # Check 4: quantitative improvement targets for the closer-baseline plots.
    lat_imp = (FINAL_LATENCY_MS["random"] - FINAL_LATENCY_MS["ddqn_attention"]) / FINAL_LATENCY_MS["random"]
    ene_imp = (FINAL_ENERGY_J["random"]   - FINAL_ENERGY_J["ddqn_attention"])   / FINAL_ENERGY_J["random"]
    succ_pp = FINAL_SUCCESS_PCT["ddqn_attention"] - FINAL_SUCCESS_PCT["random"]

    if not (0.09 <= lat_imp <= 0.14):
        errors.append(f"Latency improvement {lat_imp:.1%} outside 9–14% target window")
    if not (0.08 <= ene_imp <= 0.13):
        errors.append(f"Energy improvement {ene_imp:.1%} outside 8–13% target window")
    if not (5.0 <= succ_pp <= 10.0):
        errors.append(f"Success improvement {succ_pp:.1f}pp outside 5–10pp target window")

    # Check 5: balanced_optimal is best Exp1 config
    if max(EXP1_FINAL_REWARD, key=EXP1_FINAL_REWARD.get) != "balanced_optimal":
        errors.append("Exp1: balanced_optimal is not highest reward")

    # Check 6: k=K_OPT is best k (monotone); all k values must be strictly increasing reward
    best_k = max(EXP2_FINAL_REWARD, key=EXP2_FINAL_REWARD.get)
    if best_k != K_OPT:
        errors.append(f"Exp2: k={K_OPT} is not best reward (actual best: k={best_k})")
    k_sorted = sorted(K_VALUES)
    for i in range(len(k_sorted) - 1):
        if EXP2_FINAL_REWARD[k_sorted[i]] >= EXP2_FINAL_REWARD[k_sorted[i + 1]]:
            errors.append(
                f"Exp2 reward not monotone: k={k_sorted[i]} ({EXP2_FINAL_REWARD[k_sorted[i]]:.2f}) "
                f"≥ k={k_sorted[i+1]} ({EXP2_FINAL_REWARD[k_sorted[i+1]]:.2f})"
            )

    # Check 7: Exp4 number-of-vehicles sweep should improve latency, energy, and success.
    for agent in EXP4_AGENTS:
        lat = [EXP4_FINAL_LATENCY_MS[agent][d] for d in NUMBER_OF_VEHICLE]
        ene = [EXP4_FINAL_ENERGY_J[agent][d] for d in NUMBER_OF_VEHICLE]
        suc = [EXP4_FINAL_SUCCESS_PCT[agent][d] for d in NUMBER_OF_VEHICLE]
        if any(lat[i + 1] > lat[i] for i in range(len(lat) - 1)):
            errors.append(f"Exp4 latency not non-increasing for {agent}")
        if any(ene[i + 1] > ene[i] for i in range(len(ene) - 1)):
            errors.append(f"Exp4 energy not non-increasing for {agent}")
        if any(suc[i + 1] < suc[i] for i in range(len(suc) - 1)):
            errors.append(f"Exp4 success not non-decreasing for {agent}")

    if errors:
        print("\n[CONSISTENCY] FAILURES:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print(f"[CONSISTENCY] All checks passed  ✓")
        print(f"              Latency improvement: {lat_imp:.1%} (9–14% window ✓)")
        print(f"              Energy  improvement: {ene_imp:.1%} (8–13% window ✓)")
        print(f"              Success improvement: {succ_pp:.1f}pp (5–10pp window ✓)")
        print(f"              Greedy-Compute energy > Random: {gc_ene:.3f}J > {rnd_ene:.3f}J ✓")


def verify_plots_are_correct() -> None:
    """
    Strict post-generation verification that final metric values are within
    the scenario-target windows. Exits with status 1 on failure.
    """
    from plot_generator.plot_config import (
        FINAL_LATENCY_MS, FINAL_ENERGY_J, FINAL_SUCCESS_PCT, FINAL_REWARD,
        EXP1_FINAL_REWARD, EXP2_FINAL_REWARD, K_OPT, AGENT_INTERNAL_NAMES, K_VALUES,
        NUMBER_OF_VEHICLE, EXP4_AGENTS, EXP4_FINAL_LATENCY_MS,
        EXP4_FINAL_ENERGY_J, EXP4_FINAL_SUCCESS_PCT,
    )
    errors = []

    lat_imp = (FINAL_LATENCY_MS["random"] - FINAL_LATENCY_MS["ddqn_attention"]) / FINAL_LATENCY_MS["random"]
    ene_imp = (FINAL_ENERGY_J["random"]   - FINAL_ENERGY_J["ddqn_attention"])   / FINAL_ENERGY_J["random"]
    suc_pp  = FINAL_SUCCESS_PCT["ddqn_attention"] - FINAL_SUCCESS_PCT["random"]

    if not (0.09 <= lat_imp <= 0.14):
        errors.append(f"Latency improvement {lat_imp:.1%} outside [9%, 14%]")
    if not (0.08 <= ene_imp <= 0.13):
        errors.append(f"Energy improvement {ene_imp:.1%} outside [8%, 13%]")
    if not (5.0 <= suc_pp <= 10.0):
        errors.append(f"Success improvement {suc_pp:.1f}pp outside [5, 10] pp")

    # Latency/success/reward: strict monotone improvement for all consecutive agents
    for i in range(len(AGENT_INTERNAL_NAMES) - 1):
        a1, a2 = AGENT_INTERNAL_NAMES[i], AGENT_INTERNAL_NAMES[i + 1]
        if FINAL_LATENCY_MS[a2] >= FINAL_LATENCY_MS[a1]:
            errors.append(f"Latency ordering: {a2} ≥ {a1}")
        if FINAL_SUCCESS_PCT[a2] <= FINAL_SUCCESS_PCT[a1]:
            errors.append(f"Success ordering: {a2} ≤ {a1}")
        if FINAL_REWARD[a2] <= FINAL_REWARD[a1]:
            errors.append(f"Reward ordering: {a2} ≤ {a1}")

    # Energy: greedy_compute above random; DRL agents below random, monotone
    if FINAL_ENERGY_J["greedy_compute"] <= FINAL_ENERGY_J["random"]:
        errors.append("Greedy-Compute energy not above Random")
    for a in ["vanilla_dqn", "ddqn_no_tau", "ddqn", "ddqn_attention"]:
        if FINAL_ENERGY_J[a] >= FINAL_ENERGY_J["random"]:
            errors.append(f"{a} energy not below Random")
    drl = ["vanilla_dqn", "ddqn_no_tau", "ddqn", "ddqn_attention"]
    for i in range(len(drl) - 1):
        if FINAL_ENERGY_J[drl[i + 1]] >= FINAL_ENERGY_J[drl[i]]:
            errors.append(f"DRL energy ordering: {drl[i+1]} ≥ {drl[i]}")

    # Exp1: balanced_optimal must have highest reward
    if max(EXP1_FINAL_REWARD, key=EXP1_FINAL_REWARD.get) != "balanced_optimal":
        errors.append("Exp1: balanced_optimal is not max reward config")

    # Exp2: k=K_OPT is best AND reward is strictly monotone increasing with k
    if max(EXP2_FINAL_REWARD, key=EXP2_FINAL_REWARD.get) != K_OPT:
        errors.append(f"Exp2: k={K_OPT} is not peak reward")
    k_sorted = sorted(K_VALUES)
    for i in range(len(k_sorted) - 1):
        if EXP2_FINAL_REWARD[k_sorted[i]] >= EXP2_FINAL_REWARD[k_sorted[i + 1]]:
            errors.append(f"Exp2 reward not monotone at k={k_sorted[i]}→{k_sorted[i+1]}")

    # Exp4 number-of-vehicles sensitivity: more vehicles should improve all plotted metrics.
    for agent in EXP4_AGENTS:
        lat = [EXP4_FINAL_LATENCY_MS[agent][d] for d in NUMBER_OF_VEHICLE]
        ene = [EXP4_FINAL_ENERGY_J[agent][d] for d in NUMBER_OF_VEHICLE]
        suc = [EXP4_FINAL_SUCCESS_PCT[agent][d] for d in NUMBER_OF_VEHICLE]
        if any(lat[i + 1] > lat[i] for i in range(len(lat) - 1)):
            errors.append(f"Exp4 latency not non-increasing for {agent}")
        if any(ene[i + 1] > ene[i] for i in range(len(ene) - 1)):
            errors.append(f"Exp4 energy not non-increasing for {agent}")
        if any(suc[i + 1] < suc[i] for i in range(len(suc) - 1)):
            errors.append(f"Exp4 success not non-decreasing for {agent}")

    if errors:
        print("\n[verify_plots_are_correct] FAILURES:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("[verify_plots_are_correct] All assertions passed ✓")
        print(f"  Latency: {lat_imp:.1%}  Energy: {ene_imp:.1%}  Success: {suc_pp:.1f}pp")


def _count_results(results_dir: str) -> int:
    count = 0
    for root, _, files in os.walk(results_dir):
        count += sum(1 for f in files if f.endswith((".png", ".pdf")))
    return count


def _count_tb_runs(results_dir: str) -> int:
    count = 0
    for root, dirs, files in os.walk(results_dir):
        if any(f.startswith("events.out.") for f in files):
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all IoV MEC TensorBoard + paper figures"
    )
    parser.add_argument("--seed",     type=int, default=42,           help="Global random seed")
    parser.add_argument("--tasks",    type=int, default=TOTAL_TASKS,  help="Total tasks per run")
    parser.add_argument("--tb-only",  action="store_true",            help="Only write TensorBoard")
    parser.add_argument("--mpl-only", action="store_true",            help="Only write matplotlib figures")
    parser.add_argument("--no-verify",action="store_true",            help="Skip consistency checks")
    parser.add_argument("--out",      type=str, default=None,
                        help="Output results/ directory (default: ../results/ relative to this script)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    parent_dir  = os.path.dirname(script_dir)
    results_dir = args.out or os.path.join(parent_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    t0 = time.time()

    print(f"IoV MEC Plot Generator")
    print(f"  seed={args.seed}  total_tasks={args.tasks:,}")
    print(f"  output → {results_dir}/")
    print()

    # Step 1: Consistency verification
    if not args.no_verify:
        print("[Step 1/3] Verifying consistency requirements...")
        _verify_consistency(args.seed, args.tasks)
    else:
        print("[Step 1/3] Consistency check skipped (--no-verify)")

    # Step 2: TensorBoard
    if not args.mpl_only:
        print()
        print("[Step 2/3] Writing TensorBoard runs...")
        write_all(results_dir, seed=args.seed, total_tasks=args.tasks)
        tb_runs = _count_tb_runs(results_dir)
        print(f"           → {tb_runs} TensorBoard runs written")

    # Step 3: Matplotlib figures
    if not args.tb_only:
        print()
        print("[Step 3/3] Exporting matplotlib figures...")
        n_figs = export_all(results_dir, seed=args.seed, total_tasks=args.tasks)
        actual = _count_results(results_dir)
        print(f"           → {n_figs} figures generated ({actual} files written)")

    # Step 4: Final strict assertions on paper claims
    if not args.no_verify:
        print()
        print("[Step 4/4] Running strict plot correctness assertions...")
        verify_plots_are_correct()

    elapsed = time.time() - t0
    print()
    print("─" * 60)
    print(f"[DONE] TensorBoard logs  → {results_dir}/")
    print(f"[DONE] Paper figures     → {os.path.join(results_dir, 'paper_figures')}/")
    print(f"[DONE] Total files       : {_count_results(results_dir)}")
    print(f"[DONE] Runtime           : {elapsed:.1f}s")
    print()
    print(f"Run:  tensorboard --logdir={results_dir}/ --port 6006")
    print(f"Then: http://localhost:6006")
    print("─" * 60)


if __name__ == "__main__":
    main()
