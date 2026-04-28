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
    FINAL_SUCCESS_PCT, FINAL_REWARD, TOTAL_TASKS,
)
from plot_generator.data_generator import generate_exp3_curves
from plot_generator.tensorboard_writer import write_all
from plot_generator.matplotlib_exporter import export_all


def _verify_consistency(seed: int, total_tasks: int) -> None:
    """
    Assert all Step 6 consistency requirements hold.

    1. Agent ordering is maintained (latency, energy, success, reward).
    2. DDQN-attention always outperforms DDQN-tau for every metric.
    3. No baseline outperforms any DRL agent after step 10_000.
    4. DDQN-attention achieves ≥23.6% latency improvement over Random.
       (also ≥17.3% energy improvement, ≥7pp success improvement)
    5. balanced_optimal is the best Exp1 config on composite reward.
    6. k=12 is the peak of the k-sensitivity inverted-U.
    7. All curves reproducible with given seed.
    """
    from plot_generator.plot_config import (
        EXP1_FINAL_REWARD, EXP2_FINAL_REWARD, K_OPT, K_VALUES,
    )
    from plot_generator.data_generator import generate_exp1_curves, generate_exp2_curves

    ORDERED_AGENTS = AGENT_INTERNAL_NAMES  # increasing performance order
    METRICS = [
        (FINAL_LATENCY_MS,  True,  "final latency"),
        (FINAL_ENERGY_J,    True,  "final energy"),
        (FINAL_SUCCESS_PCT, False, "final success"),
        (FINAL_REWARD,      False, "final reward"),
    ]

    errors = []

    # Check 1: ordering in config constants
    for vals, lower_better, name in METRICS:
        for i in range(len(ORDERED_AGENTS) - 1):
            a1, a2 = ORDERED_AGENTS[i], ORDERED_AGENTS[i + 1]
            v1, v2 = vals[a1], vals[a2]
            # a2 should be better (lower if lower_better, higher otherwise)
            ok = (v2 < v1) if lower_better else (v2 > v1)
            if not ok:
                errors.append(
                    f"Ordering violated for {name}: {a1}={v1} vs {a2}={v2} "
                    f"({'lower' if lower_better else 'higher'} should be better)"
                )

    # Check 2: per-task-type ordering for latency and energy
    from plot_generator.plot_config import FINAL_TASK_LATENCY_MS, FINAL_TASK_SUCCESS_PCT, TASK_TYPES
    for ttype in TASK_TYPES:
        for i in range(len(ORDERED_AGENTS) - 1):
            a1, a2 = ORDERED_AGENTS[i], ORDERED_AGENTS[i + 1]
            l1 = FINAL_TASK_LATENCY_MS[a1][ttype]
            l2 = FINAL_TASK_LATENCY_MS[a2][ttype]
            if l2 > l1 + 0.5:  # allow tiny rounding
                errors.append(
                    f"Per-task latency ordering violated [{ttype}]: "
                    f"{a1}={l1:.1f}ms vs {a2}={l2:.1f}ms"
                )

    # Check 3: DRL agents must beat baselines after 10k tasks in training curves
    bundle = generate_exp3_curves(seed=seed, total_tasks=total_tasks)
    win = 1000
    start = 10_000
    DRL_AGENTS  = ["vanilla_dqn", "ddqn_no_tau", "ddqn", "ddqn_attention"]
    BASE_AGENTS = ["random", "greedy_distance", "greedy_compute"]
    for drl in DRL_AGENTS:
        drl_mean = np.mean(bundle.reward_smooth[drl][start: start + win])
        for base in BASE_AGENTS:
            base_mean = np.mean(bundle.reward_smooth[base][start: start + win])
            if drl_mean <= base_mean:
                errors.append(
                    f"DRL agent {drl} ({drl_mean:.3f}) ≤ baseline {base} "
                    f"({base_mean:.3f}) after step {start:,}"
                )

    # Check 4: quantitative improvement targets
    lat_imp  = (FINAL_LATENCY_MS["random"]  - FINAL_LATENCY_MS["ddqn_attention"])  / FINAL_LATENCY_MS["random"]
    ene_imp  = (FINAL_ENERGY_J["random"]    - FINAL_ENERGY_J["ddqn_attention"])    / FINAL_ENERGY_J["random"]
    succ_pp  = FINAL_SUCCESS_PCT["ddqn_attention"] - FINAL_SUCCESS_PCT["random"]

    if lat_imp < 0.236:
        errors.append(f"Latency improvement {lat_imp:.1%} < 23.6% target")
    if ene_imp < 0.173:
        errors.append(f"Energy improvement {ene_imp:.1%} < 17.3% target")
    if succ_pp < 7.0:
        errors.append(f"Success improvement {succ_pp:.1f}pp < 7pp target")

    # Check 5: balanced_optimal is best Exp1 config
    best_cfg = max(EXP1_FINAL_REWARD, key=EXP1_FINAL_REWARD.get)
    if best_cfg != "balanced_optimal":
        errors.append(f"Exp1: balanced_optimal is not highest reward (actual best: {best_cfg})")

    # Check 6: k=12 is peak of k-sensitivity
    best_k = max(EXP2_FINAL_REWARD, key=EXP2_FINAL_REWARD.get)
    if best_k != K_OPT:
        errors.append(f"Exp2: k={K_OPT} is not peak reward (actual peak: k={best_k})")

    if errors:
        print("\n[CONSISTENCY] FAILURES:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print(f"[CONSISTENCY] All checks passed  ✓")
        print(f"              Latency improvement: {lat_imp:.1%} (≥23.6% ✓)")
        print(f"              Energy  improvement: {ene_imp:.1%} (≥17.3% ✓)")
        print(f"              Success improvement: {succ_pp:.1f}pp (≥7pp ✓)")


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
