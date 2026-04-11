"""
compare.py — Offline multi-run comparison plot generator.

Usage:
    python3 compare.py \
        --runs results/ddqn_heuristic_20260408_184145_inst0.json \
               results/greedy_compute_allOffload_20260408_192923_inst0.json \
               results/greedy_distance_allOffload_20260408_200108_inst0.json \
               results/local_allLocal_20260408_203456_inst0.json /
        --metrics reward latency energy success_rate \
        --output output/comparison_2026-04-08_test_run1.png

Loads multiple per-run JSON result files (written by main.py) and produces
smoothed line-plot comparisons.  Each run is labelled "<agent> (<mode>)".
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                           # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ── helpers ─────────────────────────────────────────────────────────────────

TASK_TYPES = [
    "LOCAL_OBJECT_DETECTION",
    "COOPERATIVE_PERCEPTION",
    "ROUTE_OPTIMIZATION",
    "FLEET_TRAFFIC_FORECAST",
    "VOICE_COMMAND_PROCESSING",
    "SENSOR_HEALTH_CHECK",
]

METRIC_CHOICES = ["reward", "latency", "energy", "success_rate", "qos", "failure"]

ALL_METRICS = {
    "reward":       "Reward",
    "success_rate": "Success Rate (%)",
    "latency":      "Latency (s) per Task Type",
    "energy":       "Energy (J) per Task Type",
    "qos":          "QoS Success Rate (%)",
    "failure":      "Failure Reason Breakdown",
}


def _smooth(values: list, window: int = 50) -> np.ndarray:
    """Cumulative-mean smoothing (like a rolling window, no edge artifacts)."""
    if not values:
        return np.array([])
    arr = np.array(values, dtype=float)
    if len(arr) <= window:
        return arr
    kernel = np.ones(window) / window
    # Use 'valid' mode and pad front so output length == input length
    smoothed = np.convolve(arr, kernel, mode="valid")
    pad = np.full(window - 1, smoothed[0])
    return np.concatenate([pad, smoothed])


def _label(run: dict) -> str:
    agent = run.get("agent", "?")
    mode  = run.get("offload_mode", "?")
    ts    = run.get("timestamp", "")[:10]   # date only
    iid   = run.get("instance_id", "")
    label = f"{agent} ({mode})"
    if iid != "":
        label += f" inst{iid}"
    if ts:
        label += f"  [{ts}]"
    return label


def _load_runs(paths: list[str]) -> list[dict]:
    runs = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"[WARN] File not found, skipping: {p}", file=sys.stderr)
            continue
        with open(path) as f:
            data = json.load(f)
        data["_file"] = str(path)
        runs.append(data)
    if not runs:
        print("[ERROR] No valid result files loaded.", file=sys.stderr)
        sys.exit(1)
    return runs


# ── per-metric plot functions ────────────────────────────────────────────────

def _plot_reward(ax, runs, window):
    ax.set_title("Reward per Offloaded Task")
    ax.set_xlabel("Offloaded Tasks")
    ax.set_ylabel("Reward")
    for run in runs:
        rewards = run.get("metrics", {}).get("rewards", [])
        if not rewards:
            continue
        y = _smooth(rewards, window)
        ax.plot(y, label=_label(run))
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _plot_success_rate(ax, runs, window):
    ax.set_title("Success Rate (all tasks, rolling)")
    ax.set_xlabel("Total Tasks")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 105)
    for run in runs:
        sr = run.get("metrics", {}).get("success_rates", [])
        if not sr:
            continue
        y = _smooth(sr, window)
        ax.plot(y, label=_label(run))
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _plot_per_type(axes, runs, metric_key: str, unit: str, window: int):
    """Plot one sub-panel per task type for latency or energy."""
    for idx, task_type in enumerate(TASK_TYPES):
        ax = axes[idx]
        short = task_type.replace("_", " ").title()
        ax.set_title(short, fontsize=8)
        ax.set_xlabel("Tasks of this type", fontsize=7)
        ax.set_ylabel(unit, fontsize=7)
        ax.tick_params(labelsize=6)
        for run in runs:
            values = run.get("metrics", {}).get(metric_key, {}).get(task_type, [])
            if not values:
                continue
            y = _smooth(values, window)
            ax.plot(y, label=_label(run), linewidth=0.9)
        ax.legend(fontsize=5)
        ax.grid(True, alpha=0.3)


def _plot_qos(ax, runs, window):
    ax.set_title("QoS Success Rate (%)")
    ax.set_xlabel("Total Tasks")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 105)
    styles = {"qos1": "-", "qos2": "--", "qos3": ":"}
    for run in runs:
        qos = run.get("metrics", {}).get("qos_success_rates", {})
        lbl = _label(run)
        for q, ls in styles.items():
            values = qos.get(q, [])
            if not values:
                continue
            y = _smooth(values, window)
            ax.plot(y, linestyle=ls, label=f"{lbl} [{q}]")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)


def _plot_failure(ax, runs):
    ax.set_title("Failure Reason Breakdown")
    ax.set_ylabel("Count")

    reasons = ["DEADLINE_MISSED", "RSU_QUEUE_FULL", "SV_OUT_OF_RANGE",
               "HANDOVER_FAIL", "NONE"]
    x = np.arange(len(reasons))
    width = 0.8 / max(len(runs), 1)

    for i, run in enumerate(runs):
        fr = run.get("metrics", {}).get("failure_reasons", {})
        counts = [fr.get(r, 0) for r in reasons]
        offset = (i - len(runs) / 2 + 0.5) * width
        ax.bar(x + offset, counts, width=width * 0.9, label=_label(run))

    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "\n") for r in reasons], fontsize=7)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")


# ── main layout builder ──────────────────────────────────────────────────────

def build_figure(runs: list[dict], metrics: list[str], window: int) -> plt.Figure:
    # Determine which subplots we need
    has_reward   = "reward"       in metrics
    has_success  = "success_rate" in metrics
    has_latency  = "latency"      in metrics
    has_energy   = "energy"       in metrics
    has_qos      = "qos"          in metrics
    has_failure  = "failure"      in metrics

    # Count rows:
    # row 0: reward + success_rate (side by side, if both present)
    # row 1: latency (6 sub-panels across)
    # row 2: energy  (6 sub-panels across)
    # row 3: qos + failure (side by side)
    sections = []
    if has_reward or has_success:
        sections.append("scalar")
    if has_latency:
        sections.append("latency")
    if has_energy:
        sections.append("energy")
    if has_qos or has_failure:
        sections.append("breakdown")

    n_sections = len(sections)
    if n_sections == 0:
        print("[ERROR] No recognised metrics to plot.", file=sys.stderr)
        sys.exit(1)

    fig = plt.figure(figsize=(18, 5 * n_sections))
    fig.suptitle("Agent Comparison", fontsize=14, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(n_sections, 6, figure=fig,
                           hspace=0.55, wspace=0.4)

    row = 0
    for section in sections:
        if section == "scalar":
            # Up to 2 wide panels in this row (reward left, success right)
            col_start = 0
            if has_reward:
                ax = fig.add_subplot(gs[row, 0:3])
                _plot_reward(ax, runs, window)
                col_start = 3
            if has_success:
                ax = fig.add_subplot(gs[row, col_start : col_start + 3])
                _plot_success_rate(ax, runs, window)

        elif section == "latency":
            axes = [fig.add_subplot(gs[row, c]) for c in range(6)]
            _plot_per_type(axes, runs, "latencies", "Latency (s)", window)
            # Row label
            axes[0].set_ylabel("Latency (s)", fontsize=8)
            fig.text(0.01, (n_sections - row - 0.5) / n_sections,
                     "Latency per Task Type", va="center",
                     rotation="vertical", fontsize=9, fontweight="bold")

        elif section == "energy":
            axes = [fig.add_subplot(gs[row, c]) for c in range(6)]
            _plot_per_type(axes, runs, "energies", "Energy (J)", window)
            fig.text(0.01, (n_sections - row - 0.5) / n_sections,
                     "Energy per Task Type", va="center",
                     rotation="vertical", fontsize=9, fontweight="bold")

        elif section == "breakdown":
            col_start = 0
            if has_qos:
                ax = fig.add_subplot(gs[row, 0:3])
                _plot_qos(ax, runs, window)
                col_start = 3
            if has_failure:
                ax = fig.add_subplot(gs[row, col_start : col_start + 3])
                _plot_failure(ax, runs)

        row += 1

    return fig


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare multiple single-agent-run result JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 compare.py \\
      --runs results/ddqn_heuristic_inst0.json results/random_allOffload_inst0.json \\
      --metrics reward latency energy success_rate \\
      --output output/comparison.png

  python3 compare.py \\
      --runs results/*.json \\
      --metrics success_rate qos failure
""",
    )
    p.add_argument("--runs", nargs="+", required=True, metavar="FILE",
                   help="One or more result JSON files produced by main.py")
    p.add_argument("--metrics", nargs="+",
                   choices=METRIC_CHOICES, default=list(METRIC_CHOICES),
                   metavar="METRIC",
                   help=f"Metrics to plot. Choices: {', '.join(METRIC_CHOICES)}. "
                        "Default: all.")
    p.add_argument("--output", default="output/comparison.png",
                   help="Output image path (PNG/PDF/SVG). Default: output/comparison.png")
    p.add_argument("--window", type=int, default=50,
                   help="Smoothing window for rolling averages. Default: 50.")
    p.add_argument("--dpi", type=int, default=150,
                   help="Output image DPI. Default: 150.")
    p.add_argument("--show", action="store_true",
                   help="Display interactive plot (requires display).")
    return p.parse_args()


def main():
    args = parse_args()

    # Expand glob patterns if shell didn't (e.g. --runs results/*.json on Windows)
    paths: list[str] = []
    for pattern in args.runs:
        expanded = sorted(glob.glob(pattern))
        if expanded:
            paths.extend(str(p) for p in expanded)
        else:
            paths.append(pattern)   # let _load_runs emit the warning

    runs = _load_runs(paths)
    print(f"Loaded {len(runs)} result file(s):")
    for r in runs:
        print(f"  {_label(r)}  ({r['_file']})")

    fig = build_figure(runs, args.metrics, args.window)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved comparison plot → {out}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
