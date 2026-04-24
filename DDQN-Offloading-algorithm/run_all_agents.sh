#!/bin/bash
# =============================================================================
# run_all_agents.sh — Run all agent/simulation combinations sequentially,
#                     collect results per run, and produce a comparison plot.
#
# Usage:
#   ./run_all_agents.sh
#   ./run_all_agents.sh ddqn random
#   ./run_all_agents.sh ddqn,vanilla_dqn,greedy_distance
#
# Edit the USER CONFIGURATION section below before running.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
fi

# =============================================================================
# USER CONFIGURATION — edit this section
# =============================================================================

# Label for this batch of runs.
# Used for:
#   • DRL log subdirectory:  logs/<RUN_LABEL>/
#   • Sim log subdirectory:  <SIM_DIR>/logs/<RUN_LABEL>/
#   • Results subdirectory:  results/<RUN_LABEL>/
#   • TensorBoard subdirectory in configs/redis_config.json:
#       system.log_dir = output/<RUN_LABEL>
RUN_LABEL="${RUN_LABEL:-testnoon_all_heuristic}"

# Simulation time limit (passed to run_simulation.sh as --sim-time-limit)
SIM_TIME="7200s"

# Seconds to wait after starting the simulation before launching the DRL agent
# (gives the simulator time to connect to Redis and write initial state)
DRL_START_DELAY=4

# Capture simulator stdout/stderr into per-run log files only when explicitly
# enabled. The heuristic DDQN simulation is very chatty and can otherwise
# generate multi-giabyte logs.
CAPTURE_SIM_LOG=${CAPTURE_SIM_LOG:-0}

# Capture secondary DT stdout/stderr only when explicitly enabled.
CAPTURE_SECONDARY_DT_LOG=${CAPTURE_SECONDARY_DT_LOG:-0}

# ── Agents to run ────────────────────────────────────────────────────────────
# Format: "agent_name:SimulationConfig"
# Comment out any line to skip that run.
#
# agent_name must be one of:
#   ddqn | ddqn_no_tau | vanilla_dqn | random | greedy_compute |
#   min_latency | least_queue | greedy_distance | local
#
# SimulationConfig must match a [Config ...] section in omnetpp.ini:
#   Heuristic | AllOffload | AllLocal
RUNS=(
    "ddqn:Heuristic"
    "ddqn_no_tau:Heuristic"
    "vanilla_dqn:Heuristic"
    "random:Heuristic"
    "greedy_compute:Heuristic"
    "min_latency:Heuristic"
    "least_queue:Heuristic"
    "greedy_distance:Heuristic"
    "local:AllLocal"
)

SELECTED_AGENTS=("$@")

# ── Redis databases to flush before each run ─────────────────────────────────
# Must match the "redis_db" values of active=true instances in
# configs/redis_config.json  (currently: db 4, db 5, db 6)
# Loaded from .env when present; fallback is applied below.
if ! declare -p REDIS_DBS >/dev/null 2>&1 || [ ${#REDIS_DBS[@]} -eq 0 ]; then
    REDIS_DBS=(4 5 6)
fi

# ── compare.py settings ──────────────────────────────────────────────────────
# Output image path (relative to the DRL project directory)
COMPARE_OUTPUT="output/comparison_${RUN_LABEL}.png"

# Metrics to include in the comparison plot
# Available: reward latency energy success_rate qos failure
COMPARE_METRICS=(reward latency energy success_rate qos failure)

# ── Target improvement values for presentation (reference lines in TensorBoard)
# These are NOT used to alter measured metrics; they are logged as separate
# target scalars so measured-vs-target is explicit.
TARGET_LATENCY_IMPROVEMENT_PCT=23.6
TARGET_ENERGY_IMPROVEMENT_PCT=17.3
TARGET_SUCCESS_GAIN_PCT=7.0

# =============================================================================
# PATHS — usually no need to edit below this line
# =============================================================================

DRL_DIR="${SCRIPT_DIR}"
SIM_DIR="${SIM_DIR:-/opt/omnet/omnetpp-6.1/workspace2/IoV-Digital-Twin-TaskOffloading}"
SECONDARY_DT_SCRIPT="${SIM_DIR}/run_secondary_dt.sh"
REDIS_CONFIG_PATH="${DRL_DIR}/configs/redis_config.json"

DRL_LOG_DIR="${DRL_DIR}/logs/${RUN_LABEL}"
SIM_LOG_DIR="${SIM_DIR}/logs/${RUN_LABEL}"
RESULTS_SUBDIR="${DRL_DIR}/results/${RUN_LABEL}"

# Minimum seconds a healthy simulation is expected to run.
# If the sim exits faster than this, it is treated as a crash and the run is
# skipped rather than letting the DRL agent spin against an empty Redis.
SIM_MIN_RUNTIME=30

# Seconds to wait for the DRL agent to finish saving results after SIGINT.
# If it does not exit within this window it is force-killed.
DRL_SHUTDOWN_TIMEOUT=30

# Seconds to wait for the secondary DT wrapper to stop after SIGINT before
# escalating to SIGTERM and, if required, SIGKILL.
SECONDARY_SHUTDOWN_TIMEOUT=20

# =============================================================================
# IMPLEMENTATION
# =============================================================================

# Colour helpers (skip if stdout is not a terminal)
if [ -t 1 ]; then
    C_BOLD="\033[1m"; C_GREEN="\033[0;32m"; C_YELLOW="\033[0;33m"
    C_RED="\033[0;31m"; C_CYAN="\033[0;36m"; C_RESET="\033[0m"
else
    C_BOLD=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_CYAN=""; C_RESET=""
fi

_info()    { echo -e "${C_CYAN}[INFO]${C_RESET}  $*"; }
_ok()      { echo -e "${C_GREEN}[OK]${C_RESET}    $*"; }
_warn()    { echo -e "${C_YELLOW}[WARN]${C_RESET}  $*"; }
_error()   { echo -e "${C_RED}[ERROR]${C_RESET} $*"; }
_section() { echo -e "\n${C_BOLD}══════════════════════════════════════════════${C_RESET}"; \
             echo -e "${C_BOLD}  $*${C_RESET}"; \
             echo -e "${C_BOLD}══════════════════════════════════════════════${C_RESET}"; }

usage() {
    cat <<EOF
Usage:
  ./run_all_agents.sh
  ./run_all_agents.sh <agent> [agent...]
  ./run_all_agents.sh <agent,agent,...>

Agents:
  ddqn vanilla_dqn random greedy_compute min_latency least_queue greedy_distance local

Examples:
  ./run_all_agents.sh ddqn
  ./run_all_agents.sh ddqn random greedy_compute
  ./run_all_agents.sh ddqn,vanilla_dqn,greedy_distance
EOF
}

build_selected_runs() {
    FILTERED_RUNS=()

    if [ ${#SELECTED_AGENTS[@]} -eq 0 ]; then
        FILTERED_RUNS=("${RUNS[@]}")
        return 0
    fi

    local requested=()
    local raw part
    for raw in "${SELECTED_AGENTS[@]}"; do
        raw="${raw//,/ }"
        for part in ${raw}; do
            [ -n "${part}" ] && requested+=("${part}")
        done
    done

    if [ ${#requested[@]} -eq 0 ]; then
        _error "No valid agent names were provided."
        usage
        return 1
    fi

    local agent entry run_agent found
    for agent in "${requested[@]}"; do
        found=0
        for entry in "${RUNS[@]}"; do
            run_agent="${entry%%:*}"
            if [ "${run_agent}" = "${agent}" ]; then
                FILTERED_RUNS+=("${entry}")
                found=1
            fi
        done
        if [ "${found}" -eq 0 ]; then
            _error "Unknown agent: ${agent}"
            usage
            return 1
        fi
    done
}

# Active PIDs — used by the SIGINT trap for cleanup
_SIM_PID=""
_DRL_PID=""
_SECONDARY_PID=""

_cleanup() {
    echo ""
    _warn "Interrupted — cleaning up active processes..."
    [ -n "${_SIM_PID}" ] && kill "${_SIM_PID}" 2>/dev/null && _info "Killed sim PID ${_SIM_PID}"
    [ -n "${_SECONDARY_PID}" ] && kill "${_SECONDARY_PID}" 2>/dev/null && _info "Killed secondary DT PID ${_SECONDARY_PID}"
    [ -n "${_DRL_PID}" ] && kill -SIGINT "${_DRL_PID}" 2>/dev/null && wait "${_DRL_PID}" 2>/dev/null \
        && _info "DRL PID ${_DRL_PID} stopped"
    exit 1
}
trap _cleanup SIGINT SIGTERM

# ─── flush_redis ─────────────────────────────────────────────────────────────
flush_redis() {
    _info "Flushing Redis databases: ${REDIS_DBS[*]}"
    for db in "${REDIS_DBS[@]}"; do
        if redis-cli -n "${db}" FLUSHDB > /dev/null 2>&1; then
            _ok "  Flushed Redis DB ${db}"
        else
            _warn "  Could not flush Redis DB ${db} (is Redis running?)"
        fi
    done
}

# ─── update_redis_log_dir ──────────────────────────────────────────────────────
# Rewrites configs/redis_config.json so TensorBoard logs for this batch run
# are placed under output/<RUN_LABEL>.
update_redis_log_dir() {
    local new_log_dir="output/${RUN_LABEL}"
    _info "Setting redis_config system.log_dir to: ${new_log_dir}"

    if python3 - "${REDIS_CONFIG_PATH}" "${new_log_dir}" <<'PY'
import json
import sys

config_path = sys.argv[1]
new_log_dir = sys.argv[2]

with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

if "system" not in cfg or not isinstance(cfg["system"], dict):
    raise KeyError("Missing object: system")

cfg["system"]["log_dir"] = new_log_dir

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=4)
    f.write("\n")
PY
    then
        _ok "Updated ${REDIS_CONFIG_PATH}"
    else
        _error "Failed to update ${REDIS_CONFIG_PATH}"
        exit 1
    fi
}

# ─── _stop_drl_gracefully ─────────────────────────────────────────────────────
# Send SIGINT to the DRL process and wait up to DRL_SHUTDOWN_TIMEOUT seconds.
# Force-kills if it does not exit in time.
#   $1 = DRL PID
# Rewrites configs/redis_config.json so active agent_instances use the DBs
# configured in REDIS_DBS (typically provided via .env).
update_redis_instance_dbs() {
    _info "Setting redis_config active agent DBs to: ${REDIS_DBS[*]}"

    if python3 - "${REDIS_CONFIG_PATH}" "${REDIS_DBS[@]}" <<'PY'
import json
import sys

config_path = sys.argv[1]
db_values = [int(v) for v in sys.argv[2:]]

if not db_values:
    raise ValueError("No REDIS_DBS values were provided")

with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

redis_cfg = cfg.get("redis")
if not isinstance(redis_cfg, dict):
    raise KeyError("Missing object: redis")

instances = redis_cfg.get("agent_instances")
if not isinstance(instances, list):
    raise KeyError("Missing array: redis.agent_instances")

active_instances = [inst for inst in instances if inst.get("active", True)]
if len(db_values) < len(active_instances):
    raise ValueError(
        f"REDIS_DBS has {len(db_values)} entries but {len(active_instances)} active agent_instances exist"
    )

for idx, inst in enumerate(active_instances):
    inst["redis_db"] = db_values[idx]

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=4)
    f.write("\n")
PY
    then
        _ok "Updated redis agent DB mappings in ${REDIS_CONFIG_PATH}"
    else
        _error "Failed to update redis agent DB mappings in ${REDIS_CONFIG_PATH}"
        exit 1
    fi
}

_stop_drl_gracefully() {
    local pid="$1"
    _info "Sending SIGINT to DRL agent (waiting up to ${DRL_SHUTDOWN_TIMEOUT}s for clean shutdown)..."
    kill -SIGINT "${pid}" 2>/dev/null || return 0

    local elapsed=0
    while kill -0 "${pid}" 2>/dev/null && [ "${elapsed}" -lt "${DRL_SHUTDOWN_TIMEOUT}" ]; do
        sleep 1
        (( elapsed++ )) || true
    done

    if kill -0 "${pid}" 2>/dev/null; then
        _warn "DRL agent did not stop within ${DRL_SHUTDOWN_TIMEOUT}s — force killing (SIGKILL)"
        kill -9 "${pid}" 2>/dev/null || true
        sleep 1
    else
        _ok "DRL agent stopped cleanly after ${elapsed}s"
    fi
}

_agent_uses_tau() {
    local agent="$1"
    [[ "$agent" == "ddqn" ]]
}

_start_secondary_dt() {
    local agent="$1"
    local sim_cfg="$2"

    if ! _agent_uses_tau "$agent"; then
        return 0
    fi

    if [ ! -x "${SECONDARY_DT_SCRIPT}" ]; then
        _warn "Secondary DT script not executable: ${SECONDARY_DT_SCRIPT}"
        return 0
    fi

    local secondary_log="${SIM_LOG_DIR}/secondary_dt_${sim_cfg}_${agent}.log"
    local secondary_output_target="/dev/null"

    if [ "${CAPTURE_SECONDARY_DT_LOG}" -eq 1 ]; then
        secondary_output_target="${secondary_log}"
    fi

    _info "Starting secondary DT for tau-enabled run"
    if [ "${CAPTURE_SECONDARY_DT_LOG}" -eq 1 ]; then
        _info "  Secondary DT log: ${secondary_log}"
    else
        _info "  Secondary DT log: disabled (set CAPTURE_SECONDARY_DT_LOG=1 to capture)"
    fi
    (
        cd "${SIM_DIR}" || exit 1
        exec "${SECONDARY_DT_SCRIPT}" > "${secondary_output_target}" 2>&1
    ) &
    _SECONDARY_PID=$!
    _ok "Secondary DT started (PID: ${_SECONDARY_PID})"
}

_stop_secondary_dt() {
    if [ -z "${_SECONDARY_PID}" ]; then
        return 0
    fi

    if kill -0 "${_SECONDARY_PID}" 2>/dev/null; then
        _info "Stopping secondary DT (PID: ${_SECONDARY_PID}) with up to ${SECONDARY_SHUTDOWN_TIMEOUT}s for clean shutdown..."
        kill -SIGINT "${_SECONDARY_PID}" 2>/dev/null || true

        local elapsed=0
        while kill -0 "${_SECONDARY_PID}" 2>/dev/null && [ "${elapsed}" -lt "${SECONDARY_SHUTDOWN_TIMEOUT}" ]; do
            sleep 1
            (( elapsed++ )) || true
        done

        if kill -0 "${_SECONDARY_PID}" 2>/dev/null; then
            _warn "Secondary DT did not stop within ${SECONDARY_SHUTDOWN_TIMEOUT}s — sending SIGTERM"
            kill -SIGTERM "${_SECONDARY_PID}" 2>/dev/null || true
            sleep 2
        fi

        if kill -0 "${_SECONDARY_PID}" 2>/dev/null; then
            _warn "Secondary DT still running — force killing (SIGKILL)"
            kill -9 "${_SECONDARY_PID}" 2>/dev/null || true
            sleep 1
        fi

        wait "${_SECONDARY_PID}" 2>/dev/null || true
    fi
    _SECONDARY_PID=""
}

# ─── move_new_results ─────────────────────────────────────────────────────────
# Move any *.json files in results/ that are newer than the reference file
# into the per-run subdirectory.
#   $1 = path to reference file (created before the run started)
move_new_results() {
    local ref_file="$1"
    local moved=0

    while IFS= read -r -d '' f; do
        mv "$f" "${RESULTS_SUBDIR}/"
        _ok "  Result saved: results/${RUN_LABEL}/$(basename "$f")"
        (( moved++ )) || true
    done < <(find "${DRL_DIR}/results" -maxdepth 1 -name "*.json" -newer "${ref_file}" -print0 2>/dev/null)

    if [ "${moved}" -eq 0 ]; then
        _warn "  No new result files found after this run"
    fi
}

# ─── emit_tensorboard_improvement_summary ───────────────────────────────────
# Compute DDQN improvement against baseline = average(random, greedy_compute)
# from run result JSONs, then write measured and target scalars to TensorBoard.
#   $@ = result JSON paths (typically inst0 files from this batch)
emit_tensorboard_improvement_summary() {
    local result_files=("$@")
    local tb_out_dir="${DRL_DIR}/output/${RUN_LABEL}/comparison_summary"
    local summary_txt="${RESULTS_SUBDIR}/improvement_summary_ddqn_vs_baselines.txt"

    if [ ${#result_files[@]} -eq 0 ]; then
        _warn "No result files passed for improvement summary; skipping TensorBoard summary"
        return 0
    fi

    _info "Writing TensorBoard improvement summary (DDQN vs random+greedy_compute)"

    if python3 - "${tb_out_dir}" "${summary_txt}" \
        "${TARGET_LATENCY_IMPROVEMENT_PCT}" "${TARGET_ENERGY_IMPROVEMENT_PCT}" "${TARGET_SUCCESS_GAIN_PCT}" \
        "${result_files[@]}" <<'PY'
import json
import math
import os
import sys

from torch.utils.tensorboard import SummaryWriter


def flatten_metric_dict(metric_dict):
    vals = []
    if not isinstance(metric_dict, dict):
        return vals
    for arr in metric_dict.values():
        if isinstance(arr, list):
            vals.extend(x for x in arr if isinstance(x, (int, float)) and math.isfinite(x))
    return vals


def mean(xs):
    return (sum(xs) / len(xs)) if xs else float("nan")


tb_out_dir = sys.argv[1]
summary_txt = sys.argv[2]
target_latency = float(sys.argv[3])
target_energy = float(sys.argv[4])
target_success = float(sys.argv[5])
paths = sys.argv[6:]

agents = {
    "ddqn": {"lat": [], "en": [], "sr": []},
    "random": {"lat": [], "en": [], "sr": []},
    "greedy_compute": {"lat": [], "en": [], "sr": []},
}

for p in paths:
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        continue

    agent = data.get("agent")
    if agent not in agents:
        continue

    metrics = data.get("metrics", {})
    agents[agent]["lat"].extend(flatten_metric_dict(metrics.get("latencies", {})))
    agents[agent]["en"].extend(flatten_metric_dict(metrics.get("energies", {})))
    agents[agent]["sr"].extend(
        x for x in metrics.get("success_rates", [])
        if isinstance(x, (int, float)) and math.isfinite(x)
    )

missing = [a for a in agents if len(agents[a]["sr"]) == 0]
if missing:
    raise RuntimeError(f"Missing required agents/metrics for summary: {missing}")

ddqn_lat = mean(agents["ddqn"]["lat"])
ddqn_en = mean(agents["ddqn"]["en"])
ddqn_sr = mean(agents["ddqn"]["sr"])

base_lat = mean(agents["random"]["lat"] + agents["greedy_compute"]["lat"])
base_en = mean(agents["random"]["en"] + agents["greedy_compute"]["en"])
base_sr = mean(agents["random"]["sr"] + agents["greedy_compute"]["sr"])

latency_impr_pct = ((base_lat - ddqn_lat) / base_lat * 100.0) if base_lat > 0 else float("nan")
energy_impr_pct = ((base_en - ddqn_en) / base_en * 100.0) if base_en > 0 else float("nan")
success_gain_pp = ddqn_sr - base_sr
success_gain_rel_pct = ((ddqn_sr - base_sr) / base_sr * 100.0) if base_sr > 0 else float("nan")

os.makedirs(tb_out_dir, exist_ok=True)
writer = SummaryWriter(log_dir=tb_out_dir)
step = 0

# Measured values
writer.add_scalar("Improvement/Latency_Reduction_Pct_Measured", latency_impr_pct, step)
writer.add_scalar("Improvement/Energy_Reduction_Pct_Measured", energy_impr_pct, step)
writer.add_scalar("Improvement/TaskSuccess_Gain_PctPoints_Measured", success_gain_pp, step)
writer.add_scalar("Improvement/TaskSuccess_Gain_RelativePct_Measured", success_gain_rel_pct, step)

# Targets (reference lines)
writer.add_scalar("Improvement/Latency_Reduction_Pct_Target", target_latency, step)
writer.add_scalar("Improvement/Energy_Reduction_Pct_Target", target_energy, step)
writer.add_scalar("Improvement/TaskSuccess_Gain_PctPoints_Target", target_success, step)

# Raw means for transparency
writer.add_scalar("Means/DDQN_Latency", ddqn_lat, step)
writer.add_scalar("Means/Baseline_Latency", base_lat, step)
writer.add_scalar("Means/DDQN_Energy", ddqn_en, step)
writer.add_scalar("Means/Baseline_Energy", base_en, step)
writer.add_scalar("Means/DDQN_SuccessRate", ddqn_sr, step)
writer.add_scalar("Means/Baseline_SuccessRate", base_sr, step)

writer.close()

with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("DDQN vs Baseline (random + greedy_compute)\n")
    f.write("Baseline is the combined mean over random and greedy_compute samples.\n\n")
    f.write(f"Measured latency reduction (%): {latency_impr_pct:.3f}\n")
    f.write(f"Measured energy reduction (%): {energy_impr_pct:.3f}\n")
    f.write(f"Measured task success gain (percentage points): {success_gain_pp:.3f}\n")
    f.write(f"Measured task success gain (relative %): {success_gain_rel_pct:.3f}\n\n")
    f.write(f"Target latency reduction (%): {target_latency:.3f}\n")
    f.write(f"Target energy reduction (%): {target_energy:.3f}\n")
    f.write(f"Target task success gain (percentage points): {target_success:.3f}\n")

print(f"[OK] TensorBoard improvement summary written to: {tb_out_dir}")
print(f"[OK] Text summary written to: {summary_txt}")
PY
    then
        _ok "  TensorBoard improvement summary generated"
        _ok "  Text summary saved: results/${RUN_LABEL}/$(basename "${summary_txt}")"
    else
        _warn "  Could not generate TensorBoard improvement summary"
    fi
}

# ─── write_completion_summary ───────────────────────────────────────────────
# Compute the same completion metrics used by run_ddqn_healthcheck_7200.sh
# from a simulator log and save them into a per-run summary file.
#   $1 = simulation log path
#   $2 = summary output path
#   $3 = run label text (e.g., ddqn:Heuristic)
write_completion_summary() {
        local sim_log="$1"
        local summary_file="$2"
        local run_name="$3"

        if [ ! -f "${sim_log}" ]; then
                _warn "  Simulator log not found, skipping completion summary: ${sim_log}"
                return 0
        fi

        _info "Computing completion summary for ${run_name}"
        awk -v run_name="${run_name}" -v source_log="${sim_log}" '
                match($0,/LOCAL_RESULT: task=[^ ]+ .* status=([A-Z_]+)/,m) {
                    local_total++
                    if (m[1] == "COMPLETED_ON_TIME") local_success++
                }
                match($0,/REDIS_UPDATE: Task [^ ]+ status -> ([A-Z_]+) decision_type=([A-Z_]+)/,m) {
                    off_total++
                    if (m[1] == "COMPLETED_ON_TIME") off_success++
                    if (m[2] == "RSU") {
                        rsu_total++
                        if (m[1] == "COMPLETED_ON_TIME") rsu_success++
                    } else if (m[2] == "SERVICE_VEHICLE") {
                        sv_total++
                        if (m[1] == "COMPLETED_ON_TIME") sv_success++
                    }
                }
                function pct(success, total) {
                    return (total > 0) ? (100.0 * success / total) : 0.0
                }
                END {
                    all_total = local_total + off_total
                    all_success = local_success + off_success

                    printf("Completion Summary\n")
                    printf("Run: %s\n", run_name)
                    printf("Source log: %s\n\n", source_log)

                    printf("All tasks completion: %.2f%% (%d/%d)\n", pct(all_success, all_total), all_success, all_total)
                    printf("Local tasks completion: %.2f%% (%d/%d)\n", pct(local_success, local_total), local_success, local_total)
                    printf("All offloaded tasks completion: %.2f%% (%d/%d)\n", pct(off_success, off_total), off_success, off_total)
                    printf("RSU offloaded tasks completion: %.2f%% (%d/%d)\n", pct(rsu_success, rsu_total), rsu_success, rsu_total)
                    printf("SV offloaded tasks completion: %.2f%% (%d/%d)\n", pct(sv_success, sv_total), sv_success, sv_total)
                }
        ' "${sim_log}" | tee "${summary_file}" > /dev/null

        python3 - "${RESULTS_SUBDIR}" "${run_name}" "${summary_file}" <<'PY'
import glob
import json
import math
import os
import sys

TASK_TYPES = [
    "LOCAL_OBJECT_DETECTION",
    "COOPERATIVE_PERCEPTION",
    "ROUTE_OPTIMIZATION",
    "FLEET_TRAFFIC_FORECAST",
    "VOICE_COMMAND_PROCESSING",
    "SENSOR_HEALTH_CHECK",
]


def finite_values(values):
    return [
        float(v) for v in values
        if isinstance(v, (int, float)) and math.isfinite(float(v))
    ]


def pct(successes):
    values = finite_values(successes)
    return (sum(values) / len(values) * 100.0, len(values)) if values else (None, 0)


def mean(values):
    values = finite_values(values)
    return (sum(values) / len(values), len(values)) if values else (None, 0)


results_dir, run_name, summary_file = sys.argv[1:4]
agent, sim_cfg = run_name.split(":", 1)
sim_cfg_norm = sim_cfg.lower()

runs = []
for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        continue

    if data.get("agent") != agent:
        continue
    if str(data.get("offload_mode", "")).lower() != sim_cfg_norm:
        continue
    runs.append((path, data))

with open(summary_file, "a", encoding="utf-8") as out:
    out.write("\nAggregated Metrics Across Redis Instances\n")
    out.write("Source JSON files: ")
    if runs:
        out.write(", ".join(os.path.basename(p) for p, _ in runs) + "\n")
    else:
        out.write("none found\n")
        out.write("Note: weighted metric aggregation requires result JSON files from this run.\n")
        sys.exit(0)

    all_latencies = []
    total_tasks_from_json = 0
    for _, data in runs:
        metrics = data.get("metrics", {})
        total_tasks_from_json += int(metrics.get("total_tasks", 0) or 0)
        for values in metrics.get("latencies", {}).values():
            all_latencies.extend(values if isinstance(values, list) else [])

    overall_latency, overall_latency_count = mean(all_latencies)
    if overall_latency is None:
        out.write("Overall average latency: N/A\n")
    else:
        out.write(
            f"Overall average latency: {overall_latency:.6f}s "
            f"({overall_latency_count} tasks, weighted across instances)\n"
        )
    out.write(f"Total tasks in JSON metrics: {total_tasks_from_json}\n")

    out.write("\nAverage energy by task type:\n")
    for task_type in TASK_TYPES:
        values = []
        for _, data in runs:
            task_values = data.get("metrics", {}).get("energies", {}).get(task_type, [])
            values.extend(task_values if isinstance(task_values, list) else [])
        avg_energy, count = mean(values)
        if avg_energy is None:
            out.write(f"{task_type}: N/A (0 tasks)\n")
        else:
            total_energy = sum(finite_values(values))
            out.write(
                f"{task_type}: avg={avg_energy:.6f}J total={total_energy:.6f}J "
                f"({count} tasks)\n"
            )

    out.write("\nSuccess rate by task type:\n")
    has_success_metric = any(
        "task_type_successes" in data.get("metrics", {}) for _, data in runs
    )
    if not has_success_metric:
        out.write("N/A: rerun with updated main.py to generate task_type_successes in result JSON files.\n")
    else:
        for task_type in TASK_TYPES:
            values = []
            for _, data in runs:
                task_values = data.get("metrics", {}).get("task_type_successes", {}).get(task_type, [])
                values.extend(task_values if isinstance(task_values, list) else [])
            rate, count = pct(values)
            if rate is None:
                out.write(f"{task_type}: N/A (0 tasks)\n")
            else:
                successes = int(sum(finite_values(values)))
                out.write(f"{task_type}: {rate:.2f}% ({successes}/{count})\n")
PY

        _ok "  Completion summary saved: results/${RUN_LABEL}/$(basename "${summary_file}")"
}

# ─── run_one ─────────────────────────────────────────────────────────────────
# Execute a single agent + simulation pair.
#   $1 = agent name  (e.g. "ddqn")
#   $2 = sim config  (e.g. "Heuristic")
# Returns 1 if the simulation crashed too quickly (run skipped).
run_one() {
    local agent="$1"
    local sim_cfg="$2"

    _section "Run: agent=${agent}  sim=${sim_cfg}"

    # ── Log paths for this run ──────────────────────────────────────────────
    local sim_log="${SIM_LOG_DIR}/sim_output_single_agent_${sim_cfg}_${agent}.log"
    local drl_log="${DRL_LOG_DIR}/drl_output_single_agent_${sim_cfg}_${agent}.log"
    local completion_summary_file="${RESULTS_SUBDIR}/completion_summary_${sim_cfg}_${agent}.txt"
    local sim_output_target="/dev/null"

    if [ "${CAPTURE_SIM_LOG}" -eq 1 ]; then
        sim_output_target="${sim_log}"
    fi

    # ── Flush Redis ─────────────────────────────────────────────────────────
    flush_redis

    # ── Reference file for detecting new result JSON files ─────────────────
    local ref_file
    ref_file=$(mktemp)

    # ── Start simulation ────────────────────────────────────────────────────
    _info "Starting simulation  [config=${sim_cfg}, time=${SIM_TIME}]"
    if [ "${CAPTURE_SIM_LOG}" -eq 1 ]; then
        _info "  Sim log: ${sim_log}"
    else
        _info "  Sim log: disabled (set CAPTURE_SIM_LOG=1 to capture)"
    fi
    local sim_start_epoch
    sim_start_epoch=$(date +%s)
    (
        cd "${SIM_DIR}" || exit 1
        exec ./run_simulation.sh -u Cmdenv -c "${sim_cfg}" --sim-time-limit="${SIM_TIME}" \
            > "${sim_output_target}" 2>&1
    ) &
    _SIM_PID=$!
    _ok "Simulation started (PID: ${_SIM_PID})"

    # ── Start secondary DT (tau-enabled agents only) ──────────────────────
    _start_secondary_dt "${agent}" "${sim_cfg}"

    # ── Wait before launching DRL ───────────────────────────────────────────
    _info "Waiting ${DRL_START_DELAY}s for simulator to initialise Redis..."
    sleep "${DRL_START_DELAY}"

    # ── Crash-guard: check if the sim already exited during the startup delay
    if ! kill -0 "${_SIM_PID}" 2>/dev/null; then
        local sim_dur=$(( $(date +%s) - sim_start_epoch ))
        _error "Simulation exited after ${sim_dur}s (before DRL was started) — likely a crash."
        if [ "${CAPTURE_SIM_LOG}" -eq 1 ]; then
            _error "Check sim log for errors: ${sim_log}"
        else
            _error "Re-run with CAPTURE_SIM_LOG=1 to capture simulator output for debugging."
        fi
        _warn "Skipping DRL launch for this run."
        _stop_secondary_dt
        _SIM_PID=""
        rm -f "${ref_file}"
        return 1
    fi

    # ── Start DRL agent ─────────────────────────────────────────────────────
    _info "Starting DRL agent  [agent=${agent}]"
    _info "  DRL log: ${drl_log}"
    local drl_extra_args=()
    if [ "${agent}" = "ddqn" ] || [ "${agent}" = "ddqn_no_tau" ] || [ "${agent}" = "vanilla_dqn" ]; then
        drl_extra_args+=(--resume_training)
        _info "  Continuous training enabled: existing ${agent} model checkpoints will be reused if found"
    fi

    (
        cd "${DRL_DIR}" || exit 1
        exec python3 -u main.py --env redis --agent "${agent}" "${drl_extra_args[@]}" \
            > "${drl_log}" 2>&1
    ) &
    _DRL_PID=$!
    _ok "DRL agent started (PID: ${_DRL_PID})"

    # ── Wait for simulation to finish, then stop DRL gracefully ────────────
    _info "Waiting for simulation to finish..."
    local sim_exit_code=0
    wait "${_SIM_PID}" || sim_exit_code=$?
    local sim_dur=$(( $(date +%s) - sim_start_epoch ))
    _SIM_PID=""

    if [ "${sim_exit_code}" -ne 0 ]; then
        _warn "Simulation exited with code ${sim_exit_code} after ${sim_dur}s"
        if [ "${sim_dur}" -lt "${SIM_MIN_RUNTIME}" ]; then
            _error "Simulation ran for only ${sim_dur}s — this looks like a crash!"
            _error "Check sim log: ${sim_log}"
        fi
    else
        _ok "Simulation finished normally after ${sim_dur}s"
    fi

    _stop_secondary_dt

    # Stop DRL whether sim succeeded or crashed (results may still be partial)
    _stop_drl_gracefully "${_DRL_PID}"
    _DRL_PID=""

    # ── Collect result files ────────────────────────────────────────────────
    move_new_results "${ref_file}"
    rm -f "${ref_file}"

    # ── Completion summary (same metrics as healthcheck script) ────────────
    write_completion_summary "${sim_log}" "${completion_summary_file}" "${agent}:${sim_cfg}"

    if [ "${sim_exit_code}" -ne 0 ] && [ "${sim_dur}" -lt "${SIM_MIN_RUNTIME}" ]; then
        _warn "Run ${agent}/${sim_cfg} likely failed due to sim crash — results may be empty."
        return 1
    fi

    _ok "Run complete: ${agent}/${sim_cfg} (sim ran ${sim_dur}s)"
}

# =============================================================================
# MAIN
# =============================================================================

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

build_selected_runs || exit 1

_section "run_all_agents.sh — RUN_LABEL="
update_redis_log_dir
update_redis_instance_dbs
echo "  Runs planned : ${#FILTERED_RUNS[@]}"
echo "  Agents       : ${SELECTED_AGENTS[*]:-all}"
echo "  Sim time     : ${SIM_TIME}"
echo "  DRL log dir  : ${DRL_LOG_DIR}"
echo "  Sim log dir  : ${SIM_LOG_DIR}"
echo "  Results dir  : ${RESULTS_SUBDIR}"
echo "  Compare out  : ${COMPARE_OUTPUT}"
echo ""

# ── Create output directories ─────────────────────────────────────────────────
mkdir -p "${DRL_LOG_DIR}"
mkdir -p "${SIM_LOG_DIR}"
mkdir -p "${RESULTS_SUBDIR}"
mkdir -p "${DRL_DIR}/output"

BATCH_START=$(date +%s)
FAILED_RUNS=()
COMPLETED_RUNS=()

# ── Execute each run ──────────────────────────────────────────────────────────
for entry in "${FILTERED_RUNS[@]}"; do
    agent="${entry%%:*}"
    sim_cfg="${entry##*:}"

    if run_one "${agent}" "${sim_cfg}"; then
        COMPLETED_RUNS+=("${agent}:${sim_cfg}")
    else
        _error "Run failed: ${agent}:${sim_cfg} — continuing with next run"
        FAILED_RUNS+=("${agent}:${sim_cfg}")
        # Ensure processes are dead before next run
        [ -n "${_SIM_PID}" ] && kill "${_SIM_PID}" 2>/dev/null; _SIM_PID=""
        [ -n "${_SECONDARY_PID}" ] && kill "${_SECONDARY_PID}" 2>/dev/null; _SECONDARY_PID=""
        [ -n "${_DRL_PID}" ] && kill "${_DRL_PID}" 2>/dev/null; _DRL_PID=""
    fi

    echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────
BATCH_END=$(date +%s)
ELAPSED=$(( BATCH_END - BATCH_START ))
ELAPSED_FMT="$(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m $(( ELAPSED % 60 ))s"

_section "Batch complete — ${ELAPSED_FMT} elapsed"
_ok  "Completed runs (${#COMPLETED_RUNS[@]}): ${COMPLETED_RUNS[*]:-none}"
[ ${#FAILED_RUNS[@]} -gt 0 ] && _warn "Failed runs    (${#FAILED_RUNS[@]}): ${FAILED_RUNS[*]}"

# ── Build compare.py arguments ────────────────────────────────────────────────
_info "Collecting inst0 result files from ${RESULTS_SUBDIR}..."
mapfile -t ALL_INST0 < <(find "${RESULTS_SUBDIR}" -name "*_inst0.json" | sort)

if [ ${#ALL_INST0[@]} -eq 0 ]; then
    _warn "No inst0 result files found in ${RESULTS_SUBDIR} — skipping compare.py"
    exit 0
fi

_info "Found ${#ALL_INST0[@]} result file(s) for comparison:"
for f in "${ALL_INST0[@]}"; do
    echo "    results/${RUN_LABEL}/$(basename "$f")"
done

# Build relative paths for compare.py (run from DRL_DIR)
RELATIVE_RESULTS=()
for f in "${ALL_INST0[@]}"; do
    RELATIVE_RESULTS+=("results/${RUN_LABEL}/$(basename "$f")")
done

# ── Run compare.py ────────────────────────────────────────────────────────────
_section "Running compare.py"
_info "Output: ${COMPARE_OUTPUT}"

(
    cd "${DRL_DIR}" || exit 1
    python3 compare.py \
        --runs "${RELATIVE_RESULTS[@]}" \
        --metrics "${COMPARE_METRICS[@]}" \
        --output "${COMPARE_OUTPUT}"
)

if [ $? -eq 0 ]; then
    _ok "Comparison plot saved to: ${DRL_DIR}/${COMPARE_OUTPUT}"
else
    _error "compare.py failed — check logs above"
fi

# ── Emit TensorBoard summary for DDQN vs baselines ─────────────────────────
emit_tensorboard_improvement_summary "${ALL_INST0[@]}"
