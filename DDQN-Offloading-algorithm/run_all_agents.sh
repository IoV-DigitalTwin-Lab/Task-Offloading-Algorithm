#!/bin/bash
# =============================================================================
# run_all_agents.sh — Run all agent/simulation combinations sequentially,
#                     collect results per run, and produce a comparison plot.
#
# Usage:
#   ./run_all_agents.sh
#
# Edit the USER CONFIGURATION section below before running.
# =============================================================================

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
RUN_LABEL="test2_all_heuristic"

# Simulation time limit (passed to run_simulation.sh as --sim-time-limit)
SIM_TIME="7200s"

# Seconds to wait after starting the simulation before launching the DRL agent
# (gives the simulator time to connect to Redis and write initial state)
DRL_START_DELAY=2

# ── Agents to run ────────────────────────────────────────────────────────────
# Format: "agent_name:SimulationConfig"
# Comment out any line to skip that run.
#
# agent_name must be one of:
#   ddqn | vanilla_dqn | random | greedy_compute |
#   min_latency | least_queue | greedy_distance | local
#
# SimulationConfig must match a [Config ...] section in omnetpp.ini:
#   Heuristic | AllOffload | AllLocal
RUNS=(
    "ddqn:Heuristic"
    "vanilla_dqn:Heuristic"
    "random:Heuristic"
    "greedy_compute:Heuristic"
    "min_latency:Heuristic"
    "least_queue:Heuristic"
    "greedy_distance:Heuristic"
    "local:AllLocal"
)

# ── Redis databases to flush before each run ─────────────────────────────────
# Must match the "redis_db" values of active=true instances in
# configs/redis_config.json  (currently: db 0, db 1)
REDIS_DBS=(0 1)

# ── compare.py settings ──────────────────────────────────────────────────────
# Output image path (relative to the DRL project directory)
COMPARE_OUTPUT="output/comparison_${RUN_LABEL}.png"

# Metrics to include in the comparison plot
# Available: reward latency energy success_rate qos failure
COMPARE_METRICS=(reward latency energy success_rate qos failure)

# =============================================================================
# PATHS — usually no need to edit below this line
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRL_DIR="${SCRIPT_DIR}"
SIM_DIR="/opt/omnet/omnetpp-6.1/workspace2/IoV-Digital-Twin-TaskOffloading"
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

# Active PIDs — used by the SIGINT trap for cleanup
_SIM_PID=""
_DRL_PID=""

_cleanup() {
    echo ""
    _warn "Interrupted — cleaning up active processes..."
    [ -n "${_SIM_PID}" ] && kill "${_SIM_PID}" 2>/dev/null && _info "Killed sim PID ${_SIM_PID}"
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

    # ── Flush Redis ─────────────────────────────────────────────────────────
    flush_redis

    # ── Reference file for detecting new result JSON files ─────────────────
    local ref_file
    ref_file=$(mktemp)

    # ── Start simulation ────────────────────────────────────────────────────
    _info "Starting simulation  [config=${sim_cfg}, time=${SIM_TIME}]"
    _info "  Sim log: ${sim_log}"
    local sim_start_epoch
    sim_start_epoch=$(date +%s)
    (
        cd "${SIM_DIR}" || exit 1
        exec ./run_simulation.sh -u Cmdenv -c "${sim_cfg}" --sim-time-limit="${SIM_TIME}" \
            > "${sim_log}" 2>&1
    ) &
    _SIM_PID=$!
    _ok "Simulation started (PID: ${_SIM_PID})"

    # ── Wait before launching DRL ───────────────────────────────────────────
    _info "Waiting ${DRL_START_DELAY}s for simulator to initialise Redis..."
    sleep "${DRL_START_DELAY}"

    # ── Crash-guard: check if the sim already exited during the startup delay
    if ! kill -0 "${_SIM_PID}" 2>/dev/null; then
        local sim_dur=$(( $(date +%s) - sim_start_epoch ))
        _error "Simulation exited after ${sim_dur}s (before DRL was started) — likely a crash."
        _error "Check sim log for errors: ${sim_log}"
        _warn "Skipping DRL launch for this run."
        _SIM_PID=""
        rm -f "${ref_file}"
        return 1
    fi

    # ── Start DRL agent ─────────────────────────────────────────────────────
    _info "Starting DRL agent  [agent=${agent}]"
    _info "  DRL log: ${drl_log}"
    (
        cd "${DRL_DIR}" || exit 1
        exec python3 -u main.py --env redis --agent "${agent}" \
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

    # Stop DRL whether sim succeeded or crashed (results may still be partial)
    _stop_drl_gracefully "${_DRL_PID}"
    _DRL_PID=""

    # ── Collect result files ────────────────────────────────────────────────
    move_new_results "${ref_file}"
    rm -f "${ref_file}"

    if [ "${sim_exit_code}" -ne 0 ] && [ "${sim_dur}" -lt "${SIM_MIN_RUNTIME}" ]; then
        _warn "Run ${agent}/${sim_cfg} likely failed due to sim crash — results may be empty."
        return 1
    fi

    _ok "Run complete: ${agent}/${sim_cfg} (sim ran ${sim_dur}s)"
}

# =============================================================================
# MAIN
# =============================================================================

_section "run_all_agents.sh — RUN_LABEL=${RUN_LABEL}"
update_redis_log_dir
echo "  Runs planned : ${#RUNS[@]}"
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
for entry in "${RUNS[@]}"; do
    agent="${entry%%:*}"
    sim_cfg="${entry##*:}"

    if run_one "${agent}" "${sim_cfg}"; then
        COMPLETED_RUNS+=("${agent}:${sim_cfg}")
    else
        _error "Run failed: ${agent}:${sim_cfg} — continuing with next run"
        FAILED_RUNS+=("${agent}:${sim_cfg}")
        # Ensure processes are dead before next run
        [ -n "${_SIM_PID}" ] && kill "${_SIM_PID}" 2>/dev/null; _SIM_PID=""
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
