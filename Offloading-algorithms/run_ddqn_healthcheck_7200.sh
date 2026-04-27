#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

DRL_DIR="${SCRIPT_DIR}"
SIM_DIR="${SIM_DIR:-/opt/omnet/omnetpp-6.1/workspace-mihi/IoV-Digital-Twin-TaskOffloading}"
REDIS_CONFIG_PATH="${DRL_DIR}/configs/redis_config.json"

RUN_LABEL="${RUN_LABEL:-ddqn_7200_healthcheck_$(date +%Y%m%d_%H%M%S)}"
SIM_TIME="7200s"
DRL_START_DELAY=2
REDIS_DBS=(0 1 2)
DRL_SHUTDOWN_TIMEOUT=30

DRL_LOG_DIR="${DRL_DIR}/logs/${RUN_LABEL}"
SIM_LOG_DIR="${SIM_DIR}/logs/${RUN_LABEL}"
RESULTS_SUBDIR="${DRL_DIR}/results/${RUN_LABEL}"
TB_LOG_DIR="output/${RUN_LABEL}"

mkdir -p "${DRL_LOG_DIR}" "${SIM_LOG_DIR}" "${RESULTS_SUBDIR}"

cleanup() {
  if [[ -n "${SIM_PID:-}" ]] && kill -0 "${SIM_PID}" 2>/dev/null; then
    kill "${SIM_PID}" 2>/dev/null || true
  fi
  if [[ -n "${DRL_PID:-}" ]] && kill -0 "${DRL_PID}" 2>/dev/null; then
    kill -SIGINT "${DRL_PID}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

echo "[INFO] RUN_LABEL=${RUN_LABEL}"
echo "[INFO] TensorBoard log dir -> ${TB_LOG_DIR}"

python3 - "${REDIS_CONFIG_PATH}" "${TB_LOG_DIR}" <<'PY'
import json
import sys
cfg_path, new_dir = sys.argv[1], sys.argv[2]
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)
cfg.setdefault('system', {})['log_dir'] = new_dir
with open(cfg_path, 'w', encoding='utf-8') as f:
    json.dump(cfg, f, indent=4)
    f.write('\n')
print(f"[OK] Updated {cfg_path}: system.log_dir={new_dir}")
PY

for db in "${REDIS_DBS[@]}"; do
  redis-cli -n "${db}" FLUSHDB >/dev/null
  echo "[OK] Flushed Redis DB ${db}"
done

SIM_LOG="${SIM_LOG_DIR}/sim_ddqn_7200.log"
DRL_LOG="${DRL_LOG_DIR}/drl_ddqn_7200.log"
REF_FILE="$(mktemp)"

echo "[INFO] Starting simulator -> ${SIM_LOG}"
(
  cd "${SIM_DIR}"
  exec ./run_simulation.sh -u Cmdenv -c Heuristic --sim-time-limit="${SIM_TIME}" > "${SIM_LOG}" 2>&1
) &
SIM_PID=$!
echo "[OK] Simulator PID=${SIM_PID}"

sleep "${DRL_START_DELAY}"

if ! kill -0 "${SIM_PID}" 2>/dev/null; then
  echo "[ERROR] Simulator exited before DRL start. Check ${SIM_LOG}"
  rm -f "${REF_FILE}"
  exit 1
fi

# Look for an existing model checkpoint to resume training
if ls "${DRL_DIR}"/models/ddqn_rsu*.pth >/dev/null 2>&1; then
  echo "[INFO] Found existing model checkpoints. Resuming training independently for each RSU."
  DRL_LOAD_FLAG="--resume_training"
else
  echo "[INFO] No existing model checkpoint found. Starting fresh."
  DRL_LOAD_FLAG=""
fi

echo "[INFO] Starting DDQN agent -> ${DRL_LOG}"
(
  cd "${DRL_DIR}"
  exec python3 -u main.py --env redis --agent ddqn ${DRL_LOAD_FLAG} > "${DRL_LOG}" 2>&1
) &
DRL_PID=$!
echo "[OK] DRL PID=${DRL_PID}"

echo "[INFO] Waiting for simulator to finish..."
SIM_EXIT=0
wait "${SIM_PID}" || SIM_EXIT=$?

if [[ ${SIM_EXIT} -ne 0 ]]; then
  echo "[WARN] Simulator exited with code ${SIM_EXIT}. Check ${SIM_LOG}"
else
  echo "[OK] Simulator finished cleanly"
fi

echo "[INFO] Stopping DDQN agent gracefully"
kill -SIGINT "${DRL_PID}" 2>/dev/null || true
ELAPSED=0
while kill -0 "${DRL_PID}" 2>/dev/null && [[ ${ELAPSED} -lt ${DRL_SHUTDOWN_TIMEOUT} ]]; do
  sleep 1
  ELAPSED=$((ELAPSED+1))
done
if kill -0 "${DRL_PID}" 2>/dev/null; then
  echo "[WARN] DDQN did not stop in time, force kill"
  kill -9 "${DRL_PID}" 2>/dev/null || true
else
  echo "[OK] DDQN stopped cleanly"
fi

while IFS= read -r -d '' f; do
  mv "$f" "${RESULTS_SUBDIR}/"
  echo "[OK] Moved result -> ${RESULTS_SUBDIR}/$(basename "$f")"
done < <(find "${DRL_DIR}/results" -maxdepth 1 -name "ddqn_*.json" -newer "${REF_FILE}" -print0)
rm -f "${REF_FILE}"

SUMMARY_FILE="${RESULTS_SUBDIR}/completion_summary.txt"
if [[ -f "${SIM_LOG}" ]]; then
  echo "[INFO] Computing completion summary from simulator log..."
  awk -v run_label="${RUN_LABEL}" -v source_log="${SIM_LOG}" '
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
      printf("Run label: %s\n", run_label)
      printf("Source log: %s\n\n", source_log)

      printf("All tasks completion: %.2f%% (%d/%d)\n", pct(all_success, all_total), all_success, all_total)
      printf("Local tasks completion: %.2f%% (%d/%d)\n", pct(local_success, local_total), local_success, local_total)
      printf("All offloaded tasks completion: %.2f%% (%d/%d)\n", pct(off_success, off_total), off_success, off_total)
      printf("RSU offloaded tasks completion: %.2f%% (%d/%d)\n", pct(rsu_success, rsu_total), rsu_success, rsu_total)
      printf("SV offloaded tasks completion: %.2f%% (%d/%d)\n", pct(sv_success, sv_total), sv_success, sv_total)
    }
  ' "${SIM_LOG}" | tee "${SUMMARY_FILE}"
  echo "[OK] Saved completion summary -> ${SUMMARY_FILE}"
else
  echo "[WARN] Simulator log not found, skipping completion summary"
fi

echo "[DONE] DDQN healthcheck run complete"
echo "[INFO] TensorBoard folder: ${DRL_DIR}/${TB_LOG_DIR}"
echo "[INFO] Results folder    : ${RESULTS_SUBDIR}"
echo "[INFO] Sim log           : ${SIM_LOG}"
echo "[INFO] DRL log           : ${DRL_LOG}"
