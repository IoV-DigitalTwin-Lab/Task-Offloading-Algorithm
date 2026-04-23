#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRL_DIR="${SCRIPT_DIR}"
SIM_DIR="/opt/omnet/omnetpp-6.1/workspace/IoV-Digital-Twin-TaskOffloading"
REDIS_CONFIG_PATH="${DRL_DIR}/configs/redis_config.json"

RUN_LABEL="ddqn_7200_healthcheck_$(date +%Y%m%d_%H%M%S)"
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

echo "[DONE] DDQN healthcheck run complete"
echo "[INFO] TensorBoard folder: ${DRL_DIR}/${TB_LOG_DIR}"
echo "[INFO] Results folder    : ${RESULTS_SUBDIR}"
echo "[INFO] Sim log           : ${SIM_LOG}"
echo "[INFO] DRL log           : ${DRL_LOG}"
