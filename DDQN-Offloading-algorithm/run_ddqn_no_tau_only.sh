#!/bin/bash
# =============================================================================
# run_ddqn_no_tau_only.sh — Flush Redis and start only the ddqn_no_tau DRL.
#
# This script does NOT start the simulator. Start the simulator separately
# after this DRL process is running.
# =============================================================================

set -euo pipefail

RUN_LABEL="${RUN_LABEL:-ddqn_no_tau_only}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REDIS_CONFIG_PATH="${SCRIPT_DIR}/configs/redis_config.json"
DRL_LOG_DIR="${SCRIPT_DIR}/logs/${RUN_LABEL}"

# Redis DBs to flush before starting the agent.
# Defaults match the active instances in configs/redis_config.json.
REDIS_DBS=()

if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    . "${SCRIPT_DIR}/.env"
    set +a
fi

if ! declare -p REDIS_DBS >/dev/null 2>&1 || [ ${#REDIS_DBS[@]} -eq 0 ]; then
    REDIS_DBS=(0 1 2)
fi

_info() { echo "[INFO]  $*"; }
_ok()   { echo "[OK]    $*"; }
_warn() { echo "[WARN]  $*"; }

flush_redis() {
    _info "Flushing Redis databases: ${REDIS_DBS[*]}"
    for db in "${REDIS_DBS[@]}"; do
        if redis-cli -n "${db}" FLUSHDB > /dev/null 2>&1; then
            _ok "Flushed Redis DB ${db}"
        else
            _warn "Could not flush Redis DB ${db} (is Redis running?)"
        fi
    done
}

update_redis_log_dir() {
    local new_log_dir="output/${RUN_LABEL}"
    _info "Setting redis_config system.log_dir to: ${new_log_dir}"

    python3 - "${REDIS_CONFIG_PATH}" "${new_log_dir}" <<'PY'
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

    _ok "Updated ${REDIS_CONFIG_PATH}"
}

update_redis_instance_dbs() {
    _info "Setting redis_config active agent DBs to: ${REDIS_DBS[*]}"

    python3 - "${REDIS_CONFIG_PATH}" "${REDIS_DBS[@]}" <<'PY'
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

    _ok "Updated redis agent DB mappings in ${REDIS_CONFIG_PATH}"
}

cleanup() {
    if [ -n "${DRL_PID:-}" ] && kill -0 "${DRL_PID}" 2>/dev/null; then
        _warn "Stopping ddqn_no_tau DRL (PID: ${DRL_PID})"
        kill -SIGINT "${DRL_PID}" 2>/dev/null || true
        wait "${DRL_PID}" 2>/dev/null || true
    fi
}

trap cleanup INT TERM

update_redis_log_dir
update_redis_instance_dbs
flush_redis

mkdir -p "${DRL_LOG_DIR}"

DRL_LOG="${DRL_LOG_DIR}/ddqn_no_tau_only.log"
_info "Starting ddqn_no_tau DRL only"
_info "  DRL log: ${DRL_LOG}"

(
    cd "${SCRIPT_DIR}" || exit 1
    exec python3 -u main.py --env redis --agent ddqn_no_tau --resume_training > "${DRL_LOG}" 2>&1
) &

DRL_PID=$!
_ok "ddqn_no_tau DRL started (PID: ${DRL_PID})"
_info "Start the simulator separately in another terminal when ready."

wait "${DRL_PID}"