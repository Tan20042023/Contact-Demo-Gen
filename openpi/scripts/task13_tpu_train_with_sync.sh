#!/usr/bin/env bash
# TPU-only training launcher. Do not invoke until the user explicitly approves
# the selected run. It uses local Orbax checkpointing plus the GCS sync daemon.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <registered-task13-tpu-config-name>" >&2
  exit 2
fi

: "${TASK13_TPU_WORKDIR:=$HOME/dexjoco-task13-tpu}"
: "${TASK13_TPU_PROJECT_VENV:=$HOME/.venvs/openpi-task13-tpu-py311}"
: "${TASK13_TPU_INPUT_ROOT:=$HOME/task13_input_assets}"
: "${TASK13_TPU_LOCAL_RUNS_ROOT:=$HOME/task13_local_runs}"
: "${TASK13_TPU_GCS_RUNS_ROOT:=gs://use1/user/tanjunhao/task13_tpu_feasibility/v1/runs}"

config_name="$1"
export TASK13_TPU_INPUT_ROOT TASK13_TPU_LOCAL_RUNS_ROOT TASK13_TPU_GCS_RUNS_ROOT
export TASK13_TPU_FSDP_DEVICES="${TASK13_TPU_FSDP_DEVICES:-4}"

read -r local_checkpoint_dir gcs_checkpoint_dir < <(
  cd "${TASK13_TPU_WORKDIR}/openpi"
  "${TASK13_TPU_PROJECT_VENV}/bin/python" - "${config_name}" <<'PY'
import sys
from pathlib import Path
from openpi.training.config import get_config

config = get_config(sys.argv[1])
local = Path(config.checkpoint_dir)
gcs = (Path(config.checkpoint_base_dir).name, config.name, config.exp_name)
gcs_root = __import__("os").environ["TASK13_TPU_GCS_RUNS_ROOT"].rstrip("/")
print(str(local), f"{gcs_root}/{gcs[0]}/{gcs[1]}/{gcs[2]}")
PY
)

export TASK13_TPU_LOCAL_CHECKPOINT_DIR="${local_checkpoint_dir}"
export TASK13_TPU_GCS_CHECKPOINT_DIR="${gcs_checkpoint_dir}"
log_dir="${TASK13_TPU_LOCAL_RUNS_ROOT}/sync_logs"
mkdir -p "${log_dir}"
"${TASK13_TPU_WORKDIR}/openpi/scripts/task13_tpu_checkpoint_sync_daemon.sh" \
  >"${log_dir}/${config_name}.sync.log" 2>&1 &
sync_pid=$!

finish() {
  kill "${sync_pid}" 2>/dev/null || true
  wait "${sync_pid}" 2>/dev/null || true
  TASK13_TPU_SYNC_ONCE=1 "${TASK13_TPU_WORKDIR}/openpi/scripts/task13_tpu_checkpoint_sync_daemon.sh"
}
trap finish EXIT INT TERM

cd "${TASK13_TPU_WORKDIR}/openpi"
"${TASK13_TPU_PROJECT_VENV}/bin/python" scripts/train.py "${config_name}"
