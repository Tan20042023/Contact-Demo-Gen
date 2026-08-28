#!/usr/bin/env bash
# Run from any authenticated gcloud control host (not from a TPU worker).
# It first injects the SSH key on all workers, copies the immutable bootstrap
# script concurrently, then bootstraps all workers concurrently.
set -euo pipefail

: "${TASK13_TPU_RUN_ID:?Set a unique TPU side-track run ID}"
: "${TASK13_TPU_CODE_URI:?Set immutable GCS source archive URI}"
: "${TASK13_TPU_CODE_SHA256:?Set source archive SHA-256}"
: "${TASK13_TPU_INPUT_URI:=gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/input_assets}"
: "${TASK13_TPU_INPUT_BYTES:=30696986145}"

TPU_NAME="${TASK13_TPU_NAME:-tanjunhao-tpu}"
SSH_USER="${TASK13_TPU_SSH_USER:-tanjunhao}"
PROJECT="${TASK13_TPU_PROJECT:-whyu01}"
ZONE="${TASK13_TPU_ZONE:-us-east1-d}"
WORKERS="${TASK13_TPU_WORKERS:-all}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKER_SCRIPT="${SCRIPT_DIR}/task13_tpu_v6e16_bootstrap_worker.sh"
LOG_DIR="${TASK13_TPU_CONTROLLER_LOG_DIR:-${PWD}/task13_tpu_${TASK13_TPU_RUN_ID}_bootstrap_logs}"

[[ -x "${WORKER_SCRIPT}" ]] || { echo "CONTROLLER_FAIL: worker bootstrap not executable: ${WORKER_SCRIPT}" >&2; exit 2; }
mkdir -p "${LOG_DIR}"

read -r state health < <(
  gcloud compute tpus tpu-vm describe "${TPU_NAME}" --project="${PROJECT}" --zone="${ZONE}" \
    --format='value(state,health)'
)
[[ "${state}" == "READY" && "${health}" == "HEALTHY" ]] || {
  echo "CONTROLLER_WAIT: TPU state=${state} health=${health}" >&2
  exit 3
}

# gcloud requires an agent-held key for concurrent all-worker operations.
ssh-add "${HOME}/.ssh/google_compute_engine" >/dev/null 2>&1 || {
  echo "CONTROLLER_FAIL: ssh-add ~/.ssh/google_compute_engine before --worker=all" >&2; exit 4;
}

common=(--project="${PROJECT}" --zone="${ZONE}" --worker="${WORKERS}")
gcloud compute tpus tpu-vm ssh "${SSH_USER}@${TPU_NAME}" "${common[@]}" --command='true' --output-directory="${LOG_DIR}/first_connect"
gcloud compute tpus tpu-vm scp "${WORKER_SCRIPT}" "${SSH_USER}@${TPU_NAME}:~/task13_tpu_v6e16_bootstrap_worker.sh" "${common[@]}"

remote_env=(
  "TASK13_TPU_RUN_ID=${TASK13_TPU_RUN_ID}"
  "TASK13_TPU_CODE_URI=${TASK13_TPU_CODE_URI}"
  "TASK13_TPU_CODE_SHA256=${TASK13_TPU_CODE_SHA256}"
  "TASK13_TPU_INPUT_URI=${TASK13_TPU_INPUT_URI}"
  "TASK13_TPU_INPUT_BYTES=${TASK13_TPU_INPUT_BYTES}"
)
remote_command="chmod 700 ~/task13_tpu_v6e16_bootstrap_worker.sh && env ${remote_env[*]} ~/task13_tpu_v6e16_bootstrap_worker.sh"
gcloud compute tpus tpu-vm ssh "${SSH_USER}@${TPU_NAME}" "${common[@]}" --command="${remote_command}" --output-directory="${LOG_DIR}/bootstrap"

echo "CONTROLLER_PASS: all workers bootstrapped; logs=${LOG_DIR}"
