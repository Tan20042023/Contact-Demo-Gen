#!/usr/bin/env bash
# Launch one isolated Task13 TPU cell only after bootstrap has completed on all
# v6e-16 workers.  It never touches GPU/A100 paths or shared output prefixes.
set -euo pipefail

: "${TASK13_TPU_RUN_ID:?Set a unique run ID}"
: "${TASK13_TPU_CONFIG:?Set a registered Task13 TPU config name}"
: "${TASK13_TPU_GCS_RUN_URI:?Set a dedicated gs:// output prefix for this cell}"
: "${TASK13_TPU_SOURCE_SHA256:?Set the immutable source SHA-256}"

TPU_NAME="${TASK13_TPU_NAME:-tanjunhao-tpu}"
PROJECT="${TASK13_TPU_PROJECT:-whyu01}"
ZONE="${TASK13_TPU_ZONE:-us-east1-d}"
WORKERS="${TASK13_TPU_WORKERS:-all}"
WORK_ROOT="${TASK13_TPU_WORK_ROOT:-\$HOME/task13_v6e16}"
RUN_URI="${TASK13_TPU_GCS_RUN_URI%/}"
REPO_REL="task13_v6e16/repo-${TASK13_TPU_SOURCE_SHA256:0:12}/openpi"
VENV_REL="task13_v6e16/venv"
INPUT_REL="task13_v6e16/input_assets"

read -r state health < <(
  gcloud compute tpus tpu-vm describe "$TPU_NAME" --project="$PROJECT" --zone="$ZONE" --format='value(state,health)'
)
[[ "$state" == READY && "$health" == HEALTHY ]] || { echo "LAUNCH_WAIT: state=$state health=$health" >&2; exit 3; }

if gcloud storage ls "${RUN_URI}/LATEST.json" >/dev/null 2>&1; then
  [[ "${TASK13_TPU_RESUME:-0}" == 1 ]] || { echo "LAUNCH_FAIL: committed output exists; set TASK13_TPU_RESUME=1" >&2; exit 4; }
else
  [[ "${TASK13_TPU_RESUME:-0}" != 1 ]] || { echo "LAUNCH_FAIL: resume requested but LATEST.json is absent" >&2; exit 4; }
fi

ssh-add "${HOME}/.ssh/google_compute_engine" >/dev/null 2>&1 || true
common=(--project="$PROJECT" --zone="$ZONE" --worker="$WORKERS")
gcloud compute tpus tpu-vm ssh "$TPU_NAME" "${common[@]}" --command='test -f "$HOME/task13_v6e16/bootstrap"/*/READY.json && test -x "$HOME/task13_v6e16/venv/bin/python"'

preflight="set -euo pipefail; export PYTHONPATH=\"\$HOME/${REPO_REL}/src\"; export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TASK13_TPU_INPUT_ROOT=\"\$HOME/${INPUT_REL}\" TASK13_TPU_LOCAL_RUNS_ROOT=\"\$HOME/task13_v6e16/runs/${TASK13_TPU_RUN_ID}\" TASK13_TPU_FSDP_DEVICES=4 TASK13_TPU_NUM_WORKERS=0; cd \"\$HOME/${REPO_REL}\"; \"\$HOME/${VENV_REL}/bin/python\" -c \"import jax; jax.distributed.initialize(); from openpi.training.config import get_config; from openpi.training.data_loader import create_torch_dataset; c=get_config('${TASK13_TPU_CONFIG}'); d=c.data.create(c.assets_dirs,c.model); ds=create_torch_dataset(d,c.model); assert jax.process_count()==4 and jax.device_count()==16; print('LAUNCH_PREFLIGHT_PASS',jax.process_index(),len(ds))\""
gcloud compute tpus tpu-vm ssh "$TPU_NAME" "${common[@]}" --command="$preflight"

resume_arg=""
[[ "${TASK13_TPU_RESUME:-0}" == 1 ]] && resume_arg="--resume"
remote="set -euo pipefail; export PYTHONPATH=\"\$HOME/${REPO_REL}/src\"; export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1; export TASK13_TPU_MULTIHOST=1 TASK13_TPU_INPUT_ROOT=\"\$HOME/${INPUT_REL}\" TASK13_TPU_LOCAL_RUNS_ROOT=\"\$HOME/task13_v6e16/runs/${TASK13_TPU_RUN_ID}\" TASK13_TPU_FSDP_DEVICES=4 TASK13_TPU_NUM_WORKERS=0 TASK13_TPU_GCS_RUN_URI='${RUN_URI}' TASK13_TPU_SOURCE_SHA256='${TASK13_TPU_SOURCE_SHA256}'; cd \"\$HOME/${REPO_REL}\"; mkdir -p \"\$HOME/task13_v6e16/logs/${TASK13_TPU_RUN_ID}\"; exec nohup \"\$HOME/${VENV_REL}/bin/python\" scripts/train.py '${TASK13_TPU_CONFIG}' ${resume_arg} > \"\$HOME/task13_v6e16/logs/${TASK13_TPU_RUN_ID}/train.log\" 2>&1 &"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" "${common[@]}" --command="$remote"
echo "LAUNCH_SUBMITTED run_id=${TASK13_TPU_RUN_ID} uri=${RUN_URI} config=${TASK13_TPU_CONFIG}"
