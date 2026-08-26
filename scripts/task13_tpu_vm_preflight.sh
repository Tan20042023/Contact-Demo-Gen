#!/usr/bin/env bash
# Run only after an approved, already allocated TPU VM is reachable.
# This script performs no installation, data copy, or training.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-whyu01}"
INPUT_ROOT="${TASK13_TPU_INPUT_ROOT:-/mnt/task13/input_assets}"
INPUT_URI="${TASK13_TPU_INPUT_URI:-gs://euw4/user/tanjunhao/task13_tpu_feasibility/v1/input_assets}"
RUNS_URI="${TASK13_TPU_RUNS_ROOT:-gs://euw4/user/tanjunhao/task13_tpu_feasibility/v1/runs}"
EXPECTED_SERVICE_ACCOUNT="${EXPECTED_SERVICE_ACCOUNT:-184047521632-compute@developer.gserviceaccount.com}"
PROBE_URI="${RUNS_URI}/preflight/identity_probe_$(date -u +%Y%m%dT%H%M%SZ).txt"

fail() { printf 'PRECHECK_FAIL: %s\n' "$*" >&2; exit 1; }
note() { printf 'PRECHECK: %s\n' "$*"; }

note "project=${PROJECT_ID}"
note "host=$(hostname)"
uname -a
nproc
free -h
df -h / /home "${INPUT_ROOT%/*}" 2>/dev/null || true

METADATA_ACCOUNT="$(curl --fail --silent --show-error -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email)" \
  || fail "cannot obtain the attached service account from the metadata server"
note "attached_service_account=${METADATA_ACCOUNT}"
[[ "${METADATA_ACCOUNT}" == "${EXPECTED_SERVICE_ACCOUNT}" ]] \
  || fail "unexpected service account; recreate/inspect the TPU request without a custom --service-account"

gcloud storage ls "${INPUT_URI}/checkpoints/pi05_base/" >/dev/null \
  || fail "cannot list the sealed 22-D base prefix"
gcloud storage ls "${INPUT_URI}/checkpoints/pi05_base_action_dim_44/" >/dev/null \
  || fail "cannot list the sealed 44-D base prefix"
gcloud storage ls "${INPUT_URI}/lerobot/hammer_nail/nominal_src/" >/dev/null \
  || fail "cannot list the Hammer nominal_src prefix"

printf '%s\n' "Task13 TPU preflight ${METADATA_ACCOUNT}" | gcloud storage cp - "${PROBE_URI}" \
  || fail "cannot write under the isolated runs prefix"
gcloud storage cp "${PROBE_URI}" - >/dev/null \
  || fail "cannot read back the GCS probe object"
gcloud storage rm "${PROBE_URI}" >/dev/null \
  || fail "cannot clean up the isolated GCS probe object"
note "gcs_input_and_runs_probe=PASS"

python - <<'PY'
import jax
print(f'jax_version={jax.__version__}')
print(f'process_count={jax.process_count()} process_index={jax.process_index()}')
print(f'device_count={jax.device_count()}')
print('devices=')
for device in jax.devices():
    print(device)
if jax.process_count() != 1:
    raise SystemExit('PRECHECK_FAIL: multi-process TPU is unsupported by the current loader')
if 32 % jax.device_count() != 0:
    raise SystemExit('PRECHECK_FAIL: device count does not divide global batch 32')
PY

note "PRECHECK_PASS: no installation, copy, or training was performed"
