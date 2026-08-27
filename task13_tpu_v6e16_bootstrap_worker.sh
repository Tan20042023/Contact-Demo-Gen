#!/usr/bin/env bash
# Runs on exactly one v6e-16 TPU worker.  The controller invokes it on all
# workers concurrently. It prepares environment and read-only inputs, but never
# starts training.
set -euo pipefail

: "${TASK13_TPU_RUN_ID:?Set a unique TPU side-track run ID}"
: "${TASK13_TPU_CODE_URI:?Set immutable GCS source archive URI}"
: "${TASK13_TPU_CODE_SHA256:?Set source archive SHA-256}"
: "${TASK13_TPU_VENV_ARCHIVE_URI:?Set immutable GCS venv archive URI}"
: "${TASK13_TPU_VENV_ARCHIVE_SHA256:?Set venv archive SHA-256}"
: "${TASK13_TPU_INPUT_URI:=gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/input_assets}"
: "${TASK13_TPU_INPUT_BYTES:=30696986145}"

WORK_ROOT="${TASK13_TPU_WORK_ROOT:-$HOME/task13_v6e16}"
WORKDIR="${TASK13_TPU_WORKDIR:-${WORK_ROOT}/repo}"
VENV="${TASK13_TPU_VENV:-${WORK_ROOT}/venv}"
INPUT_ROOT="${TASK13_TPU_INPUT_ROOT:-${WORK_ROOT}/input_assets}"
INPUT_MANIFEST="${WORK_ROOT}/input_assets.ready.json"
STATE_DIR="${WORK_ROOT}/bootstrap/${TASK13_TPU_RUN_ID}"
MANIFEST="${STATE_DIR}/bootstrap_manifest.json"
LOCK="${WORK_ROOT}/bootstrap.lock"

mkdir -p "${WORK_ROOT}" "${STATE_DIR}"
exec 9>"${LOCK}"
flock -n 9 || { echo "BOOTSTRAP_FAIL: worker bootstrap lock is held" >&2; exit 2; }

command -v gcloud >/dev/null || { echo "BOOTSTRAP_FAIL: gcloud absent" >&2; exit 3; }
command -v sha256sum >/dev/null || { echo "BOOTSTRAP_FAIL: sha256sum absent" >&2; exit 3; }
command -v tar >/dev/null || { echo "BOOTSTRAP_FAIL: tar absent" >&2; exit 3; }

gcloud storage ls "${TASK13_TPU_CODE_URI}" >/dev/null
gcloud storage ls "${TASK13_TPU_VENV_ARCHIVE_URI}" >/dev/null

code_archive="${STATE_DIR}/source.tar.gz"
venv_archive="${STATE_DIR}/venv.tar.gz"
gcloud storage cp "${TASK13_TPU_CODE_URI}" "${code_archive}"
gcloud storage cp "${TASK13_TPU_VENV_ARCHIVE_URI}" "${venv_archive}"
printf '%s  %s\n' "${TASK13_TPU_CODE_SHA256}" "${code_archive}" | sha256sum -c -
printf '%s  %s\n' "${TASK13_TPU_VENV_ARCHIVE_SHA256}" "${venv_archive}" | sha256sum -c -

candidate="${STATE_DIR}/candidate.json"
python3 - "${candidate}" <<PY
import json, os, socket
from pathlib import Path
Path("${candidate}").write_text(json.dumps({
  "schema_version": "task13-tpu-v6e16-bootstrap-v1",
  "run_id": os.environ["TASK13_TPU_RUN_ID"],
  "hostname": socket.gethostname(),
  "code_uri": os.environ["TASK13_TPU_CODE_URI"],
  "code_sha256": os.environ["TASK13_TPU_CODE_SHA256"],
  "venv_uri": os.environ["TASK13_TPU_VENV_ARCHIVE_URI"],
  "venv_sha256": os.environ["TASK13_TPU_VENV_ARCHIVE_SHA256"],
  "input_uri": os.environ["TASK13_TPU_INPUT_URI"],
  "input_bytes": int(os.environ["TASK13_TPU_INPUT_BYTES"]),
}, indent=2, sort_keys=True) + "\n")
PY

if [[ -e "${WORKDIR}" || -e "${VENV}" ]]; then
  [[ -f "${MANIFEST}" ]] || { echo "BOOTSTRAP_FAIL: existing work root lacks manifest" >&2; exit 4; }
  cmp -s "${candidate}" "${MANIFEST}" || { echo "BOOTSTRAP_FAIL: existing bootstrap identity differs" >&2; exit 4; }
else
  install -d -m 0755 "${WORKDIR}" "${VENV}"
  tar -xzf "${code_archive}" -C "${WORKDIR}" --strip-components=1
  tar -xzf "${venv_archive}" -C "${VENV}"
  cp "${candidate}" "${MANIFEST}"
fi

[[ -x "${VENV}/bin/python" ]] || { echo "BOOTSTRAP_FAIL: venv Python absent" >&2; exit 5; }
[[ -d "${WORKDIR}/openpi" ]] || { echo "BOOTSTRAP_FAIL: openpi source absent" >&2; exit 5; }

if [[ -e "${INPUT_ROOT}" ]]; then
  [[ -f "${INPUT_MANIFEST}" ]] || {
    echo "BOOTSTRAP_FAIL: existing input root lacks a readiness record" >&2
    exit 6
  }
  python3 - "${INPUT_MANIFEST}" "${TASK13_TPU_INPUT_URI}" "${TASK13_TPU_INPUT_BYTES}" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1]))
if record.get("input_uri") != sys.argv[2] or record.get("input_bytes") != int(sys.argv[3]):
    raise SystemExit("BOOTSTRAP_FAIL: existing input identity differs")
PY
else
  input_staging="${INPUT_ROOT}.staging-${TASK13_TPU_RUN_ID}"
  [[ ! -e "${input_staging}" ]] || {
    echo "BOOTSTRAP_FAIL: stale input staging root exists: ${input_staging}" >&2
    exit 6
  }
  mkdir -p "${input_staging}"
  gcloud storage rsync --recursive "${TASK13_TPU_INPUT_URI}" "${input_staging}"
  input_bytes="$(find "${input_staging}" -type f -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')"
  [[ "${input_bytes}" == "${TASK13_TPU_INPUT_BYTES}" ]] || {
    echo "BOOTSTRAP_FAIL: input byte mismatch expected=${TASK13_TPU_INPUT_BYTES} actual=${input_bytes}" >&2
    exit 6
  }
  chmod -R a-w "${input_staging}"
  mv "${input_staging}" "${INPUT_ROOT}"
fi

python3 - "${INPUT_MANIFEST}" <<PY
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
  "input_root": "${INPUT_ROOT}",
  "input_uri": "${TASK13_TPU_INPUT_URI}",
  "input_bytes": int("${TASK13_TPU_INPUT_BYTES}"),
}, indent=2, sort_keys=True) + "\\n")
PY

"${VENV}/bin/python" - <<'PY'
import json, os, socket, subprocess
import jax, flax, orbax.checkpoint, torch
try:
    import lerobot, torchcodec
except Exception as exc:
    raise SystemExit(f"BOOTSTRAP_FAIL: project media import: {exc}")
subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL)
record = {
    "hostname": socket.gethostname(),
    "jax": jax.__version__, "process_count": jax.process_count(),
    "process_index": jax.process_index(), "device_count": jax.device_count(),
    "devices": [str(d) for d in jax.devices()], "flax": flax.__version__,
    "orbax": orbax.checkpoint.__version__, "torch": torch.__version__,
    "worker_run_id": os.environ["TASK13_TPU_RUN_ID"],
    "input_root": "${INPUT_ROOT}",
    "input_uri": os.environ["TASK13_TPU_INPUT_URI"],
    "input_bytes": int(os.environ["TASK13_TPU_INPUT_BYTES"]),
}
print(json.dumps(record, sort_keys=True))
PY

echo "BOOTSTRAP_PASS run_id=${TASK13_TPU_RUN_ID} host=$(hostname)"
