#!/usr/bin/env bash
# Bootstrap exactly one v6e-16 worker from immutable GCS inputs.  It never
# starts training and is safe to run concurrently through gcloud --worker=all.
set -euo pipefail

: "${TASK13_TPU_RUN_ID:?Set a unique TPU side-track run ID}"
: "${TASK13_TPU_CODE_URI:?Set immutable GCS source archive URI}"
: "${TASK13_TPU_CODE_SHA256:?Set source archive SHA-256}"
: "${TASK13_TPU_INPUT_URI:=gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/input_assets}"
: "${TASK13_TPU_INPUT_BYTES:=30696986145}"

ROOT="${TASK13_TPU_WORK_ROOT:-$HOME/task13_v6e16}"
REPO="${ROOT}/repo-${TASK13_TPU_CODE_SHA256:0:12}"
VENV="${ROOT}/venv"
INPUT="${TASK13_TPU_INPUT_ROOT:-${ROOT}/input_assets}"
STATE="${ROOT}/bootstrap/${TASK13_TPU_RUN_ID}"
READY="${STATE}/READY.json"
mkdir -p "$STATE"

if ! command -v python3.11 >/dev/null || ! command -v ffmpeg >/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3.11 python3.11-venv ffmpeg git
fi

archive="${STATE}/source.tar.gz"
gcloud storage cp "$TASK13_TPU_CODE_URI" "$archive"
printf '%s  %s\n' "$TASK13_TPU_CODE_SHA256" "$archive" | sha256sum -c -
if [[ ! -d "${REPO}/openpi" ]]; then
  mkdir -p "$REPO"
  tar -xzf "$archive" -C "$REPO" --strip-components=1
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3.11 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install 'jax[tpu]==0.5.3' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  sed -n '/^dependencies = \[/,/^\]/p' "${REPO}/openpi/pyproject.toml" | sed -n 's/^[[:space:]]*"\(.*\)",$/\1/p' | grep -v '^jax\[cuda12\]' > "${STATE}/openpi-tpu-deps.txt"
  "$VENV/bin/python" -m pip install -r "${STATE}/openpi-tpu-deps.txt" torchvision==0.22.1 lerobot==0.4.4
  "$VENV/bin/python" -m pip install -e "${REPO}/openpi" --no-deps
  "$VENV/bin/python" -m pip install -e "${REPO}/openpi/packages/openpi-client" --no-deps
fi

if [[ ! -e "$INPUT" ]]; then
  staging="${INPUT}.staging-${TASK13_TPU_RUN_ID}"
  mkdir -p "$staging"
  gcloud storage rsync --recursive "$TASK13_TPU_INPUT_URI" "$staging"
  actual="$(find "$staging" -type f -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')"
  [[ "$actual" == "$TASK13_TPU_INPUT_BYTES" ]] || { echo "BOOTSTRAP_FAIL: input bytes expected=${TASK13_TPU_INPUT_BYTES} actual=${actual}" >&2; exit 6; }
  chmod -R a-w "$staging"
  mv "$staging" "$INPUT"
fi

"$VENV/bin/python" - <<PY > "$READY"
import json, socket
from pathlib import Path
import jax, lerobot, torchcodec
root = Path("$INPUT")
print(json.dumps({
  "schema": "task13-v6e16-ready-v2", "host": socket.gethostname(),
  "source_uri": "$TASK13_TPU_CODE_URI", "source_sha256": "$TASK13_TPU_CODE_SHA256",
  "input_uri": "$TASK13_TPU_INPUT_URI", "input_bytes": sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
  "jax": jax.__version__, "local_device_count": jax.local_device_count(),
}, sort_keys=True))
PY
echo "BOOTSTRAP_PASS host=$(hostname) ready=$READY"
