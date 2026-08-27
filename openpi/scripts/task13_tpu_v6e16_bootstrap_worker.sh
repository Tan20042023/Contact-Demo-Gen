#!/usr/bin/env bash
# Prepare one v6e-16 worker for Task13 TPU-sidebranch work.
# This script never starts training. Invoke it on every worker concurrently.
set -euo pipefail

ROOT_DIR="${HOME}/task13_tpu"
VENV_DIR="${HOME}/.venvs/openpi-task13-tpu-py311"
SOURCE_URI="${TASK13_TPU_SOURCE_URI:?set TASK13_TPU_SOURCE_URI}"
SOURCE_SHA256="${TASK13_TPU_SOURCE_SHA256:?set TASK13_TPU_SOURCE_SHA256}"
BUNDLE_PATH="${ROOT_DIR}/source.tar.gz"
REPO_DIR="${ROOT_DIR}/repo-${SOURCE_SHA256:0:12}"
READY_PATH="${ROOT_DIR}/READY.json"

mkdir -p "${ROOT_DIR}"

if ! command -v python3.11 >/dev/null || ! command -v ffmpeg >/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3.11 python3.11-venv ffmpeg git
fi

gcloud storage cp "${SOURCE_URI}" "${BUNDLE_PATH}"
printf '%s  %s\n' "${SOURCE_SHA256}" "${BUNDLE_PATH}" | sha256sum -c -

if [[ ! -d "${REPO_DIR}/openpi" ]]; then
  mkdir -p "${REPO_DIR}"
  tar -xzf "${BUNDLE_PATH}" -C "${REPO_DIR}" --strip-components=1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3.11 -m venv "${VENV_DIR}"
fi
. "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install 'jax[tpu]==0.5.3' \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

sed -n '/^dependencies = \[/,/^\]/p' "${REPO_DIR}/openpi/pyproject.toml" \
  | sed -n 's/^[[:space:]]*"\(.*\)",$/\1/p' \
  | grep -v '^jax\[cuda12\]' > "${ROOT_DIR}/openpi-tpu-deps.txt"
python -m pip install -r "${ROOT_DIR}/openpi-tpu-deps.txt" torchvision==0.22.1 lerobot==0.4.4
python -m pip install -e "${REPO_DIR}/openpi" --no-deps
if [[ -d "${REPO_DIR}/packages/openpi-client" ]]; then
  python -m pip install -e "${REPO_DIR}/packages/openpi-client" --no-deps
fi

python - <<'PY'
import jax
import lerobot
import torchcodec
print(f"JAX_VERSION={jax.__version__}")
print(f"JAX_LOCAL_DEVICE_COUNT={jax.local_device_count()}")
PY
ffmpeg -version | head -1

python - <<PY > "${READY_PATH}"
import json
import socket
import subprocess

print(json.dumps({
    "host": socket.gethostname(),
    "source_uri": "${SOURCE_URI}",
    "source_sha256": "${SOURCE_SHA256}",
    "python": subprocess.check_output(["${VENV_DIR}/bin/python", "--version"], text=True).strip(),
}, sort_keys=True))
PY
echo "TASK13_TPU_ENV_READY=${READY_PATH}"
