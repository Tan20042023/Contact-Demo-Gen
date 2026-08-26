#!/usr/bin/env bash
# Validate Task13 TPU config wiring only; do not build a model or invoke training.
set -euo pipefail

WORKDIR="${TASK13_TPU_WORKDIR:-$HOME/dexjoco-task13-tpu}"
VENV="${TASK13_TPU_PROJECT_VENV:-$HOME/.venvs/openpi-task13-tpu-py311}"
INPUT_ROOT="${TASK13_TPU_INPUT_ROOT:-$HOME/task13_input_assets}"
LOCAL_RUNS_ROOT="${TASK13_TPU_LOCAL_RUNS_ROOT:-$HOME/task13_local_runs}"

export TASK13_TPU_INPUT_ROOT="${INPUT_ROOT}"
export TASK13_TPU_LOCAL_RUNS_ROOT="${LOCAL_RUNS_ROOT}"
export TASK13_TPU_FSDP_DEVICES=4

cd "${WORKDIR}/openpi"
"${VENV}/bin/python" - <<'PY'
from pathlib import Path

from openpi.training.config import get_config

config = get_config("task13_tpu_smoke_hammer_nail_nominal_src")
print(f"name={config.name}")
print(f"dataset_root={config.data.root}")
print(f"assets_dir={config.data.assets.assets_dir}")
print(f"asset_id={config.data.assets.asset_id}")
print(f"checkpoint_base_dir={config.checkpoint_base_dir}")
print(f"checkpoint_dir={config.checkpoint_dir}")
print(f"save_interval={config.save_interval} keep_period={config.keep_period}")
print(f"batch_size={config.batch_size} fsdp_devices={config.fsdp_devices}")
assert Path(config.data.root).is_dir()
assert Path(config.data.assets.assets_dir).is_dir()
assert str(config.checkpoint_base_dir).startswith("/home/")
assert str(config.checkpoint_dir).startswith("/home/")
assert config.save_interval == 100
assert config.batch_size == 32
assert config.fsdp_devices == 4
print("CONFIG_PASS: no model construction or training was performed")
PY
