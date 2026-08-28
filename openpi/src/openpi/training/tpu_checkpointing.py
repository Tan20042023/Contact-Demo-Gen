"""Commit marker for native shared-GCS Task13 TPU checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Any

import jax
from jax.experimental import multihost_utils


def enabled() -> bool:
    return os.environ.get("TASK13_TPU_MULTIHOST") == "1"


def run_uri() -> str:
    uri = os.environ.get("TASK13_TPU_GCS_RUN_URI", "").rstrip("/")
    if not uri.startswith("gs://"):
        raise ValueError("TASK13_TPU_GCS_RUN_URI must be a dedicated gs:// run prefix")
    return uri


def commit_shared_checkpoint(checkpoint_dir: str, step: int, *, provenance: dict[str, Any]) -> None:
    """Marks an Orbax-native shared-GCS step recoverable after all hosts finish."""
    uri = run_uri()
    worker = jax.process_index()
    process_count = jax.process_count()
    checkpoint_uri = f"{str(checkpoint_dir).rstrip('/')}/{step}"
    multihost_utils.sync_global_devices(f"task13-orbax-finished-{step}")

    if worker == 0:
        objects = [line for line in _output("gcloud", "storage", "ls", "--recursive", f"{checkpoint_uri}/**").splitlines() if line]
        if not objects:
            raise FileNotFoundError(f"Native Orbax checkpoint has no GCS objects: {checkpoint_uri}")
        commit = {
            "schema": "task13-v6e16-native-gcs-checkpoint-v2",
            "step": step,
            "process_count": process_count,
            "checkpoint_uri": checkpoint_uri,
            "object_count": len(objects),
            "provenance": provenance,
        }
        _write_gcs_json(commit, f"{uri}/checkpoints/{step}/COMMITTED.json")
        _write_gcs_json(commit, f"{uri}/LATEST.json")
    multihost_utils.sync_global_devices(f"task13-committed-{step}")


def validate_latest_committed(checkpoint_dir: str) -> int:
    """Validates launch metadata; Orbax restores the shared GCS step directly."""
    uri = run_uri()
    latest = json.loads(_output("gcloud", "storage", "cat", f"{uri}/LATEST.json"))
    step = int(latest["step"])
    if int(latest["process_count"]) != jax.process_count():
        raise ValueError("Committed checkpoint topology does not match this TPU slice")
    if latest.get("checkpoint_uri") != f"{str(checkpoint_dir).rstrip('/')}/{step}":
        raise ValueError("Committed checkpoint URI does not match this Task13 config")
    multihost_utils.sync_global_devices(f"task13-validated-{step}")
    return step


def _write_gcs_json(payload: dict[str, Any], uri: str) -> None:
    process = subprocess.run(
        ["gcloud", "storage", "cp", "-", uri], input=json.dumps(payload, sort_keys=True) + "\n", text=True, check=True
    )
    del process


def _run(*args: str) -> None:
    subprocess.run(args, check=True)


def _output(*args: str) -> str:
    return subprocess.check_output(args, text=True)
