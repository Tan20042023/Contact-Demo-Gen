"""Durable checkpoint transport for the isolated Task13 v6e-16 side branch.

Orbax writes to a local filesystem.  A v6e-16 slice has one such filesystem
per VM, so a completed local Orbax step is uploaded as four independently
verified contributions before it is declared recoverable in GCS.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from etils import epath
import jax
from jax.experimental import multihost_utils


def enabled() -> bool:
    return os.environ.get("TASK13_TPU_MULTIHOST") == "1"


def run_uri() -> str:
    uri = os.environ.get("TASK13_TPU_GCS_RUN_URI", "").rstrip("/")
    if not uri.startswith("gs://"):
        raise ValueError("TASK13_TPU_GCS_RUN_URI must be a dedicated gs:// run prefix")
    return uri


def upload_and_commit(checkpoint_dir: epath.Path | str, step: int, *, provenance: dict[str, Any]) -> None:
    """Uploads this worker's completed local Orbax step and commits it globally."""
    root = Path(str(checkpoint_dir))
    local_step = root / str(step)
    if not local_step.is_dir():
        raise FileNotFoundError(f"Completed local checkpoint step is absent: {local_step}")

    uri = run_uri()
    worker = jax.process_index()
    process_count = jax.process_count()
    worker_uri = f"{uri}/checkpoints/{step}/worker-{worker}"
    manifest = _manifest(local_step, step=step, worker=worker, process_count=process_count)
    manifest_path = local_step.parent / f"{step}.worker-{worker}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")

    _run("gcloud", "storage", "rsync", "--recursive", str(local_step), f"{worker_uri}/state")
    _run("gcloud", "storage", "cp", str(manifest_path), f"{worker_uri}/manifest.json")
    multihost_utils.sync_global_devices(f"task13-uploaded-{step}")

    if worker == 0:
        workers = [_verify_worker(uri, step, index, process_count) for index in range(process_count)]
        commit = {
            "schema": "task13-v6e16-checkpoint-v1",
            "step": step,
            "process_count": process_count,
            "workers": workers,
            "provenance": provenance,
        }
        commit_path = local_step.parent / f"{step}.COMMITTED.json"
        commit_path.write_text(json.dumps(commit, sort_keys=True, indent=2) + "\n")
        _run("gcloud", "storage", "cp", str(commit_path), f"{uri}/checkpoints/{step}/COMMITTED.json")
        _run("gcloud", "storage", "cp", str(commit_path), f"{uri}/LATEST.json")
    multihost_utils.sync_global_devices(f"task13-committed-{step}")


def materialize_latest_committed(checkpoint_dir: epath.Path | str) -> int:
    """Downloads only this worker's contribution of the last committed step."""
    uri = run_uri()
    latest = json.loads(_output("gcloud", "storage", "cat", f"{uri}/LATEST.json"))
    step = int(latest["step"])
    if int(latest["process_count"]) != jax.process_count():
        raise ValueError("Committed checkpoint topology does not match this TPU slice")
    _verify_worker(uri, step, jax.process_index(), jax.process_count())
    target = Path(str(checkpoint_dir)) / str(step)
    if target.exists():
        raise FileExistsError(f"Refusing to merge a committed checkpoint into existing {target}")
    target.mkdir(parents=True)
    _run("gcloud", "storage", "rsync", "--recursive", f"{uri}/checkpoints/{step}/worker-{jax.process_index()}/state", str(target))
    multihost_utils.sync_global_devices(f"task13-materialized-{step}")
    return step


def _manifest(root: Path, *, step: int, worker: int, process_count: int) -> dict[str, Any]:
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        files.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    if not files:
        raise ValueError(f"Checkpoint contribution contains no files: {root}")
    return {"step": step, "worker": worker, "process_count": process_count, "files": files}


def _verify_worker(uri: str, step: int, worker: int, process_count: int) -> dict[str, Any]:
    base = f"{uri}/checkpoints/{step}/worker-{worker}"
    manifest_text = _output("gcloud", "storage", "cat", f"{base}/manifest.json")
    manifest = json.loads(manifest_text)
    if manifest["step"] != step or manifest["worker"] != worker or manifest["process_count"] != process_count:
        raise ValueError(f"Invalid checkpoint manifest for worker {worker}")
    for entry in manifest["files"]:
        payload = subprocess.check_output(["gcloud", "storage", "cat", f"{base}/state/{entry['path']}"])
        if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"SHA verification failed for worker {worker}: {entry['path']}")
    return {"worker": worker, "manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(), "files": len(manifest["files"])}


def _run(*args: str) -> None:
    subprocess.run(args, check=True)


def _output(*args: str) -> str:
    return subprocess.check_output(args, text=True)
