"""Persistent elastic-slot controller for the Task13 Spot TPU formal campaign.

The controller treats each TPU allocation as disposable, GCS LATEST.json as
the only resume authority, and the local campaign JSON as scheduling state.  It
never touches the GPU/A100 worktree or outputs.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any


PROJECT = "whyu01"
ZONE = "us-east1-d"
ACCELERATOR = "v6e-8"
RUNTIME = "v2-alpha-tpuv6e"
RUNS_ROOT = "gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/runs"
INPUT_URI = "gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/input_assets"
INPUT_BYTES = 30_696_986_145
CODE_URI = "gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/bootstrap/60f7a53/source-layout-v2.tar.gz"
SOURCE_SHA256 = "c1e6a96abc645b1d6abb66d4e64ad225946192c80a46e8a63cc97bd812af8c85"
V6E16_PROOF_URI = (
    "gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/runs/"
    "checkpoint_contract_60f7a53_20260828a/hammer_nail_nominal_src/"
    "provenance/CHECKPOINT_CONTRACT_PASS.json"
)
PROOF_URI = (
    "gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/runs/"
    "checkpoint_contract_v6e8_60f7a53_v1/hammer_nail_nominal_src/"
    "provenance/CHECKPOINT_CONTRACT_PASS.json"
)
TARGET_STEP = 30_000
CHECKPOINT_INTERVAL = 1_000
EXPECTED_WORKERS = 1
EXPECTED_DEVICES = 8
FSDP_DEVICES = 4
SLOTS: tuple[dict[str, Any], ...] = tuple()
PROFILES = {
    "v6e8x26": {
        "accelerator": "v6e-8", "slot_count": 26, "workers": 1, "devices": 8,
        "fsdp_devices": 4, "checkpoint_interval": 1_000, "proof_uri": PROOF_URI,
    },
    "v6e4x26": {
        "accelerator": "v6e-4", "slot_count": 26, "workers": 1, "devices": 4,
        "fsdp_devices": 4, "checkpoint_interval": 1_000,
        "proof_uri": "gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/runs/checkpoint_contract_v6e4_60f7a53_v1/hammer_nail_nominal_src/provenance/CHECKPOINT_CONTRACT_PASS.json",
    },
    "v6e16x13": {
        "accelerator": "v6e-16", "slot_count": 13, "workers": 4, "devices": 16,
        "fsdp_devices": 4, "checkpoint_interval": 2_500, "proof_uri": V6E16_PROOF_URI,
    },
}
TERMINAL_QUEUE_STATES = {"SUSPENDED", "FAILED"}
TERMINAL_NODE_STATES = {"PREEMPTED", "TERMINATED", "STOPPED"}
MUTATING_CAMPAIGN_STATES = {"RUNNING", "COMPLETE_CLEANUP", "BLOCKED_CLEANUP", "STOP_REQUESTED"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def slug(value: str) -> str:
    return value.replace("_", "-")


def make_slots(count: int) -> tuple[dict[str, Any], ...]:
    return tuple(
        {"id": index, "node": f"tanjunhao-tpu{index}", "queue": f"tanjunhao-tpu{index}-qr"}
        for index in range(1, count + 1)
    )


def build_cells(campaign: str, checkpoint_interval: int) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    tasks = ("hammer_nail", "bimanual_assembly")
    conditions = ("nominal_src", "repeat", "visual", "contact", "combined")
    schedule: list[tuple[str, int, str, str]] = []
    for condition in conditions:
        for task in tasks:
            schedule.append(("p2", 42, task, condition))
    for seed in (43, 44):
        for condition in conditions[1:]:
            for task in tasks:
                schedule.append(("p3", seed, task, condition))

    for index, (phase, seed, task, condition) in enumerate(schedule, start=1):
        cell_id = f"{index:02d}-{phase}-{slug(task)}-{slug(condition)}-s{seed}"
        config = f"task13_tpu_technical_{task}_{condition}"
        exp_name = f"{task}__{condition}__seed{seed}__technical"
        cells.append(
            {
                "index": index,
                "id": cell_id,
                "phase": phase,
                "task": task,
                "condition": condition,
                "seed": seed,
                "config": config,
                "exp_name": exp_name,
                "train_args": [
                    f"--seed={seed}", f"--exp-name={exp_name}", f"--save-interval={checkpoint_interval}"
                ],
                "target_step": TARGET_STEP,
                "status": "PENDING",
                "assigned_slot": None,
                "attempt": 0,
                "run_uri": None,
                "latest_step": 0,
                "last_progress_utc": None,
                "consecutive_no_progress_failures": 0,
                "abandoned_attempts": [],
            }
        )
    assert len(cells) == 26
    assert len({c["id"] for c in cells}) == 26
    return cells


def initial_state(campaign: str, profile_name: str) -> dict[str, Any]:
    profile = PROFILES[profile_name]
    slots = make_slots(profile["slot_count"])
    return {
        "schema": "task13-tpu-formal-campaign-v1",
        "campaign": campaign,
        "campaign_status": "INITIALIZED",
        "profile": profile_name,
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "project": PROJECT,
        "zone": ZONE,
        "accelerator_type": profile["accelerator"],
        "runtime_version": RUNTIME,
        "expected_workers": profile["workers"],
        "expected_devices": profile["devices"],
        "fsdp_devices": profile["fsdp_devices"],
        "runs_root": RUNS_ROOT,
        "input_uri": INPUT_URI,
        "input_bytes": INPUT_BYTES,
        "code_uri": CODE_URI,
        "source_sha256": SOURCE_SHA256,
        "proof_uri": profile["proof_uri"],
        "target_step": TARGET_STEP,
        "checkpoint_interval": profile["checkpoint_interval"],
        "cells": build_cells(campaign, profile["checkpoint_interval"]),
        "slots": [
            {
                **slot,
                "active_cell": None,
                "allocation_id": None,
                "bootstrapped_allocation_id": None,
                "phase": "IDLE",
                "last_observation": None,
                "last_error": None,
                "controller_error_count": 0,
                "cleanup_complete": False,
            }
            for slot in slots
        ],
        "events": [],
    }


def configure_globals(state: dict[str, Any]) -> None:
    global ACCELERATOR, RUNTIME, PROOF_URI, CHECKPOINT_INTERVAL
    global EXPECTED_WORKERS, EXPECTED_DEVICES, FSDP_DEVICES, SLOTS
    ACCELERATOR = state["accelerator_type"]
    RUNTIME = state["runtime_version"]
    PROOF_URI = state["proof_uri"]
    CHECKPOINT_INTERVAL = int(state["checkpoint_interval"])
    EXPECTED_WORKERS = int(state["expected_workers"])
    EXPECTED_DEVICES = int(state["expected_devices"])
    FSDP_DEVICES = int(state["fsdp_devices"])
    SLOTS = tuple({"id": slot["id"], "node": slot["node"], "queue": slot["queue"]} for slot in state["slots"])


class CampaignStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()

    def load(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(self.path.read_text(encoding="utf-8-sig"))

    def mutate(self, fn):
        with self._lock:
            state = self.load()
            result = fn(state)
            state["updated_utc"] = utc_now()
            self._write(state)
            return result

    def initialize(self, state: dict[str, Any]) -> None:
        with self._lock:
            if self.path.exists():
                raise RuntimeError(f"Campaign state already exists: {self.path}")
            self._write(state)

    def _write(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temp, self.path)


class EventLog:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def write(self, slot: int | str, message: str) -> None:
        line = f"{utc_now()} slot={slot} {message}"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        print(line, flush=True)


@contextlib.contextmanager
def single_instance_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0)
    if handle.read(1) == b"":
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError("Another Task13 TPU supervisor instance is already running") from exc
    try:
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def run(args: list[str], *, timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()[-2000:]
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(args[:8])}\n{detail}")
    return proc


def gcloud_json_optional(args: list[str]) -> dict[str, Any] | None:
    proc = run(["gcloud", *args, "--format=json"], timeout=120, check=False)
    if proc.returncode == 0:
        return json.loads(proc.stdout)
    detail = (proc.stderr + "\n" + proc.stdout).lower()
    not_found = "not found" in detail or "could not fetch resource" in detail or "status=[404]" in detail
    if not_found:
        return None
    raise RuntimeError(f"gcloud describe failed: {detail[-2000:]}")


def get_queue(queue: str) -> dict[str, Any] | None:
    return gcloud_json_optional(
        ["compute", "tpus", "queued-resources", "describe", queue, f"--project={PROJECT}", f"--zone={ZONE}"]
    )


def get_node(node: str) -> dict[str, Any] | None:
    return gcloud_json_optional(
        ["compute", "tpus", "tpu-vm", "describe", node, f"--project={PROJECT}", f"--zone={ZONE}"]
    )


def queue_state(queue: dict[str, Any] | None) -> tuple[str, str | None]:
    if queue is None:
        return "NOT_FOUND", None
    raw = queue.get("state", {})
    if isinstance(raw, str):
        return raw, None
    return str(raw.get("state", "UNKNOWN")), raw.get("stateInitiator")


def validate_slot_resources(slot: dict[str, Any], queue: dict[str, Any] | None, node: dict[str, Any] | None) -> None:
    if queue is not None:
        specs = queue.get("tpu", {}).get("nodeSpec", [])
        ids = [spec.get("nodeId") for spec in specs]
        if ids != [slot["node"]] or "spot" not in queue:
            raise RuntimeError(f"Refuse to mutate unexpected queued resource {slot['queue']}: node_ids={ids}, spot={'spot' in queue}")
    if node is not None:
        queued = str(node.get("queuedResource", ""))
        if node.get("acceleratorType") != ACCELERATOR or not queued.endswith("/" + slot["queue"]):
            raise RuntimeError(
                f"Refuse to mutate unexpected TPU {slot['node']}: accelerator={node.get('acceleratorType')} queue={queued}"
            )


def delete_exact_slot(slot: dict[str, Any], log: EventLog) -> None:
    queue = get_queue(slot["queue"])
    node = get_node(slot["node"])
    validate_slot_resources(slot, queue, node)
    if queue is not None:
        log.write(slot["id"], f"DELETE_EXACT_QUEUE queue={slot['queue']} node={slot['node']}")
        run(
            [
                "gcloud", "compute", "tpus", "queued-resources", "delete", slot["queue"],
                f"--project={PROJECT}", f"--zone={ZONE}", "--force", "--quiet",
            ],
            timeout=1800,
        )
    elif node is not None:
        log.write(slot["id"], f"DELETE_ORPHAN_EXACT_NODE node={slot['node']}")
        run(
            [
                "gcloud", "compute", "tpus", "tpu-vm", "delete", slot["node"],
                f"--project={PROJECT}", f"--zone={ZONE}", "--quiet",
            ],
            timeout=1800,
        )

    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if get_queue(slot["queue"]) is None and get_node(slot["node"]) is None:
            return
        time.sleep(15)
    raise RuntimeError(f"Timed out waiting for exact slot deletion: {slot['queue']}")


def create_exact_slot(slot: dict[str, Any], log: EventLog) -> None:
    if get_queue(slot["queue"]) is not None or get_node(slot["node"]) is not None:
        raise RuntimeError(f"Refuse create because exact slot still exists: {slot['queue']} / {slot['node']}")
    log.write(slot["id"], f"CREATE_SPOT_QUEUE queue={slot['queue']} node={slot['node']} accelerator={ACCELERATOR}")
    run(
        [
            "gcloud", "compute", "tpus", "queued-resources", "create", slot["queue"],
            f"--node-id={slot['node']}", f"--accelerator-type={ACCELERATOR}",
            f"--runtime-version={RUNTIME}", "--spot", "--network=default",
            f"--project={PROJECT}", f"--zone={ZONE}", "--quiet",
        ],
        timeout=300,
    )


def gcs_latest(uri: str | None) -> dict[str, Any] | None:
    if not uri:
        return None
    proc = run(["gcloud", "storage", "cat", f"{uri}/LATEST.json"], timeout=120, check=False)
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid LATEST.json at {uri}") from exc


def gcs_prefix_nonempty(uri: str) -> bool:
    proc = run(["gcloud", "storage", "ls", "--recursive", f"{uri}/**"], timeout=180, check=False)
    return proc.returncode == 0 and bool(proc.stdout.strip())


def assert_latest_for_cell(latest: dict[str, Any], cell: dict[str, Any]) -> int:
    step = int(latest.get("step", -1))
    provenance = latest.get("provenance", {})
    if int(latest.get("process_count", -1)) != EXPECTED_WORKERS:
        raise RuntimeError(f"LATEST topology mismatch for {cell['id']}")
    if provenance.get("config_name") != cell["config"] or provenance.get("exp_name") != cell["exp_name"]:
        raise RuntimeError(f"LATEST provenance mismatch for {cell['id']}: {provenance}")
    if provenance.get("source_sha256") != SOURCE_SHA256:
        raise RuntimeError(f"LATEST source mismatch for {cell['id']}")
    if step < 0 or step > cell["target_step"]:
        raise RuntimeError(f"LATEST step out of range for {cell['id']}: {step}")
    return step


def worker_ips(node: dict[str, Any]) -> list[str]:
    return [
        endpoint.get("accessConfig", {}).get("externalIp")
        for endpoint in node.get("networkEndpoints", [])
        if endpoint.get("accessConfig", {}).get("externalIp")
    ]


def ssh_training_commands(node: dict[str, Any], key_path: Path) -> list[str] | None:
    ips = worker_ips(node)
    if len(ips) != EXPECTED_WORKERS:
        return None
    commands: list[str] = []
    for ip in ips:
        proc = run(
            [
                "ssh", "-i", str(key_path), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                "-o", "ConnectTimeout=20", f"tanjunhao@{ip}",
                "pgrep -af '[p]ython.*scripts/train.py' || true",
            ],
            timeout=45,
            check=False,
        )
        if proc.returncode != 0:
            return None
        commands.append(proc.stdout.strip())
    return commands


def stop_training(node: dict[str, Any], key_path: Path, log: EventLog, slot_id: int) -> None:
    for ip in worker_ips(node):
        run(
            [
                "ssh", "-i", str(key_path), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                "-o", "ConnectTimeout=20", f"tanjunhao@{ip}",
                "pkill -TERM -f '[p]ython.*scripts/train.py' || true",
            ],
            timeout=45,
            check=False,
        )
    log.write(slot_id, "STOPPED_SLOT_TRAIN_PROCESSES")


class Supervisor:
    def __init__(self, state_path: Path, log_path: Path, poll_seconds: int):
        self.store = CampaignStore(state_path)
        self.log = EventLog(log_path)
        self.poll_seconds = poll_seconds
        self.root = Path(__file__).resolve().parent
        self.key_path = Path.home() / ".ssh" / "google_compute_engine"
        self.bootstrap_script = self.root / "task13_tpu_v6e16_bootstrap_all.ps1"
        self.launch_script = self.root / "task13_tpu_v6e16_launch.ps1"

    def set_slot(self, slot_id: int, **changes: Any) -> None:
        def mutate(state):
            target = next(slot for slot in state["slots"] if slot["id"] == slot_id)
            target.update(changes)

        self.store.mutate(mutate)

    def block_campaign(self, slot_id: int, message: str) -> None:
        def mutate(state):
            state["campaign_status"] = "BLOCKED_CLEANUP"
            state["blocked_reason"] = message
            target = next(slot for slot in state["slots"] if slot["id"] == slot_id)
            target["last_error"] = message

        self.store.mutate(mutate)
        self.log.write(slot_id, f"CAMPAIGN_BLOCKED reason={message}")

    def claim_or_get_cell(self, slot_id: int) -> dict[str, Any] | None:
        result: dict[str, Any] | None = None
        newly_claimed = False

        def mutate(state):
            nonlocal result, newly_claimed
            slot = next(item for item in state["slots"] if item["id"] == slot_id)
            if slot["active_cell"]:
                result = next(cell for cell in state["cells"] if cell["id"] == slot["active_cell"])
                return
            pending = next((cell for cell in state["cells"] if cell["status"] == "PENDING"), None)
            if pending is None:
                # Work stealing is allowed only from a slot that is demonstrably
                # waiting for a replacement allocation. The state mutation is
                # atomic, so the old slot observes loss of ownership before it
                # can launch again.
                donor = next(
                    (
                        item
                        for item in state["slots"]
                        if item["id"] != slot_id
                        and item["active_cell"]
                        and item["phase"] in {"WAITING_FOR_CAPACITY", "REQUESTING_CAPACITY"}
                    ),
                    None,
                )
                if donor is not None:
                    pending = next(cell for cell in state["cells"] if cell["id"] == donor["active_cell"])
                    donor["active_cell"] = None
                    donor["phase"] = "WAITING_FOR_CAPACITY"
            if pending is None:
                return
            pending["status"] = "ASSIGNED"
            pending["assigned_slot"] = slot_id
            pending["assigned_utc"] = utc_now()
            slot["active_cell"] = pending["id"]
            slot["phase"] = "CELL_ASSIGNED"
            result = pending
            newly_claimed = True

        self.store.mutate(mutate)
        if newly_claimed and result:
            self.log.write(slot_id, f"CELL_ASSIGNED cell={result['id']}")
        return result

    def update_cell(self, cell_id: str, **changes: Any) -> None:
        def mutate(state):
            cell = next(item for item in state["cells"] if item["id"] == cell_id)
            cell.update(changes)

        self.store.mutate(mutate)

    def complete_cell(self, slot_id: int, cell_id: str, step: int) -> None:
        def mutate(state):
            cell = next(item for item in state["cells"] if item["id"] == cell_id)
            cell.update(
                status="COMPLETE", latest_step=step, completed_utc=utc_now(), assigned_slot=slot_id,
                consecutive_no_progress_failures=0,
            )
            slot = next(item for item in state["slots"] if item["id"] == slot_id)
            slot.update(active_cell=None, phase="IDLE", last_error=None)
            if all(item["status"] == "COMPLETE" for item in state["cells"]):
                state["campaign_status"] = "COMPLETE_CLEANUP"

        self.store.mutate(mutate)
        self.log.write(slot_id, f"CELL_COMPLETE cell={cell_id} step={step}")

    def prepare_attempt(self, cell: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        if cell["attempt"] == 0:
            attempt = 1
            uri = f"{RUNS_ROOT}/{self.store.load()['campaign']}/{cell['id']}/attempt-{attempt:03d}"
            self.update_cell(cell["id"], attempt=attempt, run_uri=uri)
            cell = next(item for item in self.store.load()["cells"] if item["id"] == cell["id"])

        latest = gcs_latest(cell["run_uri"])
        if latest is not None:
            step = assert_latest_for_cell(latest, cell)
            if step > cell["latest_step"]:
                self.update_cell(
                    cell["id"], latest_step=step, last_progress_utc=utc_now(), consecutive_no_progress_failures=0
                )
                cell["latest_step"] = step
            return cell, True

        if gcs_prefix_nonempty(cell["run_uri"]):
            old = {"attempt": cell["attempt"], "run_uri": cell["run_uri"], "abandoned_utc": utc_now(), "reason": "no_committed_latest"}
            attempt = cell["attempt"] + 1
            uri = f"{RUNS_ROOT}/{self.store.load()['campaign']}/{cell['id']}/attempt-{attempt:03d}"
            abandoned = [*cell.get("abandoned_attempts", []), old]
            self.update_cell(cell["id"], attempt=attempt, run_uri=uri, abandoned_attempts=abandoned)
            self.log.write(cell["assigned_slot"], f"ABANDON_PARTIAL_ATTEMPT cell={cell['id']} old_uri={old['run_uri']} new_uri={uri}")
            cell = next(item for item in self.store.load()["cells"] if item["id"] == cell["id"])
        return cell, False

    def bootstrap(self, slot: dict[str, Any], node: dict[str, Any]) -> None:
        allocation_id = str(node.get("id", node.get("createTime", "unknown")))
        if slot.get("bootstrapped_allocation_id") == allocation_id:
            return
        run_id = f"formal-{slot['node']}-{allocation_id}"
        self.set_slot(slot["id"], phase="BOOTSTRAPPING", allocation_id=allocation_id)
        self.log.write(slot["id"], f"BOOTSTRAP_START allocation={allocation_id}")
        run(
            [
                "pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(self.bootstrap_script),
                "-RunId", run_id, "-CodeUri", CODE_URI, "-CodeSha256", SOURCE_SHA256,
                "-InputUri", INPUT_URI, "-InputBytes", str(INPUT_BYTES), "-TpuName", slot["node"],
                "-Project", PROJECT, "-Zone", ZONE, "-ReadyTimeoutSeconds", "3600",
                "-ExpectedWorkers", str(EXPECTED_WORKERS),
            ],
            timeout=5400,
        )
        self.set_slot(
            slot["id"], bootstrapped_allocation_id=allocation_id, phase="READY_FOR_CELL",
            last_error=None, controller_error_count=0,
        )
        self.log.write(slot["id"], f"BOOTSTRAP_COMPLETE allocation={allocation_id}")

    def launch(self, slot: dict[str, Any], cell: dict[str, Any], resume: bool) -> None:
        current = self.store.load()
        current_cell = next(item for item in current["cells"] if item["id"] == cell["id"])
        current_slot = next(item for item in current["slots"] if item["id"] == slot["id"])
        if current_cell["assigned_slot"] != slot["id"] or current_slot["active_cell"] != cell["id"]:
            self.log.write(slot["id"], f"LAUNCH_SKIPPED_OWNERSHIP_CHANGED cell={cell['id']}")
            return
        run_id = f"{cell['id']}-a{cell['attempt']:03d}-{slot['node']}"
        args = [
            "pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(self.launch_script),
            "-RunId", run_id, "-Config", cell["config"], "-GcsRunUri", cell["run_uri"],
            "-SourceSha256", SOURCE_SHA256, "-Purpose", "formal",
            "-CheckpointContractProofUri", PROOF_URI,
            "-TrainArgsJson", json.dumps(cell["train_args"], separators=(",", ":")),
            "-TpuName", slot["node"], "-Project", PROJECT, "-Zone", ZONE,
            "-ExpectedWorkers", str(EXPECTED_WORKERS), "-ExpectedDevices", str(EXPECTED_DEVICES),
            "-FsdpDevices", str(FSDP_DEVICES),
        ]
        if resume:
            args.append("-Resume")
        self.set_slot(slot["id"], phase="LAUNCHING")
        self.log.write(slot["id"], f"TRAIN_LAUNCH cell={cell['id']} resume={resume} uri={cell['run_uri']}")
        run(args, timeout=1800)
        self.update_cell(cell["id"], status="TRAINING", last_launch_utc=utc_now())
        self.set_slot(slot["id"], phase="TRAINING", last_error=None, controller_error_count=0)

    def cleanup_slot(self, slot: dict[str, Any], stop_training_first: bool) -> None:
        node = get_node(slot["node"])
        if stop_training_first and node is not None and node.get("state") == "READY":
            stop_training(node, self.key_path, self.log, slot["id"])
        delete_exact_slot(slot, self.log)
        self.set_slot(slot["id"], cleanup_complete=True, phase="CLEANED")

        def mutate(state):
            if all(item["cleanup_complete"] for item in state["slots"]):
                if state["campaign_status"] == "COMPLETE_CLEANUP":
                    state["campaign_status"] = "COMPLETE"
                    state["completed_utc"] = utc_now()
                elif state["campaign_status"] == "BLOCKED_CLEANUP":
                    state["campaign_status"] = "BLOCKED"
                elif state["campaign_status"] == "STOP_REQUESTED":
                    state["campaign_status"] = "STOPPED"

        self.store.mutate(mutate)

    def observe_and_recover(self, slot: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool]:
        queue = get_queue(slot["queue"])
        node = get_node(slot["node"])
        validate_slot_resources(slot, queue, node)
        qstate, initiator = queue_state(queue)
        nstate = "NOT_FOUND" if node is None else str(node.get("state", "UNKNOWN"))
        health = "NOT_FOUND" if node is None else str(node.get("health", "UNKNOWN"))
        observation = {"checked_utc": utc_now(), "queue_state": qstate, "queue_initiator": initiator, "node_state": nstate, "health": health}
        self.set_slot(slot["id"], last_observation=observation, controller_error_count=0)

        preempted = qstate in TERMINAL_QUEUE_STATES or nstate in TERMINAL_NODE_STATES or (
            nstate == "READY" and health.startswith("UNHEALTHY")
        )
        if preempted:
            self.log.write(slot["id"], f"PREEMPTION_DETECTED queue={qstate} initiator={initiator} node={nstate} health={health}")
            delete_exact_slot(slot, self.log)
            self.set_slot(slot["id"], allocation_id=None, bootstrapped_allocation_id=None, phase="REQUESTING_CAPACITY")
            create_exact_slot(slot, self.log)
            return None, None, False

        if queue is None and node is None:
            self.set_slot(slot["id"], phase="REQUESTING_CAPACITY")
            create_exact_slot(slot, self.log)
            return None, None, False

        if node is None or nstate != "READY" or health != "HEALTHY":
            self.set_slot(slot["id"], phase="WAITING_FOR_CAPACITY")
            return queue, node, False

        ips = worker_ips(node)
        if node.get("acceleratorType") != ACCELERATOR or len(ips) != EXPECTED_WORKERS:
            raise RuntimeError(f"Unexpected ready topology on {slot['node']}: accelerator={node.get('acceleratorType')} workers={len(ips)}")
        return queue, node, True

    def slot_loop(self, slot_id: int) -> None:
        self.log.write(slot_id, "SLOT_WORKER_STARTED")
        while True:
            state = self.store.load()
            slot = next(item for item in state["slots"] if item["id"] == slot_id)
            status = state["campaign_status"]
            try:
                if status in {"COMPLETE", "BLOCKED", "STOPPED"}:
                    return
                if status in {"COMPLETE_CLEANUP", "BLOCKED_CLEANUP", "STOP_REQUESTED"}:
                    if not slot["cleanup_complete"]:
                        self.cleanup_slot(slot, stop_training_first=status != "COMPLETE_CLEANUP")
                    time.sleep(5)
                    continue
                if status != "RUNNING":
                    time.sleep(self.poll_seconds)
                    continue

                _, node, ready = self.observe_and_recover(slot)
                if not ready or node is None:
                    time.sleep(self.poll_seconds)
                    continue

                cell = self.claim_or_get_cell(slot_id)
                if cell is None:
                    # The other slot may still own a cell. Keep this healthy
                    # slice idle only until the global queue is truly complete.
                    if all(item["status"] == "COMPLETE" for item in self.store.load()["cells"]):
                        self.store.mutate(lambda s: s.update(campaign_status="COMPLETE_CLEANUP"))
                    time.sleep(self.poll_seconds)
                    continue

                state = self.store.load()
                slot = next(item for item in state["slots"] if item["id"] == slot_id)
                self.bootstrap(slot, node)

                # Inspect committed progress while a process is running, but
                # never classify an in-flight Orbax prefix as abandoned. A
                # partial attempt is rotated only after all training processes
                # are confirmed absent.
                latest = gcs_latest(cell.get("run_uri"))
                if latest is not None:
                    latest_step = assert_latest_for_cell(latest, cell)
                    if latest_step > int(cell.get("latest_step", 0)):
                        self.update_cell(
                            cell["id"], latest_step=latest_step, last_progress_utc=utc_now(),
                            consecutive_no_progress_failures=0,
                        )
                        cell["latest_step"] = latest_step

                commands = ssh_training_commands(node, self.key_path)
                if commands is None:
                    time.sleep(self.poll_seconds)
                    continue
                running = [command for command in commands if command]
                expected = all(cell["config"] in command and cell["exp_name"] in command for command in running)
                if len(running) == EXPECTED_WORKERS and expected:
                    self.set_slot(slot_id, phase="TRAINING")
                    time.sleep(self.poll_seconds)
                    continue
                if cell["latest_step"] >= cell["target_step"] and not running:
                    self.complete_cell(slot_id, cell["id"], cell["latest_step"])
                    continue
                if running:
                    stop_training(node, self.key_path, self.log, slot_id)
                    time.sleep(15)
                    if cell["latest_step"] >= cell["target_step"]:
                        self.complete_cell(slot_id, cell["id"], cell["latest_step"])
                        continue

                cell, resume = self.prepare_attempt(cell)

                # A missing process before target is a recoverable crash.  Stop
                # after three failures without checkpoint progress so a code or
                # data bug cannot burn Spot capacity forever.
                failures = int(cell.get("consecutive_no_progress_failures", 0)) + (1 if cell.get("last_launch_utc") else 0)
                if failures >= 3:
                    self.block_campaign(slot_id, f"cell {cell['id']} crashed three times without checkpoint progress")
                    continue
                self.update_cell(cell["id"], consecutive_no_progress_failures=failures)
                self.launch(slot, cell, resume)
                time.sleep(30)
            except Exception as exc:  # noqa: BLE001 - persist controller failures and retry safely.
                message = re.sub(r"\s+", " ", str(exc))[-1500:]
                errors = int(self.store.load()["slots"][slot_id - 1].get("controller_error_count", 0)) + 1
                self.set_slot(
                    slot_id, last_error=message, phase="CONTROLLER_RETRY", controller_error_count=errors,
                )
                self.log.write(slot_id, f"CONTROLLER_ERROR error={message}")
                if errors >= 5:
                    self.block_campaign(slot_id, f"slot {slot_id} repeated the same controller path five times: {message}")
                time.sleep(max(self.poll_seconds, 30))

    def run_forever(self) -> None:
        if not self.key_path.exists():
            raise RuntimeError(f"Compute Engine SSH key not found: {self.key_path}")
        for path in (self.bootstrap_script, self.launch_script):
            if not path.exists():
                raise RuntimeError(f"Required script missing: {path}")

        proof_proc = run(["gcloud", "storage", "cat", PROOF_URI], timeout=120, check=False)
        if proof_proc.returncode != 0:
            raise RuntimeError(
                f"Topology recovery proof is absent; complete the 100->101 gate before starting formal slots: {PROOF_URI}"
            )
        proof = json.loads(proof_proc.stdout)
        if (
            proof.get("status") != "PASS"
            or proof.get("source_sha256") != SOURCE_SHA256
            or int(proof.get("process_count", -1)) != EXPECTED_WORKERS
            or (proof.get("accelerator_type") not in (None, ACCELERATOR))
        ):
            raise RuntimeError("Topology recovery proof does not match the selected campaign profile")

        def start(state):
            if state["schema"] != "task13-tpu-formal-campaign-v1":
                raise RuntimeError("Unsupported campaign schema")
            if state["campaign_status"] == "INITIALIZED":
                state["campaign_status"] = "RUNNING"
                state["started_utc"] = utc_now()
            elif state["campaign_status"] not in MUTATING_CAMPAIGN_STATES | {"COMPLETE", "BLOCKED", "STOPPED"}:
                raise RuntimeError(f"Unsupported campaign status: {state['campaign_status']}")

        self.store.mutate(start)
        self.log.write("main", f"SUPERVISOR_STARTED profile={self.store.load()['profile']} slots={len(SLOTS)} cells=26")
        threads = [threading.Thread(target=self.slot_loop, args=(slot["id"],), daemon=False) for slot in SLOTS]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.log.write("main", f"SUPERVISOR_EXIT status={self.store.load()['campaign_status']}")


def summarize(state: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for cell in state["cells"]:
        counts[cell["status"]] = counts.get(cell["status"], 0) + 1
    return {
        "campaign": state["campaign"],
        "status": state["campaign_status"],
        "cells": counts,
        "slots": [
            {
                "id": slot["id"], "node": slot["node"], "queue": slot["queue"],
                "phase": slot["phase"], "active_cell": slot["active_cell"],
                "observation": slot["last_observation"], "last_error": slot["last_error"],
            }
            for slot in state["slots"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "run", "status", "request-stop", "self-test"))
    parser.add_argument("--campaign", default="task13-tpu-formal-26-v1")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="v6e8x26")
    parser.add_argument("--state", type=Path, default=Path(__file__).with_name("task13_tpu_formal_campaign_state.json"))
    parser.add_argument("--log", type=Path, default=Path(__file__).with_name("task13_tpu_formal_supervisor.log"))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--approve-auto-mutation", action="store_true")
    args = parser.parse_args()
    store = CampaignStore(args.state.resolve())

    if args.command == "self-test":
        state = initial_state(args.campaign, args.profile)
        assert len(state["cells"]) == 26
        assert [cell["phase"] for cell in state["cells"]].count("p2") == 10
        assert [cell["phase"] for cell in state["cells"]].count("p3") == 16
        assert len({tuple(cell["train_args"]) for cell in state["cells"]}) == 26
        print(json.dumps(summarize(state), indent=2))
        return 0
    if args.command == "init":
        store.initialize(initial_state(args.campaign, args.profile))
        print(json.dumps(summarize(store.load()), indent=2, ensure_ascii=False))
        return 0
    if not args.state.exists():
        raise RuntimeError(f"Campaign state not initialized: {args.state}")
    if args.command == "status":
        print(json.dumps(summarize(store.load()), indent=2, ensure_ascii=False))
        return 0
    if args.command == "request-stop":
        store.mutate(lambda state: state.update(campaign_status="STOP_REQUESTED", stop_requested_utc=utc_now()))
        print("STOP_REQUESTED: the running supervisor will terminate training and delete only the two exact campaign queues.")
        return 0
    if not args.approve_auto_mutation:
        raise RuntimeError("run requires --approve-auto-mutation because it may create/delete TPU queues and launch/resume training")
    if not 15 <= args.poll_seconds <= 3600:
        raise RuntimeError("poll interval must be between 15 and 3600 seconds")
    lock_path = args.state.resolve().with_suffix(args.state.suffix + ".run.lock")
    configure_globals(store.load())
    with single_instance_lock(lock_path):
        Supervisor(args.state.resolve(), args.log.resolve(), args.poll_seconds).run_forever()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(1)
