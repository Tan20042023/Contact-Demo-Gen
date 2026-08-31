# Cloud TPU Spot operations guide

This is the durable repository-wide handoff for Cloud TPU work. Read it before
creating, reconnecting to, recovering, or training on a TPU. Task13 formal
campaign details live only in `Task13_TPU_Formal_Runbook.md`.

## Safety and authority

- TPU VMs are disposable compute. Code is an immutable GCS release; inputs and
  committed checkpoints are in GCS. VM-local files are never recovery state.
- TPU work stays on `task13-tpu-feasibility-prep` and under
  `gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/`. Never modify GPU/A100
  environments, inputs, processes, checkpoints, or results.
- Starting the formal supervisor with `-ApproveAutoMutation` grants standing
  authority only for the campaign state file, its exact `tanjunhao-tpuN` /
  `tanjunhao-tpuN-qr` resources, TPU bootstrap, formal launch/resume, and exact
  queue deletion. It does not authorize other GCP, GCS, Git, GPU, or IAM changes.
- The controller validates exact queue/node ownership, Spot status, accelerator
  type and topology before deletion. It never uses a wildcard resource target.
- A `STOP_REQUESTED`, `BLOCKED`, or `COMPLETE` campaign must not allocate more
  compute. Completion and fail-closed blockage delete the exact campaign queues
  to stop charges.

## Frozen project assets

| Item | Value |
| --- | --- |
| Project / zone | `whyu01` / `us-east1-d` |
| Bucket location | `gs://use1` in `US-EAST1` |
| Inputs | `gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/input_assets/` |
| Runs | `gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/runs/` |
| Source release | `bootstrap/60f7a53/source-layout-v2.tar.gz` |
| Source SHA-256 | `c1e6a96abc645b1d6abb66d4e64ad225946192c80a46e8a63cc97bd812af8c85` |
| Input bytes | `30,696,986,145` |
| TPU runtime | `v2-alpha-tpuv6e` |
| Controller SSH user | `tanjunhao` |

The release contains multi-process JAX initialization, deterministic process
sharding, direct shared-GCS Orbax checkpointing, and registered Task13 configs.
It does not require GitHub, 5090, or a 5090-local path at runtime.

## Validated history

- Spot v6e-16: four TPU VMs, 16 devices, global batch 32. Native shared-GCS
  checkpoint save, clean four-process restore and post-restore step 101 passed.
  Proof:
  `runs/checkpoint_contract_60f7a53_20260828a/hammer_nail_nominal_src/provenance/CHECKPOINT_CONTRACT_PASS.json`.
- P1 on 2026-08-31: Hammer and bimanual nominal seed-42 each completed 1,000
  steps on separate v6e-16 slices. End-to-end runtime was about 8.8 and 9.6
  minutes, including compilation and the terminal checkpoint. Each terminal
  checkpoint was about 9.57 GB and contained 48 objects.
- This qualifies the source and v6e-16 recovery path. Every new formal topology
  must pass its own 100-step save plus clean resume to step 101 before formal
  cells start. No GPU/FSDP numerical-equivalence study is required.

## Lifecycle rules

1. The controller may run on the operator laptop or Cloud Shell. The current
   implementation uses the Windows laptop because gcloud and the Compute Engine
   SSH key are already configured. 5090 is not a runtime dependency.
2. A queue is usable only when the node reports `READY` and `HEALTHY`, the
   accelerator matches the selected profile, and the expected worker/device
   counts pass an all-worker JAX preflight.
3. Spot TPU VMs cannot be restarted after preemption. `SUSPENDED` queued
   resources are not eligible for reallocation. The supervisor deletes the
   exact old queue/node and submits a new exact queued resource.
4. New external IPs receive fresh host-key enrollment. Host-key checking is not
   disabled globally.
5. Bootstrap installs the pinned TPU environment and stages the immutable
   30.7-GB input bundle before declaring ready. It does not initialize JAX or
   launch training.
6. The supervisor never waits for all slots. Every healthy slot independently
   claims the next pending cell.

Official lifecycle reference:
https://docs.cloud.google.com/tpu/docs/spot and
https://docs.cloud.google.com/tpu/docs/queued-resources.

## Runtime environment

- Python 3.11
- `jax[tpu]==0.5.3`
- `flax==0.10.2`
- `orbax-checkpoint==0.11.23`
- `torch==2.7.1`, `torchvision==0.22.1`, `torchcodec==0.5.*`
- `lerobot==0.4.4`
- system FFmpeg

The bootstrap must validate imports, FFmpeg, source/input hashes, available
disk, GCS read/write and offline dataset construction. It must use
`PYTHONPATH` for the immutable checkout so an old editable install cannot win.

## Checkpoint and recovery contract

- All processes in one slice write one native Orbax root in GCS.
- A checkpoint is recoverable only when worker 0 writes `COMMITTED.json` and
  `LATEST.json` after every process returns from `wait_until_finished()`.
- `LATEST.json` must match source SHA, config, experiment name, process count,
  checkpoint URI and completed optimizer step.
- If preemption occurs before the first `LATEST.json`, the partial attempt is
  preserved and the scheduler creates `attempt-002`, `attempt-003`, and so on.
  It never deletes or resumes an uncommitted prefix.
- If `LATEST.json` exists, the same logical cell resumes on the same topology.
- A cell completes only when its committed step reaches 30,000 and all training
  processes exit. `max_to_keep=1` means one large Orbax step per cell remains;
  checkpoint frequency does not imply retaining every historical checkpoint.

## Fail-closed boundaries

Lifecycle loss is retried indefinitely while the campaign is running. A code,
data, proof, authentication, or deterministic training failure is different:
three crashes without checkpoint progress, or five consecutive controller-path
errors, moves the campaign to `BLOCKED_CLEANUP`. Exact queues are deleted and a
human must inspect the state/log before a new campaign is authorized.

The Windows controller must remain powered on, logged in, online, with valid
gcloud credentials. Task Scheduler restarts the supervisor after process exit or
login, but cannot act while the laptop is shut down.

## Canonical files

- `Task13_TPU_Formal_Runbook.md`: experiment and operations contract.
- `task13_tpu_formal_supervisor.py`: scheduler and recovery state machine.
- `task13_tpu_formal_supervisor.ps1`: foreground/background/Scheduled Task UI.
- `task13_tpu_v6e16_bootstrap_all.ps1` and
  `task13_tpu_v6e16_bootstrap_worker.sh`: topology-parameterized bootstrap
  (historical filename retained for release compatibility).
- `task13_tpu_v6e16_launch.ps1`: topology-parameterized launch/resume gate.
- `task13_tpu_v6e16_verify_contract.ps1`: topology-parameterized proof writer.

Historical single-host sync daemons, phase scripts, ad-hoc probes, stop-only
watchers and duplicate plans are obsolete and must not be restored except from
Git history for forensic review.
