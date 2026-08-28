# Cloud TPU Spot operations guide

This is the durable operational handoff for future TPU work in this repository.
Read it before creating, reconnecting to, or training on a Cloud TPU. It records
the Spot-VM recovery tutorial and the validated Task13 TPU run on 2026-08-26.

## Scope and safety

- Treat a Spot TPU VM as disposable compute. Persist code in Git and persist
  required inputs and completed checkpoints in GCS. Never rely on VM-local state
  after a preemption.
- Keep TPU work on a dedicated branch and output prefix. Do not alter the GPU or
  A100 worktree, environment, lockfile, inputs, or formal experiment outputs.
- Obtain explicit approval before creating a TPU, installing new dependencies,
  launching training, deleting outputs, or using a GPU for restore validation.
- Do not put datasets, checkpoints, model weights, credentials, or generated
  artifacts in Git.

## Current project conventions

| Item | Value / rule |
| --- | --- |
| GCP project | `whyu01` |
| TPU SSH user | `tanjunhao` |
| Last qualified slice | Spot `v6e-4`, `us-east1-d`, one host, topology `2x2`, four JAX devices |
| Latest allocation | Spot `v6e-16`, `us-east1-d`, topology `4x4`, four TPU VM workers; preempted during checkpoint validation on 2026-08-27 |
| Current Task13 side-branch inputs | `gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/input_assets/` |
| Current Task13 side-branch outputs | `gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/runs/` — use a new per-run child |
| TPU branch | `task13-tpu-feasibility-prep` in `Tan20042023/Contact-Demo-Gen` |
| Current checkpoint-contract release | Git `e6dda50`; GCS `bootstrap/e6dda50/source-layout-v2.tar.gz`; SHA-256 `c9847bcd819ba707740c47a3cff8fa1ff1e3e6a57a6a87c7d16be5e6ac930d8c`. **It permits only the checkpoint-contract smoke and clean-resume validation; it is not yet a qualified formal-training release.** |

Do not assume a future Spot allocation has the same IP, zone, topology, device
count, service account, or capacity. `v6e-4` is the last *qualified* recovery
profile. The most recent `v6e-16` run proved four-process JAX initialization,
sharded LeRobot input, and a real 100-step Hammer forward/backward/update loop
(all four workers completed, about 1.3 steps/s after compilation). It did **not**
qualify checkpointing: the slice was preempted during the isolated one-step
local save, before any state was made durable. Do not reuse its single-host
launcher or checkpoint daemon unchanged. A real all-worker checkpoint-to-GCS-
and-restore test remains mandatory before any long run. The TPU-native campaign,
staging, and recovery contract live in `Task13_TPU_Native_Experiment_Plan.md`.

## Spot TPU lifecycle

1. Use any authenticated **control host** (operator laptop or Cloud Shell is
   preferred). It needs `gcloud` access to `whyu01`; it is not part of the
   TPU runtime data or training path. `lab-server-5090` is only a legacy
   fallback because it happened to hold an authenticated gcloud session during
   initial setup.
2. Create the requested Spot TPU VM with an explicit project and zone.
3. Require both `state=READY` **and** `health=HEALTHY`. `READY` with
   `UNHEALTHY_MAINTENANCE` is not usable; do not bootstrap or train in that
   state.
4. **First connect with gcloud**, not bare SSH:

   ```bash
   gcloud compute tpus tpu-vm ssh TPU_NAME --project=whyu01 --zone=ZONE
   ```

   This uploads/generates the Compute Engine SSH key and establishes host keys.
5. Read the new external IP and update the `tanjunhao-tpu` SSH alias only when
   direct interactive SSH is wanted. A controller can use `gcloud compute tpus
   tpu-vm ssh/scp` without this alias. A Spot recreation normally has a
   different IP.

   ```bash
   gcloud compute tpus tpu-vm describe TPU_NAME --project=whyu01 --zone=ZONE \
     --format='value(networkEndpoints[0].accessConfig.externalIp)'
   ```

6. Record state, health, accelerator type, topology, host count, JAX
   `process_count`, device count, disk, RAM, runtime version, and attached
   service account. Require a healthy single-host VM and a device count dividing
   the configured global batch.
7. A preempted Spot TPU cannot be restarted. Recreate it and repeat this section.

All runtime inputs, weights, bootstrap archives, logs, and committed checkpoints
belong in GCS. The TPU must bootstrap directly from GCS using its attached
service account; it must never need a 5090-local path, environment, dataset, or
Git credential to train.

## Authentication and Git

Preferred current path: use the TPU VM's attached default Compute Engine service
account with bucket IAM permissions. Verify it from the TPU with a read/list and
a disposable write/read/delete under the dedicated output prefix. This avoids
long-lived key files altogether.

Fallback from the original tutorial: if the attached account cannot access a
required private bucket, keep a scoped service-account JSON and GitHub credential
JSON only on the operator's local machine; upload them transiently during rebuild,
activate the service account for `gcloud`, set `GOOGLE_APPLICATION_CREDENTIALS`
for Python clients, and keep both files out of Git, logs, documents, and chat.
Prefer least-privilege bucket access and rotate any exposed key.

For GitHub, build an immutable source archive from the TPU-specific branch and
put it in GCS before requesting a Spot slice. A fresh TPU VM downloads that
archive directly from GCS; it never needs GitHub authentication. Never embed a
PAT in a repository URL or script.

## Build the TPU environment

Create a separate Python 3.11 environment on the TPU. Do not reuse a CUDA/GPU
environment or install the repository's `jax[cuda12]` extra. Preserve the core
versions validated here unless a new compatibility review approves a change:

- `jax[tpu]==0.5.3`, matching `jaxlib`/`libtpu`
- `flax==0.10.2`
- `orbax-checkpoint==0.11.23` (the first release with per-process directory
  creation; compatible with pinned `jax==0.5.3`; checkpoint-contract
  qualification remains required)
- `torch==2.7.1`, `torchvision==0.22.1`, `torchcodec==0.5.*`
- `lerobot==0.4.4`
- system package `ffmpeg` (TorchCodec requires its shared libraries)

Install the repository editable without the CUDA JAX dependency, then verify:

1. `jax.devices()` sees the expected device count and a small collective passes.
2. `import lerobot` and `import torchcodec` pass.
3. `ffmpeg -version` works. A missing system FFmpeg makes data-loader workers
   fail at first video decode with `Could not load libtorchcodec`.
4. Import the registered training config without constructing a model or training.

`openpi-client` may advertise a NumPy 1.26.4 requirement while the TPU JAX /
LeRobot stack requires NumPy 2.x. Do not downgrade the TPU core numerical stack
just to make this optional client metadata pass `pip check`; instead document the
exception and validate the imports actually used by the TPU training path.

## v6e-16 code-audit gates

Before requesting the next slice, publish a new immutable source release that
contains all three TPU changes together: `train.py` multi-host initialization
and TPU checkpoint-step labeling; `data_loader.py` deterministic
`DistributedSampler` sharding; and the registered in-package Task13 TPU config.
Do not depend on a manually copied config file or a 5090-local checkout.

The Task13 launcher must `cd "${WORKDIR}/openpi"` before importing
`openpi.training.config`: the inherited DexJoCo config currently resolves
`config.yaml` from the process working directory. Treat an incorrect working
directory as a preflight failure, not a reason to edit GPU configuration.

The following repository-root scripts are historical single-host artifacts and
must not be used for v6e-16: `task13_tpu_phase1_bootstrap.sh`,
`task13_tpu_phase2_project_deps.sh`, `task13_tpu_vm_preflight.sh`,
`task13_tpu_checkpoint_sync_daemon.sh`, and `task13_tpu_train_with_sync.sh`.
They either assert `process_count == 1`, refer to old paths/regions, or implement
the invalid single-host checkpoint protocol. Keep them only as audit history
until an explicit cleanup decision.

`task13_tpu_v6e16_bootstrap_all.ps1` is the currently reviewed controller. Its
worker stages the immutable input URI to a read-only local cache and verifies
the release byte count before emitting readiness data. Bootstrap must be
runtime-neutral: it may import package versions but must not call
`jax.devices()` or `jax.local_device_count()`. The controller validates four
`READY.json` records before a separate, all-worker preflight initializes JAX.
A controller must reject `READY` plus `UNHEALTHY_MAINTENANCE` before copying or
launching anything.

On a VM that already has an environment from an earlier release, do not assume
its editable package points at the newly extracted checkout. Launch against the
immutable checkout explicitly, for example with
`PYTHONPATH="${REPO}/openpi/src${PYTHONPATH:+:${PYTHONPATH}}"`, and record the
loaded `openpi.training.task13_tpu_configs.__file__` in the preflight. A fresh
bootstrap may install the project editable as usual; this rule prevents a
Spot-reuse accident from silently loading stale source.

## Inputs, outputs, and checkpointing

- The bootstrap release must copy required inputs from GCS to local disk before
  declaring `READY.json`; verify manifests, make the input copy read-only, and
  keep it separate from outputs. For Task13, the cached input root is
  `/home/tanjunhao/task13_input_assets`.
- Never write into `input_assets`.
- The sealed Task13 bundle retains its original internal layout. TPU configs
  must use `datasets/task13/v1/lerobot/<task>/<condition>` for LeRobot data and
  `outputs/task13_policy_matrix/v1/assets_full/<task>/<condition>` for assets.
  The placeholder repo ID `local_repo` is valid only when that local root
  contains `meta/info.json`; otherwise LeRobot falls back to the Hugging Face
  Hub. Require an `HF_HUB_OFFLINE=1` construction preflight on every worker.
- Do **not** write the unqualified Task13 multi-host Orbax checkpoint directly
  to `gs://`. The earlier 0.11.13 path failed while initializing its temporary
  prefix; 0.11.24 is incompatible with pinned JAX 0.5.3 during save, and the
  0.11.23 local-contribution candidate is not a substitute for a passing
  recovery test.
- A single-host local Orbax directory/`UPLOAD_COMPLETE` sidecar is valid only
  for single-host slices. It is **not valid for v6e-16**: each TPU VM has a
  separate local filesystem. Do not work around this with per-worker local
  managers or `rsync` shard transport; their finalize barriers are still a
  multi-host protocol and have been observed to stall.
- For v6e-16, all four processes use one native shared GCS Orbax root under
  `runs/<campaign>/<cell>/orbax/<config>/<exp>/<step>/`. Orbax owns distributed
  save/restore and atomic step finalization. Only after every process returns
  from `wait_until_finished()` does worker 0 enumerate the finalized objects
  and write `checkpoints/<step>/COMMITTED.json` plus `LATEST.json`.
- The sealed input-assets bundle remains the sole source for Task13 norm/data
  assets on TPU. Do not run the legacy local-path asset callback inside a
  shared `gs://` Orbax checkpoint; it is intentionally omitted from this
  side-branch checkpoint because resume reads the frozen input bundle again.
- Restore reads the same shared GCS Orbax root directly. It requires `LATEST`
  to name the matching config root, then must complete an actual update on all
  four workers.
- Until the exact all-worker GCS save/restore test passes and writes a
  source-SHA-specific `CHECKPOINT_CONTRACT_PASS.json`, **no 1k or 30k Task13
  run may start**. The launcher rejects a formal configuration without that
  proof and rejects a non-empty initial output prefix. Do not pre-create a
  checkpoint directory unless a real `resume=True` policy is selected.

Use 2,500 steps for the first P1 checkpoint interval; measure save plus upload
time, then freeze either 2,500 or 5,000 for P2 before it starts. For a 100-step
smoke, save the terminal state. The saved directory must be labeled by the
post-update training step (100, not loop index 99).

## Monitoring and recovery

Run training under `nohup` or `tmux`, log into the local output root, and monitor
training and all-worker commit logs. Record compilation time separately from
steady-state step time. The validated `v6e-4` smoke took about two minutes for
first compile, then about 1.8 steps/s; the `v6e-16` Hammer P0 achieved about
1.3 steps/s after compilation. Its checkpoint size and duration are unmeasured.

On the Windows control host, `task13_tpu_ready_watcher.ps1` can persistently
poll the queued resource and node, and writes its durable result to
`task13_tpu_ready_state.json`. Use its `-InstallScheduledTask` mode when the
watch must survive an agent or terminal exit. It emits its ready event only
after `state=READY` **and** `health=HEALTHY`, then continues monitoring for a
later preemption; a state JSON file with `ready: false` is not launch
authorization. `SUSPENDED` with `stateInitiator=SERVICE` is a
terminal service-side deletion: Cloud TPU will not allocate that queue again.
**Current Task13 policy:** do not use `-AutoRecreate`. On a preemption or
`UNHEALTHY_MAINTENANCE`, the watcher records `TPU_PREEMPTED_STOP` and exits;
it must not delete, recreate, or bootstrap a TPU. A later operator decision is
required before any recovery action. It must not recreate merely because a
queue is absent, since that can be intentional cleanup.
The local watcher cannot wake a hosted Codex conversation directly, so a later
agent must read the JSON handoff before continuing.

On preemption, stop. Preserve the durable watcher state and diagnostic logs,
and wait for an explicit operator decision before any new allocation, cleanup,
bootstrap, or resume.

## Closeout

After a non-formal feasibility run, do not delete anything until the operator
specifies the retention decision. When cleanup is approved, inventory exact local
and GCS output prefixes first, delete only those paths, verify they are absent,
and preserve input assets, environment/bootstrap scripts, source branch, and this
guide for the next TPU allocation.
