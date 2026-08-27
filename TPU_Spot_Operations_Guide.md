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
- `orbax-checkpoint==0.11.13`
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

`task13_tpu_v6e16_bootstrap_all.sh` is the intended controller shape. Its
worker stages the immutable input URI to a read-only local cache and verifies
the release byte count before emitting readiness data; its first real
all-worker `READY.json` test is still a required gate. A controller must reject
`READY` plus `UNHEALTHY_MAINTENANCE` before copying or launching anything.

## Inputs, outputs, and checkpointing

- The bootstrap release must copy required inputs from GCS to local disk before
  declaring `READY.json`; verify manifests, make the input copy read-only, and
  keep it separate from outputs. For Task13, the cached input root is
  `/home/tanjunhao/task13_input_assets`.
- Never write into `input_assets`.
- Do **not** write Orbax 0.11.13 checkpoints directly to `gs://`. GCS object
  prefixes are not true empty directories; Orbax fails while initializing its
  temporary prefix before state is written.
- A single-host local Orbax directory/`UPLOAD_COMPLETE` sidecar is valid only
  for single-host slices. It is **not valid for v6e-16**: each TPU VM has a
  separate local filesystem and Orbax writes process-specific state.
- For v6e-16, each process must upload its completed local contribution under
  `checkpoints/<step>/worker-<index>/`, with a byte count and SHA manifest.
  Worker 0 writes `COMMITTED.json` only after all four manifests have been
  verified. Restore only a step with that commit record, materialize the needed
  union on every worker, then run an actual restore plus one training step.
- Until that exact all-worker save/upload/restore test passes, **no 1k or 30k
  Task13 run may start**. Do not pre-create a checkpoint directory unless a
  real `resume=True` policy is selected.

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

On preemption:

1. Recreate the VM and perform the gcloud first-connect/IP update.
2. Run the immutable GCS bootstrap release on every worker; it recreates the
   isolated environment, source, and read-only inputs without 5090.
3. Verify the attached identity's GCS access and all four `READY.json` records.
4. Download only the newest step containing `COMMITTED.json`, validate every
   worker manifest, materialize it on all workers, and use an explicitly
   approved `resume=True` configuration.

## Closeout

After a non-formal feasibility run, do not delete anything until the operator
specifies the retention decision. When cleanup is approved, inventory exact local
and GCS output prefixes first, delete only those paths, verify they are absent,
and preserve input assets, environment/bootstrap scripts, source branch, and this
guide for the next TPU allocation.
