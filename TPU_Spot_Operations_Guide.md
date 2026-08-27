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
| Active allocation | Spot `v6e-16`, `us-east1-d`, topology `4x4`, four TPU VM workers; preflight pending |
| Current Task13 side-branch inputs | `gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/input_assets/` |
| Current Task13 side-branch outputs | `gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/runs/` — use a new per-run child |
| TPU branch | `task13-tpu-feasibility-prep` in `Tan20042023/Contact-Demo-Gen` |

Do not assume a future Spot allocation has the same IP, zone, topology, device
count, service account, or capacity. `v6e-4` is the last *qualified* profile.
The active `v6e-16` is a four-worker side-track profile: do not reuse its
single-host launcher or checkpoint daemon unchanged. It needs a fresh
multi-process startup test and a real all-worker checkpoint-to-GCS-and-restore
test before any long run. The TPU-native campaign, staging, and recovery
contract live in `Task13_TPU_Native_Experiment_Plan.md`.

## Spot TPU lifecycle

1. Create the requested Spot TPU VM with an explicit project and zone.
2. **First connect with gcloud**, not bare SSH:

   ```bash
   gcloud compute tpus tpu-vm ssh TPU_NAME --project=whyu01 --zone=ZONE
   ```

   This uploads/generates the Compute Engine SSH key and establishes host keys.
3. Read the new external IP and update the `tanjunhao-tpu` SSH alias. A Spot
   recreation normally has a different IP.

   ```bash
   gcloud compute tpus tpu-vm describe TPU_NAME --project=whyu01 --zone=ZONE \
     --format='value(networkEndpoints[0].accessConfig.externalIp)'
   ```

4. Record state, health, accelerator type, topology, host count, JAX
   `process_count`, device count, disk, RAM, runtime version, and attached
   service account. Require a healthy single-host VM and a device count dividing
   the configured global batch.
5. A preempted Spot TPU cannot be restarted. Recreate it and repeat this section.

When operating through `lab-server-5090`, use its configured `proxy_on` function
before `gcloud` commands if network access fails. Its gcloud session should use
an explicit `--project=whyu01`.

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

For GitHub, use the TPU-specific repository/branch. The 5090 has GitHub SSH
access; a fresh TPU VM may not. It is safe to transfer a prepared clone or bundle
from the 5090, or configure a short-lived authorized GitHub method. Never embed a
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

## Inputs, outputs, and checkpointing

- Copy only required inputs from GCS to local disk before launching a Spot run;
  verify manifests, make the input copy read-only, and keep it separate from
  outputs. For Task13, the cached input root is
  `/home/tanjunhao/task13_input_assets`.
- Never write into `input_assets`.
- Do **not** write Orbax 0.11.13 checkpoints directly to `gs://`. GCS object
  prefixes are not true empty directories; Orbax fails while initializing its
  temporary prefix before state is written.
- Write checkpoints locally. Local POSIX Orbax commits each numbered step by
  atomic directory rename. A visible numeric step directory is therefore the
  completed checkpoint signal.
- Run `openpi/scripts/task13_tpu_checkpoint_sync_daemon.sh` alongside training.
  It copies a completed numeric step to GCS, compares total bytes, then creates
  `UPLOAD_COMPLETE`. Resume only from a remote step that has this marker.
- The launcher `openpi/scripts/task13_tpu_train_with_sync.sh CONFIG_NAME` derives
  the local and GCS paths, starts the sidecar, and runs a registered config. Do
  not pre-create the checkpoint directory: `CheckpointManager` deliberately
  rejects an existing directory unless a real resume/overwrite policy is set.

For long approved technical runs, the TPU-specific save interval is 5,000 steps.
For a 100-step smoke, save the terminal step. Measure actual checkpoint duration
and review the interval if 5,000 steps would risk too much Spot-preemption loss.

## Monitoring and recovery

Run training under `nohup` or `tmux`, log into the local output root, and monitor
both training and sync logs. Record compilation time separately from steady-state
step time. The validated Task13 smoke took about two minutes for first compile,
then about 1.8 steps/s on `v6e-4`; its terminal checkpoint was about 9.56 GB.

On preemption:

1. Recreate the VM and perform the gcloud first-connect/IP update.
2. Recreate the isolated environment and system dependencies.
3. Clone/transfer the pinned branch, restore only needed inputs from GCS, and
   verify the attached identity's GCS access.
4. Download the newest remote step containing `UPLOAD_COMPLETE`, validate it,
   and use an explicitly approved resume configuration.

## Closeout

After a non-formal feasibility run, do not delete anything until the operator
specifies the retention decision. When cleanup is approved, inventory exact local
and GCS output prefixes first, delete only those paths, verify they are absent,
and preserve input assets, environment/bootstrap scripts, source branch, and this
guide for the next TPU allocation.
